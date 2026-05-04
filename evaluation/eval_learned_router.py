#!/usr/bin/env python3
"""Learned router probe: predict best of {A, B, G} per query from query embedding.

Tests whether the +1.76pp R@10 oracle headroom over the K (uniform-mean)
ensemble is reachable from base-MiniLM query embeddings — cheap-feature
routing already failed (README "what didn't work").

Setups (all over BM25 top-50 reranked):
  A2: rerank_A (6M-MNRL) single
  B2: rerank_B (qrels-hardneg) single
  G2: rerank_G (ESCI-supervised) single
  K:  uniform mean(A, B, G) — current shipped 3-way ensemble
  always_A: always pick rerank_A (lower bound for routing)
  oracle:   per-query, pick whichever single has highest E@1 (then R@10) — upper bound
  knn:      kNN-vote router (K=10) on base-MiniLM embedding
  mlp:      MLP router on base-MiniLM embedding

70/30 router-train/router-eval split of 22,458 ESCI test queries.
"""

import json
import math
import os
import random
import statistics
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INDEX_DIR = os.path.join(SCRIPT_DIR, "combined_index_us_minilm")

GAIN = {3: 1.0, 2: 0.1, 1: 0.01, 0: 0.0}


def encode_subproc(model_path, queries):
    code = f"""
import os, json, sys
os.environ['TOKENIZERS_PARALLELISM']='false'
os.environ['OMP_NUM_THREADS']='1'
os.chdir({SCRIPT_DIR!r})
import numpy as np, torch
from sentence_transformers import SentenceTransformer
device = 'mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu')
m = SentenceTransformer({model_path!r}, device=device)
qs = json.loads(sys.stdin.read())
v = m.encode(qs, normalize_embeddings=True, batch_size=128, show_progress_bar=False)
sys.stdout.write(json.dumps(np.asarray(v, dtype=np.float32).tolist()))
"""
    out = subprocess.check_output(
        [".venv/bin/python", "-c", code],
        input=json.dumps(queries),
        stderr=subprocess.DEVNULL,
        text=True,
    )
    return np.array(json.loads(out), dtype=np.float32)


def rerank_top50(q_vecs, p_vecs, bm25_top50, k):
    n = q_vecs.shape[0]
    out_idx = np.zeros((n, k), dtype=np.int64)
    for i in range(n):
        cand = bm25_top50[i]
        cand = cand[cand >= 0]
        if len(cand) == 0:
            out_idx[i, :] = -1
            continue
        sims = q_vecs[i] @ p_vecs[cand].T
        order = np.argsort(-sims)[:k]
        topk_in_cand = cand[order]
        if len(topk_in_cand) < k:
            pad = np.full(k - len(topk_in_cand), -1, dtype=np.int64)
            topk_in_cand = np.concatenate([topk_in_cand, pad])
        out_idx[i] = topk_in_cand
    return out_idx


def rerank_top50_ensemble(q_vecs_list, p_vecs_list, bm25_top50, k):
    """Mean of three rerank streams (sumsim K)."""
    n = q_vecs_list[0].shape[0]
    out_idx = np.zeros((n, k), dtype=np.int64)
    for i in range(n):
        cand = bm25_top50[i]
        cand = cand[cand >= 0]
        if len(cand) == 0:
            out_idx[i, :] = -1
            continue
        sims = sum(qv[i] @ pv[cand].T for qv, pv in zip(q_vecs_list, p_vecs_list)) / len(
            q_vecs_list
        )
        order = np.argsort(-sims)[:k]
        topk = cand[order]
        if len(topk) < k:
            topk = np.concatenate([topk, np.full(k - len(topk), -1, dtype=np.int64)])
        out_idx[i] = topk
    return out_idx


def metrics_for(retrieved_pids, qrels_q, k_eval=10):
    e_pids = {p for p, g in qrels_q.items() if g == 3}
    es_pids = {p for p, g in qrels_q.items() if g >= 2}
    out = {}
    if es_pids:
        top_k = retrieved_pids[:k_eval]
        out["recall"] = sum(1 for p in top_k if p in es_pids) / len(es_pids)
        gains = [GAIN.get(qrels_q.get(p, 0), 0.0) for p in top_k]
        dcg = sum(g / math.log2(i + 2) for i, g in enumerate(gains))
        ideal = sorted(qrels_q.values(), reverse=True)[:k_eval]
        idcg = sum(GAIN.get(g, 0) / math.log2(i + 2) for i, g in enumerate(ideal))
        out["ndcg"] = dcg / idcg if idcg > 0 else 0.0
    if e_pids:
        out["e_at_1"] = 1.0 if retrieved_pids and retrieved_pids[0] in e_pids else 0.0
    return out


def main():
    print("loading qrels + queries + product map...", flush=True)
    qrels = defaultdict(dict)
    with open(os.path.join(SCRIPT_DIR, "esci_us_data/test_qrels.jsonl")) as f:
        for line in f:
            r = json.loads(line)
            qrels[r["query_id"]][r["product_id"]] = r["relevance"]
    qrels = dict(qrels)
    queries_all = {}
    with open(os.path.join(SCRIPT_DIR, "esci_us_data/test_queries.jsonl")) as f:
        for line in f:
            d = json.loads(line)
            queries_all[d["query_id"]] = d["query"]
    with open(os.path.join(SCRIPT_DIR, "esci_us_data/product_ids.json")) as f:
        esci_pids = json.load(f)
    with open(os.path.join(SCRIPT_DIR, "esci_us_data/titles.json")) as f:
        esci_titles_arr = json.load(f)
    with open(os.path.join(INDEX_DIR, "titles.json")) as f:
        index_titles = json.load(f)
    title_to_pid = {t: p for p, t in zip(esci_pids, esci_titles_arr)}
    faiss_pos_to_pid = [title_to_pid.get(t) for t in index_titles]

    qids = [qid for qid in queries_all if qid in qrels and any(g >= 2 for g in qrels[qid].values())]
    queries = [queries_all[qid] for qid in qids]
    print(f"  {len(qids):,} eval queries", flush=True)

    print("\nencoding queries (4 models)...", flush=True)
    t0 = time.time()
    qv_base = encode_subproc("all-MiniLM-L6-v2", queries)
    print(f"  base MiniLM (router features): {time.time() - t0:.0f}s", flush=True)
    t0 = time.time()
    qv_a = encode_subproc(os.path.join(SCRIPT_DIR, "query_model_us_full_6m_mnrl"), queries)
    print(f"  rerank_A: {time.time() - t0:.0f}s", flush=True)
    t0 = time.time()
    qv_b = encode_subproc(os.path.join(SCRIPT_DIR, "query_model_us_qrels_mnrl_hardneg"), queries)
    print(f"  rerank_B: {time.time() - t0:.0f}s", flush=True)
    t0 = time.time()
    qv_g = encode_subproc(os.path.join(SCRIPT_DIR, "query_model_us_esci_supervised"), queries)
    print(f"  rerank_G: {time.time() - t0:.0f}s", flush=True)

    print("\nloading product matrices...", flush=True)
    pv_a = np.load(os.path.join(INDEX_DIR, "rerank_A.vecs.fp16.npy")).astype(np.float32)
    pv_b = np.load(os.path.join(INDEX_DIR, "rerank_B.vecs.fp16.npy")).astype(np.float32)
    pv_g = np.load(os.path.join(INDEX_DIR, "rerank_G.vecs.fp16.npy")).astype(np.float32)

    bm25_top200_path = os.path.join(INDEX_DIR, "bm25s_top200.npy")
    if not os.path.exists(bm25_top200_path):
        bm25_top200_path = os.path.join(INDEX_DIR, "bm25_top200.npy")
    bm25_top50 = np.load(bm25_top200_path)[:, :50]

    print("\nrerank top-10 per encoder (BM25 top-50)...", flush=True)
    K_EVAL = 10
    a_pos = rerank_top50(qv_a, pv_a, bm25_top50, K_EVAL)
    b_pos = rerank_top50(qv_b, pv_b, bm25_top50, K_EVAL)
    g_pos = rerank_top50(qv_g, pv_g, bm25_top50, K_EVAL)
    k_pos = rerank_top50_ensemble([qv_a, qv_b, qv_g], [pv_a, pv_b, pv_g], bm25_top50, K_EVAL)

    def to_pids_row(row):
        return [faiss_pos_to_pid[p] if 0 <= p < len(faiss_pos_to_pid) else None for p in row]

    print("\ncomputing per-query metrics for A2/B2/G2/K...", flush=True)
    per_query = []
    for i, qid in enumerate(qids):
        ma = metrics_for(to_pids_row(a_pos[i]), qrels[qid])
        mb = metrics_for(to_pids_row(b_pos[i]), qrels[qid])
        mg = metrics_for(to_pids_row(g_pos[i]), qrels[qid])
        mk = metrics_for(to_pids_row(k_pos[i]), qrels[qid])
        per_query.append(
            {
                "qid": qid,
                "A_R": ma.get("recall", -1),
                "B_R": mb.get("recall", -1),
                "G_R": mg.get("recall", -1),
                "K_R": mk.get("recall", -1),
                "A_E1": ma.get("e_at_1", -1),
                "B_E1": mb.get("e_at_1", -1),
                "G_E1": mg.get("e_at_1", -1),
                "K_E1": mk.get("e_at_1", -1),
            }
        )

    rng = random.Random(42)
    perm = list(range(len(qids)))
    rng.shuffle(perm)
    split = int(0.7 * len(qids))
    train_idx = sorted(perm[:split])
    eval_idx = sorted(perm[split:])
    print(f"\nsplit: {len(train_idx)} train, {len(eval_idx)} eval", flush=True)

    def label_for(rec):
        scores = [
            (rec["A_E1"] if rec["A_E1"] >= 0 else 0)
            + 0.001 * (rec["A_R"] if rec["A_R"] >= 0 else 0),
            (rec["B_E1"] if rec["B_E1"] >= 0 else 0)
            + 0.001 * (rec["B_R"] if rec["B_R"] >= 0 else 0),
            (rec["G_E1"] if rec["G_E1"] >= 0 else 0)
            + 0.001 * (rec["G_R"] if rec["G_R"] >= 0 else 0),
        ]
        return int(np.argmax(scores))

    train_labels = np.array([label_for(per_query[i]) for i in train_idx])
    print(
        f"  train label dist: A={np.mean(train_labels == 0):.3f} "
        f"B={np.mean(train_labels == 1):.3f} G={np.mean(train_labels == 2):.3f}",
        flush=True,
    )

    train_X = qv_base[train_idx]
    eval_X = qv_base[eval_idx]
    eval_labels_oracle = np.array([label_for(per_query[i]) for i in eval_idx])

    def aggregate_setup(predicted_choice, eval_idx_):
        recalls = []
        e1s = []
        for ei, qi in enumerate(eval_idx_):
            choice = int(predicted_choice[ei])
            metric_keys = [("A_R", "A_E1"), ("B_R", "B_E1"), ("G_R", "G_E1")][choice]
            r = per_query[qi][metric_keys[0]]
            e = per_query[qi][metric_keys[1]]
            if r >= 0:
                recalls.append(r)
            if e >= 0:
                e1s.append(e)
        return {
            "R@10": statistics.mean(recalls) if recalls else 0,
            "E@1": statistics.mean(e1s) if e1s else 0,
            "n_es": len(recalls),
            "n_e": len(e1s),
        }

    print("\nrouter setups on eval split...", flush=True)
    n_eval = len(eval_idx)
    summary = {}

    summary["always_A"] = aggregate_setup(np.zeros(n_eval, dtype=int), eval_idx)
    summary["always_B"] = aggregate_setup(np.ones(n_eval, dtype=int), eval_idx)
    summary["always_G"] = aggregate_setup(2 * np.ones(n_eval, dtype=int), eval_idx)
    summary["oracle"] = aggregate_setup(eval_labels_oracle, eval_idx)
    summary["uniform_K"] = {
        "R@10": statistics.mean(
            [per_query[i]["K_R"] for i in eval_idx if per_query[i]["K_R"] >= 0]
        ),
        "E@1": statistics.mean(
            [per_query[i]["K_E1"] for i in eval_idx if per_query[i]["K_E1"] >= 0]
        ),
        "n_es": sum(1 for i in eval_idx if per_query[i]["K_R"] >= 0),
        "n_e": sum(1 for i in eval_idx if per_query[i]["K_E1"] >= 0),
    }

    print("\nkNN router (K=10)...", flush=True)
    train_norm = train_X / (np.linalg.norm(train_X, axis=1, keepdims=True) + 1e-8)
    eval_norm = eval_X / (np.linalg.norm(eval_X, axis=1, keepdims=True) + 1e-8)
    sims = eval_norm @ train_norm.T
    knn_K = 10
    top_idx = np.argpartition(-sims, knn_K, axis=1)[:, :knn_K]
    knn_pred = np.zeros(n_eval, dtype=int)
    for i in range(n_eval):
        votes = train_labels[top_idx[i]]
        counts = np.bincount(votes, minlength=3)
        knn_pred[i] = int(np.argmax(counts))
    summary["knn_router_k10"] = aggregate_setup(knn_pred, eval_idx)

    print("\nMLP router...", flush=True)
    import torch
    import torch.nn as nn

    device = (
        "mps"
        if torch.backends.mps.is_available()
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    Xt = torch.from_numpy(train_X).float().to(device)
    yt = torch.from_numpy(train_labels).long().to(device)
    Xe = torch.from_numpy(eval_X).float().to(device)

    torch.manual_seed(42)
    model = nn.Sequential(
        nn.Linear(qv_base.shape[1], 128),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(64, 3),
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    BATCH = 256
    n_train = Xt.shape[0]
    for epoch in range(30):
        perm_t = torch.randperm(n_train, device=device)
        total_loss = 0
        for s in range(0, n_train, BATCH):
            idx = perm_t[s : s + BATCH]
            logits = model(Xt[idx])
            loss = loss_fn(logits, yt[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item() * idx.shape[0]
        if (epoch + 1) % 5 == 0:
            print(f"  epoch {epoch + 1}/30 loss={total_loss / n_train:.4f}", flush=True)

    model.eval()
    with torch.no_grad():
        eval_logits = model(Xe).cpu().numpy()
    mlp_pred = np.argmax(eval_logits, axis=1)
    print(
        f"  MLP pred dist: A={np.mean(mlp_pred == 0):.3f} "
        f"B={np.mean(mlp_pred == 1):.3f} G={np.mean(mlp_pred == 2):.3f}",
        flush=True,
    )
    summary["mlp_router"] = aggregate_setup(mlp_pred, eval_idx)

    print(f"\n=== SUMMARY (eval split, n={n_eval}) ===", flush=True)
    name_pad = max(len(n) for n in summary)
    print(f"{'setup':<{name_pad}}  R@10    E@1     n_es")
    for name in [
        "always_A",
        "always_B",
        "always_G",
        "uniform_K",
        "knn_router_k10",
        "mlp_router",
        "oracle",
    ]:
        m = summary[name]
        print(f"{name:<{name_pad}}  {m['R@10'] * 100:5.2f}  {m['E@1'] * 100:5.2f}  {m['n_es']}")

    out = {
        "summary": summary,
        "eval_qids": [qids[i] for i in eval_idx],
        "per_query_eval": [per_query[i] for i in eval_idx],
        "mlp_pred": mlp_pred.tolist(),
        "knn_pred": knn_pred.tolist(),
        "oracle_labels": eval_labels_oracle.tolist(),
    }
    with open("/tmp/router_probe.json", "w") as f:
        json.dump(out, f)
    print("\nsaved per-query data to /tmp/router_probe.json", flush=True)


if __name__ == "__main__":
    main()
