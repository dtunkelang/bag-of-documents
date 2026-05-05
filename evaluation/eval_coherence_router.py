#!/usr/bin/env python3
"""Coherence-routing probe: route per query among rerank_A/B/G by candidate-set
coherence rather than query-text features.

Hypothesis: the +1.97pp R@10 oracle-routing headroom over uniform_K that the
query-text-feature router couldn't capture lives in the candidate set, not in
the query. Specifically, an encoder's top-K coherence (mean pairwise cosine
of its top-K retrieved products) is a measure of how strongly the cluster
hypothesis holds for that query under that encoder. Pick the encoder where it
holds most strongly.

Setups (all over BM25 top-50 reranked):
  always_A / always_B / always_G  : single-encoder lower bounds
  uniform_K                       : 3-way uniform mean (current shipped)
  coherence_route                 : per query, pick encoder with highest top-K coherence
  oracle                          : per query, pick encoder with highest E@1+0.001*R@10 (upper bound)

Same 70/30 router-train/router-eval split as eval_learned_router.py
(though coherence routing has no training step; we just evaluate on the eval split
to keep results comparable).
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
device = 'mps' if torch.backends.mps.is_available() else 'cpu'
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
        topk = cand[order]
        if len(topk) < k:
            topk = np.concatenate([topk, np.full(k - len(topk), -1, dtype=np.int64)])
        out_idx[i] = topk
    return out_idx


def rerank_top50_ensemble(q_vecs_list, p_vecs_list, bm25_top50, k):
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


def topk_coherence(top_idx_row, p_vecs):
    """Mean pairwise cosine among the top-K product vectors."""
    valid = top_idx_row[top_idx_row >= 0]
    if len(valid) < 2:
        return 0.0
    v = p_vecs[valid]
    sims = v @ v.T
    iu = np.triu_indices(len(sims), k=1)
    return float(sims[iu].mean())


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

    K_EVAL = 10
    K_COH = 10  # top-K window for coherence calculation

    print("\nencoding queries...", flush=True)
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

    bm25_path = os.path.join(INDEX_DIR, "bm25s_top200.npy")
    if not os.path.exists(bm25_path):
        bm25_path = os.path.join(INDEX_DIR, "bm25_top200.npy")
    bm25_top50 = np.load(bm25_path)[:, :50]

    print("\nrerank top-10 per encoder...", flush=True)
    a_top = rerank_top50(qv_a, pv_a, bm25_top50, K_EVAL)
    b_top = rerank_top50(qv_b, pv_b, bm25_top50, K_EVAL)
    g_top = rerank_top50(qv_g, pv_g, bm25_top50, K_EVAL)
    k_top = rerank_top50_ensemble([qv_a, qv_b, qv_g], [pv_a, pv_b, pv_g], bm25_top50, K_EVAL)

    print("\ncomputing per-query coherence under each encoder...", flush=True)
    n = len(qids)
    coh_a = np.zeros(n)
    coh_b = np.zeros(n)
    coh_g = np.zeros(n)
    for i in range(n):
        coh_a[i] = topk_coherence(a_top[i, :K_COH], pv_a)
        coh_b[i] = topk_coherence(b_top[i, :K_COH], pv_b)
        coh_g[i] = topk_coherence(g_top[i, :K_COH], pv_g)

    print(
        f"  coherence stats: A mean={coh_a.mean():.3f} B mean={coh_b.mean():.3f} "
        f"G mean={coh_g.mean():.3f}",
        flush=True,
    )

    # Routing decisions
    coh_stack = np.stack([coh_a, coh_b, coh_g], axis=1)  # (n, 3)
    coh_pred = np.argmax(coh_stack, axis=1)
    print(
        f"  coherence-router pred dist: A={np.mean(coh_pred == 0):.3f} "
        f"B={np.mean(coh_pred == 1):.3f} G={np.mean(coh_pred == 2):.3f}",
        flush=True,
    )

    # Per-query metrics for each encoder
    def to_pids_row(row):
        return [faiss_pos_to_pid[p] if 0 <= p < len(faiss_pos_to_pid) else None for p in row]

    print("\ncomputing per-query metrics...", flush=True)
    pq = []
    for i, qid in enumerate(qids):
        ma = metrics_for(to_pids_row(a_top[i]), qrels[qid])
        mb = metrics_for(to_pids_row(b_top[i]), qrels[qid])
        mg = metrics_for(to_pids_row(g_top[i]), qrels[qid])
        mk = metrics_for(to_pids_row(k_top[i]), qrels[qid])
        pq.append(
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
                "coh_a": coh_a[i],
                "coh_b": coh_b[i],
                "coh_g": coh_g[i],
                "coh_pred": int(coh_pred[i]),
            }
        )

    # Use the same eval split as the learned-router probe for direct comparability
    rng = random.Random(42)
    perm = list(range(n))
    rng.shuffle(perm)
    split = int(0.7 * n)
    eval_idx = sorted(perm[split:])
    print(f"\neval split: {len(eval_idx)} queries", flush=True)

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

    def aggregate_setup(predicted_choice, eval_idx_):
        recalls, e1s = [], []
        for ei, qi in enumerate(eval_idx_):
            choice = int(predicted_choice[ei])
            keys = [("A_R", "A_E1"), ("B_R", "B_E1"), ("G_R", "G_E1")][choice]
            r = pq[qi][keys[0]]
            e = pq[qi][keys[1]]
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

    n_eval = len(eval_idx)
    summary = {}
    summary["always_A"] = aggregate_setup(np.zeros(n_eval, dtype=int), eval_idx)
    summary["always_B"] = aggregate_setup(np.ones(n_eval, dtype=int), eval_idx)
    summary["always_G"] = aggregate_setup(2 * np.ones(n_eval, dtype=int), eval_idx)
    summary["uniform_K"] = {
        "R@10": statistics.mean([pq[i]["K_R"] for i in eval_idx if pq[i]["K_R"] >= 0]),
        "E@1": statistics.mean([pq[i]["K_E1"] for i in eval_idx if pq[i]["K_E1"] >= 0]),
        "n_es": sum(1 for i in eval_idx if pq[i]["K_R"] >= 0),
        "n_e": sum(1 for i in eval_idx if pq[i]["K_E1"] >= 0),
    }
    coh_pred_eval = np.array([pq[i]["coh_pred"] for i in eval_idx])
    summary["coherence_route"] = aggregate_setup(coh_pred_eval, eval_idx)

    oracle_pred = np.array([label_for(pq[i]) for i in eval_idx])
    summary["oracle"] = aggregate_setup(oracle_pred, eval_idx)

    print(f"\n=== SUMMARY (eval split, n={n_eval}) ===", flush=True)
    name_pad = max(len(s) for s in summary)
    print(f"{'setup':<{name_pad}}  R@10    E@1     n_es")
    for name in [
        "always_A",
        "always_B",
        "always_G",
        "uniform_K",
        "coherence_route",
        "oracle",
    ]:
        m = summary[name]
        print(f"{name:<{name_pad}}  {m['R@10'] * 100:5.2f}  {m['E@1'] * 100:5.2f}  {m['n_es']}")

    # Diagnostic: agreement between coherence-route and oracle
    print("\ncoherence-route vs oracle agreement on eval split:", flush=True)
    agree = sum(1 for i in eval_idx if pq[i]["coh_pred"] == label_for(pq[i]))
    print(f"  {agree}/{n_eval} = {agree / n_eval:.1%}", flush=True)

    out = {
        "summary": summary,
        "eval_qids": [qids[i] for i in eval_idx],
        "per_query_eval": [pq[i] for i in eval_idx],
    }
    with open("/tmp/coherence_router_probe.json", "w") as f:
        json.dump(out, f)
    print("\nsaved per-query data to /tmp/coherence_router_probe.json", flush=True)


if __name__ == "__main__":
    main()
