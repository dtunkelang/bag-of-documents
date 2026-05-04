#!/usr/bin/env python3
"""Evaluate FAISS-hardneg-trained MiniLM vs no-hardneg and qrels-hardneg.

Setups:
  A0. base MiniLM retriever                                   (15.60 baseline)
  A1. 6M-MNRL retriever (rerank_A, no explicit hardneg)       (current prod)
  B1. qrels-hardneg retriever (rerank_B)                      (existing variant)
  K1. FAISS-hardneg retriever (rerank_K_faisshn)              (NEW)
  A2. BM25 top-50 + rerank_A single
  B2. BM25 top-50 + rerank_B single
  K2. BM25 top-50 + rerank_K_faisshn single                   (NEW)

Saves per-query metrics to /tmp/faisshn_skip30_probe_per_query.json for bootstrap.
"""

import json
import math
import os
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


def brute_top_k(q_vecs, p_vecs, k):
    n = q_vecs.shape[0]
    out_idx = np.zeros((n, k), dtype=np.int64)
    BATCH = 64
    for start in range(0, n, BATCH):
        end = min(start + BATCH, n)
        sims = q_vecs[start:end] @ p_vecs.T
        top = np.argpartition(-sims, k, axis=1)[:, :k]
        for i in range(end - start):
            row = sims[i, top[i]]
            order = np.argsort(-row)
            out_idx[start + i] = top[i, order]
        if (start // BATCH) % 20 == 0:
            print(f"    brute_top_k {end}/{n}", flush=True)
    return out_idx


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
        top3 = retrieved_pids[:3]
        out["e_at_3"] = sum(1 for p in top3 if p in e_pids) / min(3, len(e_pids))
    return out


def aggregate(orderings_by_setup, qids, qrels_by_qid):
    summary = {}
    per_query = {}
    for name, orderings in orderings_by_setup.items():
        recalls, ndcgs, e1s, e3s = [], [], [], []
        pq_recalls, pq_e1 = [], []
        for qi, qid in enumerate(qids):
            m = metrics_for(orderings[qi], qrels_by_qid[qid])
            r = m.get("recall", None)
            pq_recalls.append(r if r is not None else -1)
            pq_e1.append(m.get("e_at_1", -1))
            if "recall" in m:
                recalls.append(m["recall"])
                ndcgs.append(m["ndcg"])
            if "e_at_1" in m:
                e1s.append(m["e_at_1"])
                e3s.append(m["e_at_3"])
        summary[name] = {
            "n_es": len(recalls),
            "n_e": len(e1s),
            "R@10": statistics.mean(recalls) if recalls else 0,
            "nDCG@10": statistics.mean(ndcgs) if ndcgs else 0,
            "E@1": statistics.mean(e1s) if e1s else 0,
            "E@3": statistics.mean(e3s) if e3s else 0,
        }
        per_query[name] = {"recall": pq_recalls, "e_at_1": pq_e1}
    return summary, per_query


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

    K_RETRIEVE = 100
    K_EVAL = 10

    print("\nencoding queries...", flush=True)
    t0 = time.time()
    qv_base = encode_subproc("all-MiniLM-L6-v2", queries)
    print(f"  base MiniLM: {time.time() - t0:.0f}s", flush=True)
    t0 = time.time()
    qv_a = encode_subproc(os.path.join(SCRIPT_DIR, "query_model_us_full_6m_mnrl"), queries)
    print(f"  rerank_A (6M-MNRL): {time.time() - t0:.0f}s", flush=True)
    t0 = time.time()
    qv_b = encode_subproc(os.path.join(SCRIPT_DIR, "query_model_us_qrels_mnrl_hardneg"), queries)
    print(f"  rerank_B (qrels-hn): {time.time() - t0:.0f}s", flush=True)
    t0 = time.time()
    qv_k = encode_subproc(
        os.path.join(SCRIPT_DIR, "query_model_us_full_6m_mnrl_faisshn_skip30"), queries
    )
    print(f"  rerank_K_faisshn: {time.time() - t0:.0f}s", flush=True)

    print("\nloading product matrices...", flush=True)
    pv_a = np.load(os.path.join(INDEX_DIR, "rerank_A.vecs.fp16.npy")).astype(np.float32)
    print(f"  rerank_A: {pv_a.shape}", flush=True)
    pv_b = np.load(os.path.join(INDEX_DIR, "rerank_B.vecs.fp16.npy")).astype(np.float32)
    print(f"  rerank_B: {pv_b.shape}", flush=True)
    pv_k = np.load(os.path.join(INDEX_DIR, "rerank_K_faisshn_skip30.vecs.fp16.npy")).astype(
        np.float32
    )
    print(f"  rerank_K_faisshn: {pv_k.shape}", flush=True)

    bm25_top200_path = os.path.join(INDEX_DIR, "bm25s_top200.npy")
    if not os.path.exists(bm25_top200_path):
        bm25_top200_path = os.path.join(INDEX_DIR, "bm25_top200.npy")
    bm25_top200 = np.load(bm25_top200_path)
    bm25_top50 = bm25_top200[:, :50]

    print("\nretrieval setups (top-100)...", flush=True)
    print("  A0 (base MiniLM)...", flush=True)
    base_pos = brute_top_k(qv_base, pv_a, K_RETRIEVE)
    print("  A1 (rerank_A)...", flush=True)
    a_pos = brute_top_k(qv_a, pv_a, K_RETRIEVE)
    print("  B1 (rerank_B)...", flush=True)
    b_pos = brute_top_k(qv_b, pv_b, K_RETRIEVE)
    print("  K1 (rerank_K_faisshn)...", flush=True)
    k_pos = brute_top_k(qv_k, pv_k, K_RETRIEVE)

    print("\nrerank setups (BM25 top-50 reranked)...", flush=True)
    a_rerank = rerank_top50(qv_a, pv_a, bm25_top50, K_EVAL)
    b_rerank = rerank_top50(qv_b, pv_b, bm25_top50, K_EVAL)
    k_rerank = rerank_top50(qv_k, pv_k, bm25_top50, K_EVAL)

    def to_pids(orderings):
        out = []
        for row in orderings:
            out.append(
                [faiss_pos_to_pid[p] if p >= 0 and p < len(faiss_pos_to_pid) else None for p in row]
            )
        return out

    orderings = {
        "A0_base_minilm_retriever": to_pids(base_pos[:, :K_EVAL]),
        "A1_rerank_A_retriever": to_pids(a_pos[:, :K_EVAL]),
        "B1_rerank_B_qrelshn_retriever": to_pids(b_pos[:, :K_EVAL]),
        "K1_rerank_K_faisshn_retriever": to_pids(k_pos[:, :K_EVAL]),
        "A2_bm25top50_rerank_A": to_pids(a_rerank),
        "B2_bm25top50_rerank_B_qrelshn": to_pids(b_rerank),
        "K2_bm25top50_rerank_K_faisshn": to_pids(k_rerank),
    }

    summary, per_query = aggregate(orderings, qids, qrels)

    print("\n=== SUMMARY ===", flush=True)
    name_pad = max(len(n) for n in summary)
    print(f"{'setup':<{name_pad}}  R@10    nDCG@10  E@1     E@3     n_es")
    for name, m in summary.items():
        print(
            f"{name:<{name_pad}}  {m['R@10'] * 100:5.2f}  {m['nDCG@10'] * 100:6.2f}   "
            f"{m['E@1'] * 100:5.2f}  {m['E@3'] * 100:5.2f}  {m['n_es']}"
        )

    out_path = "/tmp/faisshn_skip30_probe_per_query.json"
    with open(out_path, "w") as f:
        json.dump({"qids": qids, "summary": summary, "per_query": per_query}, f)
    print(f"\nsaved per-query to {out_path}", flush=True)


if __name__ == "__main__":
    main()
