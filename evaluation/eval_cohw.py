#!/usr/bin/env python3
"""Evaluate a coherence-weighted variant of rerank_A as both single-stream
and as a drop-in replacement for rerank_A in the 3-way uniform-mean ensemble.

Usage:
  python evaluation/eval_cohw.py --model query_model_us_full_6m_mnrl_cohw \
      --vecs combined_index_us_minilm/rerank_K_cohw_alpha1.vecs.fp16.npy \
      --tag alpha1
"""

import argparse
import json
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


def metrics_for(retrieved_pids, qrels_q, k_eval=10):
    e_pids = {p for p, g in qrels_q.items() if g == 3}
    es_pids = {p for p, g in qrels_q.items() if g >= 2}
    out = {}
    if es_pids:
        top_k = retrieved_pids[:k_eval]
        out["recall"] = sum(1 for p in top_k if p in es_pids) / len(es_pids)
    if e_pids:
        out["e_at_1"] = 1.0 if retrieved_pids and retrieved_pids[0] in e_pids else 0.0
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="cohw model dir")
    ap.add_argument("--vecs", required=True, help="cohw catalog vecs path")
    ap.add_argument("--tag", required=True, help="output tag (e.g. alpha1)")
    args = ap.parse_args()

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

    print("\nencoding queries with rerank_A, B, G, cohw...", flush=True)
    t0 = time.time()
    qv_a = encode_subproc(os.path.join(SCRIPT_DIR, "query_model_us_full_6m_mnrl"), queries)
    print(f"  rerank_A: {time.time() - t0:.0f}s", flush=True)
    t0 = time.time()
    qv_b = encode_subproc(os.path.join(SCRIPT_DIR, "query_model_us_qrels_mnrl_hardneg"), queries)
    print(f"  rerank_B: {time.time() - t0:.0f}s", flush=True)
    t0 = time.time()
    qv_g = encode_subproc(os.path.join(SCRIPT_DIR, "query_model_us_esci_supervised"), queries)
    print(f"  rerank_G: {time.time() - t0:.0f}s", flush=True)
    t0 = time.time()
    qv_h = encode_subproc(os.path.join(SCRIPT_DIR, args.model), queries)
    print(f"  cohw ({args.tag}): {time.time() - t0:.0f}s", flush=True)

    pv_a = np.load(os.path.join(INDEX_DIR, "rerank_A.vecs.fp16.npy")).astype(np.float32)
    pv_b = np.load(os.path.join(INDEX_DIR, "rerank_B.vecs.fp16.npy")).astype(np.float32)
    pv_g = np.load(os.path.join(INDEX_DIR, "rerank_G.vecs.fp16.npy")).astype(np.float32)
    pv_h = np.load(args.vecs).astype(np.float32)

    bm25_path = os.path.join(INDEX_DIR, "bm25s_top200.npy")
    if not os.path.exists(bm25_path):
        bm25_path = os.path.join(INDEX_DIR, "bm25_top200.npy")
    bm25_top50 = np.load(bm25_path)[:, :50]

    K = 10
    print("\nrerank top-10 streams...", flush=True)
    a_top = rerank_top50(qv_a, pv_a, bm25_top50, K)
    h_top = rerank_top50(qv_h, pv_h, bm25_top50, K)
    k_uniform = rerank_top50_ensemble([qv_a, qv_b, qv_g], [pv_a, pv_b, pv_g], bm25_top50, K)
    k_swap = rerank_top50_ensemble([qv_h, qv_b, qv_g], [pv_h, pv_b, pv_g], bm25_top50, K)

    def to_pids(ord_array):
        return [
            [faiss_pos_to_pid[p] if 0 <= p < len(faiss_pos_to_pid) else None for p in row]
            for row in ord_array
        ]

    setups = {
        "BM25_top50+rerank_A": to_pids(a_top),
        f"BM25_top50+cohw_{args.tag}": to_pids(h_top),
        "uniform_K_3way": to_pids(k_uniform),
        f"uniform_K_swap_cohw_{args.tag}": to_pids(k_swap),
    }

    summary = {}
    per_query = {}
    for name, ords in setups.items():
        recalls, e1s = [], []
        pq_r, pq_e = [], []
        for qi, qid in enumerate(qids):
            m = metrics_for(ords[qi], qrels[qid])
            r = m.get("recall", -1)
            e = m.get("e_at_1", -1)
            pq_r.append(r if r >= 0 else -1)
            pq_e.append(e if e >= 0 else -1)
            if r >= 0:
                recalls.append(r)
            if e >= 0:
                e1s.append(e)
        summary[name] = {
            "R@10": statistics.mean(recalls) if recalls else 0,
            "E@1": statistics.mean(e1s) if e1s else 0,
            "n_es": len(recalls),
            "n_e": len(e1s),
        }
        per_query[name] = {"recall": pq_r, "e_at_1": pq_e}

    print(f"\n=== SUMMARY (cohw {args.tag}) ===", flush=True)
    name_pad = max(len(n) for n in summary)
    print(f"{'setup':<{name_pad}}  R@10    E@1     n_es")
    for name in setups:
        m = summary[name]
        print(f"{name:<{name_pad}}  {m['R@10'] * 100:5.2f}  {m['E@1'] * 100:5.2f}  {m['n_es']}")

    out_path = f"/tmp/cohw_{args.tag}_per_query.json"
    with open(out_path, "w") as f:
        json.dump({"qids": qids, "summary": summary, "per_query": per_query}, f)
    print(f"\nsaved to {out_path}", flush=True)


if __name__ == "__main__":
    main()
