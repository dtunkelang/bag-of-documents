#!/usr/bin/env python3
"""ESCI-test eval: base retrieval ± BoD reranker, R@K and nDCG@K.

Pipeline:
  1. Encode all test queries with base MiniLM (subprocess, MPS), retrieve top-K
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

     from FAISS (combined_index_us_minilm).
  2. Save base retrievals to disk.
  3. For each rerank model: subprocess loads model, reads retrievals in batches,
     encodes query + candidates per batch on MPS, computes new top-K, writes
     reordered pids to disk.
  4. Master reads outputs and computes R@K / nDCG@K.

This batched-with-disk-staging design keeps memory bounded for full-75K runs.
"""

import argparse
import json
import math
import os
import statistics
import subprocess
import time
from collections import defaultdict

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INDEX_DIR = os.path.join(SCRIPT_DIR, "combined_index_us_minilm")


def encode_subproc(model_path, items_path, out_path, device="auto", batch_size=128):
    """Run a subprocess that encodes items from JSON file, writes vectors to .npy."""
    code = f"""
import os, json, sys
os.environ['TOKENIZERS_PARALLELISM']='false'
os.environ['OMP_NUM_THREADS']='1'
os.chdir({SCRIPT_DIR!r})
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
device = {device!r}
if device == 'auto':
    device = 'mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu')
print(f'  device: {{device}}', flush=True)
m = SentenceTransformer({model_path!r}, device=device)
with open({items_path!r}) as f:
    items = json.load(f)
v = m.encode(items, normalize_embeddings=True, batch_size={batch_size}, show_progress_bar=False)
np.save({out_path!r}, np.asarray(v, dtype=np.float32))
print('  encoded', len(items), 'items')
"""
    subprocess.check_call([".venv/bin/python", "-c", code])
    return np.load(out_path)


def rerank_subproc(
    model_path, queries_path, candidates_path, out_path, k_retrieve, k_top, device="auto", batch=256
):
    """Subprocess loads model, reads queries + candidates, rerank-encodes in batches,
    writes top-K positions per query to .npy.
    """
    code = f"""
import os, json, sys
os.environ['TOKENIZERS_PARALLELISM']='false'
os.environ['OMP_NUM_THREADS']='1'
os.chdir({SCRIPT_DIR!r})
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
device = {device!r}
if device == 'auto':
    device = 'mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu')
print(f'  device: {{device}}', flush=True)
m = SentenceTransformer({model_path!r}, device=device)
with open({queries_path!r}) as f:
    queries = json.load(f)
with open({candidates_path!r}) as f:
    candidates_per_q = json.load(f)
n = len(queries)
k_retrieve = {k_retrieve}
k_top = {k_top}
out = np.zeros((n, k_top), dtype=np.int32)
B = {batch}
import time
t0 = time.time()
for start in range(0, n, B):
    end = min(start + B, n)
    q_batch = queries[start:end]
    c_batch = candidates_per_q[start:end]
    flat = [c for sub in c_batch for c in sub]
    qv = m.encode(q_batch, normalize_embeddings=True, batch_size=128, show_progress_bar=False)
    cv = m.encode(flat, normalize_embeddings=True, batch_size=128, show_progress_bar=False)
    qv = np.asarray(qv, dtype=np.float32)
    cv = np.asarray(cv, dtype=np.float32)
    offset = 0
    for i, sub in enumerate(c_batch):
        n_sub = len(sub)
        if n_sub == 0:
            offset += n_sub
            continue
        sub_vecs = cv[offset:offset + n_sub]
        sims = sub_vecs @ qv[i]
        order = np.argsort(-sims)[:k_top]
        # pad if fewer candidates than k_top
        for j, pos in enumerate(order):
            out[start + i, j] = int(pos)
        if n_sub < k_top:
            for j in range(n_sub, k_top):
                out[start + i, j] = -1
        offset += n_sub
    if (start // B) % 10 == 0:
        elapsed = time.time() - t0
        rate = (end / elapsed) if elapsed > 0 else 0
        eta = (n - end) / rate if rate > 0 else 0
        print(f'    {{end}}/{{n}}  ({{end/n*100:.1f}}%)  rate={{rate:.0f}} q/s  eta={{eta:.0f}}s', flush=True)
np.save({out_path!r}, out)
print('  wrote rerank positions, shape:', out.shape)
"""
    subprocess.check_call([".venv/bin/python", "-c", code])
    return np.load(out_path)


def load_qrels(path):
    by_query = defaultdict(dict)
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            by_query[r["query_id"]][r["product_id"]] = r["relevance"]
    return dict(by_query)


def load_queries(path):
    out = {}
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            out[d["query_id"]] = d["query"]
    return out


GAIN = {3: 1.0, 2: 0.1, 1: 0.01, 0: 0.0}


def metrics_for_query(retrieved_pids, qrels, k=10):
    relevant_e_s = {pid for pid, g in qrels.items() if g >= 2}
    if not relevant_e_s:
        return None
    top_k = retrieved_pids[:k]
    recall = sum(1 for pid in top_k if pid in relevant_e_s) / len(relevant_e_s)
    gains = [GAIN.get(qrels.get(pid, 0), 0.0) for pid in top_k]
    dcg = sum(g / math.log2(i + 2) for i, g in enumerate(gains))
    ideal_grades = sorted(qrels.values(), reverse=True)[:k]
    ideal_gains = [GAIN.get(g, 0) for g in ideal_grades]
    idcg = sum(g / math.log2(i + 2) for i, g in enumerate(ideal_gains))
    return {"recall": recall, "ndcg": dcg / idcg if idcg > 0 else 0.0}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--rerank-model",
        action="append",
        default=[],
        help="Path to a BoD model to use as reranker (can be passed multiple times).",
    )
    parser.add_argument("--limit", type=int, default=0, help="Limit queries (0=all)")
    parser.add_argument("--k-retrieve", type=int, default=100, help="Top-K from base FAISS")
    parser.add_argument("--k-eval", type=int, default=10, help="K for R@K / nDCG@K")
    parser.add_argument("--workdir", default="/tmp/rerank_eval", help="Scratch directory")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.workdir, exist_ok=True)

    print("loading queries + qrels + product titles...", flush=True)
    queries_all = load_queries(os.path.join(SCRIPT_DIR, "esci_us_data/test_queries.jsonl"))
    qrels = load_qrels(os.path.join(SCRIPT_DIR, "esci_us_data/test_qrels.jsonl"))

    with open(os.path.join(SCRIPT_DIR, "esci_us_data/product_ids.json")) as f:
        esci_pids = json.load(f)
    with open(os.path.join(SCRIPT_DIR, "esci_us_data/titles.json")) as f:
        esci_titles_arr = json.load(f)

    with open(os.path.join(INDEX_DIR, "titles.json")) as f:
        index_titles = json.load(f)
    title_to_pid = {t: p for p, t in zip(esci_pids, esci_titles_arr)}
    faiss_pos_to_pid = [title_to_pid.get(t) for t in index_titles]
    misses = sum(1 for p in faiss_pos_to_pid if p is None)
    print(f"  {len(index_titles):,} index products; {misses:,} not in pid map", flush=True)

    qids = []
    for qid in queries_all:
        if qid not in qrels:
            continue
        if any(g >= 2 for g in qrels[qid].values()):
            qids.append(qid)
    if args.limit:
        import random

        rng = random.Random(args.seed)
        rng.shuffle(qids)
        qids = qids[: args.limit]
    print(f"  {len(qids):,} eval queries (with E or S in qrels)", flush=True)

    queries = [queries_all[qid] for qid in qids]
    queries_path = os.path.join(args.workdir, "queries.json")
    with open(queries_path, "w") as f:
        json.dump(queries, f)

    # Step 1: base encoding -> FAISS retrieval -> save candidates per query
    print("\nstep 1: base query encoding (subprocess, MPS)...", flush=True)
    t0 = time.time()
    base_q_vecs = encode_subproc(
        "all-MiniLM-L6-v2", queries_path, os.path.join(args.workdir, "base_q_vecs.npy")
    )
    print(f"  done in {time.time() - t0:.0f}s", flush=True)

    print("step 1b: FAISS retrieval...", flush=True)
    import faiss

    idx = faiss.read_index(os.path.join(INDEX_DIR, "index.faiss"))
    if hasattr(idx, "hnsw"):
        idx.hnsw.efSearch = 128
    elif hasattr(idx, "nprobe"):
        idx.nprobe = 32
    t0 = time.time()
    D, I = idx.search(base_q_vecs, args.k_retrieve)
    print(f"  retrieved in {time.time() - t0:.0f}s", flush=True)

    # Save base retrievals: candidate titles per query (for reranker subprocs) and pids (for metrics)
    base_retrieved_titles = []
    base_retrieved_pids = []
    for qi in range(len(queries)):
        ts = []
        ps = []
        for fpos in I[qi]:
            if fpos < 0:
                continue
            ts.append(index_titles[fpos])
            ps.append(faiss_pos_to_pid[fpos])
        base_retrieved_titles.append(ts)
        base_retrieved_pids.append(ps)

    cands_path = os.path.join(args.workdir, "candidates.json")
    with open(cands_path, "w") as f:
        json.dump(base_retrieved_titles, f)

    # Step 2: per-rerank reordering
    rerank_orderings_pids = {}  # name -> list[list[pid]] reordered top-k_eval
    for model_path in args.rerank_model:
        name = os.path.basename(model_path.rstrip("/"))
        out_npy = os.path.join(args.workdir, f"rerank_{name}.npy")
        print(f"\nstep 2: rerank with {name}...", flush=True)
        t0 = time.time()
        positions = rerank_subproc(
            model_path,
            queries_path,
            cands_path,
            out_npy,
            args.k_retrieve,
            args.k_eval,
        )
        print(f"  rerank done in {time.time() - t0:.0f}s", flush=True)

        # Translate positions -> pids
        new_orderings = []
        for qi in range(len(queries)):
            row = positions[qi]
            pids = []
            for pos in row:
                if pos < 0:
                    break
                pids.append(base_retrieved_pids[qi][int(pos)])
            new_orderings.append(pids)
        rerank_orderings_pids[name] = new_orderings

    # Step 3: metrics
    def aggregate(orderings):
        recalls = []
        ndcgs = []
        for qi, qid in enumerate(qids):
            m = metrics_for_query(orderings[qi], qrels[qid], k=args.k_eval)
            if m:
                recalls.append(m["recall"])
                ndcgs.append(m["ndcg"])
        return {
            "n": len(recalls),
            "R@k": statistics.mean(recalls) if recalls else 0.0,
            "nDCG@k": statistics.mean(ndcgs) if ndcgs else 0.0,
        }

    print(f"\n\n{'=' * 70}")
    print(f"=== Results: R@{args.k_eval} (E+S relevant), nDCG@{args.k_eval} ===")
    print(f"{'=' * 70}")
    print(f"{'Model':<45} {'n':>6} {'R@k':>8} {'nDCG@k':>9}")
    print("-" * 75)
    base_res = aggregate(base_retrieved_pids)
    print(
        f"{'base (MiniLM)':<45} {base_res['n']:>6} {base_res['R@k']:>8.2%} {base_res['nDCG@k']:>9.4f}"
    )
    for name, ordered in rerank_orderings_pids.items():
        r = aggregate(ordered)
        delta_r = r["R@k"] - base_res["R@k"]
        delta_n = r["nDCG@k"] - base_res["nDCG@k"]
        print(
            f"{'base + rerank(' + name + ')':<45} {r['n']:>6} {r['R@k']:>8.2%} "
            f"{r['nDCG@k']:>9.4f}  (ΔR@k {delta_r:+.2%}, ΔnDCG {delta_n:+.4f})"
        )


if __name__ == "__main__":
    main()
