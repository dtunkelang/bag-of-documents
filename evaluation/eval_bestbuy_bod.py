#!/usr/bin/env python3
"""Evaluate base vs BoD-trained retrieval on BestBuy ACM holdout.

Reads bestbuy_acm_data/{product_ids.json, titles.json, holdout_queries.jsonl,
holdout_qrels.jsonl} and a pair of sentence-transformers models (base + BoD).
Encodes the catalog once per model, encodes the holdout queries, and reports
R@10 and E@1 (top-1 hit on any positive SKU) for each.

The base catalog is cached at base_catalog.vecs.fp16.npy by build_bestbuy_bags.py.
The BoD catalog is encoded fresh from the trained model directory.

Validates the CHS prediction: BestBuy's SCHS=0.525 falls in the GREEN band
(predicted BoD-positive). A positive R@10 lift here is empirical evidence
that the CHS metric rank-orders BoD-readiness as expected.

Usage:
    python evaluation/eval_bestbuy_bod.py --bod-model query_model_bestbuy_bod
"""

import argparse
import json
import os
import time
from collections import defaultdict

import numpy as np
import torch
from sentence_transformers import SentenceTransformer


def encode_catalog(model_path, titles, batch_size=128):
    device = (
        "mps"
        if torch.backends.mps.is_available()
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"  loading {model_path} on {device}...", flush=True)
    m = SentenceTransformer(model_path, device=device)
    t0 = time.time()
    v = m.encode(
        titles, normalize_embeddings=True, batch_size=batch_size, show_progress_bar=True
    ).astype(np.float32)
    print(f"  encoded {len(titles):,} titles in {time.time() - t0:.0f}s", flush=True)
    return m, v


def encode_queries(model, queries, batch_size=256):
    t0 = time.time()
    v = model.encode(
        queries, normalize_embeddings=True, batch_size=batch_size, show_progress_bar=True
    ).astype(np.float32)
    print(f"  encoded {len(queries):,} queries in {time.time() - t0:.0f}s", flush=True)
    return v


def evaluate(qv, pv, qids, qrels_by_qid, pids, k=10):
    """Returns (R@k, E@1) over queries with at least one positive in catalog."""
    pid_to_idx = {p: i for i, p in enumerate(pids)}
    pos_idxs = []
    for qid in qids:
        idxs = {pid_to_idx[p] for p in qrels_by_qid.get(qid, []) if p in pid_to_idx}
        pos_idxs.append(idxs)

    n_eval = sum(1 for s in pos_idxs if s)
    print(f"  evaluating {n_eval:,}/{len(qids):,} queries with >=1 positive in catalog", flush=True)

    recall_hits = 0
    e1_hits = 0
    chunk = 1024
    for start in range(0, len(qids), chunk):
        end = min(start + chunk, len(qids))
        sims = qv[start:end] @ pv.T
        top_k = np.argpartition(-sims, k, axis=1)[:, :k]
        for j, gi in enumerate(range(start, end)):
            pos = pos_idxs[gi]
            if not pos:
                continue
            row = sims[j]
            top_k_row = top_k[j]
            top_k_sorted = top_k_row[np.argsort(-row[top_k_row])]
            if int(top_k_sorted[0]) in pos:
                e1_hits += 1
            if any(int(i) in pos for i in top_k_sorted):
                recall_hits += 1

    return recall_hits / n_eval, e1_hits / n_eval, n_eval


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="bestbuy_acm_data")
    ap.add_argument("--base-model", default="all-MiniLM-L6-v2")
    ap.add_argument("--bod-model", required=True, help="Path to BoD-trained model dir.")
    ap.add_argument("--k", type=int, default=10)
    args = ap.parse_args()

    # Load data.
    print("loading data...", flush=True)
    with open(os.path.join(args.data_dir, "product_ids.json")) as f:
        pids = json.load(f)
    with open(os.path.join(args.data_dir, "titles.json")) as f:
        titles = json.load(f)
    qids = []
    queries = []
    with open(os.path.join(args.data_dir, "holdout_queries.jsonl")) as f:
        for line in f:
            d = json.loads(line)
            qids.append(d["query_id"])
            queries.append(d["query"])
    qrels_by_qid = defaultdict(list)
    with open(os.path.join(args.data_dir, "holdout_qrels.jsonl")) as f:
        for line in f:
            r = json.loads(line)
            qrels_by_qid[r["query_id"]].append(r["product_id"])
    print(
        f"  catalog={len(pids):,}  holdout queries={len(qids):,}  "
        f"qrels rows={sum(len(v) for v in qrels_by_qid.values()):,}",
        flush=True,
    )

    # Try to use the cached base catalog.
    base_cache = os.path.join(args.data_dir, "base_catalog.vecs.fp16.npy")
    if os.path.exists(base_cache):
        print(f"\nloading cached base catalog from {base_cache}...", flush=True)
        base_pv = np.load(base_cache).astype(np.float32)
        base_model = SentenceTransformer(
            args.base_model,
            device="mps" if torch.backends.mps.is_available() else "cpu",
        )
    else:
        print(f"\nencoding base catalog with {args.base_model}...", flush=True)
        base_model, base_pv = encode_catalog(args.base_model, titles)

    print("encoding base queries...", flush=True)
    base_qv = encode_queries(base_model, queries)
    del base_model
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

    print("\nevaluating base...", flush=True)
    base_r, base_e1, n_eval = evaluate(base_qv, base_pv, qids, qrels_by_qid, pids, k=args.k)
    print(f"  base: R@{args.k}={base_r:.4f}  E@1={base_e1:.4f}  (n={n_eval:,})", flush=True)
    del base_pv, base_qv

    print(f"\nencoding BoD catalog with {args.bod_model}...", flush=True)
    bod_model, bod_pv = encode_catalog(args.bod_model, titles)
    print("encoding BoD queries...", flush=True)
    bod_qv = encode_queries(bod_model, queries)

    print("\nevaluating BoD...", flush=True)
    bod_r, bod_e1, _ = evaluate(bod_qv, bod_pv, qids, qrels_by_qid, pids, k=args.k)
    print(f"  BoD:  R@{args.k}={bod_r:.4f}  E@1={bod_e1:.4f}  (n={n_eval:,})", flush=True)

    # Headline.
    print("\n" + "=" * 64)
    print(f"BestBuy ACM holdout retrieval ({n_eval:,} queries)")
    print("=" * 64)
    print(f"  base  ({args.base_model:>40s})  R@{args.k}={base_r:.4f}  E@1={base_e1:.4f}")
    print(f"  BoD   ({args.bod_model:>40s})  R@{args.k}={bod_r:.4f}  E@1={bod_e1:.4f}")
    print(
        f"  delta:                                         "
        f"R@{args.k}={bod_r - base_r:+.4f}  E@1={bod_e1 - base_e1:+.4f}",
        flush=True,
    )
    print(
        f"\nCHS prediction (SCHS=0.525, GREEN): BoD positive — {'CONFIRMED' if bod_r > base_r else 'REJECTED'}"
    )


if __name__ == "__main__":
    main()
