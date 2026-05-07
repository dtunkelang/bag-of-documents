#!/usr/bin/env python3
"""Measure base R@10 distribution on a corpus — no BoD training needed.

Companion to evaluation/diagnose_lift.py. The framework's prediction
"stronger base -> smaller base-blind subset" can be tested with the base
encoder alone: encode catalog + queries, compute per-query R@10, bucket
by difficulty.

Useful for: quick "what would the base-blind size look like if we swapped
the base encoder?" probes without paying the BoD training cost.

Usage:
    python evaluation/measure_base_difficulty.py \\
        --catalog combined_index_us_minilm/titles.json \\
        --product-ids esci_us_data/product_ids.json \\
        --qrels esci_us_data/test_qrels.jsonl --min-relevance 3 \\
        --queries esci_us_data/test_queries.jsonl \\
        --encoder BAAI/bge-base-en-v1.5 \\
        --vecs-cache combined_index_us_minilm/bge_base_catalog.vecs.fp16.npy \\
        --label esci_us_bge_base
"""

import argparse
import json
import os
from collections import defaultdict

import numpy as np
import torch
from sentence_transformers import SentenceTransformer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--catalog", required=True)
    ap.add_argument("--product-ids", required=True)
    ap.add_argument("--qrels", required=True)
    ap.add_argument("--queries", required=True)
    ap.add_argument("--min-relevance", type=int, default=1)
    ap.add_argument("--encoder", required=True)
    ap.add_argument("--vecs-cache", default=None, help="optional .npy cache for catalog vecs")
    ap.add_argument("--label", default="corpus")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--chunk", type=int, default=512)
    args = ap.parse_args()

    print("loading data...", flush=True)
    with open(args.catalog) as f:
        titles = json.load(f)
    with open(args.product_ids) as f:
        pids = json.load(f)
    pid_to_idx = {p: i for i, p in enumerate(pids)}
    queries_by_qid = {}
    with open(args.queries) as f:
        for line in f:
            d = json.loads(line)
            queries_by_qid[d["query_id"]] = d["query"]
    pos = defaultdict(set)
    field = None
    with open(args.qrels) as f:
        for line in f:
            r = json.loads(line)
            if field is None:
                field = "product_id" if "product_id" in r else "doc_id"
            if r[field] not in pid_to_idx:
                continue
            if r["relevance"] < args.min_relevance:
                continue
            pos[r["query_id"]].add(pid_to_idx[r[field]])

    qids = sorted(queries_by_qid)
    queries = [queries_by_qid[q] for q in qids]
    n_pos_q = sum(1 for v in pos.values() if v)
    print(f"  catalog={len(pids):,}  queries={len(queries):,}  pos-bearing={n_pos_q:,}", flush=True)

    device = "mps" if torch.backends.mps.is_available() else "cpu"

    if args.vecs_cache and os.path.exists(args.vecs_cache):
        print(f"loading cached catalog vecs {args.vecs_cache}...", flush=True)
        pv = np.load(args.vecs_cache).astype(np.float32)
    else:
        print(f"encoding catalog with {args.encoder} on {device}...", flush=True)
        m = SentenceTransformer(args.encoder, device=device)
        pv = m.encode(
            titles, normalize_embeddings=True, batch_size=128, show_progress_bar=True
        ).astype(np.float32)
        if args.vecs_cache:
            np.save(args.vecs_cache, pv.astype(np.float16))
            print(f"  cached at {args.vecs_cache}", flush=True)
        del m
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

    print("encoding queries...", flush=True)
    m = SentenceTransformer(args.encoder, device=device)
    qv = m.encode(
        queries, normalize_embeddings=True, batch_size=256, show_progress_bar=True
    ).astype(np.float32)
    del m
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

    print("matmul + bucketing...", flush=True)
    per_q = []
    n_chunks = (len(qids) + args.chunk - 1) // args.chunk
    for ci, start in enumerate(range(0, len(qids), args.chunk)):
        end = min(start + args.chunk, len(qids))
        sims = qv[start:end] @ pv.T
        topk = np.argpartition(-sims, args.k, axis=1)[:, : args.k]
        del sims
        for j, gi in enumerate(range(start, end)):
            qid = qids[gi]
            g = pos.get(qid, set())
            if not g:
                continue
            h = len({int(x) for x in topk[j]} & g)
            per_q.append((qid, queries[gi], len(g), h))
        if (ci + 1) % 10 == 0 or ci + 1 == n_chunks:
            print(f"  {ci + 1}/{n_chunks} chunks", flush=True)

    print("\n" + "=" * 78)
    print(
        f"Base difficulty distribution — {args.label}  "
        f"(R@{args.k} on relevance>={args.min_relevance})"
    )
    print("=" * 78)

    buckets = defaultdict(list)
    for r in per_q:
        ratio = r[3] / r[2] if r[2] else 0
        if ratio == 0:
            b = "0.0 (base misses entirely)"
        elif ratio < 0.5:
            b = "0.0-0.5"
        elif ratio < 1.0:
            b = "0.5-1.0"
        else:
            b = "1.0 (base perfect)"
        buckets[b].append(r)

    total = sum(len(v) for v in buckets.values())
    print(f"  {'bucket':<28} {'n':>8} {'mean R@10':>12}")
    for k_ in [
        "0.0 (base misses entirely)",
        "0.0-0.5",
        "0.5-1.0",
        "1.0 (base perfect)",
    ]:
        if k_ in buckets:
            rows = buckets[k_]
            n = len(rows)
            mean_r = sum(r[3] / r[2] if r[2] else 0 for r in rows) / n
            pct = 100.0 * n / total
            print(f"  {k_:<28} {n:>4,} ({pct:>4.1f}%)  {mean_r:>12.3f}")

    overall = sum(r[3] / r[2] if r[2] else 0 for r in per_q) / len(per_q)
    print(f"\n  overall R@{args.k} (n={len(per_q):,}): {overall:.3f}")


if __name__ == "__main__":
    main()
