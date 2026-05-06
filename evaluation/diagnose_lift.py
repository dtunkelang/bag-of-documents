#!/usr/bin/env python3
"""Generic per-bucket BoD lift diagnostic — works on any (corpus, qrels, BoD) tuple.

Generalizes diagnose_bestbuy_lift.py / diagnose_esci_lift.py. Buckets each
query by base R@10 difficulty so we can decompose lift into base-blind
subset size × rescue rate (the two-factor framework from CHS_RESULTS.md
Pattern 5).

Examples:
    # ESCI-US (already cached)
    python evaluation/diagnose_lift.py \\
        --catalog combined_index_us_minilm/titles.json \\
        --product-ids esci_us_data/product_ids.json \\
        --qrels esci_us_data/test_qrels.jsonl --min-relevance 3 \\
        --queries esci_us_data/test_queries.jsonl \\
        --base-vecs combined_index_us_minilm/base_catalog.vecs.fp16.npy \\
        --bod-vecs combined_index_us_minilm/rerank_A.vecs.fp16.npy \\
        --base-model all-MiniLM-L6-v2 \\
        --bod-model query_model_us_full_6m_mnrl \\
        --label esci_us

    # NFCorpus (small, no cached vecs — encode fresh)
    python evaluation/diagnose_lift.py \\
        --catalog nfcorpus_data/titles.json \\
        --product-ids nfcorpus_data/product_ids.json \\
        --qrels nfcorpus_data/test_qrels.jsonl --min-relevance 1 \\
        --queries nfcorpus_data/test_queries.jsonl \\
        --base-model all-MiniLM-L6-v2 \\
        --bod-model query_model_nfcorpus_bod \\
        --label nfcorpus
"""

import argparse
import json
import os
from collections import defaultdict

import numpy as np
import torch
from sentence_transformers import SentenceTransformer


def encode(model, texts, batch_size=128):
    return model.encode(
        texts, normalize_embeddings=True, batch_size=batch_size, show_progress_bar=True
    ).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--catalog", required=True, help="titles.json")
    ap.add_argument("--product-ids", required=True, help="product_ids.json (or doc_ids)")
    ap.add_argument("--qrels", required=True)
    ap.add_argument("--queries", required=True)
    ap.add_argument("--min-relevance", type=int, default=1)
    ap.add_argument("--base-model", required=True)
    ap.add_argument("--bod-model", required=True)
    ap.add_argument("--base-vecs", default=None, help="optional cached base catalog .npy")
    ap.add_argument("--bod-vecs", default=None, help="optional cached BoD catalog .npy")
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
    print(
        f"  catalog={len(pids):,}  queries={len(queries):,}  pos-bearing={n_pos_q:,}",
        flush=True,
    )

    device = "mps" if torch.backends.mps.is_available() else "cpu"

    if args.base_vecs and os.path.exists(args.base_vecs):
        print(f"loading cached base vecs {args.base_vecs}...", flush=True)
        base_pv = np.load(args.base_vecs).astype(np.float32)
    else:
        print(f"encoding catalog with {args.base_model}...", flush=True)
        m = SentenceTransformer(args.base_model, device=device)
        base_pv = encode(m, titles)
        del m

    if args.bod_vecs and os.path.exists(args.bod_vecs):
        print(f"loading cached BoD vecs {args.bod_vecs}...", flush=True)
        bod_pv = np.load(args.bod_vecs).astype(np.float32)
    else:
        print(f"encoding catalog with {args.bod_model}...", flush=True)
        m = SentenceTransformer(args.bod_model, device=device)
        bod_pv = encode(m, titles)
        del m

    print("encoding queries (base)...", flush=True)
    base = SentenceTransformer(args.base_model, device=device)
    base_qv = encode(base, queries, batch_size=256)
    del base

    print("encoding queries (BoD)...", flush=True)
    bod = SentenceTransformer(args.bod_model, device=device)
    bod_qv = encode(bod, queries, batch_size=256)
    del bod
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

    print("matmul + bucketing...", flush=True)
    per_q = []
    n_chunks = (len(qids) + args.chunk - 1) // args.chunk
    for ci, start in enumerate(range(0, len(qids), args.chunk)):
        end = min(start + args.chunk, len(qids))
        bsim = base_qv[start:end] @ base_pv.T
        btopk = np.argpartition(-bsim, args.k, axis=1)[:, : args.k]
        del bsim
        dsim = bod_qv[start:end] @ bod_pv.T
        dtopk = np.argpartition(-dsim, args.k, axis=1)[:, : args.k]
        del dsim
        for j, gi in enumerate(range(start, end)):
            qid = qids[gi]
            g = pos.get(qid, set())
            if not g:
                continue
            bh = len({int(x) for x in btopk[j]} & g)
            dh = len({int(x) for x in dtopk[j]} & g)
            per_q.append((qid, queries[gi], len(g), bh, dh))
        if (ci + 1) % 10 == 0 or ci + 1 == n_chunks:
            print(f"  {ci + 1}/{n_chunks} chunks", flush=True)

    def agg(rows):
        if not rows:
            return None
        n = len(rows)
        base_r = sum(r[3] / r[2] if r[2] else 0 for r in rows) / n
        bod_r = sum(r[4] / r[2] if r[2] else 0 for r in rows) / n
        return n, base_r, bod_r, bod_r - base_r

    print("\n" + "=" * 78)
    print(f"BoD lift breakdown — {args.label}  (R@{args.k} on relevance>={args.min_relevance})")
    print("=" * 78)

    by_diff = defaultdict(list)
    for r in per_q:
        ratio = r[3] / r[2] if r[2] else 0
        if ratio == 0:
            bucket = "0.0 (base misses entirely)"
        elif ratio < 0.5:
            bucket = "0.0-0.5"
        elif ratio < 1.0:
            bucket = "0.5-1.0"
        else:
            bucket = "1.0 (base perfect)"
        by_diff[bucket].append(r)
    total = sum(len(v) for v in by_diff.values())
    print(f"  {'bucket':<28} {'n':>8} {'base':>8} {'BoD':>8} {'Δ':>8}")
    for k_ in [
        "0.0 (base misses entirely)",
        "0.0-0.5",
        "0.5-1.0",
        "1.0 (base perfect)",
    ]:
        if k_ in by_diff:
            a = agg(by_diff[k_])
            pct = 100.0 * a[0] / total
            print(f"  {k_:<28} {a[0]:>4,} ({pct:>4.1f}%)  {a[1]:>8.3f} {a[2]:>8.3f} {a[3]:>+8.3f}")

    a = agg(per_q)
    print(f"\n  overall (n={a[0]:,}): base={a[1]:.3f}  BoD={a[2]:.3f}  Δ={a[3]:+.3f}")


if __name__ == "__main__":
    main()
