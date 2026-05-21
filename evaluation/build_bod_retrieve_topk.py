#!/usr/bin/env python3
"""Build a top-K candidate-pool cache using a bi-encoder retriever instead of
BM25. Mirrors the bm25s_top200.npy/bm25s_qids.json output format so
evaluation/eval_cc_cross_lingual.py can be pointed at a dense-retrieval pool
via --bm25-suffix.

Use case: test whether layering a CE on top of the *best* retriever (e.g.,
the BoD bi-encoder) gives different fusion behavior than layering on BM25.
This is the missing leg of the CC architecture comparison on BestBuy.

Usage:
    .venv/bin/python evaluation/build_bod_retrieve_topk.py \\
        --data-dir bestbuy_acm_data \\
        --bod-model query_model_bestbuy_bod \\
        --bod-vecs bestbuy_bod_catalog.vecs.fp16.npy \\
        --queries-file test_queries_1k.jsonl \\
        --qrels-file test_qrels_1k.jsonl \\
        --min-relevance 1 \\
        --out-suffix _bodret_1k \\
        --top-k 200
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse  # noqa: E402
import json  # noqa: E402
from collections import defaultdict  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
from sentence_transformers import SentenceTransformer  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--bod-model", required=True)
    ap.add_argument("--bod-vecs", required=True, help="filename inside data-dir")
    ap.add_argument("--queries-file", default="test_queries.jsonl")
    ap.add_argument("--qrels-file", default="test_qrels.jsonl")
    ap.add_argument("--min-relevance", type=int, default=2)
    ap.add_argument("--top-k", type=int, default=200)
    ap.add_argument(
        "--out-suffix",
        default="_bodret",
        help="suffix → bm25s_top{K}{suffix}.npy + bm25s_qids{suffix}.json",
    )
    ap.add_argument("--query-prefix", default="")
    args = ap.parse_args()

    data = Path(args.data_dir).resolve()
    print(f"loading {data}/{args.qrels_file} + {args.queries_file}...", flush=True)
    qrels = defaultdict(dict)
    with open(data / args.qrels_file) as f:
        for line in f:
            r = json.loads(line)
            qrels[r["query_id"]][r["product_id"]] = r["relevance"]
    queries_all = {}
    with open(data / args.queries_file) as f:
        for line in f:
            d = json.loads(line)
            queries_all[d["query_id"]] = d["query"]
    eval_qids = [
        qid
        for qid in queries_all
        if qid in qrels and any(g >= args.min_relevance for g in qrels[qid].values())
    ]
    queries = [queries_all[qid] for qid in eval_qids]
    print(f"  {len(eval_qids):,} eval queries", flush=True)

    catalog = np.load(data / args.bod_vecs, mmap_mode="r")
    print(f"  catalog: {catalog.shape}", flush=True)

    device = (
        "mps"
        if torch.backends.mps.is_available()
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"encoding queries with {args.bod_model} on {device}...", flush=True)
    st_kwargs = {"trust_remote_code": True} if "nomic" in args.bod_model.lower() else {}
    m = SentenceTransformer(args.bod_model, device=device, **st_kwargs)
    prefixed = [args.query_prefix + q for q in queries] if args.query_prefix else queries
    qv = m.encode(
        prefixed, normalize_embeddings=True, batch_size=64, show_progress_bar=False
    ).astype(np.float32)
    print(f"  qv shape: {qv.shape}", flush=True)

    cat_fp32 = np.asarray(catalog).astype(np.float32)
    print(
        f"scoring {len(qv):,} x {len(cat_fp32):,} = {len(qv) * len(cat_fp32):,} pairs...",
        flush=True,
    )
    sims = qv @ cat_fp32.T  # (N_q, N_c)
    print(f"  sims: {sims.shape}", flush=True)

    print(f"top-{args.top_k}...", flush=True)
    top_unsorted = np.argpartition(-sims, args.top_k - 1, axis=1)[:, : args.top_k]
    top_sorted = np.zeros_like(top_unsorted)
    for i in range(len(top_unsorted)):
        order = np.argsort(-sims[i, top_unsorted[i]])
        top_sorted[i] = top_unsorted[i, order]

    top_path = data / f"bm25s_top{args.top_k}{args.out_suffix}.npy"
    qids_path = data / f"bm25s_qids{args.out_suffix}.json"
    np.save(top_path, top_sorted.astype(np.int64))
    with open(qids_path, "w") as f:
        json.dump(eval_qids, f)
    print(f"saved {top_path}: {top_sorted.shape}", flush=True)
    print(f"saved {qids_path}", flush=True)


if __name__ == "__main__":
    main()
