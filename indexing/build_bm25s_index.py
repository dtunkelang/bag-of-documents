#!/usr/bin/env python3
"""Build a saved bm25s index over the catalog with optimized (k1, b) params,
and precompute top-K for the ESCI test queries.

Replaces the tantivy retriever in the production path. The (k1, b)=(0.3, 0.6)
combo is the sweep winner (CC3-50 R@10 21.61% vs tantivy 21.32%) — see
evaluation/eval_bm25_sweep.py.

Outputs (under combined_index_us_minilm/):
  - bm25s_index/                       saved bm25s index (load via BM25.load)
  - bm25s_top200.npy                   top-200 indices for ESCI test queries
  - bm25s_for_bags_top100.npy          top-100 indices for the bags' source queries
                                        (so the same retriever feeds bag construction
                                        and serving — only built if queries.jsonl exists)

Usage:
    python indexing/build_bm25s_index.py
    python indexing/build_bm25s_index.py --k1 0.3 --b 0.6
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import time  # noqa: E402

import bm25s  # noqa: E402
import numpy as np  # noqa: E402
import Stemmer  # noqa: E402

SCRIPT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INDEX_DIR = os.path.join(SCRIPT_DIR, "combined_index_us_minilm")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k1", type=float, default=0.3)
    ap.add_argument("--b", type=float, default=0.6)
    ap.add_argument("--index-dir", default=INDEX_DIR)
    ap.add_argument("--top-k-eval", type=int, default=200, help="top-K for ESCI test cache")
    args = ap.parse_args()

    print(f"loading titles from {args.index_dir}/titles.json...", flush=True)
    with open(os.path.join(args.index_dir, "titles.json")) as f:
        titles = json.load(f)
    print(f"  {len(titles):,} titles", flush=True)

    stemmer = Stemmer.Stemmer("english")

    print("tokenizing titles (Snowball english + en stopwords)...", flush=True)
    t0 = time.time()
    title_tokens = bm25s.tokenize(titles, stopwords="en", stemmer=stemmer, show_progress=False)
    print(f"  done in {time.time() - t0:.0f}s", flush=True)

    print(f"building bm25s index (k1={args.k1}, b={args.b})...", flush=True)
    t0 = time.time()
    idx = bm25s.BM25(k1=args.k1, b=args.b)
    idx.index(title_tokens, show_progress=False)
    print(f"  done in {time.time() - t0:.0f}s", flush=True)

    save_dir = os.path.join(args.index_dir, "bm25s_index")
    print(f"saving index to {save_dir}/...", flush=True)
    os.makedirs(save_dir, exist_ok=True)
    idx.save(save_dir, show_progress=False)
    # Persist k1/b explicitly so reload paths can verify.
    with open(os.path.join(save_dir, "bm25_params.json"), "w") as f:
        json.dump({"k1": args.k1, "b": args.b, "stemmer": "english", "stopwords": "en"}, f)
    print(f"  saved (size: {sum(p.stat().st_size for p in Path(save_dir).iterdir()) / 1e6:.0f} MB)")

    # Precompute ESCI test top-K cache for eval. Use the SAME qid filter
    # as eval_mnrl_retriever.py and indexing/precompute_bm25_top_k.py
    # (qids in qrels with at least one E+S judgment) so the resulting array
    # aligns position-wise with the eval's `queries` list.
    test_queries_path = os.path.join(SCRIPT_DIR, "esci_us_data/test_queries.jsonl")
    test_qrels_path = os.path.join(SCRIPT_DIR, "esci_us_data/test_qrels.jsonl")
    if os.path.exists(test_queries_path) and os.path.exists(test_qrels_path):
        print("\ntokenizing ESCI test queries (filtered to eval qids)...", flush=True)
        from collections import defaultdict

        qrels = defaultdict(dict)
        with open(test_qrels_path) as f:
            for line in f:
                r = json.loads(line)
                qrels[r["query_id"]][r["product_id"]] = r["relevance"]
        queries_all = {}
        with open(test_queries_path) as f:
            for line in f:
                d = json.loads(line)
                queries_all[d["query_id"]] = d["query"]
        # Same filter as eval_mnrl_retriever.py — preserves dict insertion order.
        eval_qids = [
            qid for qid in queries_all if qid in qrels and any(g >= 2 for g in qrels[qid].values())
        ]
        queries = [queries_all[qid] for qid in eval_qids]
        query_tokens = bm25s.tokenize(queries, stopwords="en", stemmer=stemmer, show_progress=False)

        print(f"retrieving top-{args.top_k_eval} for {len(queries):,} eval queries...", flush=True)
        t0 = time.time()
        results, _ = idx.retrieve(query_tokens, k=args.top_k_eval, show_progress=False)
        print(f"  done in {time.time() - t0:.0f}s", flush=True)

        out_path = os.path.join(args.index_dir, f"bm25s_top{args.top_k_eval}.npy")
        np.save(out_path, np.asarray(results, dtype=np.int64))
        print(f"  saved {out_path}: shape={np.asarray(results).shape}", flush=True)

        qids_path = os.path.join(args.index_dir, "bm25s_qids.json")
        with open(qids_path, "w") as f:
            json.dump(eval_qids, f)
        print(f"  saved {qids_path}", flush=True)


if __name__ == "__main__":
    main()
