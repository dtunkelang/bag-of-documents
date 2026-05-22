#!/usr/bin/env python3
"""BM25 retrieval over a catalog (titles.json), parallel to dense retrievers.

Outputs JSONL of {query_id, query, top_pids: [...], top_scores: [...]}
which can then be (a) judged via llm_relevance_judge to populate a candidate pool
or (b) merged into the eval_pilot_b_retrievers comparison harness.

Uses bm25s (pyserini-style implementation) with default hyperparams; both
tokenization (lowercase, English stemming, English stopwords) and index params
(k1=1.5, b=0.75) are configurable.

Usage:
  .venv/bin/python evaluation/bm25_retrieve.py \\
      --data-dir jobs_data \\
      --queries-file eval_queries.jsonl \\
      --ids-file doc_ids.json \\
      --k 50 \\
      --output /tmp/jobs_bm25_top50.jsonl
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402


def build_index(titles, k1=1.5, b=0.75):
    import bm25s
    from Stemmer import Stemmer

    stemmer = Stemmer("english")
    print(f"  tokenizing {len(titles):,} docs (stem=en, stopwords=en)...", flush=True)
    t0 = time.time()
    title_tok = bm25s.tokenize(titles, stopwords="en", stemmer=stemmer, show_progress=False)
    print(f"  done in {time.time() - t0:.1f}s", flush=True)
    print(f"  indexing BM25 with k1={k1} b={b}...", flush=True)
    t0 = time.time()
    idx = bm25s.BM25(k1=k1, b=b)
    idx.index(title_tok, show_progress=False)
    print(f"  done in {time.time() - t0:.1f}s", flush=True)
    return idx, stemmer


def retrieve(idx, stemmer, queries, k):
    import bm25s

    t0 = time.time()
    qtok = bm25s.tokenize(queries, stopwords="en", stemmer=stemmer, show_progress=False)
    res_idx, res_scores = idx.retrieve(qtok, k=k, show_progress=False)
    print(
        f"  retrieved top-{k} for {len(queries):,} queries in {time.time() - t0:.1f}s",
        flush=True,
    )
    return res_idx, res_scores


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--titles-file", default="titles.json")
    ap.add_argument("--queries-file", required=True, help="JSONL with query_id+query")
    ap.add_argument("--ids-file", default=None, help="default: doc_ids.json else product_ids.json")
    ap.add_argument("--k", type=int, default=50)
    ap.add_argument("--k1", type=float, default=1.5)
    ap.add_argument("--b", type=float, default=0.75)
    ap.add_argument("--output", required=True, help="JSONL output")
    args = ap.parse_args()

    data = Path(args.data_dir)

    if args.ids_file:
        ids_path = data / args.ids_file
    elif (data / "doc_ids.json").exists():
        ids_path = data / "doc_ids.json"
    else:
        ids_path = data / "product_ids.json"
    with open(ids_path) as f:
        pids = json.load(f)
    with open(data / args.titles_file) as f:
        titles = json.load(f)
    print(f"catalog: {len(pids):,} docs from {ids_path}", flush=True)

    queries = []
    with open(data / args.queries_file) as f:
        for line in f:
            d = json.loads(line)
            queries.append((d["query_id"], d["query"]))
    print(f"queries: {len(queries):,} from {args.queries_file}", flush=True)

    print("\nbuilding BM25 index...", flush=True)
    idx, stemmer = build_index(titles, k1=args.k1, b=args.b)

    print(f"\nretrieving top-{args.k}...", flush=True)
    res_idx, res_scores = retrieve(idx, stemmer, [q for _, q in queries], k=args.k)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as fout:
        for i, (qid, q) in enumerate(queries):
            top_pids = [pids[int(p)] for p in res_idx[i]]
            top_scores = [float(s) for s in res_scores[i]]
            fout.write(
                json.dumps(
                    {
                        "query_id": qid,
                        "query": q,
                        "top_pids": top_pids,
                        "top_scores": top_scores,
                    }
                )
                + "\n"
            )

    # Sanity: distribution of top scores
    s = np.asarray(res_scores)
    print(
        f"\nwrote {len(queries):,} rows to {args.output}\n"
        f"  top-1 score: mean={s[:, 0].mean():.3f} median={np.median(s[:, 0]):.3f}\n"
        f"  top-K score: mean={s[:, -1].mean():.3f} median={np.median(s[:, -1]):.3f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
