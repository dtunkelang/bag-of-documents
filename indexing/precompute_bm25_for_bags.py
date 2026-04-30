#!/usr/bin/env python3
"""Precompute BM25 top-K positions for the queries in bags.jsonl.

Mirrors precompute_bm25_top_k.py but reads queries from bags.jsonl rather
than the ESCI test set, so we can mine BM25 hits as hard negatives during
reranker training.

Output: combined_index_us_minilm/bm25_for_bags_top100.npy
        - shape (N_BAGS, 100), int32, FAISS-aligned product positions
        - row order matches the order bags.jsonl produces (positive bags only:
          num_results >= 1)
        - companion file bm25_for_bags_qids.json maps row -> query string
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import re  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
import tantivy  # noqa: E402

SCRIPT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INDEX_DIR = os.path.join(SCRIPT_DIR, "combined_index_us_minilm")
TANTIVY_PATH = os.path.join(INDEX_DIR, "tantivy_index")

PUNCT_RE = re.compile(r"[^\w\s]")


def safe_parse(idx, q):
    try:
        return idx.parse_query(q, ["title"])
    except ValueError:
        cleaned = PUNCT_RE.sub(" ", q).strip()
        if not cleaned:
            return None
        try:
            return idx.parse_query(cleaned, ["title"])
        except ValueError:
            return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bags", default=os.path.join(SCRIPT_DIR, "bags.jsonl"))
    ap.add_argument("--tantivy-path", default=TANTIVY_PATH)
    ap.add_argument("--output", default=os.path.join(INDEX_DIR, "bm25_for_bags_top100.npy"))
    ap.add_argument("--qids-output", default=os.path.join(INDEX_DIR, "bm25_for_bags_qids.json"))
    ap.add_argument("--k", type=int, default=100)
    args = ap.parse_args()

    print("loading bags...", flush=True)
    queries = []
    with open(args.bags) as f:
        for line in f:
            d = json.loads(line)
            if d.get("num_results", 0) >= 1:
                queries.append(d["query"])
    print(f"  {len(queries):,} bags with >=1 positive", flush=True)

    print(f"loading tantivy index {args.tantivy_path}...", flush=True)
    idx = tantivy.Index.open(args.tantivy_path)
    idx.reload()
    s = idx.searcher()
    print(f"  {s.num_docs:,} docs", flush=True)

    print("loading titles + building title -> position map...", flush=True)
    with open(os.path.join(INDEX_DIR, "titles.json")) as f:
        titles = json.load(f)
    title_to_pos = {}
    for i, t in enumerate(titles):
        if t not in title_to_pos:
            title_to_pos[t] = i

    out = np.full((len(queries), args.k), -1, dtype=np.int32)
    n_empty = 0
    n_unparseable = 0
    t0 = time.time()
    for qi, q in enumerate(queries):
        parsed = safe_parse(idx, q)
        if parsed is None:
            n_unparseable += 1
            continue
        hits = s.search(parsed, limit=args.k).hits
        if not hits:
            n_empty += 1
            continue
        positions = []
        for _, addr in hits:
            title = s.doc(addr)["title"][0]
            pos = title_to_pos.get(title)
            if pos is not None:
                positions.append(pos)
                if len(positions) == args.k:
                    break
        for j, p in enumerate(positions):
            out[qi, j] = p
        if (qi + 1) % 10000 == 0:
            elapsed = time.time() - t0
            rate = (qi + 1) / elapsed
            eta = (len(queries) - (qi + 1)) / rate
            print(f"  {qi + 1:,}/{len(queries):,}  rate={rate:.0f} q/s  eta={eta:.0f}s", flush=True)

    print(
        f"  done {time.time() - t0:.0f}s. unparseable={n_unparseable}, empty={n_empty}", flush=True
    )
    np.save(args.output, out)
    with open(args.qids_output, "w") as f:
        json.dump(queries, f)
    print(f"wrote {args.output}  shape={out.shape}", flush=True)
    print(f"wrote {args.qids_output}  ({len(queries):,} queries)", flush=True)


if __name__ == "__main__":
    main()
