#!/usr/bin/env python3
"""Precompute BM25 top-K positions for the 22,458 ESCI test queries.

Reads a tantivy index and writes an (N_QUERIES, K) int32 array of FAISS-aligned
product positions, with -1 padding when fewer than K hits. Order matches the
qid order produced by eval_mnrl_retriever.py so the array can be loaded
directly without re-deriving qids.

Usage:
    # default: top-100 against the en_stem index
    python precompute_bm25_top_k.py

    # top-200 against the default-tokenizer index, alternate output
    python precompute_bm25_top_k.py \\
        --tantivy-path combined_index_us_minilm/tantivy_index_default \\
        --output combined_index_us_minilm/bm25_default_top200.npy \\
        --k 200
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import json
import os
import re
import time
from collections import defaultdict

import numpy as np
import tantivy

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INDEX_DIR = os.path.join(SCRIPT_DIR, "combined_index_us_minilm")

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
    ap.add_argument("--tantivy-path", default=os.path.join(INDEX_DIR, "tantivy_index"))
    ap.add_argument("--output", default=os.path.join(INDEX_DIR, "bm25_top100.npy"))
    ap.add_argument("--k", type=int, default=100)
    args = ap.parse_args()

    print("loading qrels + queries...", flush=True)
    qrels = defaultdict(dict)
    with open(os.path.join(SCRIPT_DIR, "esci_us_data/test_qrels.jsonl")) as f:
        for line in f:
            r = json.loads(line)
            qrels[r["query_id"]][r["product_id"]] = r["relevance"]
    queries_all = {}
    with open(os.path.join(SCRIPT_DIR, "esci_us_data/test_queries.jsonl")) as f:
        for line in f:
            d = json.loads(line)
            queries_all[d["query_id"]] = d["query"]

    qids = [qid for qid in queries_all if qid in qrels and any(g >= 2 for g in qrels[qid].values())]
    queries = [queries_all[qid] for qid in qids]
    print(f"  {len(qids):,} eval queries", flush=True)

    print(f"loading tantivy index {args.tantivy_path}...", flush=True)
    idx = tantivy.Index.open(args.tantivy_path)
    idx.reload()
    s = idx.searcher()
    print(f"  {s.num_docs:,} docs", flush=True)

    print("loading titles.json + building title -> first-position map...", flush=True)
    with open(os.path.join(INDEX_DIR, "titles.json")) as f:
        titles = json.load(f)
    title_to_pos = {}
    for i, t in enumerate(titles):
        if t not in title_to_pos:
            title_to_pos[t] = i
    print(f"  {len(titles):,} titles, {len(title_to_pos):,} unique", flush=True)

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
        if (qi + 1) % 5000 == 0:
            elapsed = time.time() - t0
            rate = (qi + 1) / elapsed
            eta = (len(queries) - (qi + 1)) / rate
            print(f"  {qi + 1:,}/{len(queries):,}  rate={rate:.0f} q/s  eta={eta:.0f}s", flush=True)

    elapsed = time.time() - t0
    print(
        f"  done {elapsed:.0f}s. unparseable={n_unparseable}, empty={n_empty}",
        flush=True,
    )

    np.save(args.output, out)
    print(f"wrote {args.output}  shape={out.shape}", flush=True)


if __name__ == "__main__":
    main()
