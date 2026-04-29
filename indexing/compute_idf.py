#!/usr/bin/env python3
"""Compute per-token document frequency / IDF over the catalog titles.

Tokenization matches utils.tokenize_query so the IDF lookup keys are
consistent with what generate_keyword_combos receives at retrieval time.

Output: <index_path>/idf.json with structure:
    {
      "n_docs": <int>,
      "df": {token: doc_frequency, ...}
    }

IDF is derivable from (df, n_docs) at use time so the file is small.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import json
import os
import time
from collections import Counter

from utils import fmt_duration, tokenize_query


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "index_path",
        nargs="?",
        default="combined_index_amazon",
        help="Index directory containing titles.json (default: combined_index_amazon)",
    )
    args = parser.parse_args()

    titles_path = os.path.join(args.index_path, "titles.json")
    out_path = os.path.join(args.index_path, "idf.json")
    if not os.path.exists(titles_path):
        raise SystemExit(f"missing {titles_path}")

    print(f"loading {titles_path}...", flush=True)
    t0 = time.time()
    with open(titles_path) as f:
        titles = json.load(f)
    print(f"  {len(titles):,} titles in {fmt_duration(time.time() - t0)}", flush=True)

    print("counting document frequencies...", flush=True)
    t0 = time.time()
    df = Counter()
    for i, title in enumerate(titles):
        tokens = set(tokenize_query(title))
        df.update(tokens)
        if (i + 1) % 1_000_000 == 0:
            print(
                f"  {i + 1:,} titles processed ({(i + 1) / (time.time() - t0):,.0f}/sec)",
                flush=True,
            )
    print(f"  done in {fmt_duration(time.time() - t0)}; vocab size {len(df):,}", flush=True)

    print(f"writing {out_path}...", flush=True)
    with open(out_path, "w") as f:
        json.dump({"n_docs": len(titles), "df": dict(df)}, f)

    size_mb = os.path.getsize(out_path) / 1e6
    print(f"  wrote {size_mb:.1f} MB")


if __name__ == "__main__":
    main()
