#!/usr/bin/env python3
"""Augment bags.jsonl with BM25-mined hard negatives per bag.

For each bag, take the BM25 top-K hits (from the precomputed
bm25_for_bags_top100.npy) and treat the top N that are NOT in the bag's
positive set as hard negatives. These are products BM25 ranks highly but
the bag construction pipeline (hybrid retrieval + CE filtering) rejected.

Output: bags_with_hardnegs.jsonl
  Same format as bags.jsonl, with one new field per bag:
    "hardnegs": [list of titles, length up to --n-hardnegs]
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402

import numpy as np  # noqa: E402

SCRIPT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INDEX_DIR = os.path.join(SCRIPT_DIR, "combined_index_us_minilm")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bags", default=os.path.join(SCRIPT_DIR, "bags.jsonl"))
    ap.add_argument("--bm25", default=os.path.join(INDEX_DIR, "bm25_for_bags_top100.npy"))
    ap.add_argument("--qids", default=os.path.join(INDEX_DIR, "bm25_for_bags_qids.json"))
    ap.add_argument("--titles", default=os.path.join(INDEX_DIR, "titles.json"))
    ap.add_argument("--output", default=os.path.join(SCRIPT_DIR, "bags_with_hardnegs.jsonl"))
    ap.add_argument("--n-hardnegs", type=int, default=10)
    args = ap.parse_args()

    print(f"loading bm25 cache + qid order from {args.bm25}...", flush=True)
    bm25 = np.load(args.bm25)
    with open(args.qids) as f:
        bm25_qids = json.load(f)
    print(f"  bm25 shape: {bm25.shape}, {len(bm25_qids):,} qids", flush=True)
    qid_to_row = {q: i for i, q in enumerate(bm25_qids)}

    print(f"loading titles from {args.titles}...", flush=True)
    with open(args.titles) as f:
        titles = json.load(f)
    print(f"  {len(titles):,} titles", flush=True)

    n_total = 0
    n_with_hardnegs = 0
    hardneg_counts = []
    print(f"reading bags + writing {args.output}...", flush=True)
    with open(args.bags) as fin, open(args.output, "w") as fout:
        for line in fin:
            bag = json.loads(line)
            n_total += 1
            query = bag["query"]
            positive_titles = {r["title"] for r in bag.get("results", [])}

            row = qid_to_row.get(query)
            hardnegs = []
            if row is not None:
                for pos in bm25[row]:
                    if pos < 0:
                        continue
                    title = titles[int(pos)]
                    if title in positive_titles:
                        continue
                    hardnegs.append(title)
                    if len(hardnegs) >= args.n_hardnegs:
                        break

            bag["hardnegs"] = hardnegs
            if hardnegs:
                n_with_hardnegs += 1
                hardneg_counts.append(len(hardnegs))
            fout.write(json.dumps(bag) + "\n")

    print(f"\ndone. {n_total:,} bags written.", flush=True)
    print(f"  {n_with_hardnegs:,} ({n_with_hardnegs / n_total:.1%}) have >=1 hardneg", flush=True)
    if hardneg_counts:
        print(
            f"  hardnegs/bag: mean={np.mean(hardneg_counts):.1f}, "
            f"median={int(np.median(hardneg_counts))}, max={max(hardneg_counts)}",
            flush=True,
        )


if __name__ == "__main__":
    main()
