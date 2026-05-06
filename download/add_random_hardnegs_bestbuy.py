#!/usr/bin/env python3
"""Augment bestbuy_acm_data/bags.jsonl with random non-clicked SKUs as hardnegs.

BestBuy clickthrough data is positives-only (no explicit negatives), and we
don't have a precomputed BM25 cache for the BestBuy catalog. For an MNRL
training run, even uniformly-sampled non-positive titles are useful: they
act as extra in-batch negatives per triplet.

Output: bestbuy_acm_data/bags_with_hardnegs.jsonl, in the format
training/finetune_with_hardnegs.py expects.
"""

import argparse
import json
import os
import random


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="bestbuy_acm_data")
    ap.add_argument("--n-hardnegs", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    titles_path = os.path.join(args.data_dir, "titles.json")
    bags_in = os.path.join(args.data_dir, "bags.jsonl")
    bags_out = os.path.join(args.data_dir, "bags_with_hardnegs.jsonl")

    with open(titles_path) as f:
        titles = json.load(f)
    print(f"loaded {len(titles):,} titles", flush=True)

    rng = random.Random(args.seed)
    n_total = 0
    n_with = 0
    print(f"writing {bags_out}...", flush=True)
    with open(bags_in) as fin, open(bags_out, "w") as fout:
        for line in fin:
            bag = json.loads(line)
            n_total += 1
            pos_titles = {r["title"] for r in bag.get("results", [])}
            hardnegs = []
            tries = 0
            while len(hardnegs) < args.n_hardnegs and tries < args.n_hardnegs * 5:
                t = titles[rng.randrange(len(titles))]
                if t not in pos_titles and t not in hardnegs:
                    hardnegs.append(t)
                tries += 1
            bag["hardnegs"] = hardnegs
            if hardnegs:
                n_with += 1
            fout.write(json.dumps(bag) + "\n")

    print(f"  wrote {n_total:,} bags ({n_with:,} with >=1 hardneg)", flush=True)


if __name__ == "__main__":
    main()
