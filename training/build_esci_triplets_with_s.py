#!/usr/bin/env python3
"""Build (query, positive, hardneg) triplets from ESCI train qrels with E+S
as positives.

Variant of build_esci_triplets.py. Includes Substitute (S) products as
additional positives alongside Exact (E). Tests whether the binary E/I
training in rerank_G was leaving signal on the table.

For each train query with at least one (E or S) judgment AND at least one
I judgment:
  - positives: products with relevance in {3 (E), 2 (S)}
  - hardnegs:  products with relevance=0 (I)

Output format mirrors bags_with_hardnegs.jsonl so finetune_with_hardnegs.py
consumes it without changes.

Usage:
    python training/build_esci_triplets_with_s.py \\
        --output esci_triplets_with_s.jsonl
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
from collections import defaultdict  # noqa: E402

SCRIPT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qrels", default=os.path.join(SCRIPT_DIR, "esci_us_data/train_qrels.jsonl"))
    ap.add_argument(
        "--queries",
        default=os.path.join(SCRIPT_DIR, "esci_us_data/train_queries.jsonl"),
    )
    ap.add_argument(
        "--product-ids",
        default=os.path.join(SCRIPT_DIR, "esci_us_data/product_ids.json"),
    )
    ap.add_argument("--titles", default=os.path.join(SCRIPT_DIR, "esci_us_data/titles.json"))
    ap.add_argument("--output", default=os.path.join(SCRIPT_DIR, "esci_triplets_with_s.jsonl"))
    args = ap.parse_args()

    print("loading queries...", flush=True)
    queries = {}
    with open(args.queries) as f:
        for line in f:
            d = json.loads(line)
            queries[d["query_id"]] = d["query"]
    print(f"  {len(queries):,} queries", flush=True)

    print("loading product_ids + titles...", flush=True)
    with open(args.product_ids) as f:
        product_ids = json.load(f)
    with open(args.titles) as f:
        titles = json.load(f)
    pid_to_title = dict(zip(product_ids, titles))
    print(f"  {len(pid_to_title):,} products", flush=True)

    print("loading qrels...", flush=True)
    by_query = defaultdict(lambda: {"E": [], "S": [], "C": [], "I": []})
    label_map = {3: "E", 2: "S", 1: "C", 0: "I"}
    with open(args.qrels) as f:
        for line in f:
            r = json.loads(line)
            qid = r["query_id"]
            label = label_map.get(r["relevance"])
            title = pid_to_title.get(r["product_id"])
            if label and title:
                by_query[qid][label].append(title)
    print(f"  {len(by_query):,} queries with judgments", flush=True)

    print(f"writing triplets to {args.output}...", flush=True)
    n_written = 0
    n_skipped_no_pos = 0
    n_skipped_no_i = 0
    n_e_pos = 0
    n_s_pos = 0
    with open(args.output, "w") as fout:
        for qid, labels in by_query.items():
            positives = labels["E"] + labels["S"]
            if not positives:
                n_skipped_no_pos += 1
                continue
            if not labels["I"]:
                n_skipped_no_i += 1
                continue
            query = queries.get(qid)
            if not query:
                continue
            n_e_pos += len(labels["E"])
            n_s_pos += len(labels["S"])
            bag = {
                "query": query,
                "results": [{"title": t} for t in positives],
                "hardnegs": labels["I"],
                "num_results": len(positives),
            }
            fout.write(json.dumps(bag) + "\n")
            n_written += 1

    print(f"\n  {n_written:,} queries kept", flush=True)
    print(f"  {n_skipped_no_pos:,} skipped (no E or S judgments)", flush=True)
    print(f"  {n_skipped_no_i:,} skipped (no I judgments)", flush=True)
    print(
        f"  positives breakdown: {n_e_pos:,} E + {n_s_pos:,} S = {n_e_pos + n_s_pos:,}", flush=True
    )


if __name__ == "__main__":
    main()
