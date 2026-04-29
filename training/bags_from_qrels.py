#!/usr/bin/env python3
"""
Build bag-of-documents training bags directly from human relevance judgments (qrels).

For each query with enough relevant documents, this script:
  1. Gathers labeled relevant products (graded relevance >= min_relevance)
  2. Encodes them with a sentence transformer
  3. Computes the spherical-mean centroid + specificity
  4. Writes a bag in the same JSONL format compute_bags.py produces

Differences from compute_bags.py:
  - No cross-encoder (qrels ARE the relevance signal)
  - No hybrid retrieval (we only consider human-labeled products)
  - Much faster (no CE forward passes)

Usage:
    python bags_from_qrels.py \\
        --titles esci_es_data/titles.json \\
        --product-ids esci_es_data/product_ids.json \\
        --queries esci_es_data/train_queries.jsonl \\
        --qrels esci_es_data/train_qrels.jsonl \\
        --output bags.jsonl \\
        --model paraphrase-multilingual-MiniLM-L12-v2 \\
        --min-relevance 2 --k 50
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import json
from collections import defaultdict

import numpy as np
from sentence_transformers import SentenceTransformer


def main():
    parser = argparse.ArgumentParser(description="Build bags directly from qrels")
    parser.add_argument("--titles", required=True, help="titles.json (list of product titles)")
    parser.add_argument(
        "--product-ids", required=True, help="product_ids.json (parallel list of product_ids)"
    )
    parser.add_argument("--queries", required=True, help="queries.jsonl (query_id, query)")
    parser.add_argument(
        "--qrels", required=True, help="qrels.jsonl (query_id, product_id, relevance)"
    )
    parser.add_argument("--output", required=True, help="Output bags.jsonl")
    parser.add_argument(
        "--model",
        default="paraphrase-multilingual-MiniLM-L12-v2",
        help="Sentence transformer for encoding (default: multilingual MiniLM)",
    )
    parser.add_argument(
        "--min-relevance",
        type=int,
        default=2,
        help="Minimum grade to include in bag (default: 2 = Exact or Substitute)",
    )
    parser.add_argument(
        "--k", type=int, default=50, help="Max members per bag (default: 50, same as compute_bags)"
    )
    parser.add_argument(
        "--min-members",
        type=int,
        default=2,
        help="Skip queries with fewer than this many relevant members (default: 2)",
    )
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()

    print(f"Loading corpus from {args.titles}...")
    with open(args.titles) as f:
        titles = json.load(f)
    with open(args.product_ids) as f:
        product_ids = json.load(f)
    pid_to_idx = {pid: i for i, pid in enumerate(product_ids)}
    print(f"  {len(titles):,} products")

    print(f"Loading queries from {args.queries}...")
    queries = {}
    with open(args.queries) as f:
        for line in f:
            q = json.loads(line)
            queries[q["query_id"]] = q["query"]
    print(f"  {len(queries):,} queries")

    print(f"Loading qrels from {args.qrels} (min grade {args.min_relevance})...")
    query_relevant = defaultdict(list)  # query_id -> [(product_idx, grade)]
    missing_products = 0
    with open(args.qrels) as f:
        for line in f:
            q = json.loads(line)
            if q["relevance"] < args.min_relevance:
                continue
            if q["product_id"] not in pid_to_idx:
                missing_products += 1
                continue
            query_relevant[q["query_id"]].append((pid_to_idx[q["product_id"]], q["relevance"]))
    if missing_products:
        print(f"  Skipped {missing_products} qrels with products not in corpus")

    # Keep queries with enough members
    usable_queries = {
        q: rels for q, rels in query_relevant.items() if len(rels) >= args.min_members
    }
    print(f"  {len(usable_queries):,} queries with >= {args.min_members} relevant members")

    # Sort members by grade (descending) and keep top k
    print("\nAssembling bag members...")
    all_titles_to_encode = []  # flat list
    bag_member_slices = []  # (query_id, [(idx_in_flat, grade)])
    for qid, rels in usable_queries.items():
        rels = sorted(rels, key=lambda x: -x[1])[: args.k]
        start = len(all_titles_to_encode)
        for prod_idx, _grade in rels:
            all_titles_to_encode.append(titles[prod_idx])
        bag_member_slices.append((qid, rels, start, len(all_titles_to_encode)))
    print(f"  {len(all_titles_to_encode):,} total member encodings needed")

    print(f"\nLoading model {args.model}...")
    model = SentenceTransformer(args.model)

    print(f"Encoding {len(all_titles_to_encode):,} product titles...")
    vectors = model.encode(
        all_titles_to_encode,
        normalize_embeddings=True,
        batch_size=args.batch_size,
        show_progress_bar=True,
    )

    print(f"\nWriting bags to {args.output}...")
    n_written = 0
    with open(args.output, "w") as f:
        for qid, rels, start, end in bag_member_slices:
            member_vecs = vectors[start:end]
            centroid = member_vecs.mean(axis=0)
            centroid = centroid / np.linalg.norm(centroid)
            specificity = float(np.mean([centroid @ v for v in member_vecs]))
            bag = {
                "query": queries[qid],
                "num_results": len(rels),
                "query_vector": centroid.tolist(),
                "specificity": specificity,
                "results": [
                    {"title": titles[p], "ce_score": float(g)} for p, g in rels
                ],  # "ce_score" here is the qrel grade
            }
            f.write(json.dumps(bag, ensure_ascii=False) + "\n")
            n_written += 1
    print(f"Wrote {n_written:,} bags")


if __name__ == "__main__":
    main()
