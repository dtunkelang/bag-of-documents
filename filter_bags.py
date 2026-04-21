#!/usr/bin/env python3
"""
Derive a higher-CE-threshold bag file from an existing bag file that has per-member
ce_score fields. Filters bag members by score, re-encodes survivors to recompute
the centroid and specificity.

Input bags must have been produced by the version of compute_bags that stores
ce_score per result. Running this on older bag files will fail with a clear error.

Usage:
    python filter_bags.py bags.jsonl bags_t2.jsonl --threshold 2.0
    python filter_bags.py bags.jsonl bags_t3.jsonl --threshold 3.0
"""

import argparse
import json

import numpy as np
from sentence_transformers import SentenceTransformer

DEFAULT_MODEL = "all-MiniLM-L6-v2"


def main():
    parser = argparse.ArgumentParser(
        description="Derive a higher-threshold bag file from an existing bag file with CE scores"
    )
    parser.add_argument("input", help="Input bags JSONL (with ce_score per member)")
    parser.add_argument("output", help="Output bags JSONL")
    parser.add_argument(
        "--threshold",
        type=float,
        required=True,
        help="New CE threshold (must be >= original threshold used to build input bags)",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Model for re-encoding centroids (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--batch-size", type=int, default=128, help="Batch size for sentence encoding"
    )
    args = parser.parse_args()

    print(f"Loading bags from {args.input}...")
    with open(args.input) as f:
        bags = [json.loads(line) for line in f]
    print(f"  Loaded {len(bags):,} bags")

    # Verify input has scores
    for bag in bags:
        if bag["num_results"] > 0:
            if "ce_score" not in bag["results"][0]:
                raise SystemExit(
                    "Input bags lack 'ce_score' on members. "
                    "Re-run compute_bags with the updated code before using filter_bags."
                )
            break

    # Collect all surviving (bag_idx, title, score) entries for batched encoding
    all_kept = []  # list of (bag_idx, title, score)
    for i, bag in enumerate(bags):
        for r in bag["results"]:
            if r["ce_score"] >= args.threshold:
                all_kept.append((i, r["title"], r["ce_score"]))

    print(f"  {len(all_kept):,} members pass threshold {args.threshold}")

    # Encode all surviving titles in one batched call
    print(f"Loading model {args.model} and encoding...")
    model = SentenceTransformer(args.model)
    titles = [t for _, t, _ in all_kept]
    if titles:
        vectors = model.encode(
            titles,
            normalize_embeddings=True,
            batch_size=args.batch_size,
            show_progress_bar=True,
        )
    else:
        vectors = np.zeros((0, 384), dtype=np.float32)

    # Group by bag
    bag_members = {}  # bag_idx -> list of (title, score, vector)
    for (bag_idx, title, score), vec in zip(all_kept, vectors):
        bag_members.setdefault(bag_idx, []).append((title, score, vec))

    # Write output
    print(f"Writing {args.output}...")
    n_empty = 0
    n_kept = 0
    with open(args.output, "w") as f:
        for i, bag in enumerate(bags):
            members = bag_members.get(i, [])
            if not members:
                out = {
                    "query": bag["query"],
                    "num_results": 0,
                    "query_vector": bag["query_vector"],
                    "specificity": 0,
                    "results": [],
                }
                n_empty += 1
            else:
                titles_ = [t for t, _, _ in members]
                scores_ = [s for _, s, _ in members]
                vecs_ = np.stack([v for _, _, v in members])
                centroid = vecs_.mean(axis=0)
                centroid = centroid / np.linalg.norm(centroid)
                specificity = float(np.mean([centroid @ v for v in vecs_]))
                out = {
                    "query": bag["query"],
                    "num_results": len(members),
                    "query_vector": centroid.tolist(),
                    "specificity": specificity,
                    "results": [{"title": t, "ce_score": s} for t, s in zip(titles_, scores_)],
                }
                n_kept += 1
            f.write(json.dumps(out) + "\n")

    print(f"Done. {n_kept:,} bags with members, {n_empty:,} empty at threshold {args.threshold}.")


if __name__ == "__main__":
    main()
