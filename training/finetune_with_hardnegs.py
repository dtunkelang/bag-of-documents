#!/usr/bin/env python3
"""Train a query encoder with MNRL + explicit hard negatives mined from BM25.

Reads bags_with_hardnegs.jsonl (output of training/build_bags_with_hardnegs.py).
For each bag, samples N triplets of the form (query, positive, hardneg) where
  - positive: a random member of bag["results"]
  - hardneg: a random member of bag["hardnegs"] (BM25 hits not in the bag)

These triplets feed sentence-transformers' MultipleNegativesRankingLoss, which
pulls the query embedding toward the positive and pushes it away from the
hardneg AND in-batch negatives.

The result is a sibling reranker to query_model_us_full_6m_mnrl: same base
(MiniLM), same bag corpus, but trained with explicit BM25-mined hard negatives
rather than query-to-query MNRL.

Usage:
    python training/finetune_with_hardnegs.py \\
        bags_with_hardnegs.jsonl \\
        query_model_us_full_6m_mnrl_bm25hn \\
        --epochs 10 --batch-size 32 --triplets-per-bag 5
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import random  # noqa: E402
import time  # noqa: E402

import torch  # noqa: E402
from sentence_transformers import (  # noqa: E402
    InputExample,
    SentenceTransformer,
    losses,
)
from torch.utils.data import DataLoader  # noqa: E402

SCRIPT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_triplets(bags_path, n_per_bag, seed):
    """Yield InputExample(texts=[query, positive, hardneg]) — n_per_bag per bag."""
    rng = random.Random(seed)
    triplets = []
    skipped_no_pos = 0
    skipped_no_neg = 0
    with open(bags_path) as f:
        for line in f:
            bag = json.loads(line)
            results = bag.get("results", [])
            hardnegs = bag.get("hardnegs", [])
            positives = [r["title"] for r in results if r.get("title")]
            if not positives:
                skipped_no_pos += 1
                continue
            if not hardnegs:
                skipped_no_neg += 1
                continue
            for _ in range(n_per_bag):
                pos = rng.choice(positives)
                neg = rng.choice(hardnegs)
                triplets.append(InputExample(texts=[bag["query"], pos, neg]))
    print(
        f"  built {len(triplets):,} triplets "
        f"(skipped: no_pos={skipped_no_pos}, no_neg={skipped_no_neg})",
        flush=True,
    )
    return triplets


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("bags_file")
    ap.add_argument("output_dir")
    ap.add_argument("--base-model", default="all-MiniLM-L6-v2")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--triplets-per-bag", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    print(f"loading bags from {args.bags_file}...", flush=True)
    t0 = time.time()
    triplets = load_triplets(args.bags_file, args.triplets_per_bag, args.seed)
    print(f"  load took {time.time() - t0:.0f}s", flush=True)

    device = (
        "mps"
        if torch.backends.mps.is_available()
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"\nloading base model {args.base_model} on {device}...", flush=True)
    model = SentenceTransformer(args.base_model, device=device)

    train_loader = DataLoader(triplets, shuffle=True, batch_size=args.batch_size)
    train_loss = losses.MultipleNegativesRankingLoss(model)

    n_steps = len(train_loader) * args.epochs
    warmup_steps = int(0.1 * n_steps)
    print(
        f"\ntraining: {len(triplets):,} triplets, {args.epochs} epochs, "
        f"batch={args.batch_size}, lr={args.lr}, "
        f"{n_steps:,} total steps (warmup={warmup_steps:,})",
        flush=True,
    )

    model.fit(
        train_objectives=[(train_loader, train_loss)],
        epochs=args.epochs,
        warmup_steps=warmup_steps,
        optimizer_params={"lr": args.lr},
        output_path=args.output_dir,
        show_progress_bar=True,
    )

    print(f"\nsaved model to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
