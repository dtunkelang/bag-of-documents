#!/usr/bin/env python3
"""Distill the LiYuan ESCI cross-encoder into a MiniLM bi-encoder using
MarginMSE loss. Goal: capture CE's full-attention knowledge in a fast
bi-encoder so we can match CC4-100 quality at CC3-50 latency.

Training data is built from CE scores on a sample of ESCI *train*
queries (combined_index_us_minilm/ce_train_*.npy, produced by
evaluation/score_ce_train.py). For each train query, sample N pairs of
top-100 candidates (d_high, d_low) where ce(d_high) > ce(d_low). Label =
ce(d_high) - ce(d_low). Loss = MSE between (sim(q, d_high) - sim(q,
d_low)) and that CE-margin label. The bi-encoder learns to mimic the
cross-encoder's relative ordering. Eval is on the held-out test set.

Output: query_model_us_ce_distilled/

Usage:
    python training/distill_ce_into_bi_encoder.py
    python training/distill_ce_into_bi_encoder.py --epochs 5 --pairs-per-query 5
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import random  # noqa: E402
import time  # noqa: E402

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np  # noqa: E402
import torch  # noqa: E402
from sentence_transformers import (  # noqa: E402
    InputExample,
    SentenceTransformer,
    losses,
)
from torch.utils.data import DataLoader  # noqa: E402

SCRIPT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INDEX_DIR = os.path.join(SCRIPT_DIR, "combined_index_us_minilm")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-model", default="all-MiniLM-L6-v2")
    ap.add_argument("--output-dir", default=os.path.join(SCRIPT_DIR, "query_model_us_ce_distilled"))
    ap.add_argument("--pairs-per-query", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Load TRAIN queries with their CE scores (no test-set leakage).
    print("loading train queries + cached ce_train artifacts...", flush=True)
    with open(os.path.join(INDEX_DIR, "ce_train_queries.json")) as f:
        queries = json.load(f)
    with open(os.path.join(INDEX_DIR, "titles.json")) as f:
        index_titles = json.load(f)
    candidate_pos = np.load(os.path.join(INDEX_DIR, "ce_train_candidates.npy"))
    ce_scores = np.load(os.path.join(INDEX_DIR, "ce_train_scores.npy"))
    valid = candidate_pos >= 0
    print(f"  {len(queries):,} queries, candidates={candidate_pos.shape}", flush=True)

    # Build training triplets: per query, sample N pairs (d_high, d_low)
    # where ce(d_high) > ce(d_low). Label = ce_high - ce_low.
    print(f"sampling {args.pairs_per_query} pairs/query...", flush=True)
    examples = []
    n_skipped = 0
    for qi, q in enumerate(queries):
        idx_valid = np.where(valid[qi])[0]
        if idx_valid.size < 2:
            n_skipped += 1
            continue
        scores = ce_scores[qi, idx_valid]
        for _ in range(args.pairs_per_query):
            i, j = random.sample(range(idx_valid.size), 2)
            if scores[i] == scores[j]:
                continue
            if scores[i] > scores[j]:
                hi, lo = i, j
            else:
                hi, lo = j, i
            pos = int(candidate_pos[qi, idx_valid[hi]])
            neg = int(candidate_pos[qi, idx_valid[lo]])
            margin = float(scores[hi] - scores[lo])
            examples.append(
                InputExample(texts=[q, index_titles[pos], index_titles[neg]], label=margin)
            )
    print(f"  built {len(examples):,} triplets ({n_skipped} queries skipped)", flush=True)

    # Loss target: scale CE-margins to match cosine-margin range (-1..2).
    # CE scores from the LiYuan regression model are ~0..1 (sigmoid-bounded).
    # Margins are typically 0.05..0.6. The MarginMSE loss will learn the scale
    # via the bi-encoder, but training is more stable if labels are scaled
    # reasonably. We multiply margin by a fixed factor.
    print("setting up model + loss...", flush=True)
    device = (
        "mps"
        if torch.backends.mps.is_available()
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    model = SentenceTransformer(args.base_model, device=device)
    print(f"  model: {args.base_model} on {device}", flush=True)

    # MarginMSELoss expects label = score(q, pos) - score(q, neg) where score
    # is the *teacher*'s score. The student learns to match this margin.
    loss_fn = losses.MarginMSELoss(model=model)

    train_loader = DataLoader(examples, shuffle=True, batch_size=args.batch_size)

    print(f"training {args.epochs} epochs, batch={args.batch_size}, lr={args.lr}...", flush=True)
    t0 = time.time()
    model.fit(
        train_objectives=[(train_loader, loss_fn)],
        epochs=args.epochs,
        warmup_steps=int(0.1 * len(train_loader) * args.epochs),
        optimizer_params={"lr": args.lr},
        show_progress_bar=True,
    )
    print(f"  trained in {time.time() - t0:.0f}s", flush=True)

    print(f"saving to {args.output_dir}/...", flush=True)
    model.save(args.output_dir)
    print("done.")


if __name__ == "__main__":
    main()
