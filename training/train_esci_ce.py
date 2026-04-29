#!/usr/bin/env python3
"""
Train a multilingual cross-encoder on Amazon ESCI graded relevance judgments.

Follows LiYuan's approach (regression on ESCI grades) but with a multilingual
base model so the CE handles US, ES, and JP product text uniformly.

ESCI labels -> grades:
  Exact       -> 1.0
  Substitute  -> 0.67
  Complement  -> 0.33
  Irrelevant  -> 0.0

Usage:
    python train_esci_ce.py --output-dir models/esci-multilingual-ce
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import os
import random

from datasets import load_dataset
from sentence_transformers import CrossEncoder, InputExample
from torch.utils.data import DataLoader

LABEL_MODES = {
    # Original ESCI-style regression targets (baseline)
    "regression": {"Exact": 1.0, "Substitute": 0.67, "Complement": 0.33, "Irrelevant": 0.0},
    # Widened — pushes E vs S apart for sharper threshold separation
    "wide": {"Exact": 1.0, "Substitute": 0.3, "Complement": 0.1, "Irrelevant": 0.0},
    # Binary — E only is "relevant"; everything else is treated as irrelevant
    "binary": {"Exact": 1.0, "Substitute": 0.0, "Complement": 0.0, "Irrelevant": 0.0},
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir", default="models/esci-multilingual-ce", help="Where to save the CE"
    )
    parser.add_argument(
        "--base-model",
        default="microsoft/Multilingual-MiniLM-L12-H384",
        help="HF model id to use as CE base (default: Multilingual MiniLM)",
    )
    parser.add_argument(
        "--locales",
        default="us,es,jp",
        help="Comma-separated ESCI product locales to include (default: all)",
    )
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument(
        "--max-length", type=int, default=128, help="Max sequence length (default: 128)"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=0, help="Cap total training examples (0=all)")
    parser.add_argument(
        "--label-mode",
        default="regression",
        choices=list(LABEL_MODES.keys()),
        help="Target encoding for ESCI grades. 'regression' (default) reproduces the "
        "original LiYuan-style baseline. 'wide' pushes E vs S apart for sharper "
        "threshold separation. 'binary' treats only E as relevant.",
    )
    args = parser.parse_args()
    label_grade = LABEL_MODES[args.label_mode]
    print(f"Label mode: {args.label_mode} -> {label_grade}")

    locales = set(args.locales.split(","))
    print(f"Locales: {locales}")

    print("Loading ESCI train data (tasksource/esci)...")
    ds = load_dataset("tasksource/esci", split="train")
    print(f"  raw train rows: {len(ds):,}")

    examples = []
    for row in ds:
        if row["product_locale"] not in locales:
            continue
        label = label_grade[row["esci_label"]]
        title = row["product_title"] or ""
        examples.append(InputExample(texts=[row["query"], title], label=label))
    print(f"  usable examples after locale filter: {len(examples):,}")

    if args.limit and args.limit < len(examples):
        rng = random.Random(args.seed)
        rng.shuffle(examples)
        examples = examples[: args.limit]
        print(f"  capped to {len(examples):,} examples")

    # Hold out 5% for eval during training
    rng = random.Random(args.seed)
    rng.shuffle(examples)
    n_val = max(500, len(examples) // 20)
    val_examples = examples[:n_val]
    train_examples = examples[n_val:]
    print(f"  train: {len(train_examples):,} | val: {len(val_examples):,}")

    model = CrossEncoder(args.base_model, num_labels=1, max_length=args.max_length)

    train_loader = DataLoader(train_examples, shuffle=True, batch_size=args.batch_size)
    warmup = int(0.1 * len(train_loader) * args.epochs)

    os.makedirs(args.output_dir, exist_ok=True)
    print(
        f"\nTraining: epochs={args.epochs}, batch={args.batch_size}, lr={args.lr}, warmup={warmup}"
    )
    print(f"Steps per epoch: {len(train_loader):,}")

    model.fit(
        train_dataloader=train_loader,
        epochs=args.epochs,
        warmup_steps=warmup,
        optimizer_params={"lr": args.lr},
        output_path=args.output_dir,
        save_best_model=True,
        show_progress_bar=True,
    )
    # Explicit save (belt & suspenders)
    model.save(args.output_dir)

    print(f"\nCE saved to {args.output_dir}")

    # Quick sanity check
    print("\nSanity check on held-out val (first 5 examples):")
    val_pairs = [(e.texts[0], e.texts[1]) for e in val_examples[:5]]
    val_labels = [e.label for e in val_examples[:5]]
    preds = model.predict(val_pairs)
    for (q, t), true, pred in zip(val_pairs, val_labels, preds):
        print(f"  true={true:.2f} pred={pred:.3f}  '{q[:40]}' <-> '{t[:60]}'")


if __name__ == "__main__":
    main()
