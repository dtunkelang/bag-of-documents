#!/usr/bin/env python3
"""LoRA fine-tune a large base encoder on BoD-style triplets.

Variant of `finetune_with_hardnegs.py` that wraps the base transformer in a
LoRA adapter (peft). Trains only adapter weights (~5-10M params instead of
~600M for Solon-large/Algolia), so optimizer state fits MPS unified memory.

Saves a sentence-transformers-compatible model at output_dir with the
adapter merged into the base weights.
"""

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402
from peft import LoraConfig, get_peft_model  # noqa: E402
from sentence_transformers import (  # noqa: E402
    InputExample,
    SentenceTransformer,
    losses,
)
from torch.utils.data import DataLoader  # noqa: E402


def load_triplets(bags_path, n_per_bag, seed, max_triplets=None):
    rng = random.Random(seed)
    triplets = []
    with open(bags_path) as f:
        for line in f:
            bag = json.loads(line)
            results = bag.get("results", [])
            hardnegs = bag.get("hardnegs", [])
            positives = [r["title"] for r in results if r.get("title")]
            if not positives or not hardnegs:
                continue
            for _ in range(n_per_bag):
                pos = rng.choice(positives)
                neg = rng.choice(hardnegs)
                triplets.append(InputExample(texts=[bag["query"], pos, neg]))
    if max_triplets and len(triplets) > max_triplets:
        rng.shuffle(triplets)
        triplets = triplets[:max_triplets]
    return triplets


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("bags_file")
    ap.add_argument("output_dir")
    ap.add_argument("--base-model", required=True)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4, help="higher LR for LoRA")
    ap.add_argument("--triplets-per-bag", type=int, default=2)
    ap.add_argument(
        "--max-triplets",
        type=int,
        default=None,
        help="cap dataset to this many triplets (random subsample, uses --seed). "
        "Useful for short cycles to stay below memory-pressure slowdown threshold.",
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-seq-length", type=int, default=256)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument(
        "--checkpoint-save-steps",
        type=int,
        default=500,
        help="save a checkpoint every N steps (also keeps last 4 by default)",
    )
    ap.add_argument("--checkpoint-save-total-limit", type=int, default=4)
    ap.add_argument(
        "--resume-from",
        default=None,
        help="resume LoRA adapter weights from a previously saved checkpoint "
        "dir (containing adapter_model.safetensors). Optimizer state resets; "
        "saves ~95%% of training progress vs cold restart.",
    )
    ap.add_argument(
        "--target-modules",
        default="query,key,value",
        help="comma-separated module name substrings to target (XLM-RoBERTa-large default fits Solon)",
    )
    args = ap.parse_args()

    print(f"loading bags from {args.bags_file}...", flush=True)
    t0 = time.time()
    triplets = load_triplets(args.bags_file, args.triplets_per_bag, args.seed, args.max_triplets)
    print(f"  {len(triplets):,} triplets in {time.time() - t0:.0f}s", flush=True)

    device = (
        "mps"
        if torch.backends.mps.is_available()
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"loading {args.base_model} on {device}...", flush=True)
    model = SentenceTransformer(args.base_model, device=device)
    model.max_seq_length = args.max_seq_length

    # Wrap the underlying transformer in LoRA
    transformer = model._first_module().auto_model
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=[m.strip() for m in args.target_modules.split(",")],
        lora_dropout=args.lora_dropout,
        bias="none",
    )
    peft_model = get_peft_model(transformer, lora_config)
    model._first_module().auto_model = peft_model
    peft_model.print_trainable_parameters()

    if args.resume_from:
        adapter_path = os.path.join(args.resume_from, "adapter_model.safetensors")
        if not os.path.exists(adapter_path):
            raise FileNotFoundError(
                f"--resume-from {args.resume_from!r} missing adapter_model.safetensors"
            )
        print(f"resuming LoRA adapter from {args.resume_from}", flush=True)
        # peft.load_adapter handles the `.default.` naming convention internally.
        peft_model.load_adapter(args.resume_from, adapter_name="default", is_trainable=True)
        # Verify the active adapter has non-zero norms (load happened).
        first_lora_param = next(
            (p for n, p in peft_model.named_parameters() if "lora_A" in n and "default" in n),
            None,
        )
        if first_lora_param is None or first_lora_param.abs().sum().item() == 0:
            raise RuntimeError("resume failed: LoRA params look unloaded (zero)")
        print("  resume verified (first lora_A param non-zero)", flush=True)

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

    os.makedirs(args.output_dir, exist_ok=True)
    ckpt_path = args.output_dir + "_ckpts"
    os.makedirs(ckpt_path, exist_ok=True)
    print(f"  checkpoints every {args.checkpoint_save_steps} steps -> {ckpt_path}", flush=True)
    model.fit(
        train_objectives=[(train_loader, train_loss)],
        epochs=args.epochs,
        warmup_steps=warmup_steps,
        optimizer_params={"lr": args.lr},
        output_path=args.output_dir,
        checkpoint_path=ckpt_path,
        checkpoint_save_steps=args.checkpoint_save_steps,
        checkpoint_save_total_limit=args.checkpoint_save_total_limit,
        show_progress_bar=True,
    )

    # Merge LoRA into base weights for inference compatibility, then save
    # the merged model in sentence-transformers format.
    print("\nmerging LoRA into base weights...", flush=True)
    merged = peft_model.merge_and_unload()
    model._first_module().auto_model = merged
    model.save(args.output_dir)
    print(f"saved merged model to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
