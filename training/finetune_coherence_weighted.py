#!/usr/bin/env python3
"""Train MiniLM with coherence-weighted bag sampling for MNRL.

For each bag, compute intra-bag cosine (mean pairwise cosine of the bag's
labeled positive products under rerank_A's product embeddings) — this is the
bag's *coherence* under the cluster hypothesis. Diffuse bags carry weaker
training signal; coherent bags carry sharper signal.

Then sample N total triplets (matching the baseline 6M-MNRL training scale)
with bag-selection probability proportional to coherence^alpha. This keeps
total training scale fixed but emphasizes coherent bags.

alpha=0  : uniform sampling (matches baseline rerank_A training)
alpha=1  : linear weighting by coherence
alpha=2+ : aggressive emphasis on highly coherent bags

Training otherwise identical to finetune_with_hardnegs.py: MNRL with explicit
hardnegs from BM25 mining, MiniLM-L6 base, batch=32, 1 epoch.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import random  # noqa: E402
import time  # noqa: E402

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


def compute_bag_coherences(bags, title_to_pos, p_vecs):
    """Mean pairwise cosine of labeled positive products in each bag."""
    coherences = np.zeros(len(bags), dtype=np.float32)
    n_no_data = 0
    for bi, bag in enumerate(bags):
        positions = []
        for r in bag.get("results", []):
            t = r.get("title")
            if t and t in title_to_pos:
                positions.append(title_to_pos[t])
        if len(positions) < 2:
            n_no_data += 1
            continue
        v = p_vecs[positions]
        sims = v @ v.T
        iu = np.triu_indices(len(sims), k=1)
        coherences[bi] = float(sims[iu].mean())
    print(
        f"  coherences: mean={coherences.mean():.3f}, "
        f"std={coherences.std():.3f}, "
        f"min={coherences.min():.3f}, "
        f"max={coherences.max():.3f}, "
        f"n_no_data={n_no_data}",
        flush=True,
    )
    return coherences


def sample_triplets_weighted(bags, weights, total_triplets, seed):
    """Sample total_triplets triplets, choosing bag with prob proportional to weights."""
    rng = random.Random(seed)
    nz = weights > 0
    bag_indices = np.where(nz)[0]
    bag_probs = weights[nz] / weights[nz].sum()

    np_rng = np.random.default_rng(seed)
    sampled_bag_idx = np_rng.choice(bag_indices, size=total_triplets, p=bag_probs)

    triplets = []
    skipped = 0
    for bi in sampled_bag_idx:
        bag = bags[bi]
        positives = [r["title"] for r in bag.get("results", []) if r.get("title")]
        hardnegs = bag.get("hardnegs", [])
        if not positives or not hardnegs:
            skipped += 1
            continue
        pos = rng.choice(positives)
        neg = rng.choice(hardnegs)
        triplets.append(InputExample(texts=[bag["query"], pos, neg]))

    print(f"  sampled {len(triplets):,} triplets (skipped {skipped} for missing data)", flush=True)
    return triplets


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("bags_file")
    ap.add_argument("output_dir")
    ap.add_argument("--base-model", default="all-MiniLM-L6-v2")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument(
        "--total-triplets",
        type=int,
        default=370000,
        help="Total triplets to sample (default 370K matches baseline 6M-MNRL training scale)",
    )
    ap.add_argument(
        "--alpha",
        type=float,
        default=1.0,
        help="Coherence exponent. 0 = uniform sampling. 1 = linear. >1 = aggressive emphasis.",
    )
    ap.add_argument(
        "--catalog-vecs",
        default=os.path.join(INDEX_DIR, "rerank_A.vecs.fp16.npy"),
        help="Catalog vectors used to compute bag coherence (default rerank_A)",
    )
    ap.add_argument("--titles", default=os.path.join(INDEX_DIR, "titles.json"))
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    print(f"loading bags from {args.bags_file}...", flush=True)
    bags = []
    with open(args.bags_file) as f:
        for line in f:
            bags.append(json.loads(line))
    print(f"  {len(bags):,} bags", flush=True)

    print(f"loading title map from {args.titles}...", flush=True)
    with open(args.titles) as f:
        index_titles = json.load(f)
    title_to_pos = {t: i for i, t in enumerate(index_titles)}

    print(f"loading catalog vecs from {args.catalog_vecs}...", flush=True)
    p_vecs = np.load(args.catalog_vecs).astype(np.float32)
    print(f"  catalog shape: {p_vecs.shape}", flush=True)

    print("\ncomputing bag coherences...", flush=True)
    t0 = time.time()
    coh = compute_bag_coherences(bags, title_to_pos, p_vecs)
    print(f"  took {time.time() - t0:.0f}s", flush=True)

    weights = np.maximum(coh, 0) ** args.alpha
    print(
        f"\nsampling triplets with alpha={args.alpha}, total_triplets={args.total_triplets:,}...",
        flush=True,
    )
    triplets = sample_triplets_weighted(bags, weights, args.total_triplets, args.seed)
    if not triplets:
        print("ERROR: no triplets sampled", file=sys.stderr)
        sys.exit(1)

    device = (
        "mps"
        if torch.backends.mps.is_available()
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"\nloading {args.base_model} on {device}...", flush=True)
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
