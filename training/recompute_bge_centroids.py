#!/usr/bin/env python3
"""Recompute bag centroids using BGE-base (or any other base) instead of MiniLM.

bags.jsonl ships with `query_vector` = MiniLM-encoded bag centroid (384-dim).
The cosine-to-centroid training recipe (--loss cos in finetune_query_model.py)
uses this as the target, so dim has to match the model being trained.

For BGE-base (768-dim) we recompute: encode each bag's member titles with
BGE, take the spherical mean, write the result back as a new bags file.

Usage:
    python training/recompute_bge_centroids.py \\
        --bags bags.jsonl \\
        --base-model BAAI/bge-base-en-v1.5 \\
        --output bags_bge_centroids.jsonl
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
from sentence_transformers import SentenceTransformer  # noqa: E402

SCRIPT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bags", default=os.path.join(SCRIPT_DIR, "bags.jsonl"))
    ap.add_argument("--base-model", default="BAAI/bge-base-en-v1.5")
    ap.add_argument("--output", default=os.path.join(SCRIPT_DIR, "bags_bge_centroids.jsonl"))
    ap.add_argument("--batch-size", type=int, default=128)
    args = ap.parse_args()

    device = (
        "mps"
        if torch.backends.mps.is_available()
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"loading {args.base_model} on {device}...", flush=True)
    model = SentenceTransformer(args.base_model, device=device)

    # Pass 1: collect all unique titles across all bags so we can encode each once.
    print(f"reading bags from {args.bags} to collect unique titles...", flush=True)
    unique_titles = []
    seen = set()
    n_bags = 0
    with open(args.bags) as f:
        for line in f:
            n_bags += 1
            bag = json.loads(line)
            for r in bag.get("results", []):
                t = r.get("title")
                if t and t not in seen:
                    seen.add(t)
                    unique_titles.append(t)
    print(f"  {n_bags:,} bags, {len(unique_titles):,} unique titles", flush=True)

    # Pass 2: encode the unique titles in one batched pass.
    print(f"encoding {len(unique_titles):,} titles...", flush=True)
    t0 = time.time()
    title_vecs = model.encode(
        unique_titles,
        normalize_embeddings=True,
        batch_size=args.batch_size,
        show_progress_bar=True,
    )
    title_vecs = np.asarray(title_vecs, dtype=np.float32)
    print(f"  encoded in {time.time() - t0:.0f}s, shape={title_vecs.shape}", flush=True)
    title_to_vec = dict(zip(unique_titles, title_vecs))

    # Pass 3: rewrite bags with new centroids + specificity.
    print(f"writing centroids to {args.output}...", flush=True)
    n_with_results = 0
    n_skipped = 0
    with open(args.bags) as fin, open(args.output, "w") as fout:
        for line in fin:
            bag = json.loads(line)
            members = [r.get("title") for r in bag.get("results", []) if r.get("title")]
            vecs = np.stack([title_to_vec[t] for t in members]) if members else None
            if vecs is None or len(vecs) == 0:
                bag["query_vector"] = None
                bag["specificity"] = 0.0
                n_skipped += 1
            else:
                centroid = vecs.mean(axis=0)
                norm = np.linalg.norm(centroid)
                if norm > 0:
                    centroid = centroid / norm
                bag["query_vector"] = centroid.tolist()
                # Specificity = mean cos(centroid, member); members already normalized.
                bag["specificity"] = float(np.mean(vecs @ centroid))
                n_with_results += 1
            fout.write(json.dumps(bag) + "\n")

    print(
        f"  wrote {n_with_results:,} bags with centroids, {n_skipped:,} skipped (empty)",
        flush=True,
    )


if __name__ == "__main__":
    main()
