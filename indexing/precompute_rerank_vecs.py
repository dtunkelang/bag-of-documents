#!/usr/bin/env python3
"""Pre-compute reranker embeddings for every product in an index.

For a given reranker model, encode all titles in <index>/titles.json and save
the result as a fp16 numpy array. demo.py loads these precomputed matrices at
startup and uses lookup+dot at query time instead of encoding candidates live.

This is the production-deployable form of the BoD-as-reranker pipeline:
per-query overhead drops from ~200 forward passes to 2 query encodes + 2
indexed dot products.

Usage:
    python precompute_rerank_vecs.py query_model_us_full_6m_mnrl \\
        combined_index_us_minilm/query_model_us_full_6m_mnrl.vecs.fp16.npy
"""

import argparse
import json
import os
import time

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path", help="Reranker model directory")
    parser.add_argument("output_npy", help="Output .npy file path (will store fp16)")
    parser.add_argument(
        "--index-dir",
        default=os.path.join(SCRIPT_DIR, "combined_index_us_minilm"),
        help="Index directory whose titles.json to encode",
    )
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default="auto", choices=["auto", "mps", "cuda", "cpu"])
    args = parser.parse_args()

    if args.device == "auto":
        if torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"
    else:
        device = args.device

    print(f"device: {device}", flush=True)
    print(f"loading model from {args.model_path}...", flush=True)
    model = SentenceTransformer(args.model_path, device=device)

    titles_path = os.path.join(args.index_dir, "titles.json")
    print(f"loading {titles_path}...", flush=True)
    with open(titles_path) as f:
        titles = json.load(f)
    print(f"  {len(titles):,} titles", flush=True)

    print(f"encoding (batch={args.batch_size})...", flush=True)
    t0 = time.time()
    vecs = model.encode(
        titles,
        normalize_embeddings=True,
        batch_size=args.batch_size,
        show_progress_bar=True,
    )
    elapsed = time.time() - t0
    rate = len(titles) / max(elapsed, 1e-3)
    print(f"  done in {elapsed:.0f}s ({rate:.0f} titles/sec)", flush=True)

    print(f"saving fp16 vecs to {args.output_npy}...", flush=True)
    np.save(args.output_npy, np.asarray(vecs, dtype=np.float16))
    size_mb = os.path.getsize(args.output_npy) / 1e6
    print(f"  wrote {size_mb:.0f} MB ({vecs.shape})")


if __name__ == "__main__":
    main()
