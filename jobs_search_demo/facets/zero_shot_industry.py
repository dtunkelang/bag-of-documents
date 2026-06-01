#!/usr/bin/env python3
"""Zero-shot industry classification via bge-small label embeddings + cosine.

Reads the pre-computed 347900x384 bge-small doc vectors and matmuls against
embeddings of hand-written industry descriptions. Output: per-doc JSONL with
{idx, industry, margin, top2}.

Usage:
  python zero_shot_industry.py --out zero_shot.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

sys.path.insert(0, str(Path(__file__).parent))
from industry_labels import get_descriptions  # noqa: E402
from taxonomy import INDUSTRY  # noqa: E402

DOC_VECS = Path("/Users/dtunkelang/bagofdocs/unified_jobs/bge_catalog.vecs.fp16.npy")
MODEL = "BAAI/bge-small-en-v1.5"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch", type=int, default=4096, help="doc batch for matmul")
    args = ap.parse_args()

    t0 = time.time()
    print(f"loading model {MODEL}", flush=True)
    model = SentenceTransformer(MODEL)

    descs = get_descriptions(INDUSTRY)
    print(f"embedding {len(descs)} label descriptions", flush=True)
    # BGE asks for instruction prefix on queries; here labels = queries.
    label_vecs = model.encode(
        descs,
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).astype(np.float32)
    print(f"label_vecs shape: {label_vecs.shape}", flush=True)

    print(f"mmap-loading {DOC_VECS}", flush=True)
    docs = np.load(DOC_VECS, mmap_mode="r")
    n, d = docs.shape
    print(f"docs: {n:,} x {d}", flush=True)
    assert d == label_vecs.shape[1], "embedding dim mismatch"

    print(f"writing to {args.out}", flush=True)
    with open(args.out, "w") as out_f:
        for start in range(0, n, args.batch):
            end = min(start + args.batch, n)
            batch = np.asarray(docs[start:end], dtype=np.float32)
            # Both sides L2-normalized -> dot = cosine.
            sims = batch @ label_vecs.T  # (B, 28)
            top2_idx = np.argsort(-sims, axis=1)[:, :2]
            for i in range(end - start):
                a, b = top2_idx[i]
                margin = float(sims[i, a] - sims[i, b])
                rec = {
                    "idx": start + i,
                    "industry": INDUSTRY[a],
                    "score": float(sims[i, a]),
                    "margin": margin,
                    "top2": INDUSTRY[b],
                }
                out_f.write(json.dumps(rec) + "\n")
            if (end // args.batch) % 10 == 0:
                rate = end / (time.time() - t0)
                print(f"  {end:,}/{n:,} ({rate:,.0f} docs/s)", flush=True)
    print(f"done in {time.time() - t0:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
