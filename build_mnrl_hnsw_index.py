#!/usr/bin/env python3
"""Build a FAISS HNSW index from cached 6M-MNRL product embeddings.

Reads rerank_A.vecs.fp16.npy and writes rerank_A.index.faiss alongside it,
enabling sub-100ms retrieval through the 6M-MNRL product space (vs ~190s
brute force for 22K queries).
"""

import os
import time

import faiss
import numpy as np

INDEX_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "combined_index_us_minilm")
VECS_PATH = os.path.join(INDEX_DIR, "rerank_A.vecs.fp16.npy")
OUT_PATH = os.path.join(INDEX_DIR, "rerank_A.index.faiss")


def main():
    print(f"loading {VECS_PATH}...", flush=True)
    vecs = np.load(VECS_PATH).astype(np.float32)
    n, d = vecs.shape
    print(f"  {n:,} × {d} float32 ({vecs.nbytes / 1e9:.1f} GB)", flush=True)

    # Vectors are already L2-normalized at precompute time, but renormalize
    # defensively in case of fp16 round-trip drift.
    faiss.normalize_L2(vecs)

    M = 32
    print(f"building HNSW (M={M}, METRIC_INNER_PRODUCT)...", flush=True)
    idx = faiss.IndexHNSWFlat(d, M, faiss.METRIC_INNER_PRODUCT)
    idx.hnsw.efConstruction = 200

    t0 = time.time()
    idx.add(vecs)
    print(f"  added {idx.ntotal:,} vectors in {time.time() - t0:.0f}s", flush=True)

    print(f"writing {OUT_PATH}...", flush=True)
    faiss.write_index(idx, OUT_PATH)
    size = os.path.getsize(OUT_PATH) / 1e9
    print(f"  wrote {size:.2f} GB", flush=True)


if __name__ == "__main__":
    main()
