#!/usr/bin/env python3
"""Build a canonical mapping for the te3 query cache via FAISS-NN.

For each unique cache key, find te3 neighbors with cosine >= THRESH using a
FAISS IndexHNSWFlat (cosine via normalized inner product on fp32). Build an
undirected dup-graph, compute connected components, pick one canonical per
component (shortest token count, ties broken by source-preference and then
alphabetically).

Output: unified_jobs/te3_cache_canonical.json
  { "<non_canonical_key>": "<canonical_key>", ... }
Only non-canonical keys are written.

Usage:
  .venv/bin/python download/build_te3_cache_canonical.py [--thresh 0.97] [--topk 16]
"""

import argparse
import json
import sys
import time
from pathlib import Path

import faiss
import numpy as np

ROOT = Path(__file__).resolve().parent.parent


def load_cache_keys_and_vecs():
    """Mirror demo_jobs._load_te3_query_cache key/vec extraction (one row per unique key)."""
    triples = []  # (vec_path, id_path, source_tag)
    for d in ["jobs_data", "jobs_data_linkedin", "jobs_data_jobstreet", "jobs_data_usajobs"]:
        for split in ["train", "eval"]:
            triples.append(
                (
                    ROOT / d / f"{split}_queries_te3_1024.vecs.fp16.npy",
                    ROOT / d / f"{split}_queries_te3_1024.ids.json",
                    "synth",
                )
            )
    for stem in ("aug_titles", "aug_combos", "head_torso", "head_torso2"):
        triples.append(
            (
                ROOT / "unified_jobs" / f"{stem}_te3_1024.vecs.fp16.npy",
                ROOT / "unified_jobs" / f"{stem}_te3_1024.ids.json",
                "aug",
            )
        )

    parts = []
    queries: list[str] = []
    src_per_row: list[str] = []
    for vp, ip, tag in triples:
        if not vp.exists():
            print(f"  missing {vp.name}; skipping", file=sys.stderr)
            continue
        v = np.load(vp)
        with open(ip) as f:
            ids = json.load(f)
        if v.shape[0] != len(ids):
            raise SystemExit(f"size mismatch in {vp}")
        parts.append(v)
        queries.extend(ids)
        src_per_row.extend([tag] * len(ids))
    vecs = np.concatenate(parts, axis=0)

    # Build the same dedup as demo (aug overwrites synth)
    key_to_row: dict[str, int] = {}
    key_source: dict[str, str] = {}
    for i, q in enumerate(queries):
        k = q.strip().lower()
        if k not in key_to_row or (src_per_row[i] == "aug" and key_source.get(k) == "synth"):
            key_to_row[k] = i
            key_source[k] = src_per_row[i]

    keys = list(key_to_row.keys())
    rows = np.array([key_to_row[k] for k in keys], dtype=np.int64)
    uniq_vecs = vecs[rows]  # (n_keys, 1024) fp16
    return keys, uniq_vecs, key_source


def build_dup_pairs(vecs_fp32: np.ndarray, thresh: float, topk: int):
    """FAISS HNSW NN on normalized vectors → inner-product == cosine. Return list of (i, j) pairs (i<j) with cos>=thresh."""
    n, dim = vecs_fp32.shape
    # Normalize to unit length so IP == cosine.
    faiss.normalize_L2(vecs_fp32)
    index = faiss.IndexHNSWFlat(dim, 32, faiss.METRIC_INNER_PRODUCT)
    index.hnsw.efConstruction = 80
    index.hnsw.efSearch = 96
    t0 = time.time()
    index.add(vecs_fp32)
    print(f"  HNSW build: {time.time() - t0:.1f}s for {n:,} vectors", flush=True)

    t0 = time.time()
    D, I = index.search(vecs_fp32, topk)  # (n, topk)
    print(f"  HNSW search: {time.time() - t0:.1f}s top-{topk}", flush=True)

    pairs: list[tuple[int, int]] = []
    for i in range(n):
        for d, j in zip(D[i], I[i]):
            if j < 0 or j == i:
                continue
            if d < thresh:
                continue
            a, b = (i, j) if i < j else (j, i)
            pairs.append((a, b))
    pairs = list(set(pairs))
    print(f"  unique pairs at cos>={thresh}: {len(pairs):,}", flush=True)
    return pairs


def union_find(n, pairs):
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in pairs:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb
    return [find(i) for i in range(n)]


def pick_canonical(component_members: list[int], keys: list[str], key_source: dict) -> int:
    """Prefer: aug source > shortest token count > shorter chars > alphabetical."""

    def rank(i):
        k = keys[i]
        return (
            0 if key_source[k] == "aug" else 1,
            len(k.split()),
            len(k),
            k,
        )

    return min(component_members, key=rank)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--thresh", type=float, default=0.97)
    ap.add_argument("--topk", type=int, default=16)
    ap.add_argument("--out", default="unified_jobs/te3_cache_canonical.json")
    args = ap.parse_args()

    print("loading cache...", flush=True)
    keys, vecs_fp16, key_source = load_cache_keys_and_vecs()
    print(f"  {len(keys):,} unique keys, vecs={vecs_fp16.shape} {vecs_fp16.dtype}", flush=True)

    vecs_fp32 = vecs_fp16.astype(np.float32, copy=True)
    pairs = build_dup_pairs(vecs_fp32, args.thresh, args.topk)

    print("union-find on dup graph...", flush=True)
    root = union_find(len(keys), pairs)
    components: dict[int, list[int]] = {}
    for i, r in enumerate(root):
        components.setdefault(r, []).append(i)

    multi = {r: m for r, m in components.items() if len(m) > 1}
    print(
        f"  components: {len(components):,} total, {len(multi):,} multi-member (= dup clusters)",
        flush=True,
    )

    mapping: dict[str, str] = {}
    for members in multi.values():
        canon_i = pick_canonical(members, keys, key_source)
        canon_key = keys[canon_i]
        for j in members:
            if j != canon_i:
                mapping[keys[j]] = canon_key

    print(f"  non-canonical keys written: {len(mapping):,}", flush=True)

    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(mapping, f)
    print(f"wrote {out_path}", flush=True)

    # Sample diagnostics
    samples = list(mapping.items())[:15]
    print("\nsample (non_canonical -> canonical):")
    for nc, c in samples:
        print(f"  {nc:50s}  ->  {c}")


if __name__ == "__main__":
    main()
