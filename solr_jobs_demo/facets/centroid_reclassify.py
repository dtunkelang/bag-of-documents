#!/usr/bin/env python3
"""Centroid-based reclassification of role_family for the 'other' bucket.

For each role_family value that regex confidently assigned, compute the centroid
of those titles in bge-small (title-only) embedding space. Re-classify titles
currently labeled 'other' by nearest centroid. Only reassign if the best cosine
exceeds CONF_THRESHOLD; otherwise keep as 'other'.
"""

import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

FACETS_IN = Path("/Users/dtunkelang/bagofdocs/solr_jobs_demo/facets/facets.jsonl")
FACETS_OUT = Path("/Users/dtunkelang/bagofdocs/solr_jobs_demo/facets/facets.v2.jsonl")
TITLE_VECS = Path("/Users/dtunkelang/bagofdocs/solr_jobs_demo/facets/title_bge.vecs.fp16.npy")
CONF_THRESHOLD = 0.65  # min cosine to override 'other'
MIN_SEEDS_PER_FAMILY = 50  # don't build a centroid from fewer seeds than this


def main() -> int:
    print("loading title embeddings ...", flush=True)
    title_vecs = np.load(TITLE_VECS, mmap_mode="r")
    print(f"  {title_vecs.shape}", flush=True)

    print("loading facets ...", flush=True)
    family_by_idx: dict[int, str] = {}
    all_records: list[dict] = []
    with open(FACETS_IN) as f:
        for line in f:
            d = json.loads(line)
            all_records.append(d)
            family_by_idx[d["idx"]] = d["role_family"]
    print(f"  {len(all_records):,} records", flush=True)

    # Group indices by family.
    by_family: dict[str, list[int]] = {}
    for idx, fam in family_by_idx.items():
        by_family.setdefault(fam, []).append(idx)

    print("\nseed counts per family:")
    for fam, idxs in sorted(by_family.items(), key=lambda x: -len(x[1])):
        print(f"  {len(idxs):>7,}  {fam}")

    # Build centroids for families with enough seeds, excluding 'other'.
    print(f"\nbuilding centroids (min {MIN_SEEDS_PER_FAMILY} seeds) ...", flush=True)
    families: list[str] = []
    centroids_list: list[np.ndarray] = []
    for fam, idxs in by_family.items():
        if fam == "other":
            continue
        if len(idxs) < MIN_SEEDS_PER_FAMILY:
            print(f"  skip {fam} (only {len(idxs)} seeds)", flush=True)
            continue
        # Sample up to 5000 seeds to keep this fast — centroid is stable.
        if len(idxs) > 5000:
            idxs_sample = list(np.random.default_rng(42).choice(idxs, 5000, replace=False))
        else:
            idxs_sample = idxs
        seed_vecs = title_vecs[np.array(idxs_sample)].astype(np.float32)
        c = seed_vecs.mean(axis=0)
        c /= max(np.linalg.norm(c), 1e-9)
        families.append(fam)
        centroids_list.append(c)
    centroids = np.stack(centroids_list)
    print(f"  {len(families)} centroids x {centroids.shape[1]} dims", flush=True)

    # Find indices currently labeled 'other'.
    other_idxs = np.array(sorted(by_family.get("other", [])))
    print(f"\nreclassifying {len(other_idxs):,} 'other' titles ...", flush=True)
    t0 = time.time()
    # In chunks (keeps peak memory reasonable).
    chunk = 50_000
    reassigned: dict[int, tuple[str, float]] = {}
    for i in range(0, len(other_idxs), chunk):
        block_idxs = other_idxs[i : i + chunk]
        block = title_vecs[block_idxs].astype(np.float32)
        sims = block @ centroids.T  # (chunk, n_families)
        best = sims.argmax(axis=1)
        conf = sims.max(axis=1)
        for j, b, c in zip(block_idxs, best, conf):
            if c >= CONF_THRESHOLD:
                reassigned[int(j)] = (families[b], float(c))
    print(f"  done in {time.time() - t0:.1f}s", flush=True)
    print(
        f"  {len(reassigned):,} reassigned (>= {CONF_THRESHOLD}); "
        f"{len(other_idxs) - len(reassigned):,} still 'other'",
        flush=True,
    )

    # Distribution of reassignments.
    counter = Counter(v[0] for v in reassigned.values())
    print("\nreassignment distribution:")
    for fam, n in counter.most_common():
        print(f"  {n:>5,}  {fam}")

    # Write updated facets file.
    print(f"\nwriting {FACETS_OUT} ...", flush=True)
    n_changed = 0
    with open(FACETS_OUT, "w") as f:
        for rec in all_records:
            idx = rec["idx"]
            if idx in reassigned:
                rec["role_family"] = reassigned[idx][0]
                rec["role_family_source"] = "centroid"
                n_changed += 1
            else:
                rec["role_family_source"] = "regex"
            f.write(json.dumps(rec) + "\n")
    print(f"done. {n_changed:,}/{len(all_records):,} records reclassified.", flush=True)

    # Final distribution
    final = Counter(r["role_family"] for r in all_records)
    total = sum(final.values())
    print("\nfinal role_family distribution:")
    for v, n in final.most_common():
        print(f"  {n:>7,} ({100 * n / total:>5.1f}%)  {v}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
