"""Propagate 500 seed industry labels to all ~50K employer slugs via bge-small kNN.

Steps:
  1) Mean-pool bge_catalog vectors by source_slug -> per-slug centroid.
  2) For each unlabeled slug, find k=10 nearest labeled-slug centroids by cosine.
  3) Vote (sum of cosine similarity per class). Assign argmax label.
  4) Also emit a confidence proxy (top-class share of total positive vote mass).
  5) Write a CSV: slug, n_jobs, industry, conf_score, top1_seed, top1_sim.
"""

import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(".")
DOC_IDS = ROOT / "unified_jobs/doc_ids.json"
VECS = ROOT / "unified_jobs/bge_catalog.vecs.fp16.npy"
SEEDS = ROOT / "unified_jobs/top500_slugs_labeled_v2.csv"
OUT_SLUG = ROOT / "unified_jobs/slug_industry_labels.csv"
OUT_DOC = ROOT / "unified_jobs/doc_industry_labels.tsv"

K = 10  # neighbors to vote among


def main() -> None:
    print("loading doc_ids and vectors...")
    with open(DOC_IDS) as f:
        ids = json.load(f)
    vecs = np.load(VECS, mmap_mode="r")
    assert len(ids) == vecs.shape[0]
    print(f"  {len(ids):,} docs, dim={vecs.shape[1]}")

    print("aggregating per-slug centroids (sum then normalize)...")
    slug_sum: dict[str, np.ndarray] = {}
    slug_n: Counter[str] = Counter()
    slug_src: dict[str, str] = {}
    for i, did in enumerate(ids):
        src, slug, _ = did.split(":", 2)
        v = vecs[i].astype(np.float32)
        if slug in slug_sum:
            slug_sum[slug] += v
        else:
            slug_sum[slug] = v.copy()
            slug_src[slug] = src
        slug_n[slug] += 1
    print(f"  {len(slug_sum):,} unique slugs")

    slugs = list(slug_sum.keys())
    centroids = np.stack([slug_sum[s] / slug_n[s] for s in slugs]).astype(np.float32)
    centroids /= np.linalg.norm(centroids, axis=1, keepdims=True).clip(min=1e-8)
    print(f"  centroid matrix: {centroids.shape}")

    print("loading seed labels...")
    slug_to_idx = {s: i for i, s in enumerate(slugs)}
    seed_label: dict[str, str] = {}
    missing = 0
    with SEEDS.open() as f:
        for r in csv.DictReader(f):
            if r["industry"] == "other":  # don't propagate `other`
                continue
            if r["slug"] in slug_to_idx:
                seed_label[r["slug"]] = r["industry"]
            else:
                missing += 1
    print(f"  {len(seed_label)} seed slugs hit the centroid matrix ({missing} not found)")

    seed_slugs = list(seed_label.keys())
    seed_idx = np.array([slug_to_idx[s] for s in seed_slugs])
    seed_vecs = centroids[seed_idx]
    seed_labels = np.array([seed_label[s] for s in seed_slugs])

    print(f"computing cosine sims ({centroids.shape[0]:,} x {seed_vecs.shape[0]})...")
    sims = centroids @ seed_vecs.T  # [N_slug, N_seed]
    print(f"  sims matrix: {sims.shape} {sims.dtype}")

    # 1-NN per class: for each slug, pick the class whose nearest seed has highest sim.
    # this removes the class-base-rate issue (consulting has 71 seeds, banking 22 -- shouldn't matter).
    print("1-NN per class...")
    # for each (slug, class) pair, find the max sim to any seed of that class
    classes = sorted(set(seed_labels.tolist()))
    cls_to_seeds = {c: np.where(seed_labels == c)[0] for c in classes}
    # max-sim per class via masked argmax
    rows = sims.shape[0]
    per_class_max = np.full((rows, len(classes)), -1.0, dtype=np.float32)
    for ci, c in enumerate(classes):
        idxs = cls_to_seeds[c]
        per_class_max[:, ci] = sims[:, idxs].max(axis=1)
    cls_arr = np.array(classes)
    best_ci = per_class_max.argmax(axis=1)
    best_sim = per_class_max[np.arange(rows), best_ci]
    # runner-up sim for a margin-based confidence
    per_class_max_copy = per_class_max.copy()
    per_class_max_copy[np.arange(rows), best_ci] = -1.0
    runner_up_sim = per_class_max_copy.max(axis=1)
    margin = best_sim - runner_up_sim  # higher = more confident
    propagated = []
    for i in range(rows):
        if i % 10000 == 0:
            print(f"  {i:,}/{rows:,}")
        lbl = str(cls_arr[best_ci[i]])
        # find which seed exactly was the top-1
        top_seed_local_idx = int(sims[i, cls_to_seeds[lbl]].argmax())
        top_seed = seed_slugs[cls_to_seeds[lbl][top_seed_local_idx]]
        propagated.append(
            (
                slugs[i],
                slug_n[slugs[i]],
                slug_src[slugs[i]],
                lbl,
                float(margin[i]),
                top_seed,
                float(best_sim[i]),
            )
        )

    # override propagated labels for seeds themselves
    seed_set = set(seed_label.keys())
    for i, p in enumerate(propagated):
        if p[0] in seed_set:
            propagated[i] = (p[0], p[1], p[2], seed_label[p[0]], 1.0, p[0], 1.0)

    print(f"writing {OUT_SLUG}...")
    with OUT_SLUG.open("w", newline="") as f:
        w = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
        w.writerow(["slug", "n_jobs", "source", "industry", "conf_score", "top1_seed", "top1_sim"])
        for slug, n, src, lbl, conf, t1, ts in propagated:
            w.writerow([slug, n, src, lbl, f"{conf:.3f}", t1, f"{ts:.3f}"])
    print(f"  wrote {len(propagated):,} rows")

    print(f"writing {OUT_DOC} (doc_id\\tindustry) ...")
    slug_to_label = {p[0]: p[3] for p in propagated}
    with OUT_DOC.open("w") as f:
        for did in ids:
            _, slug, _ = did.split(":", 2)
            f.write(f"{did}\t{slug_to_label.get(slug, 'other')}\n")

    # distribution summary
    ind = Counter(p[3] for p in propagated)
    doc_ind = Counter(slug_to_label[did.split(":", 2)[1]] for did in ids)
    print("\n=== per-slug industry distribution ===")
    for k, v in ind.most_common():
        print(f"  {v:6,d}  {k}")
    print("\n=== per-doc industry distribution ===")
    for k, v in doc_ind.most_common():
        print(f"  {v:7,d}  {k}")


if __name__ == "__main__":
    main()
