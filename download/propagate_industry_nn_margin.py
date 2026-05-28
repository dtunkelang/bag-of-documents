"""NN-with-margin-gate propagation over bge-small slug embeddings.

Reuses cached embeddings from propagate_industry_bge_logreg.py. For each
unclassified slug, finds top-K nearest labeled slugs by cosine, votes the
label, and accepts only when:
  top-1 cosine >= --sim-floor
  margin(top-1, top-K) >= --margin
  top-1 label is the majority of top-K labels (>= --vote-frac)

Outputs unified_jobs/slug_industry_labels_nn_margin.csv with one row per
accepted prediction: slug, industry, top1_sim, margin, vote_frac, top_labels.
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
UJ = REPO / "unified_jobs"
TEXT_BAG = UJ / "slug_text_bag.csv"
EMB_CACHE = UJ / "slug_text_bag_bge.npy"
OVERRIDES = UJ / "slug_industry_overrides.csv"
TAIL3 = UJ / "tail3_unclassified_for_labeling.csv"


def load_text_bag_slugs() -> list[str]:
    slugs = []
    with TEXT_BAG.open() as f:
        r = csv.DictReader(f)
        for row in r:
            slugs.append(row["slug"])
    return slugs


def load_overrides() -> dict[str, str]:
    out = {}
    with OVERRIDES.open() as f:
        r = csv.DictReader(f)
        for row in r:
            s = row["slug"].strip()
            ind = row["industry"].strip()
            if s and ind:
                out[s] = ind
    return out


def load_tail3_slugs() -> list[str]:
    slugs = []
    with TAIL3.open() as f:
        r = csv.DictReader(f)
        for row in r:
            slugs.append(row["slug"])
    return slugs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=10, help="top-K neighbors to consider")
    ap.add_argument("--sim-floor", type=float, default=0.55, help="min top-1 cosine")
    ap.add_argument("--margin", type=float, default=0.05, help="top1 - topK cosine margin")
    ap.add_argument("--vote-frac", type=float, default=0.6, help="majority vote fraction in top-K")
    ap.add_argument("--out", type=Path, default=UJ / "slug_industry_labels_nn_margin.csv")
    args = ap.parse_args()

    print("loading slug text bag and embeddings ...")
    slugs = load_text_bag_slugs()
    X = np.load(EMB_CACHE).astype(np.float32)
    print(f"  {len(slugs):,} slugs, embedding shape {X.shape}")
    assert len(slugs) == X.shape[0], "slug/embedding length mismatch"
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    X = X / np.clip(norms, 1e-8, None)

    overrides = load_overrides()
    targets = load_tail3_slugs()
    print(f"  {len(overrides):,} labeled overrides, {len(targets):,} tail-3 targets")

    slug_to_idx = {s: i for i, s in enumerate(slugs)}
    label_idx = [slug_to_idx[s] for s in overrides if s in slug_to_idx]
    label_industries = [overrides[slugs[i]] for i in label_idx]
    target_idx = [slug_to_idx[s] for s in targets if s in slug_to_idx]
    missing_t = len(targets) - len(target_idx)
    missing_l = len(overrides) - len(label_idx)
    print(f"  labeled in embedding cache: {len(label_idx):,} (missing {missing_l})")
    print(f"  targets in embedding cache: {len(target_idx):,} (missing {missing_t})")

    L = X[label_idx]
    T = X[target_idx]
    print(f"computing {T.shape[0]:,} x {L.shape[0]:,} cosine ...")
    S = T @ L.T  # shape (n_targets, n_labeled)

    K = args.k
    print(f"top-{K} neighbors per target ...")
    topk_idx = np.argpartition(-S, K, axis=1)[:, :K]
    rows = np.arange(S.shape[0])[:, None]
    topk_sim = S[rows, topk_idx]
    order = np.argsort(-topk_sim, axis=1)
    topk_idx = topk_idx[rows, order]
    topk_sim = topk_sim[rows, order]

    accepted = []
    for ti, (sim_row, idx_row) in enumerate(zip(topk_sim, topk_idx)):
        top1 = float(sim_row[0])
        topK = float(sim_row[-1])
        if top1 < args.sim_floor:
            continue
        if (top1 - topK) < args.margin:
            continue
        labels = [label_industries[i] for i in idx_row]
        c = Counter(labels)
        majority_label, count = c.most_common(1)[0]
        vote_frac = count / K
        if vote_frac < args.vote_frac:
            continue
        if labels[0] != majority_label:
            continue
        accepted.append(
            {
                "slug": slugs[target_idx[ti]],
                "industry": majority_label,
                "top1_sim": round(top1, 4),
                "topK_sim": round(topK, 4),
                "margin": round(top1 - topK, 4),
                "vote_frac": round(vote_frac, 2),
                "top1_neighbor": slugs[label_idx[idx_row[0]]],
                "top_labels": " | ".join(f"{lab}:{ct}" for lab, ct in c.most_common(5)),
            }
        )

    print(
        f"\naccepted {len(accepted):,}/{len(target_idx):,} targets ({100 * len(accepted) / max(1, len(target_idx)):.1f}%)"
    )
    by_industry = Counter(r["industry"] for r in accepted)
    for ind, n in by_industry.most_common():
        print(f"  {n:4d}  {ind}")

    with args.out.open("w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "slug",
                "industry",
                "top1_sim",
                "topK_sim",
                "margin",
                "vote_frac",
                "top1_neighbor",
                "top_labels",
            ],
        )
        w.writeheader()
        for r in accepted:
            w.writerow(r)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
