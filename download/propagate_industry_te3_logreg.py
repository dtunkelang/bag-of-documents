"""TE3-embedding logistic-regression industry classifier.

Trains a multinomial logreg on per-DOC TE3 vectors using the 1,404
hand-labeled slugs in slug_industry_overrides.csv as ground truth — every
doc under a labeled slug inherits the slug's industry as its training label.

Reports slug-grouped 5-fold CV macro/micro F1 (no slug leak between
train/test), then predicts on docs of unclassified slugs. Per-slug industry
is the doc-majority winner gated by share-of-docs and mean confidence.

Outputs:
  slug_industry_labels_te3_logreg.csv  (slug-level predictions)
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import normalize

REPO = Path(__file__).resolve().parent.parent
UJ = REPO / "unified_jobs"


def load_corpus():
    with (UJ / "doc_ids.json").open() as f:
        doc_ids = json.load(f)
    vecs = np.load(UJ / "te3_catalog.vecs.fp16.npy", mmap_mode="r")
    assert len(doc_ids) == vecs.shape[0]
    return doc_ids, vecs


def load_overrides():
    out = {}
    with (UJ / "slug_industry_overrides.csv").open() as f:
        r = csv.DictReader(f)
        for row in r:
            slug = row["slug"].strip()
            ind = row["industry"].strip()
            if slug and ind:
                out[slug] = ind
    return out


def build_slug_index(doc_ids):
    idx = defaultdict(list)
    for i, did in enumerate(doc_ids):
        parts = did.split(":", 2)
        if len(parts) >= 2:
            idx[parts[1]].append(i)
    return idx


def build_doc_matrix(vecs, rows):
    """Load a subset of fp16 rows as fp32 + L2-normalize."""
    sub = vecs[rows].astype(np.float32)
    return normalize(sub, norm="l2")


def cv_eval(X, y, groups, n_splits=5, C=1.0):
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)
    oof = np.empty_like(y, dtype=object)
    for fold, (tr, te) in enumerate(sgkf.split(X, y, groups), 1):
        clf = LogisticRegression(solver="lbfgs", max_iter=1000, C=C)
        clf.fit(X[tr], y[tr])
        oof[te] = clf.predict(X[te])
        f1 = f1_score(y[te], oof[te], average="macro", zero_division=0)
        print(f"  fold {fold}: macro-F1 = {f1:.3f}  (n_train={len(tr)}, n_test={len(te)})")
    macro = f1_score(y, oof, average="macro", zero_division=0)
    micro = f1_score(y, oof, average="micro", zero_division=0)
    print(f"\nOOF macro-F1: {macro:.3f}    micro-F1 (= accuracy): {micro:.3f}\n")
    print(classification_report(y, oof, zero_division=0))
    return macro, micro


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--doc-prob-floor",
        type=float,
        default=0.5,
        help="Per-doc top1 probability floor before counting toward slug majority",
    )
    ap.add_argument(
        "--slug-share-floor",
        type=float,
        default=0.5,
        help="Fraction of confident docs that must vote for the winning class",
    )
    ap.add_argument(
        "--slug-mean-prob-floor",
        type=float,
        default=0.55,
        help="Mean top1 probability among winning-class docs",
    )
    ap.add_argument("--C", type=float, default=1.0, help="Logreg inverse regularization")
    ap.add_argument("--min-jobs", type=int, default=1)
    ap.add_argument("--out", default=str(UJ / "slug_industry_labels_te3_logreg.csv"))
    ap.add_argument("--skip-cv", action="store_true")
    args = ap.parse_args()

    print("Loading TE3 catalog and doc_ids …")
    doc_ids, vecs = load_corpus()
    overrides = load_overrides()
    slug_index = build_slug_index(doc_ids)

    train_slugs = sorted(s for s in overrides if s in slug_index)
    train_rows = []
    train_groups = []
    train_labels = []
    for s in train_slugs:
        rows = slug_index[s]
        train_rows.extend(rows)
        train_groups.extend([s] * len(rows))
        train_labels.extend([overrides[s]] * len(rows))
    print(
        f"Training docs: {len(train_rows):,} from {len(train_slugs)} slugs "
        f"({len(set(train_labels))} classes)"
    )

    print("Building train matrix …")
    X_train = build_doc_matrix(vecs, train_rows)
    y_train = np.array(train_labels)
    g_train = np.array(train_groups)

    if not args.skip_cv:
        print(f"\nStratifiedGroupKFold CV (5-fold, group=slug, C={args.C}):")
        cv_eval(X_train, y_train, g_train, n_splits=5, C=args.C)

    print(f"Fitting full model on all {len(X_train):,} docs …")
    clf = LogisticRegression(solver="lbfgs", max_iter=1000, C=args.C)
    clf.fit(X_train, y_train)
    classes = clf.classes_

    target_slugs = sorted(s for s in slug_index if s not in overrides)
    target_docs = sum(len(slug_index[s]) for s in target_slugs)
    print(f"\nPredicting on {len(target_slugs):,} unclassified slugs ({target_docs:,} docs) …")

    n_emit = 0
    classified_docs = 0
    with open(args.out, "w", newline="") as fout:
        w = csv.writer(fout)
        w.writerow(
            [
                "slug",
                "n_jobs",
                "industry",
                "share",
                "mean_p",
                "n_confident",
                "method",
            ]
        )
        for s in target_slugs:
            rows = slug_index[s]
            n_jobs = len(rows)
            if n_jobs < args.min_jobs:
                continue
            X = build_doc_matrix(vecs, rows)
            proba = clf.predict_proba(X)
            top1_idx = proba.argmax(axis=1)
            top1_p = proba[np.arange(len(proba)), top1_idx]
            confident_mask = top1_p >= args.doc_prob_floor
            n_conf = int(confident_mask.sum())

            if n_conf == 0:
                w.writerow([s, n_jobs, "unclassified", "0", "0", "0", "te3_logreg_low_conf"])
                continue

            votes = Counter(classes[top1_idx[confident_mask]])
            winner, n_winner = votes.most_common(1)[0]
            share = n_winner / n_conf
            winner_mean_p = top1_p[confident_mask & (classes[top1_idx] == winner)].mean()

            if share >= args.slug_share_floor and winner_mean_p >= args.slug_mean_prob_floor:
                w.writerow(
                    [
                        s,
                        n_jobs,
                        winner,
                        f"{share:.3f}",
                        f"{winner_mean_p:.3f}",
                        n_conf,
                        "te3_logreg",
                    ]
                )
                n_emit += 1
                classified_docs += n_jobs
            else:
                w.writerow(
                    [
                        s,
                        n_jobs,
                        "unclassified",
                        f"{share:.3f}",
                        f"{winner_mean_p:.3f}",
                        n_conf,
                        "te3_logreg_low_conf",
                    ]
                )

    print(f"\nEmitted {n_emit:,} confident slug labels ({classified_docs:,} docs).")
    print(f"Output: {args.out}")


if __name__ == "__main__":
    main()
