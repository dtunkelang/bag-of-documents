"""bge-small + logistic-regression industry classifier on per-slug text bags.

Loads the per-slug text bag (slug tokens x3 + top-50 titles) used by the
TF-IDF baseline, embeds each slug with bge-small-en-v1.5, and trains
multinomial logreg on the 1,404 hand-labeled slugs. Reports stratified
5-fold CV macro/micro F1, then predicts on unclassified slugs gated by a
top-1 probability threshold.

Outputs:
  unified_jobs/slug_text_bag_bge.npy         (cached embeddings, fp16)
  unified_jobs/slug_industry_labels_bge.csv  (slug-level predictions)
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import normalize

REPO = Path(__file__).resolve().parent.parent
UJ = REPO / "unified_jobs"
TEXT_BAG = UJ / "slug_text_bag.csv"
OVERRIDES = UJ / "slug_industry_overrides.csv"
EMB_CACHE = UJ / "slug_text_bag_bge.npy"


def load_text_bag():
    slugs, texts, n_jobs = [], [], []
    with TEXT_BAG.open() as f:
        r = csv.DictReader(f)
        for row in r:
            slugs.append(row["slug"])
            texts.append(row["text"])
            n_jobs.append(int(row["n_jobs"]))
    return slugs, texts, n_jobs


def load_overrides():
    out = {}
    with OVERRIDES.open() as f:
        r = csv.DictReader(f)
        for row in r:
            s = row["slug"].strip()
            ind = row["industry"].strip()
            if s and ind:
                out[s] = ind
    return out


def embed(texts, model_name, batch_size, device):
    print(f"Loading {model_name} on {device} …")
    model = SentenceTransformer(model_name, device=device)
    print(f"Embedding {len(texts):,} slug bags …")
    vecs = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype(np.float16)
    return vecs


def cv_eval(X, y, n_splits=5, C=1.0, class_weight=None):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    oof = np.empty_like(y, dtype=object)
    for fold, (tr, te) in enumerate(skf.split(X, y), 1):
        clf = LogisticRegression(solver="lbfgs", max_iter=2000, C=C, class_weight=class_weight)
        clf.fit(X[tr], y[tr])
        oof[te] = clf.predict(X[te])
        f1 = f1_score(y[te], oof[te], average="macro", zero_division=0)
        print(f"  fold {fold}: macro-F1 = {f1:.3f}")
    macro = f1_score(y, oof, average="macro", zero_division=0)
    micro = f1_score(y, oof, average="micro", zero_division=0)
    print(f"\nOOF macro-F1: {macro:.3f}   micro-F1 (= accuracy): {micro:.3f}\n")
    print(classification_report(y, oof, zero_division=0))
    return macro, micro


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="BAAI/bge-small-en-v1.5")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--device", default=None)
    ap.add_argument(
        "--prob-threshold",
        type=float,
        default=0.5,
        help="Top-1 probability floor for assigning a class",
    )
    ap.add_argument("--C", type=float, default=1.0)
    ap.add_argument("--class-weight", default=None, choices=[None, "balanced"])
    ap.add_argument("--rebuild-embeddings", action="store_true")
    ap.add_argument("--skip-cv", action="store_true")
    ap.add_argument("--out", default=str(UJ / "slug_industry_labels_bge.csv"))
    args = ap.parse_args()

    if args.device is None:
        args.device = "mps" if torch.backends.mps.is_available() else "cpu"

    slugs, texts, n_jobs = load_text_bag()
    print(f"Loaded {len(slugs):,} slug text bags")

    if EMB_CACHE.exists() and not args.rebuild_embeddings:
        print(f"Loading cached embeddings from {EMB_CACHE}")
        vecs = np.load(EMB_CACHE).astype(np.float32)
        assert vecs.shape[0] == len(slugs), f"cache len {vecs.shape[0]} != slugs {len(slugs)}"
    else:
        vecs = embed(texts, args.model, args.batch_size, args.device).astype(np.float32)
        np.save(EMB_CACHE, vecs.astype(np.float16))
        print(f"Saved embeddings to {EMB_CACHE}")
    vecs = normalize(vecs, norm="l2")

    overrides = load_overrides()
    slug_to_i = {s: i for i, s in enumerate(slugs)}
    labeled_idx = [slug_to_i[s] for s in overrides if s in slug_to_i]
    y_labeled = np.array([overrides[slugs[i]] for i in labeled_idx])
    X_labeled = vecs[labeled_idx]
    print(f"Labeled slugs: {len(labeled_idx)}, classes: {len(set(y_labeled))}")

    if not args.skip_cv:
        print(f"\nStratifiedKFold 5-fold CV (C={args.C}, class_weight={args.class_weight}):")
        cv_eval(X_labeled, y_labeled, n_splits=5, C=args.C, class_weight=args.class_weight)

    print(f"Fitting full model on {len(X_labeled)} labeled slugs …")
    clf = LogisticRegression(
        solver="lbfgs", max_iter=2000, C=args.C, class_weight=args.class_weight
    )
    clf.fit(X_labeled, y_labeled)
    classes = clf.classes_

    target_idx = [i for i, s in enumerate(slugs) if s not in overrides]
    X_target = vecs[target_idx]
    print(f"\nPredicting on {len(target_idx):,} unclassified slugs …")
    proba = clf.predict_proba(X_target)
    top1_idx = proba.argmax(axis=1)
    top1_p = proba[np.arange(len(proba)), top1_idx]
    proba_copy = proba.copy()
    proba_copy[np.arange(len(proba)), top1_idx] = -1.0
    top2_p = proba_copy.max(axis=1)

    n_emit = 0
    classified_docs = 0
    with open(args.out, "w", newline="") as fout:
        w = csv.writer(fout)
        w.writerow(["slug", "n_jobs", "industry", "top1_p", "top2_p", "method"])
        for j, i in enumerate(target_idx):
            s = slugs[i]
            nj = n_jobs[i]
            p1, p2 = float(top1_p[j]), float(top2_p[j])
            if p1 >= args.prob_threshold:
                ind = classes[top1_idx[j]]
                method = "bge_logreg"
                n_emit += 1
                classified_docs += nj
            else:
                ind = "unclassified"
                method = "bge_logreg_low_conf"
            w.writerow([s, nj, ind, f"{p1:.4f}", f"{p2:.4f}", method])
    print(f"\nEmitted {n_emit:,} confident slug labels ({classified_docs:,} docs).")
    print(f"Output: {args.out}")


if __name__ == "__main__":
    main()
