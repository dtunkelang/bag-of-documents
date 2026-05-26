"""Round-2 propagation: re-use confident round-1 labels as expanded seeds.

Round-1 confident methods (seed, rule, tfidf_hi) become seeds for a second TF-IDF
nearest-neighbor pass over the round-1 'unclassified' slugs. Idea: rule overrides
classified ~2900 slugs by name pattern, and many other slugs are TF-IDF-close to
those without matching the keyword rule themselves.
"""

import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

ROOT = Path(".")
META = ROOT / "unified_jobs/metadata.jsonl"
ROUND1 = ROOT / "unified_jobs/slug_industry_labels_tfidf.csv"
OUT_SLUG = ROOT / "unified_jobs/slug_industry_labels_round2.csv"
OUT_DOC = ROOT / "unified_jobs/doc_industry_labels_round2.tsv"

TOP_K_TITLES = 50
MARGIN_HI = 0.05
MARGIN_LO = 0.015
SIM_FLOOR = 0.20
# F2: dropped tfidf_hi from round-1 confident set — round2 chains only off curated
# {seed, rule} sources to prevent error compounding. tfidf_hi labels still survive
# from round1 but no longer serve as round2 seeds. See unified_jobs/AUDIT_REPORT.md.
ROUND1_CONFIDENT = {"seed", "rule"}


def slug_tokens(slug: str) -> str:
    return re.sub(r"[-_./]+", " ", slug).lower()


def main() -> None:
    print("collecting per-slug title bags...")
    slug_titles: dict[str, Counter[str]] = defaultdict(Counter)
    slug_n: Counter[str] = Counter()
    slug_src: dict[str, str] = {}
    with META.open() as f:
        for line in f:
            d = json.loads(line)
            slug = d.get("source_slug")
            if not slug:
                continue
            slug_n[slug] += 1
            slug_src[slug] = d.get("source", "?")
            title = (d.get("title") or "").strip()
            if title:
                slug_titles[slug][title.lower()] += 1
    slugs = sorted(slug_titles.keys())
    slug_to_idx = {s: i for i, s in enumerate(slugs)}
    print(f"  {len(slugs):,} slugs")

    print("loading round-1 labels...")
    round1: dict[str, dict[str, str]] = {}
    with ROUND1.open() as f:
        for r in csv.DictReader(f):
            round1[r["slug"]] = r
    confident = {s: r for s, r in round1.items() if r["method"] in ROUND1_CONFIDENT}
    print(
        f"  round-1 confident: {len(confident):,}  unclassified: {len(round1) - len(confident):,}"
    )

    print("building TF-IDF documents (same recipe as round-1)...")
    docs: list[str] = []
    for s in slugs:
        slug_text = (slug_tokens(s) + " ") * 3
        top_titles = [t for t, _ in slug_titles[s].most_common(TOP_K_TITLES)]
        docs.append(slug_text + " ".join(top_titles))

    vec = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.5,
        sublinear_tf=True,
        token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z]+\b",
        max_features=100_000,
    )
    X = vec.fit_transform(docs)
    X = normalize(X, norm="l2", axis=1)
    print(f"  X shape: {X.shape}")

    # round-2 seeds: confident round-1 slugs
    seed_slugs = sorted(confident.keys())
    seed_idx = np.array([slug_to_idx[s] for s in seed_slugs])
    seed_labels = np.array([confident[s]["industry"] for s in seed_slugs])
    Xs = X[seed_idx]
    print(f"  round-2 seed matrix: {Xs.shape}")

    # only re-classify round-1 unclassified slugs
    target_slugs = [s for s in slugs if round1[s]["method"] not in ROUND1_CONFIDENT]
    target_idx = np.array([slug_to_idx[s] for s in target_slugs])
    Xt = X[target_idx]
    print(f"  targets to re-classify: {Xt.shape[0]:,}")

    print("similarities...")
    sims = (Xt @ Xs.T).toarray()

    classes = sorted(set(seed_labels.tolist()))
    cls_to_seeds = {c: np.where(seed_labels == c)[0] for c in classes}
    n_t = sims.shape[0]
    per_class_max = np.full((n_t, len(classes)), -1.0, dtype=np.float32)
    per_class_argmax = np.zeros((n_t, len(classes)), dtype=np.int32)
    for ci, c in enumerate(classes):
        idxs = cls_to_seeds[c]
        sub = sims[:, idxs]
        per_class_argmax[:, ci] = idxs[sub.argmax(axis=1)]
        per_class_max[:, ci] = sub.max(axis=1)
    best_ci = per_class_max.argmax(axis=1)
    best_sim = per_class_max[np.arange(n_t), best_ci]
    pcm_copy = per_class_max.copy()
    pcm_copy[np.arange(n_t), best_ci] = -1.0
    runner_sim = pcm_copy.max(axis=1)
    margin = best_sim - runner_sim
    cls_arr = np.array(classes)

    round2_label: dict[str, tuple[str, float, float, str, str]] = {}
    for i, slug in enumerate(target_slugs):
        prop_label = str(cls_arr[best_ci[i]])
        prop_sim = float(best_sim[i])
        prop_margin = float(margin[i])
        top_seed_idx = int(per_class_argmax[i, best_ci[i]])
        top_seed = seed_slugs[top_seed_idx]
        # F1: add SIM_FLOOR gate to round2_hi — previously only margin was checked,
        # which let low-absolute-similarity matches through (e.g. grammarly@0.27 via
        # twinspires with margin 0.08). Now both margin AND sim must clear.
        if prop_margin >= MARGIN_HI and prop_sim >= SIM_FLOOR:
            method = "round2_hi"
        elif prop_margin >= MARGIN_LO and prop_sim >= SIM_FLOOR:
            method = "round2_med"
        else:
            method = "low_margin"
        round2_label[slug] = (
            prop_label if method != "low_margin" else "unclassified",
            prop_margin,
            prop_sim,
            top_seed,
            method,
        )

    # merge: keep round-1 confident as is, replace round-1 unclassified with round-2
    merged = []
    for slug in slugs:
        r1 = round1[slug]
        if r1["method"] in ROUND1_CONFIDENT:
            merged.append(
                (
                    slug,
                    slug_n[slug],
                    slug_src[slug],
                    r1["industry"],
                    float(r1["margin"]),
                    r1["top1_seed"],
                    float(r1["top1_sim"]),
                    r1["method"],
                )
            )
        else:
            lbl, m, s_, ts, method = round2_label[slug]
            merged.append((slug, slug_n[slug], slug_src[slug], lbl, m, ts, s_, method))

    with OUT_SLUG.open("w", newline="") as f:
        w = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
        w.writerow(
            ["slug", "n_jobs", "source", "industry", "margin", "top1_seed", "top1_sim", "method"]
        )
        for row in merged:
            slug, n, src, lbl, mm, ts, s_, meth = row
            w.writerow([slug, n, src, lbl, f"{mm:.4f}", ts, f"{s_:.4f}", meth])
    print(f"wrote {OUT_SLUG}")

    slug_to_label = {row[0]: row[3] for row in merged}
    print(f"writing {OUT_DOC} ...")
    with OUT_DOC.open("w") as fout, META.open() as fin:
        for line in fin:
            d = json.loads(line)
            slug = d.get("source_slug")
            doc_id = d.get("id")
            if not slug or not doc_id:
                continue
            fout.write(f"{doc_id}\t{slug_to_label.get(slug, 'unclassified')}\n")

    method_dist = Counter(row[7] for row in merged)
    label_dist = Counter(row[3] for row in merged)
    doc_label_dist: Counter[str] = Counter()
    for slug, n in slug_n.items():
        doc_label_dist[slug_to_label.get(slug, "unclassified")] += n

    print("\n=== by method ===")
    for k, n in method_dist.most_common():
        print(f"  {n:6,d}  {k}")
    print("\n=== per-slug labels ===")
    for k, n in label_dist.most_common():
        print(f"  {n:6,d}  {k}")
    print("\n=== per-doc labels ===")
    for k, n in doc_label_dist.most_common():
        print(f"  {n:7,d}  {k}")
    total = sum(doc_label_dist.values())
    classified = total - doc_label_dist.get("unclassified", 0)
    print(f"\nclassified: {classified:,} / {total:,} = {classified / total:.1%}")


if __name__ == "__main__":
    main()
