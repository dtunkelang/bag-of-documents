"""Industry-label propagation v2 using TF-IDF over slug-name tokens + top job titles.

Pipeline:
  1) Per slug, collect a "document" = (slug-name tokens expanded) + (deduped job titles joined).
  2) Fit TfidfVectorizer (1-2 grams, English, min_df=2) on all 50K slug documents.
  3) For each slug, find nearest seed by cosine similarity in TF-IDF space.
  4) Apply slug-name keyword overrides (strong signals: bank, health, school, federal-agency...).
  5) Margin gate: if (top1_sim - runner_up_sim_other_class) < threshold AND no keyword override,
     label as 'unclassified' rather than guess.
  6) Emit slug CSV + per-doc TSV + summary stats.
"""

import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

ROOT = Path(".")
META = ROOT / "unified_jobs/metadata.jsonl"
SEEDS = ROOT / "unified_jobs/top500_slugs_labeled_v2.csv"
# F8 overrides: hand-labeled high-impact slugs from the audit's low_margin tail.
# Loaded as additional seeds, treated identically to top500 seeds.
OVERRIDES = ROOT / "unified_jobs/slug_industry_overrides.csv"
OUT_SLUG = ROOT / "unified_jobs/slug_industry_labels_tfidf.csv"
OUT_DOC = ROOT / "unified_jobs/doc_industry_labels_tfidf.tsv"

TOP_K_TITLES = 50  # top-frequency titles per slug to include in TF-IDF document
# Two-tier gating: high-confidence = margin >= MARGIN_HI;
# medium-confidence = margin >= MARGIN_LO AND best_sim >= SIM_FLOOR.
# Below both -> unclassified.
MARGIN_HI = 0.05
MARGIN_LO = 0.015
SIM_FLOOR = 0.20
# Round-2 propagation: re-use confident round-1 labels as seeds, then re-propagate.
# Only round-1 methods listed here qualify as round-2 seeds (excludes low_margin and tfidf_med).
ROUND2_SEED_METHODS = {"seed", "rule", "tfidf_hi"}
RUN_ROUND_2 = True

# --- slug-name keyword rules: strong overrides regardless of TF-IDF ---
SLUG_KEYWORD_RULES: list[tuple[str, str]] = [
    # public sector / military
    (
        r"\b(department-of|dept-of|us-department|usda|nasa|noaa|fbi|cia|dhs|epa|hhs|hud)\b",
        "public_sector_government",
    ),
    (
        r"\b(army|navy|airforce|air-force|marines|marine-corps|coast-guard|national-guard)\b",
        "public_sector_government",
    ),
    (
        r"\b(bureau-of|federal-|state-of|city-of|county-of|government|gov-of)\b",
        "public_sector_government",
    ),
    (r"\b(municipal|municipality|city-government|state-government)\b", "public_sector_government"),
    # education
    (
        r"\b(school-district|public-schools|charter-school|elementary|high-school)\b",
        "education_k12",
    ),
    (
        r"\b(university|college-of|state-university|community-college|institute-of-technology)\b",
        "education_higher",
    ),
    # banking
    (
        r"\b(bank-of|banking-group|bancorp|bankshares|national-bank|state-bank|maybank|hsbc|jpm|wells-fargo|citibank|barclays)\b",
        "finance_banking",
    ),
    (r"-bank$", "finance_banking"),
    (r"^bank-", "finance_banking"),
    # insurance
    (
        r"\b(insurance-company|insurance-group|life-insurance|reinsurance|mutual-insurance)\b",
        "finance_insurance",
    ),
    # healthcare provider
    (
        r"\b(health-system|medical-center|hospital|hospitals|healthcare|home-health|nursing-home|urgent-care|dental-care)\b",
        "healthcare_provider",
    ),
    (r"-health$", "healthcare_provider"),
    (r"-healthcare$", "healthcare_provider"),
    # pharma
    (
        r"\b(pharmaceuticals|pharma-inc|biopharma|biotech|biosciences|therapeutics)\b",
        "healthcare_pharma_biotech",
    ),
    # legal
    (r"\b(law-firm|law-office|attorneys-at-law|llp-law)\b", "legal_services"),
    # construction / real estate
    (
        r"\b(construction-co|construction-company|construction-services|construction-group|homebuilders|builders)\b",
        "real_estate_construction",
    ),
    (r"\b(real-estate|realty|brokerage|reit-)\b", "real_estate_construction"),
    # energy / utilities
    (
        r"\b(power-company|electric-utility|utility-company|oil-and-gas|solar-energy|wind-energy|renewable-energy)\b",
        "energy_utilities",
    ),
    # telecom
    (r"\b(telecommunications|telecom-group|wireless-co|broadband)\b", "telecommunications"),
    # transportation/logistics
    (
        r"\b(logistics-inc|logistics-company|freight-services|trucking-co|shipping-line|cargo-services)\b",
        "transportation_logistics",
    ),
    (r"\b(airlines|airways|aviation-co)\b", "transportation_logistics"),
    (r"-airlines?$", "transportation_logistics"),
    # hospitality
    (
        r"\b(hotels-group|hotels-and-resorts|resort-and-spa|hospitality-group)\b",
        "hospitality_food_service",
    ),
    (r"-coffee$", "hospitality_food_service"),
    (r"-roasters$", "hospitality_food_service"),
    # staffing
    (
        r"\b(agensi-pekerjaan|staffing-agency|talent-agency|recruiting-firm|placement-services)\b",
        "consulting_professional_services",
    ),
    (r"-recruiting$", "consulting_professional_services"),
    (r"-staffing$", "consulting_professional_services"),
    (r"-recruiters$", "consulting_professional_services"),
]


def slug_tokens(slug: str) -> str:
    """Expand slug into tokens for the TF-IDF document."""
    return re.sub(r"[-_./]+", " ", slug).lower()


def apply_keyword_rule(slug: str) -> str | None:
    s = slug.lower()
    for pat, label in SLUG_KEYWORD_RULES:
        if re.search(pat, s):
            return label
    return None


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
    print(f"  {len(slug_titles):,} unique slugs")

    slugs = sorted(slug_titles.keys())
    slug_to_idx = {s: i for i, s in enumerate(slugs)}

    print("building TF-IDF documents...")
    docs: list[str] = []
    for s in slugs:
        # repeat slug-tokens to give them more weight than any single title
        slug_text = (slug_tokens(s) + " ") * 3
        # top-K titles by frequency, joined
        top_titles = [t for t, _ in slug_titles[s].most_common(TOP_K_TITLES)]
        docs.append(slug_text + " ".join(top_titles))
    print(f"  {len(docs):,} documents, mean len={int(np.mean([len(d) for d in docs]))} chars")

    print("fitting TfidfVectorizer (1-2 grams)...")
    vec = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.5,
        sublinear_tf=True,
        token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z]+\b",
        max_features=100_000,
    )
    X = vec.fit_transform(docs)
    # row-normalize so dot product == cosine
    from sklearn.preprocessing import normalize

    X = normalize(X, norm="l2", axis=1)
    print(f"  X shape: {X.shape}  nnz={X.nnz:,}")

    print("loading seed labels...")
    seed_label: dict[str, str] = {}
    with SEEDS.open() as f:
        for r in csv.DictReader(f):
            if r["industry"] == "other":
                continue
            if r["slug"] in slug_to_idx:
                seed_label[r["slug"]] = r["industry"]
    n_primary = len(seed_label)
    print(f"  {n_primary} primary seed slugs available")
    if OVERRIDES.exists():
        n_added = 0
        with OVERRIDES.open() as f:
            for r in csv.DictReader(f):
                if r["industry"] == "other":
                    continue
                if r["slug"] in slug_to_idx:
                    seed_label[r["slug"]] = r["industry"]
                    n_added += 1
        print(f"  +{n_added} override seeds from {OVERRIDES.name}")
        print(f"  {len(seed_label)} total seeds")

    seed_slugs = sorted(seed_label.keys())
    seed_idx = np.array([slug_to_idx[s] for s in seed_slugs])
    seed_labels = np.array([seed_label[s] for s in seed_slugs])
    Xs = X[seed_idx]
    print(f"  seed matrix: {Xs.shape}")

    print("computing similarities (sparse @ sparse.T) ...")
    sims = (X @ Xs.T).toarray()  # 50K x ~500
    print(f"  sims: {sims.shape} {sims.dtype}")

    classes = sorted(set(seed_labels.tolist()))
    cls_to_seeds = {c: np.where(seed_labels == c)[0] for c in classes}

    # max sim per class for each slug
    n_slugs, _ = sims.shape
    per_class_max = np.full((n_slugs, len(classes)), -1.0, dtype=np.float32)
    per_class_argmax = np.zeros((n_slugs, len(classes)), dtype=np.int32)
    for ci, c in enumerate(classes):
        idxs = cls_to_seeds[c]
        sub = sims[:, idxs]
        per_class_argmax[:, ci] = idxs[sub.argmax(axis=1)]
        per_class_max[:, ci] = sub.max(axis=1)

    best_ci = per_class_max.argmax(axis=1)
    best_sim = per_class_max[np.arange(n_slugs), best_ci]
    pcm_copy = per_class_max.copy()
    pcm_copy[np.arange(n_slugs), best_ci] = -1.0
    runner_sim = pcm_copy.max(axis=1)
    margin = best_sim - runner_sim
    cls_arr = np.array(classes)

    print("composing labels (rule overrides + margin gate)...")
    propagated = []
    seed_set = set(seed_label.keys())
    n_rule_override = 0
    n_unclassified = 0
    n_seed = 0
    for i, slug in enumerate(slugs):
        if slug in seed_set:
            propagated.append(
                (slug, slug_n[slug], slug_src[slug], seed_label[slug], 1.0, slug, 1.0, "seed")
            )
            n_seed += 1
            continue
        rule_label = apply_keyword_rule(slug)
        prop_label = str(cls_arr[best_ci[i]])
        prop_sim = float(best_sim[i])
        prop_margin = float(margin[i])
        top_seed_idx = int(per_class_argmax[i, best_ci[i]])
        top_seed = seed_slugs[top_seed_idx]
        if rule_label is not None:
            propagated.append(
                (slug, slug_n[slug], slug_src[slug], rule_label, 1.0, "(rule)", prop_sim, "rule")
            )
            n_rule_override += 1
        elif prop_margin >= MARGIN_HI:
            propagated.append(
                (
                    slug,
                    slug_n[slug],
                    slug_src[slug],
                    prop_label,
                    prop_margin,
                    top_seed,
                    prop_sim,
                    "tfidf_hi",
                )
            )
        elif prop_margin >= MARGIN_LO and prop_sim >= SIM_FLOOR:
            propagated.append(
                (
                    slug,
                    slug_n[slug],
                    slug_src[slug],
                    prop_label,
                    prop_margin,
                    top_seed,
                    prop_sim,
                    "tfidf_med",
                )
            )
        else:
            propagated.append(
                (
                    slug,
                    slug_n[slug],
                    slug_src[slug],
                    "unclassified",
                    prop_margin,
                    top_seed,
                    prop_sim,
                    "low_margin",
                )
            )
            n_unclassified += 1

    print(f"  seed-direct: {n_seed}")
    print(f"  rule-override: {n_rule_override}")
    print(f"  unclassified: {n_unclassified}")
    print(f"  tfidf-propagated: {n_slugs - n_seed - n_rule_override - n_unclassified}")
    by_method = Counter(p[7] for p in propagated)
    print(f"  by method: {dict(by_method)}")

    with OUT_SLUG.open("w", newline="") as f:
        w = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
        w.writerow(
            ["slug", "n_jobs", "source", "industry", "margin", "top1_seed", "top1_sim", "method"]
        )
        for row in propagated:
            slug, n, src, lbl, m, ts, s_, meth = row
            w.writerow([slug, n, src, lbl, f"{m:.4f}", ts, f"{s_:.4f}", meth])
    print(f"  wrote {OUT_SLUG}")

    # per-doc TSV — load doc_ids in the order they appear in metadata.jsonl
    slug_to_label = {p[0]: p[3] for p in propagated}
    print(f"writing {OUT_DOC} ...")
    with OUT_DOC.open("w") as fout, META.open() as fin:
        for line in fin:
            d = json.loads(line)
            slug = d.get("source_slug")
            doc_id = d.get("id")
            if not slug or not doc_id:
                continue
            fout.write(f"{doc_id}\t{slug_to_label.get(slug, 'unclassified')}\n")

    # distribution summary
    slug_dist = Counter(p[3] for p in propagated)
    doc_dist: Counter[str] = Counter()
    for slug, n in slug_n.items():
        doc_dist[slug_to_label.get(slug, "unclassified")] += n

    print("\n=== per-slug industry distribution ===")
    for k, v in slug_dist.most_common():
        print(f"  {v:6,d}  {k}")
    print("\n=== per-doc industry distribution ===")
    for k, v in doc_dist.most_common():
        print(f"  {v:7,d}  {k}")
    total_docs = sum(doc_dist.values())
    classified = total_docs - doc_dist.get("unclassified", 0)
    print(f"\nclassified docs: {classified:,} / {total_docs:,} = {classified / total_docs:.1%}")


if __name__ == "__main__":
    main()
