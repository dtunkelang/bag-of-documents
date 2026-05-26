# Industry-label propagation audit — 2026-05-25

50,754 employer slugs, 347,900 docs, 28-label taxonomy. Currently live on `dtunkelang/jobs-search-solr`.

## TL;DR

Round-2 propagation is the weak link. `round2_hi` (20,847 slugs / 78,183 docs) carries an estimated **~45% error rate** in this sample, and `round2_med` carries **~70%**. Together they back ~33% of all classified docs.

Mechanism: round2 seeds itself off `{seed, rule, tfidf_hi}` (see `download/propagate_industry_round2.py:29`). Any error in tfidf_hi (~20%) becomes a propagation source, and round2 amplifies it because the matching is on bag-of-titles, which is dominated by generic SaaS/engineering tokens. Chain example: `twinspires` was wrongly tfidf_hi'd to `finance_banking`; round2 then matched `grammarly` to `twinspires` and tagged grammarly `finance_banking` with margin 0.0795.

## Per-tier sample (n=20 each tier where applicable, n=10 seed)

| Tier         | Slugs   | Docs    | Est error | Notes |
|--------------|---------|---------|-----------|-------|
| seed         | 499     | 91,933  | 0% (def.) | LLM-labeled top employers |
| rule         | 2,906   | 14,959  | ~10-15%   | name-keyword rules; misfires: `kansas-health-system` → education_higher (it's a hospital), `hearst-health` → healthcare_provider (it's media) |
| tfidf_hi     | 5,105   | 43,784  | ~20%      | bag-of-titles + slug tokens; misses: `aldar` (UAE real estate) → finance_banking, `renoco-homes` → tech_software_internet, `applytoboltwise` → finance_banking |
| round2_hi    | 20,847  | 78,183  | **~45%**  | misleading "hi" — compound chaining errors. Misses: `grammarly` → finance_banking, `oxosmedical` → tech_software_internet (medical device co), `hearst-television` → healthcare_provider (chained from rule misfire on hearst-health), `nirvana-holdings-berhad` → nonprofit, `paragon-union-berhad` → tech (auto parts) |
| round2_med   | 8,846   | 36,295  | **~70%**  | low margin + bag-of-titles. Misses: `wizehire` → legal_services (HR software), `aerospike` → automotive (database co), `prosperity-life` → nonprofit (life insurance), `elite-steel` → tech_software_internet |
| low_margin   | 12,550  | 82,746  | 0% (silent) | safely unclassified — but misses obvious large slugs like `cisco`, `amtrak`, `fda`, `yamaha-corporation`, `sheppard-pratt` |

**Estimated misclassified docs**: ~70,000 (≈20% of corpus). Live facet counts on the Space are correspondingly biased: consulting/tech/healthcare are inflated.

## Attractor patterns

Top-3 seed industries (tech/consulting/healthcare) account for ~48-51% of slugs across all tiers — distribution is roughly preserved, NOT a pure attractor pattern.

Real over-attractions (round2_hi share / seed share):

| Industry                          | Ratio | round2_hi slugs |
|-----------------------------------|-------|-----------------|
| real_estate_construction          | 2.59x | 1,836 |
| legal_services                    | 2.40x | 301 |
| consulting_professional_services  | 1.51x | 4,468 |
| education_k12                     | 1.40x | 350 |

real_estate_construction over-attraction is consistent with the audit: several engineering/manufacturing slugs (`reframesystems`, `dev-technology-group`, `taprite-inc`) landed there because their titles contain "engineer/architect/quality" tokens that match construction seeds.

## Root causes (ranked)

1. **Chain compounding** — `tfidf_hi` errors become round2 seeds (`propagate_industry_round2.py:29`). 20% × 20% interaction at the boundary is real.
2. **Bag-of-titles is too generic** — most slugs share "Senior Engineer", "Account Executive", "Sales Manager", "Director, Marketing". The TF-IDF signal that survives is generic SaaS language, which biases toward tech/consulting/marketing-adjacent classes.
3. **Low absolute similarity at the high-margin gate** — `round2_hi` margin ≥ 0.05 but `top1_sim` is often only ~0.15-0.25. The gate trusts margin without floor on absolute similarity (only `round2_med` has `SIM_FLOOR=0.20`).
4. **Slug-name signal is underweighted** — slug tokens are triplicated (`* 3` at `propagate_industry_round2.py:69`) but bag-of-titles still dominates; for small-n slugs the slug name often carries the strongest signal but loses to title-token mass.
5. **No reciprocity check** — if A is closest to B but B is NOT close to A's class, current code still labels A. A k-NN majority vote would soften this.

## Recommended fixes (in priority order)

### Cheap, no model retrain
- **F1. Add `top1_sim ≥ 0.25` floor to `round2_hi`** (currently no floor). Demotes low-absolute-sim "hi" matches to low_margin. Cost: ~5min code change. Expected: cuts round2_hi by ~30%, error rate drops materially.
- **F2. Drop `tfidf_hi` from `ROUND1_CONFIDENT`** — round2 should chain only off `{seed, rule}` to prevent error compounding. Cost: 5min. Expected: round2_hi shrinks ~50%, errors drop ~half.
- **F3. Raise margin gate** — bump `MARGIN_HI` from 0.05 to 0.08; demote affected to `low_margin`. Cost: 5min. Expected: smaller, cleaner round2_hi.

### Mid-effort, no LLM
- **F4. Add description tokens** (not just titles) to per-slug bag — descriptions carry domain language (drug names, financial instruments, equipment terms) that titles strip. Cost: ~30min for code, ~20min for re-vectorize. Risk: descriptions also have boilerplate.
- **F5. k=5 majority vote with sim-weighting** instead of nearest-seed. Cost: ~20min. Reduces single-neighbor sensitivity.

### High-effort but high-precision
- **F6. LLM re-label the round2_hi tail** — ~20k slugs at GPT-4o-mini scale (~$5-10 estimated). Best precision recovery. Aligns with the OpenAI-API-available memory.
- **F7. Replace round2 with embedding-similarity reclassify** using `bge-small` over titles+descriptions, with calibrated thresholds and per-class centroids. Cost: ~1-2hrs. The `solr_jobs_demo/facets/centroid_reclassify.py` already exists for this — was it abandoned for a reason?

### Coverage gap (orthogonal)
- **F8. LLM-label the top-N slugs in `low_margin`** by n_jobs descending — cisco, amtrak, fda, etc. ~100 slugs would recover several thousand mis-unclassified docs. Cost: minutes of GPT-4o-mini.

## Recommendation

**Ship F2 + F8 today** as the minimal fix:
- F2 prevents compounding (round2 chain only off seed+rule = 3,405 slugs, the trusted set)
- F8 recovers the highest-impact low_margin large slugs

**Queue F6 (LLM re-label) for a longer follow-up** once the basic improvements ship.

Estimated combined improvement: round2_hi error rate drops to ~25% (from ~45%) and round2_med drops to ~50% (from ~70%); facet counts become trustworthy enough for production demo.

## Audit artifacts

- `/tmp/audit_sample.json` — 110 sampled slug rows (stratified by method × kind)
- `/tmp/audit_titles_report.txt` — per-slug top-5 titles for visual inspection
- `/tmp/audit_titles.py` — reproducer
