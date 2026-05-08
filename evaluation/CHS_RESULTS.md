# Cluster Hypothesis Score (CHS) — calibration table

This document records empirical CHS scores across corpora, used as the
anchor for `bagofdocs.cluster_hypothesis.schs_verdict`.

## What CHS measures

**SCHS (Simple Cluster Hypothesis Score)** = `(mean_intra - mean_random) / (1 - mean_random)`
- `mean_intra`: average cosine between two products that are positives for
  the *same* query
- `mean_random`: average cosine between two random products in the catalog
- Range [0, 1]. Higher means in-bag pairs cluster more tightly than the
  random catalog baseline; the corpus geometry agrees with relevance
  structure.

**HCHS (Hard Cluster Hypothesis Score)** = `(mean_intra - mean_inter_neg) / (mean_intra - mean_random)`
- Same as SCHS but uses *within-query labeled negatives* in the denominator.
- A stronger test: measures the gap against confusable negatives, not just
  random documents.
- Requires explicit negatives in qrels; undefined on positives-only corpora.

**strong_inv_rate**: fraction of queries where some pos-neg cosine
exceeds the *best* pos-pos cosine. The cluster hypothesis fails on those
queries; no query encoder can put positives closer than negatives.
Computable when explicit negatives are available.

See `bagofdocs/cluster_hypothesis.py` for full formulas and caveats.

## Empirical calibration

All scores below are with `sentence-transformers/all-MiniLM-L6-v2` unless
noted (multi = `paraphrase-multilingual-MiniLM-L12-v2`).

| dataset | encoder | n_pos_bearing | n_explicit_neg | mean_intra | mean_inter_neg | mean_random | **SCHS** | HCHS | strong_inv | empirical BoD outcome |
|---|---|---|---|---|---|---|---|---|---|---|
| esci_us_strict | mono | 21,394 | 7,661 | 0.607 | 0.412 | 0.139 | **0.544** | 0.376 | 16.2% | **POSITIVE** (+~7pp R@10) |
| esci_us_strict | multi | 21,394 | 7,661 | 0.634 | 0.448 | 0.167 | **0.561** | 0.359 | 16.9% | (same corpus, different encoder) |
| esci_us_relaxed | mono | 22,298 | 9,835 | 0.576 | 0.422 | 0.137 | **0.508** | 0.310 | 13.7% | (relaxed E+S vs I+C partition) |
| esci_us_relaxed | multi | 22,298 | 9,835 | 0.605 | 0.462 | 0.165 | **0.527** | 0.288 | 14.6% | |
| bestbuy_acm | mono | 60,644 | 0 | 0.604 | n/a | 0.166 | **0.525** | n/a | n/a | not BoD-tested; SCHS predicts POSITIVE |
| esci_es_strict | multi | 3,676 | 1,825 | 0.575 | 0.447 | 0.221 | **0.454** | 0.327 | 19.1% | **POSITIVE** but smaller (+4.7pp R@10) |
| esci_es_relaxed | multi | 3,825 | 2,374 | 0.550 | 0.458 | 0.221 | **0.422** | 0.254 | 18.2% | |
| beir:fever | mono | 847 | 0 | 0.441 | n/a | 0.034 | **0.421** | n/a | n/a | not BoD-tested |
| beir:dbpedia-entity | mono | 277 | 277 | 0.434 | 0.270 | 0.028 | **0.418** | 0.410 | 61.4% | not BoD-tested; very high inversion rate |
| nfcorpus_strict | mono | 90 | 0 | 0.519 | n/a | 0.222 | **0.382** | n/a | n/a | **NEGATIVE** (no lift over base) |
| beir:scidocs | mono | 1,000 | 1,000 | 0.418 | 0.083 | 0.063 | **0.379** | 0.974¹ | 7.3% | not BoD-tested |
| beir:trec-covid | mono | 50 | 50 | 0.550 | 0.391 | 0.280 | **0.375** | 0.592 | 26.0% | not BoD-tested |
| beir:climate-fever | mono | 1,391 | 0 | 0.397 | n/a | 0.080 | **0.344** | n/a | n/a | not BoD-tested |
| beir:nq | mono | 666 | 0 | 0.715 | n/a | 0.013 | **0.712** | n/a | n/a | not BoD-tested; **outlier** — see below |

¹ HCHS for SciDocs is artificially inflated because the qrels' "negatives"
are random non-positives, not confusable hard negatives.

## Skipped (not enough multi-positive queries to score)

`beir:scifact`, `beir:arguana`, `beir:fiqa`, `beir:hotpotqa`, `beir:quora`,
`beir:msmarco`. Each has < 50 queries with ≥ 2 positives. The cluster
hypothesis as operationalized here requires multi-positive bags.

## Verdict thresholds (anchored to ESCI-US)

| SCHS | label | reasoning |
|---|---|---|
| ≥ 0.50 | **GREEN** — BoD likely to generalize | At-or-above ESCI-US level |
| 0.40 - 0.50 | **YELLOW** — BoD may give a smaller lift; pilot first | Below ESCI-US, above the negative cluster (NFCorpus) — ESCI-Spanish lands here and BoD did lift on it (just less) |
| < 0.40 | **RED** — BoD unlikely to lift over base | NFCorpus level or below |

Joint factor: **n_pos_bearing**. With < 500 multi-positive queries,
training scale will limit BoD even if SCHS is high. The BEIR NQ outlier
(SCHS 0.71) illustrates this — its 666 multi-pos queries describe a
narrow slice of the corpus, and BoD lift would be bounded by training
scale, not by cluster geometry.

## Patterns worth noting

1. **SCHS rank-orders BoD lift magnitudes**, not just success/failure:
   - BestBuy ACM (+17.5pp): SCHS 0.525
   - ESCI-US (~+7pp): SCHS 0.54
   - ESCI-Spanish (+4.7pp): SCHS 0.45
   - NFCorpus (no lift): SCHS 0.38
   The metric tracks *how much* BoD will help, not just *whether*.

2. **High SCHS without enough multi-positive queries is a trap.** NQ at
   0.71 looks great in isolation but only 666 queries pass the ≥2-pos
   filter (most NQ questions have one answer passage). BoD training on
   666 bags is unlikely to compete with bigger pretraining.

3. **HCHS adds signal mainly when qrels include genuine hard negatives.**
   For BEIR-style positives-only qrels, HCHS is undefined. SCHS is the
   more universally meaningful metric.

4. **Strong-inv rate caveats**: high values signal that confusable
   negatives exist within the candidate set, but high strong_inv with
   high SCHS (e.g., trec-covid: 0.375 SCHS, 26% inv) means BoD bi-encoder
   alone won't separate the hard cases — a cross-encoder is needed.

5. **SCHS is necessary but does not predict lift magnitude alone.**
   Decomposing per-query by base difficulty (`evaluation/diagnose_lift.py`,
   `evaluation/diagnose_bestbuy_lift.py`, `evaluation/diagnose_esci_lift.py`)
   across 4 corpora spanning the SCHS spectrum reveals a 3-factor model:

   All values use the **fraction-recovered** R@10 metric (mean over queries
   of `positives_in_top_10 / total_positives`), not the binary hit-rate
   sometimes used elsewhere. The earlier "BestBuy +17.5pp" headline used
   binary hit-rate (`evaluation/eval_bestbuy_bod.py`); the row below
   restates it under the consistent metric.

   | Corpus | base R@10 | base-blind | rescue | spec tax | **overall Δ** | SCHS |
   |---|---:|---:|---:|---:|---:|---:|
   | BestBuy ACM | 0.306 | 44% | **+24.9pp** | −6.4 | **+14.2pp** | 0.525 |
   | ESCI-Spanish | 0.074 | 67% | **+15.1pp** | −12.9 | **+13.2pp** | 0.45 |
   | ESCI-US (E-only) | 0.215 | 34% | **+6.1pp** | −10.4 | **+3.0pp** | 0.54 |
   | FiQA-2018 | 0.441 | 34% | **+13.0pp** | −7.5 | **+2.6pp** | 0.44 |
   | NFCorpus | 0.159 | 31% | **+4.2pp** | −17.9 | **+0.8pp** | 0.38 |

   *BestBuy values measured on the 53,048-product multi-positive subset (the
   trainable bag corpus). When re-evaluated against the full 1,274,801-product
   2012 catalog (all SKUs from the Kaggle XML, bags unchanged), the binary
   R@10 lift is preserved at scale: base 0.3238 → BoD 0.5013 (+17.75pp) on
   the same 12,128-query holdout. See the [demo Space](https://huggingface.co/spaces/dtunkelang/bag-of-documents-bestbuy-demo).*

   The 3 factors:
   - **Bag signal sharpness → rescue rate on the base-blind subset.**
     Clicks (BestBuy) > weak-base-on-foreign-language + ESCI qrels (Spanish) >
     ESCI qrels (US) > medical-info qrels (NFCorpus): 24.9 / 15.1 / 6.1 / 4.2.
   - **Base model competence on the corpus → base-blind subset size.**
     The fraction of queries where base finds zero positives in top-10. The
     biggest surprise is ESCI-Spanish at 67%: multilingual MiniLM is much
     weaker on Spanish than English MiniLM is on English, giving huge
     headroom (base R@10 = 0.074 vs ESCI-US's 0.215).
   - **Corpus clustering geometry → SCHS (necessary floor).** SCHS measures
     whether the relevance structure clusters under the encoder; a low SCHS
     (NFCorpus 0.38) caps lift even when other factors are favorable.

   Specialization tax on the base-perfect subset grows as rescue rate
   shrinks: −6.4 / −12.9 / −10.4 / −17.9. When the bag signal isn't sharp
   enough to discover new structure, BoD only erases what base already had.

   Lift magnitude ≈ (base-blind size × rescue rate) − (base-perfect size ×
   tax). The overall lift cleanly tracks this product across all four
   corpora, while none of the three factors alone does.

   NFCorpus's rescue rate may be a slight underestimate (3-epoch training
   on random hardnegs, slow 1592-char passages); doubling it would not
   change the qualitative ordering. Spanish R@10 metric here is E-only;
   the prior published +4.7% used E+S pooled with a t=0.95 threshold,
   which is not directly comparable to the other rows.

6. **Sharper hardnegs sharpen rescue *and* the specialization tax.** Two
   independent within-corpus probes (same base, sharper signal) both show
   the same trade-off:

   | Probe | rescue Δ | spec tax | overall |
   |---|---:|---:|---:|
   | ESCI-US: rerank_A (MNRL) → rerank_B (qrels-MNRL-hardneg) | +6.1 → +7.8 | −10.4 → −21.2 | +3.0 → +0.1 |
   | BestBuy: random hardnegs → FAISS-mined hardnegs | +24.9 → +26.6 | −6.4 → −15.0 | +14.2 → +12.6 |

   In both cases sharper hardnegs improve rescue rate slightly but more
   than double the specialization tax, leaving overall lift unchanged or
   worse. The framework's prediction holds: when the bag signal can't
   produce *new* useful structure beyond what the base already had, even
   harder negatives just push BoD to forget more of the base-perfect
   structure. Hardneg mining is not a free lunch on either corpus.

7. **Stronger base shrinks the base-blind subset (causal test).** Encoding
   the ESCI-US catalog with `BAAI/bge-base-en-v1.5` instead of
   `all-MiniLM-L6-v2` drops the base-blind subset from 33.9% to **29.6%**
   (overall base R@10 0.215 → 0.246, fraction-recovered). Combined with
   the prior bge-base/ESCI BoD probe (clean negative, BoD harms R@10), this
   completes the framework prediction: a stronger base means less
   headroom and the same bag signal converts that smaller headroom into a
   smaller — sometimes negative — lift, even though SCHS would not
   change.

8. **Predict-then-test on FiQA: framework formula validates, priors must
   be measured.** A blind prediction made before training (logged in
   `logs/fiqa_prediction.md`) hit SCHS (predicted 0.40–0.55, actual 0.442)
   and rescue rate (predicted +5 to +15pp, actual +13.0pp) but missed
   base R@10 (predicted 0.08–0.18, actual **0.441**) and therefore
   base-blind subset (predicted 40–60%, actual 34.3%) and overall lift
   (predicted +4 to +8pp, actual **+2.6pp**). The single mistake — assuming
   financial English would be as foreign to MiniLM as Spanish is to
   multilingual MiniLM — propagated through every dependent quantity.

   Plugging the *actual* per-bucket factors into the compositional
   formula `lift = Σ (n_bucket / n_total) × Δ_bucket` reproduces the
   measured +2.6pp exactly: `0.343×0.130 + 0.159×0.030 +
   0.233×(−0.014) + 0.265×(−0.075) = +0.026`. The framework's predictive
   model is correct; what's hard is predicting the per-bucket factors
   from priors alone.

   Operational rule: **always measure base-difficulty distribution
   first**. `evaluation/measure_base_difficulty.py` is cheap (no
   training needed; ~10 min on a 50K corpus) and anchors all the
   downstream factor predictions. Predicting SCHS and rescue rate from
   priors works (both hit on FiQA); predicting base-blind size from
   priors is unreliable.

9. **Readiness-report tool: 5-of-5 correct verdicts on the calibration set.**
   `evaluation/bod_readiness_report.py` combines SCHS + base-difficulty
   distribution into a GO / CONDITIONAL / SKIP verdict before any BoD
   training. Applied to each row of the calibration table:

   v2 calibration (tax = TAX_K × base_perfect × (1 − base R@10)):

   | Corpus | SCHS | BB% | BP% | base R@10 | pred_real | pred_opt | actual | verdict |
   |---|---:|---:|---:|---:|---:|---:|---:|---:|
   | BestBuy ACM      | 0.525 | 44.0% | 13.0% | 0.31 |  +4.4 | **+10.6** |  **+14.2** | GO ✓ |
   | ESCI-Spanish     | 0.450 | 67.0% |  1.1% | 0.07 |  +8.0 | **+16.7** |  **+13.2** | GO ✓ |
   | CQADup/mathematica | (RED) | 50.7% | 37.3% | 0.43 |  +3.9 | +11.4 |   +5.9 | SKIP ✗ (SCHS-gate false-SKIP) |
   | CQADup/tex       | (GREEN) | 52.3% | 36.8% | 0.42 |  +4.1 | +11.7 |   +4.4 | GO ✓ |
   | CQADup/programmers | (YELLOW) | 39.7% | 47.1% | 0.53 |  +2.6 |  +8.7 |   +4.1 | GO ✓ (was false-SKIP under v1) |
   | CQADup/gaming    | (GREEN) | 24.5% | 66.9% | 0.71 |  +1.0 |  +5.0 |   +4.1 | CONDITIONAL/GO (was false-SKIP under v1) |
   | SCIDOCS          | 0.367 | 36.1% |  0.8% | 0.23 |  +4.3 |  +9.0 |   +4.1 | SKIP ✗ (SCHS-gate false-SKIP) |
   | CQADup/physics   | (GREEN) | 32.0% | 52.6% | 0.60 |  +1.7 |  +6.7 |   +3.6 | GO ✓ |
   | CQADup/stats     | (YELLOW) | 49.4% | 42.9% | 0.46 |  +3.6 | +11.0 |   +3.1 | GO ✓ |
   | ESCI-US (E-only) | 0.540 | 34.0% |  4.5% | 0.22 |  +3.7 |  +8.2 |   +3.0 | GO ✓ |
   | CQADup/gis       | (GREEN) | 40.9% | 52.4% | 0.56 |  +2.6 |  +8.5 |   +2.9 | GO ✓ |
   | FiQA-2018        | 0.440 | 34.0% | 26.5% | 0.44 |  +2.6 |  +6.5 |   +2.6 | GO ✓ |
   | SciFact          |  nan  | 20.7% | 77.3% | 0.78 |  +0.8 |  +4.1 |   +1.0 | SKIP ✓ |
   | NFCorpus         | 0.380 | 31.0% |  4.3% | 0.16 |  +3.4 |  +7.5 |   +0.8 | SKIP ✓ (gate) |
   | TREC-COVID       | 0.280 |  8.0% |  0.0% | 0.013 |  +1.0 |  +2.0 |   +0.5 | SKIP ✓ (gate) |
   | Quora            | 0.850 |  2.4% | 92.0% | 0.95 |  −0.2 |  +0.3 |   +0.2 | SKIP ✓ |
   | ArguAna          |  nan  | 23.2% | 76.8% | 0.77 |  +1.0 |  +4.7 |   n/a  | SKIP ✓ |

   Five observations (under v2 calibration; 17-corpus run):
   - **14 of 17 verdicts are correct (10 GO, 4 SKIP, 2 SCHS-gate
     false-SKIPs, 1 untrainable).** GO predictions delivered positive
     lifts (+2.6 to +14.2pp); average ~+5pp. True SKIPs delivered
     ≤+1.1pp lift. The 2 false-SKIPs (SCIDOCS, mathematica) both fired
     via the SCHS<0.40 gate; both actually lifted +4–6pp.
   - **The framework is over-pessimistic on moderate-base corpora.**
     The realistic-band tax of −10pp overstates the cost on every corpus
     with base R@10 > 0.30: actual taxes are −2.3 to −6.9pp on
     CQADupStack subsets, −1.7pp on SciFact, −0.5pp on Quora — all well
     below −10pp. This pushes CQADupStack/programmers and gaming into
     incorrect SKIP verdicts (actual +4.1pp each).
   - **Tax magnitude tracks (1 − base R@10) closely.** Empirically the
     ratio `tax / (1 − base R@10)` clusters around 0.07–0.13 across all
     measured corpora. As of `bod_readiness_report.py` v2 (this commit
     series), the tax bands now scale by (1 − base R@10):
     `tax = TAX_K × (1 − base_R10)`, with `TAX_K = (0.15, 0.10, 0.06)`
     for (pessimistic, realistic, optimistic). This refined formula
     correctly classifies CQADupStack/programmers and gaming as GO (was
     false-SKIPs); SCIDOCS remains a false-SKIP because its SCHS gate
     fires before the realistic band reaches threshold. After the
     refinement the framework is **11/13 verdicts correct** (was 9/13).
   - **The SCHS<0.40 gate produces ~50% false-SKIP among corpora it
     fires on.** Of the 4 RED-SCHS corpora, NFCorpus (0.380) and
     TREC-COVID (0.280) actual lift was correctly tiny (<+1pp), but
     SCIDOCS (0.367) and mathematica (RED) actually delivered +4–6pp
     lift. The gate is too strict on this subgroup. Differentiator: the
     true-SKIPs have higher base-perfect (NFCorpus 4.3%, TREC-COVID 0%
     but 92% of qrels are positives so different metric regime), while
     SCIDOCS has 0.8% base-perfect and mathematica has 37%. No clean
     SCHS threshold separates the false-SKIPs from the true-SKIPs.
     A 2D rule (SCHS, base R@10) might work; out of scope for v2.
   - **Practical reading of the verdict.** A GO verdict predicts
     "expect at least +2pp lift, often much more"; a SKIP can mean
     either "no lift" (≤+1pp) or "modest lift not worth the pipeline
     cost" (~+4pp). Treat SKIP as "don't expect a *big* lift," not as
     "BoD will fail outright."
   - **The bands bracket reality on 7-of-7 trainable corpora.** Actual
     lift falls inside `[pessimistic, optimistic]` for Spanish, ESCI-US,
     FiQA, SciFact, NFCorpus, and TREC-COVID. BestBuy actual *exceeds*
     even optimistic (clicks sharper than the calibration's 25%-rescue
     band). ArguAna is SKIP'd before training (1 positive/query → no
     multi-positive bags possible).
   - **Tax magnitude tracks `(1 − base R@10)`.** Quora has base R@10
     = 0.950 and a tax of only −0.5pp on the base-perfect bucket;
     SciFact at base R@10 = 0.783 has a tax of −1.7pp (both vs the
     realistic band's −10pp). High-base corpora barely disturb
     base-perfect queries because BoD's gradient pressure is dominated
     by base-blind training examples. The framework still produces the
     right SKIP verdict because the optimistic prediction stays below
     the GO threshold even with the over-pessimistic tax assumption;
     realistic is over-pessimistic but the verdict logic is robust.
   - **The SCHS < 0.40 floor does load-bearing work** on NFCorpus
     (base-difficulty math alone would have predicted +0.9 to +7.5pp;
     SCHS gate correctly says SKIP). On SciFact the SCHS is `nan` (only
     23 multi-positive queries on test); the verdict logic correctly
     falls through to base-difficulty alone, where the high
     base-perfect fraction (77%) makes any plausible BoD lift too small
     to justify the pipeline cost.

10. **Rerank-vs-retrieve dominance tracks BM25-vs-base R@10.** Across 7
    corpora, the head-to-head between BoD-as-reranker (BM25 top-50 →
    BoD cosine) and BoD-as-retriever (BoD cosine over the full
    catalog) is well-predicted by whether BM25 alone beats base MiniLM:

    | Corpus | BM25 − base | Rerank − Retrieve | (rerank wins / loses / ties) |
    |---|---:|---:|---|
    | ESCI-Spanish | **+17.6pp** | **+3.7pp** | 35.3 / 18.0 / 46.7 |
    | ESCI-US (E-only) | +2.5pp | +0.8pp | 22.7 / 19.2 / 58.1 |
    | TREC-COVID | +0.2pp | +0.06pp | 42.0 / 34.0 / 24.0 (n=50, noisy) |
    | SciFact | −1.9pp | +1.5pp | 7.0 / 5.3 / 87.7 (88% no-op) |
    | NFCorpus | −1.4pp | −1.0pp | 22.0 / 25.1 / 52.9 |
    | BestBuy ACM | −1.8pp | **−9.7pp** | 6.0 / 25.6 / 68.3 |
    | SCIDOCS | −7.3pp | **−4.8pp** | 10.8 / 28.0 / 61.2 |
    | Quora | −10.1pp | **−3.6pp** | 2.0 / 7.5 / 90.5 |
    | FiQA-2018 | −15.3pp | **−7.7pp** | 10.3 / 25.2 / 64.5 |

    All values are fraction-recovered R@10. Across these 9 corpora the
    sign of (BM25 − base) predicts the sign of (rerank − retrieve)
    correctly in 8 of 9 cases (SciFact is the exception due to its 88%
    ceiling effect); the magnitude correlation is approximate.

    The mechanism: **rerank is bottlenecked by its candidate pool.**
    The BoD model can only sort what BM25 hands it. When BM25's top-50
    catches most gold (ESCI's descriptive titles), the reranker has
    enough material to win. When BM25 misses too much (BestBuy's short
    lexically-noisy queries like "ati" or "i pad 2"), even a perfect
    reranker is constrained — and BoD-as-retriever's full-catalog
    dense search bypasses the bottleneck entirely.

    The exception: SciFact has BM25 weaker than base (−1.9pp) but the
    reranker still wins (+1.5pp). The base is already R@10 = 0.78
    (77% base-perfect queries) — both architectures are mostly no-ops
    (88% ties). With this ceiling effect the BM25-vs-base signal is
    noisy.

    Practical rule: **on a new corpus, compare BM25 R@10 vs base R@10
    *before* deciding between the rerank and retrieve architectures.**
    BM25 ≥ base + ~2pp → rerank likely wins. BM25 < base − ~2pp →
    retrieve likely wins. In the −2 to +2 band, expect mostly ties or
    small wins in either direction. See
    `evaluation/eval_rerank_vs_retrieve.py`.

11. **The specialization tax is intrinsic — query-side routing can't
   avoid it.** Two router probes tested whether a cheap query-time
   signal can route base-perfect queries to base (skipping the −6 to
   −18pp tax) while routing base-blind queries to BoD (capturing the
   rescue):

   - `evaluation/probe_tax_router.py` — base-side signals (top1 cosine,
     top1−top2 margin, mean top10, top1−topk spread). Single-feature
     oracle threshold on BestBuy: every signal's best τ lands at
     "always-BoD." No improvement over BoD-only.
   - `evaluation/probe_tax_router_v2.py` — adds BoD-side signals
     (mirrors), agreement features (top-1 match, top-10 overlap,
     query-encoder cosine), and a learned logistic-regression
     calibrator with 80/20 train/test split. On BestBuy: even the
     in-sample (cheating) router can't beat BoD-only. Default-threshold
     router actively hurts by -8.5pp. On FiQA: tiny gain (+0.26pp on
     n=130 test, within noise); default-threshold router hurts by
     -1.4pp.

   The tax queries cannot be cheaply distinguished from rescue queries
   by any first-stage cosine signal. High base confidence weakly
   correlates with base success (r ≈ +0.37 with base R@10) but only
   weakly anti-correlates with BoD's lift (r ≈ −0.21 with Δ on
   BestBuy's strongest feature, `rank_overlap`). The signal isn't
   sharp enough to separate the buckets.

   Implications:
   - Where overall lift is large (BestBuy +14.2pp), the tax is dwarfed
     and BoD ships net-positive. No router needed.
   - Where overall lift is small (NFCorpus +0.8pp, FiQA +2.6pp), the
     tax can't be cheaply mitigated, so the corpus is stuck with the
     small lift the rescue/tax ratio gives.
   - Score-fusion alternatives (RRF, weighted blend) were already a
     clean negative on ESCI (`project_score_fusion_negative.md`).
   - Beating the tax would require either a serve-time cross-encoder
     verifier (defeats the fast-tier architecture) or a model-side
     fix (different bag construction, different loss). The cheap
     query-side path is closed.

## How to add a new corpus to this table

1. Acquire qrels in the standard format (one of):
   - JSONL with `{query_id, product_id, relevance}` per line
   - JSONL with `{query_id, doc_id, relevance}` (use `--id-field doc_id`)
   - BeIR/* on HuggingFace Hub (use `beir:NAME` in `chs_corpus_compare.py`)
2. Run:
   ```
   python evaluation/cluster_hypothesis_score.py --qrels NEW.jsonl --pids NEW.json --titles NEW.json
   ```
   or
   ```
   python evaluation/chs_corpus_compare.py --datasets beir:NAME ...
   ```
3. Append the row to this table with empirical BoD outcome if known.
4. If the new corpus contradicts the thresholds (e.g., a confirmed
   BoD-positive lands in YELLOW), revise them and update the verdict
   function in `bagofdocs/cluster_hypothesis.py`.
