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

8a. **Rescue rate is predictable from bag stats below base R@10 = 0.85
   (LOO RMSE 2.66pp / LOO R²=0.74; in-sample R²=0.796 / RMSE=2.34pp).**
   Pattern 9 left rescue rate as the unmodeled factor in the readiness
   tool — the realistic band assumes 12pp universally even though the
   measured range is 4–25pp. A linear regression over 19 calibration
   corpora (Quora excluded; see regime gate below) finds three
   pre-training proxies that explain 80% of the in-sample variance:

   ```
   rescue_pp ≈ 5.46 × log10(n_bags) − 0.01 × median_bag_size
              + 52.14 × median_bag_specificity − 46.87
   ```

   - `n_bags`: count of multi-positive queries (qrels-only, free).
   - `median_bag_size`: median # positives per multi-positive query
     (qrels-only, free).
   - `median_bag_specificity`: median intra-bag cosine to centroid
     under the base encoder (requires encoding bag members, but the
     base encoder is already loaded for the base-difficulty step).

   **Regime gate (`base R@10 < 0.85`):** the all-15 fit had LOO
   RMSE 3.74pp / LOO R² 0.537, dragged down entirely by Quora (base
   R@10 = 0.95, LOO residual −7.7pp — the linear model has no support
   at the extreme high-base tail). Refitting on the 14 corpora below
   0.85 lifts LOO R² to 0.78 and tightens RMSE to 2.64pp. Above 0.85,
   `predict_rescue_rate()` returns None and the readiness tool falls
   back to the wide v1 5/12/25pp bands. Threshold sweep (in
   `probe_rescue_predictors.py`) confirms 0.85 is the right cutoff:
   LOO R² is flat at 0.78 across 0.70 ≤ thr ≤ 0.90.

   Worst in-sample residual within regime: mathematica (+6.2pp). All
   other predictions land within ±3pp.

   The qrels-only ablation (drop `median_bag_specificity`) looks
   reasonable in-sample (R²=0.608 / RMSE=3.44pp) but **fails LOOCV**
   (LOO R² = −0.534 — worse than predicting the corpus mean). Bag
   specificity is the only feature that survives LOOCV; without it, the
   model is just memorising. Implication: do not advertise a "qrels-only
   fallback" — always run the bag-encoding step (the base encoder is
   already loaded for base-difficulty anyway). A `+ base_r10` variant
   also fails LOOCV (LOO R² = −0.561), confirming it adds nothing as
   a feature. (`base_r10` *is* used as the regime gate, not as input.)

   Practical: the readiness tool replaces the wide rescue band
   (5/12/25pp) with a point estimate ± LOO RMSE (2.64pp) when in regime,
   falling back to the wide bands above the threshold. Implemented in
   `evaluation/bod_readiness_report.py` (`compute_bag_stats` +
   `predict_rescue_rate`); regression, ablations, LOOCV, and the
   threshold sweep live in `probe_rescue_predictors.py`. End-to-end on
   FiQA: predicted rescue 10.8pp ±2.6pp, measured 13.0pp (within band).

8b. **Predict-then-test on FiQA: framework formula validates, priors must
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

8d. **Pure predict-then-test on CQADup/unix (never in calibration):
   rescue prediction hits within LOO band.** Generated all priors
   BEFORE training, then ran the full bag/hardneg/MNRL/diagnose
   pipeline:

   | Prior | Predicted | Measured | Δ |
   |---|---:|---:|---:|
   | Base R@10 | 0.550 | 0.550 | 0.000 |
   | Base-blind subset | 38.8% | 38.7% | −0.1pp |
   | Base-perfect subset | 49.4% | 49.5% | +0.1pp |
   | **Rescue rate** | **12.0 ±2.6pp** | **13.1pp** | **+1.1pp ✓ in LOO band** |
   | Realistic Δ R@10 | +2.4pp | — | — |
   | Optimistic Δ R@10 | +4.4pp | — | — |
   | Overall Δ R@10 | — | **+4.7pp** | slightly above optimistic |

   The prediction was made with the gated 14-corpus regression and
   uses train_qrels for bag stats (the methodology fix this session
   shipped in `evaluation/bod_readiness_report.py` — without it,
   bag stats would have come from the smaller test_qrels and the
   prediction would have been ~2pp lower).

   Verdict was CONDITIONAL when actual was +4.7pp — the tool errs
   conservative by design (the optimistic-band threshold for GO is
   +5pp, just above what unix delivered). No change recommended.

   **Second predict-then-test on CQADup/webmasters (also never in
   calibration):**

   | Prior | Predicted | Measured | Δ |
   |---|---:|---:|---:|
   | Base R@10 | 0.524 | 0.524 | 0.000 |
   | Base-blind subset | 38.9% | 38.9% | 0.0pp |
   | Base-perfect subset | 46.8% | 46.8% | 0.0pp |
   | **Rescue rate** | **9.3 ±2.6pp** | **10.2pp** | **+0.9pp ✓ in LOO band** |
   | Realistic Δ R@10 | +1.4pp | — | — |
   | Optimistic Δ R@10 | +3.3pp | — | — |
   | Overall Δ R@10 | — | **+4.9pp** | above optimistic |

   The rescue prediction hits, but the realistic tax (4.8pp) was much
   higher than the actual tax (2.3pp), so the overall Δ overshot the
   optimistic band. Same pattern as unix.

   **Three more predict-then-tests on CQADupStack (android, english,
   wordpress)** widen the validation set:

   | Corpus | Predicted rescue | Measured | Δ | Band | Overall Δ |
   |---|---:|---:|---:|---|---:|
   | unix | 12.0 ±2.6 | 13.1 | +1.1 | ✓ | +4.7pp |
   | webmasters | 9.3 ±2.6 | 10.2 | +0.9 | ✓ | +4.9pp |
   | android | 11.6 ±2.6 | 7.1 | −4.5 | ✗ | +2.2pp |
   | english | 11.8 ±2.6 | 16.1 | +4.3 | ✗ | +5.7pp |
   | wordpress | 9.3 ±2.6 | 6.8 | −2.5 | edge | +2.8pp |

   **Out-of-sample stats**: 3 of 5 within the ±2.57pp LOO band; rescue
   RMSE 3.03pp (vs LOO 2.57pp — degraded ~0.5pp on truly held-out
   corpora). The two misses run opposite directions (android
   overshoots, english undershoots), so the residual is noise around
   the mean rather than systematic bias. Both misses have small
   n_bags (191 and 485) where the regression's variance is highest.

   **Tax over-estimation is less consistent than the first two
   datapoints suggested.** unix and webmasters delivered overall lifts
   above their optimistic bands; english did too. android landed
   exactly at its optimistic-band threshold (+2.2 vs +2.3). wordpress
   landed between realistic and optimistic. Pattern is "mostly
   slightly under-estimated" rather than the strong "always above
   optimistic" claim from N=2 — not worth a tax-band refit yet.

8c. **End-to-end sweep validation: predictor RMSE 2.09pp on the 14
   in-regime calibration corpora — tighter than the LOO band (2.57pp),
   as expected for in-sample fit.** Running `evaluation/sweep_readiness.py`
   with the gated 15-corpus regression and the train-qrels methodology
   gives RMSE = 2.09pp / MAE = 1.64pp / 11 of 14 within ±2.57pp.

   This is a meaningful improvement from the v2 sweep (RMSE 2.82pp,
   9/14 in band) which used test-qrels for bag stats. The fix: the
   predictor was calibrated against bags built from train_qrels (since
   `training/bags_from_qrels.py` reads train), so the readiness tool
   should compute bag stats from train_qrels too when available. The
   `--train-qrels` flag in `bod_readiness_report.py` (and threaded
   through `sweep_readiness.py` and `run_beir_chain.sh`) handles this.

   Three previous outside-band cases were rescued by the fix:

   - **ESCI-Spanish**: −4.6pp → −2.1pp ✓
   - **SciFact**: −4.4pp → −0.8pp ✓
   - **NFCorpus**: −4.2pp (clamped at 0) → −0.3pp ✓

   The three outside-band cases in v3 are all near the edge or
   structural:

   - **FiQA-2018** (Δ +2.8pp): just barely outside; flipped from a v2
     undershoot (−2.2pp) when train-qrels gave more bags.
   - **CQADup/gaming** (Δ −2.6pp): same as v2; right at the band edge.
   - **CQADup/mathematica** (Δ −5.1pp): structural — MiniLM is
     unusually noisy on math/LaTeX text, so `median_spec` understates
     effective bag tightness. Tested 6 alternative spec features
     (`p10/p25/std/iqr_spec` as additions/replacements) — none rescue
     mathematica, most degrade LOOCV. Domain-encoder-fit is the
     missing signal; no available bag-stat feature captures it.

   False-SKIPs in v3: SCIDOCS, mathematica, SciFact, NFCorpus (the v2
   tax band still over-penalises high-base-perfect corpora). The tool
   errs conservative — four false-SKIPs in 16 corpora vs zero
   false-GOs. ArguAna correctly hard-SKIPs on the n_bags=0 path.

8e. **The predictor's calibration is encoder-specific, not
   encoder-agnostic — by construction.** This was the lesson from
   trying to "upgrade" the predictor's bag-stat encoder from MiniLM
   to `BAAI/bge-base-en-v1.5`. Worth stating up-front because the
   architectural confusion bit me before the data confirmed it.

   **The setup, in plain terms.** The predictor learns a regression:

       bag-stats → measured rescue rate

   The TARGET (rescue rate) was measured by:
   1. Building bags from train_qrels.
   2. Fine-tuning **MiniLM-L6** with MNRL on those bags.
   3. Comparing the fine-tuned model's R@10 against MiniLM-base R@10
      on the test set's base-blind queries.

   So every rescue number in the calibration table is implicitly
   "MiniLM-rescue" — *how much MiniLM specifically can be improved
   by fine-tuning on this corpus's bags*. It is **not** "rescue under
   any encoder you might pick."

   The FEATURE (`median_spec`) is the intra-bag mean cosine, computed
   by encoding the catalog with some chosen base encoder.

   When the feature encoder and the rescue encoder match (both MiniLM),
   the regression learns a coherent thing: "given how MiniLM sees the
   bags, predict how much MiniLM-BoD lifts the corpus." When you swap
   the feature encoder to bge-base but leave the target as MiniLM-rescue,
   you're asking the regression to learn "given how bge-base sees the
   bags, predict how much MiniLM-BoD lifts the corpus" — a much weaker
   connection because bge-base's view of bag tightness has no
   architectural reason to track MiniLM's headroom for fine-tuning.

   **What the data showed.** Re-encoded all 19 calibration catalogs
   under bge-base and refit the regression. Results (gated, base R@10
   < 0.85, Quora excluded):

   | metric | MiniLM features | bge-base features |
   |---|---:|---:|
   | in-sample R² | 0.798 | 0.788 |
   | in-sample RMSE | 2.33pp | 2.38pp |
   | LOO R² | **0.745** | 0.619 |
   | LOO RMSE | **2.61pp** | 3.19pp |

   - bge-base produces uniformly higher `median_spec` (0.85–0.97 vs
     MiniLM 0.64–0.96) — every corpus's bags look "tight enough"
     to a stronger encoder, so the *range* compresses. The regression
     compensates by inflating `W_SPEC` to +159.8 (vs MiniLM's +54.8)
     and the intercept to −152.1 (vs −46.9), which makes the predictor
     hyper-sensitive to small spec differences and torches out-of-sample
     variance.
   - Mathematica's residual stays at +6.1pp (was +5.9pp under MiniLM)
     — a stronger encoder is not the missing ingredient for
     domain-encoder-fit.
   - Per-corpus deltas are direction-mixed (SciFact, gaming, android
     improve; TREC-COVID +5.7pp worse, NFCorpus −3.1pp, FiQA −1.3pp)
     — consistent with "the new feature is uncorrelated with the old
     target" rather than "the new feature is better/worse globally."

   **What would make this apples-to-apples?** Either:

   - **Re-measure the target.** Fine-tune *bge-base* with MNRL on each
     calibration corpus's bags, evaluate against bge-base R@10, refit
     bge-base features against bge-base rescue. Then the regression
     would learn "given how bge-base sees the bags, predict how much
     bge-base-BoD lifts." Apples-to-apples. But: prior memory
     (`project_bge_base_bod_probe.md`) shows bge-base + MNRL HURTS
     ESCI by 11pp R@10 — the pre-trained prior dominates the
     370K-triplet fine-tune. So we may not have a calibration set
     where bge-base BoD even works.

   - **Encoder-invariant features.** Replace `median_spec` with a
     signal that doesn't depend on a specific embedding space —
     lexical overlap, IDF-weighted bag tightness, average title
     length. Then bag-stats predict rescue regardless of which
     encoder is in play. Untested.

   **Practical takeaway: keep MiniLM as the predictor's bag-stat
   encoder.** This is consistent with the framework (encoder-specific
   prediction is fine — most users will fine-tune the same base they
   plan to deploy). It just means we should not interpret a future
   "what if I use a stronger encoder" question as something the
   current predictor can answer; that requires re-running the full
   train + measure pipeline under that encoder. Script + data:
   `evaluation/bge_base_bag_stats.py`, `logs/bge_base_bag_stats.tsv`.

8f. **bge-base BoD is signal-bound, not inherently bad.** The prior
   bge-base BoD probe on ESCI-US (May 2026, recorded as a "clean
   negative") used ~370K qrels-derived triplets and lost −11pp R@10
   to MiniLM-BoD. The diagnosis at the time was "pre-trained prior
   dominates the fine-tune; training scale is binding." Today's
   experiment falsifies the broader generalization: under the
   click-derived BestBuy signal at comparable scale, bge-base BoD
   *beats* MiniLM-BoD on every dimension.

   | metric | MiniLM-BoD | bge-base-BoD | Δ |
   |---|---:|---:|---:|
   | base R@10 (out of box) | 0.306 | 0.345 | +3.9pp |
   | base-blind subset | 44.0% | 40.3% | −3.7pp |
   | **rescue rate** | **+24.9pp** | **+28.2pp** | **+3.3pp better** |
   | spec tax (base-perfect) | −6.4pp | −3.8pp | 2.6pp less tax |
   | **overall Δ R@10** | **+14.2pp** | **+14.9pp** | **+0.7pp better** |

   bge-base-BoD pays less spec tax on the base-perfect subset, rescues
   harder on base-blind queries, and starts from a higher base — a
   compound win, not a marginal one. Even though bge-base has less
   relative headroom (higher base), its sharper representations let
   it convert click signal into bigger per-query gains.

   **Reframing the failure mode.** The variable that flipped between
   ESCI (loss) and BestBuy (win) was bag *signal sharpness*, not
   training set size:

   - ESCI-US: ~370K triplets, qrels-graded supervision (E vs S/I/C)
   - BestBuy: ~240K triplets, click-derived supervision (60K queries
     × ~50 clicked SKUs per query)

   The triplet counts are similar; the *information content per
   triplet* is what differs. Click signal is sharper because each
   positive is a verified user choice; qrels signal is noisier because
   "exact match" labels include category-level matches that hurt
   training. Stronger pretraining + sharper signal = compound
   advantage; stronger pretraining + noisier signal = the prior
   dominates the fine-tune.

   **Open questions** (queued, not run):
   - bge-base-BoD on CQADup/gaming, where MiniLM rescue was +16.1pp
     on a CE-filtered qrels signal at much smaller training scale.
     Would test whether scale alone matters when signal is mid-quality.
   - bge-base-BoD on ESCI-Spanish, where base R@10 is extremely low
     (0.07), so headroom is huge. Counter-pressure: bge-base is
     English-only.

   Artifacts: `query_model_bestbuy_bge_bod/` (the trained model),
   `logs/bestbuy_bge_bod_*.log` (encode / train / diagnose), local
   only. Disk: ~470 MB for the model.

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

12. **BEIR readiness sweep on 3 untouched paradigms: all SKIP, but
   for different reasons — and HotpotQA surfaces a sharp "false-SKIP
   zone" candidate.** Ran the readiness tool (no training) on three
   large BEIR datasets not in the calibration set, each from a
   different paradigm than what's already measured:

   | Corpus | n_docs | SCHS | base R@10 | BB% | BP% | rescue (pred) | BM25 vs base | Verdict |
   |---|---:|---:|---:|---:|---:|---:|---|---|
   | DBPedia-entity | 4.6M | 0.406 | 0.187 | 50.7% | 9.6% | 2.2 ±2.7pp | either | SKIP |
   | HotpotQA | 5.2M | 0.377 | 0.512 | 23.1% | 19.6% | **23.3 ±2.7pp** | rerank | SKIP |
   | Climate-FEVER | 5.4M | 0.344 | 0.097 | 60.0% | 0.7% | 10.4 ±2.7pp | retrieve | SKIP |

   All three SKIP, but the *reasons* differ in informative ways:

   - **DBPedia-entity** (entity retrieval): SCHS just above the 0.40
     floor, but predicted rescue is small (2.2pp). The verdict is
     SKIP because the optimistic-band lift won't justify training,
     not because clustering is broken. Reasonable "marginal corpus"
     case.

   - **HotpotQA** (multi-hop QA): SCHS 0.377 is just below the floor,
     but predicted rescue is **23.3pp — second-highest in the entire
     calibration set, only BestBuy click data (24.9pp) is higher.**
     n_bags = 85,000 with median_size = 2 reflects HotpotQA's
     "supporting passage pairs" structure (each question has two
     supporting docs). The sharp signal would predict a strong lift,
     but the SCHS floor blocks training. This is the **textbook
     false-SKIP-zone case**: high signal sharpness + weak clustering
     geometry. Worth a pilot training run if anyone wants to test
     whether the SCHS floor is over-conservative for multi-hop
     paradigms.

   - **Climate-FEVER** (claim verification): SCHS 0.344 well below
     floor, predicted rescue 10.4pp would be respectable if SCHS
     allowed it. Has known qrels noise from prior work (1391
     supporting-evidence judgments for 1535 claims) — clustering
     score may itself be depressed by label noise. Retrieve
     architecture (BM25 < base) — typical for paraphrased-claim
     retrieval.

   **Infrastructure findings from the sweep:**
   - `bod_readiness_report.py` originally double-encoded the catalog
     (compute_chs and base_difficulty each ran their own encode).
     For 5M+ doc corpora this was fatal: the first encode finished
     but the second never started because of OOM/disk pressure.
     Refactored to a single up-front encode shared via the existing
     `cache_vecs` parameter on compute_chs. Wall-clock per dataset
     halved.
   - Each dataset's catalog encode is the dominant cost (~3 hr on
     MiniLM-L6 + MPS for 5M docs). Predict-only sweeps on BEIR-scale
     corpora are practical at ~3-4 hr each, not feasible at higher
     throughput without changing the encoder or hardware.

   Data: `logs/{dbpedia_entity,hotpotqa,climate_fever}_readiness.log`.
   Run script: `overnight_beir_readiness.sh` (canonical; includes the
   `--chunk 64` and refactored single-encode fixes from this session).

13. **HotpotQA pilot training confirms the false-SKIP-zone — the SCHS
   floor at 0.40 is over-conservative for multi-hop-QA paradigms.**
   Pattern 12 flagged HotpotQA as a candidate false-SKIP: SCHS 0.377
   just below the floor, but predicted rescue **23.3pp ±2.7pp** (second
   only to BestBuy click data). The readiness verdict was SKIP. We
   trained BoD anyway as a deliberate pilot.

   **Setup (under-trained on purpose):**
   - 5,000 bags (random sample of the 85,000 available — to avoid the
     MPS leak that killed the full-corpus run at iter 79).
   - 1 epoch, batch=16, max_seq=128 — minimum viable training config.
   - Diagnosed against a 500K-doc subsampled catalog (all 13,783 gold
     docs + 486,217 random fill from the full 5.2M-doc catalog). The
     full-catalog BoD encode died at 89% to a jetsam OOM; the subsample
     preserves the base-vs-BoD Δ since both retrieve from the same pool.

   **Result:**

   | Bucket | n | base | BoD | Δ |
   |---|---:|---:|---:|---:|
   | 0.0 (base-blind) | 799 (10.8%) | 0.000 | **0.351** | **+35.1pp rescue** |
   | 0.5-1.0 | 3,921 (53.0%) | 0.500 | 0.612 | +11.2pp |
   | 1.0 (base-perfect) | 2,685 (36.3%) | 1.000 | 0.946 | −5.4pp tax |
   | **overall** | 7,405 | 0.627 | **0.705** | **+7.8pp** |

   The rescue rate on base-blind queries is **+35.1pp** — *higher* than
   the readiness predictor's 23.3pp ±2.7pp band (the predictor under-
   estimated). And this is under-trained BoD (5K bags / 1 epoch vs the
   85K bags / 3 epochs the full pipeline would run). Full training is
   likely to lift rescue further.

   **Implication: the SCHS 0.40 floor is correct as a default but wrong
   for high-rescue-prediction paradigms.** The current readiness tool
   uses SCHS as a hard gate — `verdict = SKIP if SCHS < 0.40 regardless`.
   HotpotQA shows that a corpus with SCHS just below the floor AND a
   predicted rescue rate well above the calibration mean is a false-SKIP.

   Two ways to revise (not yet implemented; need more data points
   before committing):
   1. **Lower the floor** to e.g. 0.35 globally. Risk: opens up
      false-GOs on corpora with low SCHS *and* low predicted rescue.
   2. **Override the floor** when predicted rescue is high. Rule of
      thumb candidate: "If predicted rescue > 1.5× the
      calibration-mean rescue (~10pp), bypass the SCHS floor and
      use the standard GO/CONDITIONAL/SKIP logic." HotpotQA's 23.3pp
      predicted rescue would clear this bar; DBPedia-entity's 2.2pp
      would not. Climate-FEVER's 10.4pp is borderline.

   N=1 isn't enough to choose between revisions. The honest framing
   for now: HotpotQA is a documented false-SKIP; the readiness tool
   should print a warning when SCHS is in [0.30, 0.40) AND predicted
   rescue > 15pp. Implementing this warning is small (~10 lines).

   **Infrastructure salvage technique** (worth noting for future runs):
   when a full-catalog encode dies near completion, build a subsampled
   catalog with all gold docs + random fill and re-encode. Preserves
   the BoD/base Δ measurement at a fraction of the compute.

   Run logs: `logs/hotpotqa_pilot_{train,diagnostic}.log`,
   `logs/hotpotqa_salvage_{status,diagnostic}.log`.

14. **HyDE complements BoD on SciFact — they rescue mostly DIFFERENT
   queries.** HyDE (Hypothetical Document Embeddings) is the closest
   philosophical cousin to BoD: both represent the query in document
   space rather than query space. We tested them head-to-head on
   SciFact at resource-matched scale — local Llama 3.1 8B Q4 (via
   Ollama) for HyDE's hypothetical-passage generation, MiniLM-L6 BoD
   for the supervised path.

   **Aggregate R@10 on SciFact (n=300 test queries):**

   | Retriever | R@10 | Δ vs base |
   |---|---:|---:|
   | base (MiniLM raw query) | 0.783 | — |
   | BoD (MiniLM fine-tuned on bags) | 0.793 | +1.0pp |
   | HyDE (Llama 3.1 8B → passage → MiniLM) | **0.841** | **+5.8pp** |

   HyDE beats BoD by 4.8pp overall on this corpus. Rescue rate on
   base-blind queries (n=62, 20.7% of pos-bearing):
   - **HyDE: +36.3pp** (rescues 23/62 = 37.1% of blind queries)
   - **BoD: +12.1pp** (rescues 8/62 = 12.9%)

   **The overlap analysis — Pattern 14 headline:**

   |  | HyDE rescues | HyDE misses |
   |---|---:|---:|
   | **BoD rescues** | 4 | 4 |
   | **BoD misses** | 19 | 35 |

   - **Only 4 queries overlap between BoD's 8 rescues and HyDE's 23.**
   - HyDE rescues 19 queries BoD doesn't; BoD rescues 4 queries HyDE
     doesn't.
   - Union rescue = 27 queries (43.5% of base-blind) — almost 2× either
     method alone.

   The two methods target different failure modes. BoD learns from
   labeled bag structure (multiple correct passages clustering near a
   centroid). HyDE leans on the LLM's prior knowledge — for SciFact's
   biomedical queries, Llama can generate factually accurate hypothetical
   passages that embed close to the gold scientific abstracts.

   **Caveats:**
   - **Biomedical SciFact is near-best-case for HyDE.** Llama has
     strong prior knowledge in scientific domains. Test corpora where
     Llama's prior is weak (product SKUs, programmer Q&A, niche
     forum text) likely flip the comparison.
   - **Inference-time cost asymmetry.** BoD is zero-overhead vs base
     bi-encoder. HyDE costs an LLM call per query (~7-10 sec at 20
     tok/s local Llama). For production, HyDE is only viable when
     the corpus value justifies the latency.
   - **Resource-matched comparison.** Both methods are at "laptop
     scale" — local Llama 8B Q4 + MiniLM. Production HyDE with
     GPT-4-class LLMs would likely raise its rescue rate further.

   **Implication for the framework:** the natural follow-up is an
   *ensemble*: train BoD AND generate HyDE passages, then either
   score-fuse (RRF / weighted mean of rankings) or run both retrievers
   and merge top-k. Pattern 14 establishes the necessary condition
   (complementarity at the query level); whether the ensemble actually
   delivers depends on score calibration.

   **Score-fusion (RRF) ablation on SciFact:**

   | Method | R@10 | base-blind rescue |
   |---|---:|---:|
   | base | 0.783 | — |
   | BoD | 0.793 | 12.9% |
   | **HyDE** | **0.841** | **37.1%** |
   | RRF(BoD, HyDE) | 0.832 | 27.4% |
   | RRF(base, BoD, HyDE) | 0.818 | 21.0% |
   | **UNION (oracle upper bound)** | **0.872** | **43.5%** |

   **RRF under-performs HyDE alone** (0.832 vs 0.841). The mechanism
   is clear: BoD's rescue rate on SciFact is so much weaker than HyDE's
   that fusing them dilutes HyDE's high-quality rankings with BoD's
   noisier ones. RRF works best when components are roughly equal
   quality; SciFact is too lopsided. **But the oracle UNION (+8.9pp
   overall, 43.5% rescue) confirms the complementarity headroom is
   real** — RRF just isn't the fusion strategy that captures it. A
   quality-weighted fusion, learned reranker, or query-router that
   sends queries to whichever method is more likely to rescue them
   would be needed to reach the union ceiling.

   **Open questions** (queued, not run):
   - Does the complementarity hold on non-biomedical corpora (FiQA,
     NFCorpus, CQADupStack)? Untested.
   - On corpora where Llama's prior is weak (BestBuy click data,
     niche technical forums), does HyDE under-perform BoD as expected?
     Untested.
   - Quality-weighted or learned fusion to capture the UNION ceiling?
     RRF was a clean negative; richer fusion methods unexplored.

   Pipeline: `evaluation/eval_hyde.py` (HyDE generation + eval),
   `evaluation/diagnose_lift.py` (BoD per-query JSONL),
   `evaluation/diagnose_hyde_vs_bod.py` (overlap table),
   `evaluation/eval_rrf_ensemble.py` (RRF + union). Run via Ollama
   on localhost:11434; LLM model: `llama3.1:8b-instruct-q4_K_M`.

   **FiQA-2018 follow-up (financial QA, non-biomedical):**

   | Retriever | R@10 | Δ vs base | base-blind rescue |
   |---|---:|---:|---:|
   | base | 0.441 | — | — |
   | BoD | 0.467 | +2.6pp | 20.7% |
   | **HyDE** | **0.405** | **−3.6pp** | 21.2% |
   | RRF(BoD, HyDE) | 0.471 | +3.0pp | 24.8% |
   | RRF(base, BoD, HyDE) | **0.474** | +3.3pp | 19.8% |
   | UNION (oracle) | 0.563 | +12.2pp | 33.3% |

   **HyDE alone LOSES on FiQA (−3.6pp)** — the inverse of SciFact. The
   mechanism: financial questions reference specific entities (companies,
   instruments, regulations) and Llama 8B's prior generates plausible
   passages that retrieve *similar but wrong* documents. The spec tax
   on base-perfect queries is −21.7pp — HyDE destroys nearly a quarter
   of the queries base was getting right.

   But two things flip vs SciFact:
   1. **BoD wins modestly on FiQA (+2.6pp)** — supervised bag training
      adds value when the LLM prior is weak.
   2. **RRF actually helps on FiQA** — RRF(BoD, HyDE) at +3.0pp beats
      BoD alone. RRF(base, BoD, HyDE) reaches +3.3pp. The reason: BoD
      and HyDE rescue rates are *balanced* on FiQA (20.7% vs 21.2%),
      whereas SciFact was lopsided (12.9% vs 37.1%). RRF works when
      the components are roughly equal quality.

   **Overlap on FiQA base-blind subset (n=222):** BoD-rescues 46 ∩
   HyDE-rescues 47 = 19 overlap. BoD-only 27, HyDE-only 28, neither
   148. Union = 74 (33.3%) — the oracle ceiling. Still substantial
   complementarity, but with more overlap than SciFact (where only
   4/8 BoD rescues were also HyDE rescues).

   **BestBuy ACM follow-up (1K random test-query subsample; product
   search where LLM prior is weakest):**

   | Retriever | R@10 | Δ vs base | rescue% |
   |---|---:|---:|---:|
   | base | 0.314 | — | — |
   | **BoD** | **0.537** | **+22.3pp** | **66.4%** |
   | HyDE | 0.307 | −0.7pp | 21.4% |
   | RRF(BoD, HyDE) | 0.460 | +14.6pp | 49.4% |
   | RRF(base, BoD, HyDE) | 0.416 | +10.2pp | 34.8% |
   | UNION (oracle) | 0.583 | +26.9pp | 70.2% |

   Predicted before running (we explicitly wrote it down): HyDE would
   pay heavy spec tax on product-specific queries because Llama 8B
   generates plausible *generic* product descriptions that retrieve
   similar-but-wrong SKUs. **Confirmed**: tax on base-perfect is
   −17.5pp, rescue on base-blind only +9.6pp; net −0.7pp.

   **Overlap on BestBuy base-blind (n=441):** BoD-rescues 293 ∩
   HyDE-rescues 94 = 77 overlap. **82% of HyDE's rescues are also
   BoD rescues** — HyDE is essentially subsumed. This is the inverse
   of SciFact (only 17% of HyDE rescues were also BoD rescues, i.e.
   HyDE found mostly NEW queries). FiQA sat in between.

   **Three-corpus picture — the dominant lever for each method:**

   | Corpus | BoD | HyDE | RRF | Dominant lever |
   |---|---:|---:|---:|---|
   | SciFact (biomed) | +1.0pp | **+5.8pp** | +4.9pp (loses to HyDE) | LLM-prior strong → HyDE wins |
   | FiQA (finance) | +2.6pp | −3.6pp | **+3.0pp** (beats both) | balanced → RRF wins |
   | BestBuy (clicks) | **+22.3pp** | −0.7pp | +14.6pp (loses to BoD) | bag signal sharp → BoD wins |

   **The two governing axes:**

   1. **Bag signal sharpness drives BoD magnitude.** Click data
      (BestBuy) gives BoD +22.3pp. Graded qrels (FiQA) gives +2.6pp.
      Noisy multi-positive qrels (SciFact) gives +1.0pp. This matches
      the rescue-rate predictor's reading of bag stats.
   2. **LLM-prior strength drives HyDE magnitude.** Strong domain
      knowledge (SciFact biomedicine) gives HyDE +5.8pp. Mid-knowledge
      (FiQA finance) gives −3.6pp. Weak (BestBuy SKUs) gives −0.7pp
      with a heavy tax. As the LLM prior weakens, HyDE's tax on
      base-perfect grows because it generates plausible-but-wrong
      neighbors.

   **The overlap pattern reflects which lever is dominant:**

   - SciFact: HyDE dominates → almost disjoint rescues (BoD picks up
     queries the LLM doesn't know; HyDE picks up the rest).
   - BestBuy: BoD dominates → HyDE rescues subsumed (Llama mostly
     surfaces things BoD already had via the training signal).
   - FiQA: balanced → moderate overlap, complementarity in the middle.

   **RRF only wins when components are balanced.** Lopsided pairs
   (SciFact, BestBuy) lose to the dominant component. The oracle
   UNION numbers confirm there's headroom in all three cases, but
   capturing it requires quality-aware fusion (learned reranker,
   query-router, or weighted fusion), not RRF.

   **Practical framework recommendation:** the framework should
   include LLM-prior strength as a fourth factor alongside bag-signal
   sharpness, base-model competence, and clustering geometry. The
   prediction rule: BoD wins when bag signal is sharp; HyDE wins when
   LLM has strong prior in the corpus's domain; either-or — pick the
   side that has the stronger signal for your corpus. RRF as a default
   fusion is a clean negative across all three corpora tested.

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
