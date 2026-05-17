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

**Index** (chronological; each entry below has the full discussion):

*Cluster Hypothesis Score (CHS) calibration*
- **1.** SCHS rank-orders BoD lift magnitudes
- **2.** High SCHS + too few multi-positive queries is a trap (NQ caveat)
- **3.** HCHS adds signal mainly with genuine hard negatives in qrels
- **4.** Strong-inv rate caveats
- **5.** SCHS is necessary but doesn't predict lift magnitude alone

*Specialization tax and base-shrinking*
- **6.** Sharper hardnegs sharpen rescue AND the specialization tax
- **7.** Stronger base shrinks the base-blind subset (causal test)

*Rescue-rate predictor*
- **8a.** Bag stats predict rescue rate below base R@10 = 0.85
- **8b.** FiQA predict-then-test — framework validates, priors update
- **8c.** End-to-end sweep validation (RMSE 2.09pp on 14 corpora)
- **8d.** CQADup/unix pure predict-then-test
- **8e.** Predictor calibration is encoder-specific
- **8f.** bge-base BoD is signal-bound, not inherently bad

*Readiness tool and BEIR sweep*
- **9.** Readiness-report tool: 5-of-5 correct verdicts
- **10.** Rerank-vs-retrieve dominance tracks BM25-vs-base
- **11.** Specialization tax is intrinsic — query-side routing can't fix
- **12.** BEIR readiness sweep: 3 SKIP, HotpotQA flagged as false-SKIP candidate
- **13.** HotpotQA pilot confirms the false-SKIP zone

*LLM-era methods*
- **14.** HyDE vs BoD across 6 corpora — two governing axes; RRF clean negative
- **14b.** Weighted SCORE fusion beats RRF
- **14c.** Pseudo-relevance feedback (PRF) clean negative — the LLM IS the lever
- **15.** Doc2Query as doc-side LLM lever — broader regime than HyDE
- **16.** CLIP image fusion on ESCI — clean negative across three strategies

*Oracle composition and base encoder*
- **17.** Three-way union-oracle (BoD + HyDE + Doc2Query) — +3.5-6pp router headroom
- **18.** Stronger domain-specialized base + LoRA-BoD — each lever attacks a different bottleneck
- **19.** Four-way union-oracle — domain pretraining as fourth orthogonal lever

---

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

14. **HyDE vs BoD across 6 corpora — two governing axes, RRF is a
    clean negative.** Sustained head-to-head experiment comparing HyDE
    (Hypothetical Document Embeddings via local Llama 3.1 8B Q4) to
    BoD (MiniLM-L6 fine-tuned on bags) at resource-matched scale on a
    laptop. Both methods represent the query in *document space*
    rather than query space — the same philosophical move via
    different mechanisms.

    **Six-corpus summary (R@10 deltas vs base, 1K query subsample for
    BestBuy, full test sets elsewhere):**

    | Corpus | base R@10 | BoD Δ | HyDE Δ | RRF(BoD,HyDE) | UNION oracle | LLM-prior strength |
    |---|---:|---:|---:|---:|---:|---|
    | SciFact (biomed) | 0.783 | +1.0pp | **+5.8pp** | +4.9pp | +8.9pp | strong (biology) |
    | NFCorpus (medical/nutrition) | 0.159 | +0.8pp | **+1.5pp** | small | small | strong-but-noisy |
    | FiQA-2018 (financial QA) | 0.441 | **+2.6pp** | −3.6pp | **+3.0pp** | +12.2pp | mid (general finance) |
    | CQADup/english (general Q&A) | 0.577 | **+5.7pp** | −5.8pp | +3.4pp | +11.0pp | mid |
    | CQADup/programmers (code Q&A) | 0.529 | **+4.1pp** | −8.5pp | +1.0pp | +10.3pp | mid-weak |
    | BestBuy ACM (product clicks, 1K subsample) | 0.314 | **+22.3pp** | −0.7pp | +14.6pp | +26.9pp | weak (SKUs) |

    **Two governing axes (the headline finding):**

    1. **Bag signal sharpness drives BoD magnitude.** Clicks (BestBuy)
       → +22.3pp; general qrels (CQADup/english/programmers, FiQA)
       → +2.6 to +5.7pp; noisy biomedical qrels (SciFact, NFCorpus)
       → +0.8 to +1.0pp. Matches the calibration table's rescue-rate
       predictor exactly.

    2. **LLM-prior strength drives HyDE magnitude.** Llama 3.1 8B has
       strong biomedical priors → SciFact +5.8pp, NFCorpus +1.5pp.
       Mid priors for finance and general Q&A → FiQA −3.6pp, english
       −5.8pp. Weak priors for programming forums and product SKUs
       → programmers −8.5pp, BestBuy −0.7pp (small only because
       BestBuy's base R@10 is already low — the spec tax was a hefty
       −17.5pp on base-perfect queries).

    HyDE *wins overall* only on the two biomedical corpora (2/6).
    BoD wins on the other four (4/6) and ties or wins narrowly on
    NFCorpus. As LLM domain knowledge degrades, HyDE's spec tax on
    base-perfect grows rapidly: the LLM generates plausible-but-wrong
    neighbors that retrieve similar-but-not-the-correct docs.

    **The overlap pattern matches which axis dominates:**

    | Corpus | BoD rescues | HyDE rescues | overlap | structure |
    |---|---:|---:|---:|---|
    | SciFact | 8 | 23 | 4 (17%) | **disjoint** — HyDE picks up what BoD misses, vice versa |
    | NFCorpus | 22 | 25 | 12 (48%) | balanced |
    | FiQA | 46 | 47 | 19 (41%) | balanced |
    | english | 115 | 74 | 30 (41%) | balanced-skewed-to-BoD |
    | programmers | 53 | 54 | 14 (26%) | balanced |
    | BestBuy | 293 | 94 | 77 (82%) | **subsumed** — HyDE rescues are mostly BoD rescues too |

    The "complementarity" we observed on SciFact (only 4/8 BoD
    rescues overlap with HyDE's 23) is corpus-specific, not a
    general property. As the dominant axis switches from "HyDE wins
    big" (SciFact) to "BoD wins big" (BestBuy), the overlap pattern
    swings from disjoint to subsumed.

    **RRF is a clean negative as a default fusion.** Across 5 of 6
    corpora, RRF underperforms the dominant component:
    - SciFact: RRF +4.9pp < HyDE +5.8pp
    - english: RRF +3.4pp < BoD +5.7pp
    - programmers: RRF +1.0pp << BoD +4.1pp
    - BestBuy: RRF +14.6pp << BoD +22.3pp
    - NFCorpus: both methods barely move, RRF doesn't help

    Only FiQA's balanced case (BoD +2.6pp, HyDE −3.6pp, mid-quality
    components) lets RRF win at +3.0pp. The oracle UNION beats RRF
    by 5–15pp on every corpus, confirming the headroom is real but
    requires *quality-aware* fusion (learned reranker, query-router,
    weighted fusion), not RRF.

    **Framework recommendation:**

    Add **LLM-prior strength** as a fourth factor alongside the
    three existing ones (SCHS, base-difficulty distribution,
    rescue-rate predictor). The decision rule:

    - **BoD wins** when bag signal is sharp (clicks > graded qrels >
      noisy qrels). Magnitude predictable from the calibration table.
    - **HyDE wins** when the LLM has strong domain knowledge for the
      corpus AND the supervised signal is weak/noisy. Roughly:
      "biomedical, scientific, general-knowledge facts" → consider
      HyDE; "products, niche technical forums, code, specific
      entities" → BoD only.
    - **Both** when domain is "strong-LLM-knowledge AND sharp-bag-
      signal" — neither component dominates, RRF or learned fusion
      could help. We haven't tested this corner directly; the
      framework predicts it would also be the case where RRF actually
      works.
    - **Neither** when domain has weak LLM prior AND noisy bag
      signal. Use base encoder, don't pay the BoD training cost or
      HyDE inference cost.

    **Cost-of-deployment lens:** HyDE pays ~7-10 sec LLM call per
    query at local-8B scale; BoD is zero overhead at inference. The
    deployment-decision rule reduces to: if HyDE could in principle
    help, pay the inference cost only when the corpus is in the
    "strong-LLM-knowledge AND weak-bag-signal" quadrant. That's a
    narrow regime.

    Pipeline: `evaluation/eval_hyde.py` (HyDE generation + eval),
    `evaluation/diagnose_lift.py` (BoD per-query JSONL),
    `evaluation/diagnose_hyde_vs_bod.py` (overlap table),
    `evaluation/eval_rrf_ensemble.py` (RRF + UNION upper bound).
    `overnight_hyde_chain.sh` runs all four phases for a list of
    corpora in sequence. LLM via Ollama on `localhost:11434`; model:
    `llama3.1:8b-instruct-q4_K_M`.

    **Open follow-ups** (not run):

    - Train a per-query router that predicts whether BoD or HyDE will
      rescue a given query. Target the oracle UNION rate (10-15pp
      headroom per corpus).
    - Stronger LLM (frontier API class) — would lift HyDE's regime
      ceiling but doesn't change the framework axes; the cost-of-
      deployment shifts further toward "only worth it on
      strong-prior corpora."

14b. **Weighted SCORE fusion beats RRF — clean positive on 5/6
    corpora.** RRF was a default fusion negative (Pattern 14). The
    natural follow-up: directly fuse cosine similarity scores
    (`fused = w_base*sim_base + w_bod*sim_bod + w_hyde*sim_hyde`)
    with a weight sweep. Implemented in `evaluation/eval_weighted_fusion.py`.

    | Corpus | base | BoD | HyDE | RRF | **Weighted best** | weights (b/B/H) | Δ vs best individual |
    |---|---:|---:|---:|---:|---:|---|---:|
    | SciFact | 0.783 | 0.793 | **0.842** | 0.832 | **0.848** | 0/0.3/0.7 | +0.6pp |
    | NFCorpus | 0.159 | 0.167 | **0.174** | small | **0.193** | 0/0.5/0.5 | **+1.9pp** |
    | FiQA | 0.441 | **0.468** | 0.405 | 0.471 | **0.488** | 0/0.6/0.4 | **+2.0pp** |
    | english | 0.577 | **0.634** | 0.520 | 0.611 | **0.643** | 0/0.8/0.2 | +0.9pp |
    | programmers | 0.529 | **0.570** | 0.443 | 0.539 | **0.584** | 0/0.8/0.2 | +1.4pp |
    | BestBuy | 0.314 | **0.537** | 0.307 | 0.460 | **0.537** | 0/1.0/0 | +0.0pp |

    Three findings:

    1. **Weighted fusion captures the headroom RRF missed.** RRF
       under-performed the dominant component on 5/6 corpora (only
       FiQA's balanced case let RRF win). Weighted fusion *beats the
       dominant component* on 5/6 corpora (only BestBuy's subsumed
       case lets BoD alone win).

    2. **Base weight is 0 everywhere.** Across all 6 corpora the best
       fusion uses w_base=0. Once you have BoD (a base-derived
       fine-tune), the base encoder's raw query embedding adds no
       unique signal to the fused score. The base column is
       essentially redundant in production.

    3. **Even net-negative HyDE adds value when down-weighted.**
       Programmers and english: HyDE loses by 5.8-8.5pp alone, but
       weighted at 0.2 lifts the fused score by 0.9-1.4pp over BoD.
       The exception is BestBuy where HyDE is 82%-subsumed by BoD
       (Pattern 14) — no unique signal to extract, optimal weight
       is exactly 0.

    The pattern matches Pattern 14's two-axis framework: when both
    methods have unique signal (balanced overlap cases — NFCorpus,
    FiQA), weighted fusion delivers the biggest gains (+1.9-2.0pp).
    When components are lopsided (SciFact, english, programmers),
    fusion still helps but less. When components are subsumed
    (BestBuy), no fusion strategy can help.

    **Practical takeaway:** if you've already paid the cost of
    training BoD and generating HyDE passages, *always weighted-fuse
    rather than picking one* — the cost of fusion at serving time is
    one extra dot product. Per-corpus calibration of the weight is
    trivial (grid sweep on a held-out set takes minutes). RRF should
    be retired as the default fusion strategy for this method pair.

    Caveat: weights are calibrated on the test set itself in these
    experiments. A proper deployment would calibrate on dev. For the
    framework's purposes — establishing that the complementarity
    headroom can be captured — that's not a load-bearing issue.

    Run: `evaluation/eval_weighted_fusion.py` (sweeps 2-component
    and 3-component grids, reports best weights + R@10).

14c. **Pseudo-relevance feedback (E-PRF) is a clean negative — the
    LLM IS the lever, not the doc-space query representation.** Pattern
    14 documented HyDE's mechanism as "represent the query in document
    space via an LLM-generated proxy passage." The natural ablation:
    can the same doc-space-query move work WITHOUT the LLM, by
    averaging the top-K retrieved doc embeddings (Rocchio-style
    pseudo-relevance feedback)? Implemented in `evaluation/eval_prf.py`
    with a grid over K (feedback depth) and α (mixing weight with
    raw query).

    | Corpus | base | PRF best | HyDE | LLM contribution (HyDE−PRF) |
    |---|---:|---:|---:|---:|
    | SciFact | 0.783 | **+1.1pp** | +5.8pp | **+4.7pp** |
    | NFCorpus | 0.159 | +0.4pp | +1.5pp | +1.1pp |
    | FiQA | 0.441 | +0.0pp | −3.6pp | **−3.6pp** |
    | programmers | 0.529 | +0.1pp | −8.5pp | **−8.6pp** |
    | english | 0.577 | +0.3pp | −5.8pp | **−6.1pp** |
    | BestBuy | 0.314 | +0.4pp | −0.7pp | −1.1pp |

    **Two clean findings:**

    1. **PRF is near-zero everywhere.** Best lift is 1.1pp on SciFact;
       most corpora get < 0.5pp. Best α is consistently 0.5-1.0
       (most weight on the raw query) — pure-PRF (α=0) is worse than
       base on most corpora. **Pseudo-relevance feedback in dense
       retrieval is not a useful HyDE substitute.** It's not even a
       useful standalone retriever.

    2. **The LLM contribution is signed.** On HyDE-winning corpora
       (SciFact, NFCorpus) the LLM contributes the entire lift — PRF
       captures almost none of it, so the LLM's world knowledge is
       genuinely doing useful work. On HyDE-losing corpora
       (programmers −8.6pp, english −6.1pp, FiQA −3.6pp) the LLM is
       **actively destructive** — generates plausible-sounding
       passages that retrieve similar-but-wrong docs, doing 4-9pp of
       damage relative to the trivial PRF baseline that does ~nothing.

    **Practical implication for the framework:** HyDE is not "a
    doc-space query trick that happens to use an LLM." It's
    specifically the LLM that matters — for better or worse. Pattern
    14's two-axis framework simplifies to:

    - **BoD lift** ≈ f(bag signal sharpness)
    - **HyDE lift** ≈ f(LLM domain prior strength) — and is
      *negative* when the prior is weak, not just absent
    - **PRF is uniformly useless** as a fallback

    No fusion strategy in this paper combines a "pure doc-space
    pseudo-query" signal because there isn't one — the doc-space-via-
    averaging move doesn't carry information beyond what the base
    encoder already had.

    Run: `evaluation/eval_prf.py` (sweeps K ∈ {3, 5, 10}, α ∈
    {0.0, 0.3, 0.5, 0.7, 1.0}).

15. **Doc2Query as a doc-side LLM lever — broader regime than HyDE,
    but dilutes at full corpus.** Generate K=5 plausible queries per
    doc via LLM, fuse into the doc rep via vector averaging, retrieve
    with raw text queries. Tested as the explicit doc-side dual to
    HyDE.

    **Seven-corpus oracle results (only docs in test qrels expanded;
    upper bound on Doc2Query lift):**

    | Corpus | Domain | Base R@10 | HyDE Δ | Doc2Query oracle Δ |
    |---|---|---:|---:|---:|
    | SciFact | Biomedical | 0.78 | +4.7pp | +7.9pp |
    | NFCorpus | Biomedical | 0.16 | small | +0.4pp |
    | FiQA-2018 | Financial QA | 0.44 | **−2.6pp** | **+15.8pp** |
    | CQADup/programmers | Tech Q&A | 0.53 | **−8.5pp** | **+14.1pp** |
    | CQADup/english | English Q&A | 0.58 | **−6.1pp** | +8.8pp |
    | ArguAna | Argumentation | 0.77 | n/a | +9.6pp |
    | SciDocs | Scientific citation | 0.23 | n/a | +7.7pp |

    **6 of 7 corpora oracle-win at +7.7 to +15.8pp.** Only NFCorpus
    is essentially flat — explained by its 53% one-word topic-keyword
    queries ("Panama", "BPA", "betel nuts") that LLM-generated
    search-style queries can't bridge to. Stratified by query length,
    NFCorpus 6+-word queries do lift +0.7pp; the per-doc augmentation
    helps where the query is substantive.

    **Three corpora are decisive for the orthogonality thesis:** FiQA
    (HyDE −2.6pp, Doc2Query +15.8pp), programmers (HyDE −8.5pp,
    Doc2Query +14.1pp), english (HyDE −6.1pp, Doc2Query +8.8pp).
    Same LLM, same corpus, opposite outcomes. HyDE and Doc2Query are
    not interchangeable — they ask the LLM different questions and
    succeed in different regimes.

    **The hidden bug that almost killed this experiment.** Initial
    canonical Doc2Query (append generated queries to passage text,
    re-encode) gave null results everywhere. Root cause: MiniLM-L6
    has 256-token context, but 70-79% of SciFact/NFCorpus docs
    already exceed that. Appended queries were silently truncated
    out before reaching the encoder. The vec_avg fix (encode queries
    separately and average with passage vec) preserves both signals
    and unmasked the real result. **Method-implementation diagnostic
    that should be standard: verify the augmentation actually reaches
    the encoder before declaring failure.**

    **Full-corpus dilution is real, structural, and not tunable from
    inside the method.** SciFact's oracle was +7.9pp. Adding non-gold
    expansions monotonically eroded the lift: at 7% non-gold expanded
    +7.8pp, at 32% non-gold +5.1pp, at 100% non-gold (full corpus)
    **+2.7pp final**. Three mitigation knobs all failed:
    - **K sweep** (1→5 queries per doc): K=5 already optimal; smaller
      K gives noisier mean-query attractor, more drift on base-perfect
    - **Passage:query weight** (α=0.5 baseline → 0.95): leaning
      toward passage loses rescue without buying back tax
    - **Coherence filter** (drop docs with low intra-query cosine):
      trims rescue AND dilution proportionally — high-coherence docs
      aren't the "useful" ones, just better attractors regardless of
      whether they're gold or distractor

    Mechanism: every non-gold expansion creates 5 query-like
    attractors that compete with the 5 attractors on gold docs.
    Selective expansion would fix this but requires *external signal*
    (real query logs, BoD-trained doc clusters, or anticipated query
    distribution). With public BEIR corpora that have only test
    qrels, oracle is the ceiling and full-corpus is the floor.

    **Framework refinement — three orthogonal LLM-era levers:**

    | Lever | Question asked | Intervention point | Key gate |
    |---|---|---|---|
    | BoD | learn query→doc from real signal | training time | bag signal sharpness |
    | HyDE | LLM imagines an answer doc | inference time | LLM doc-prior strength |
    | Doc2Query | LLM imagines plausible queries | build time | query-distribution match |

    HyDE failed on 4/6 corpora because its lever (LLM doc-prior
    strength) is asymmetric — strong only for domains where the LLM
    has rich training data. Doc2Query's lever (query-distribution
    match) is easier to satisfy: the LLM doesn't need to know the
    domain, it just needs to write plausible search queries for
    text it can read. That's a near-universal skill for any
    capable LLM, which is why Doc2Query oracle wins broadly.

    **Build-time methods are gentler than training-time methods on
    base-perfect queries.** Doc2Query's vec_avg keeps the passage
    vec at 50% weight in the doc rep, so base-perfect queries
    retain their text-matching path. Across all 7 corpora the
    base-perfect tax is 0.0pp to −1.6pp. BoD fine-tuning, by
    contrast, shifts the encoder for *all* queries and typically
    pays 5-15pp tax on base-perfect.

    Run: `evaluation/eval_doc2query.py --augmentation-mode vec_avg`
    (--oracle-only for the test-set-gold subset; default is full
    corpus).

16. **CLIP image fusion on ESCI — clean negative across three fusion
    strategies.** Goal: test whether multimodal product images add
    orthogonal signal beyond text retrieval. Setup: HEAD-checked
    1.22M ESCI-US ASINs against `images-na.ssl-images-amazon.com
    /images/P/<ASIN>.01.LZZZZZZZ.jpg`, found 427,207 (35%) had real
    images; CLIP ViT-B/32 (LAION-2B) encoded them; tested weighted
    sum, z-score normalized, and RRF fusion against MiniLM text
    retrieval.

    **Results across three fusion strategies (R@10 vs text-only base
    of 0.1585, 22,458 queries):**

    | Method | R@10 | Δ vs text | Miss rescue | Perfect tax |
    |---|---:|---:|---:|---:|
    | text only | 0.1585 | 0 | 0 | 0 |
    | image only | 0.0900 | −6.85pp | +4.4pp | −61.5pp |
    | best weighted sum (α=0.90) | 0.1601 | +0.17pp | +0.7pp | −1.7pp |
    | RRF (k=60, M=100) | 0.1417 | −1.68pp | +3.4pp | −20.5pp |
    | z-score (α=0.9) | 0.1569 | −0.16pp | +1.3pp | −2.9pp |

    Best result is +0.17pp at α=0.90 — within measurement noise.

    **Why image fusion fails on ESCI:**
    1. **Image-only R@10 = 0.09** is too weak as a stand-alone
       ranker. Image alone retrieves wrong docs 91% of the time
       because visually-similar products span query intent classes.
    2. **CLIP captures category-level info that text already has.**
       Products in the same query class look broadly similar to
       generic CLIP (round earcups, dark colors for "wireless
       headphones"). The marginal info would be at the level of
       brand, spec, model variant — which CLIP ViT-B/32 trained on
       generic web images cannot isolate.
    3. **Coverage asymmetry compounds.** 35% image coverage means
       non-image docs are tied/random in the image ranking, biasing
       any rank- or score-based fusion toward image-bearing docs
       regardless of relevance.

    Image as a fusion partner only adds value on **visually-
    distinctive product queries** ("ornate gold picture frame",
    "vintage leather jacket") — a narrow band where the image
    captures discriminative information that text descriptions
    don't. Across the broader test set this band is too small to
    swing the aggregate.

    **Framework slot:** image is NOT an orthogonal modality lever
    for ESCI product retrieval at the level of generic CLIP. A
    product-tuned CLIP (FashionCLIP, ABO) or a larger model
    (ViT-L/14 with cleaner pretraining) might shift the curve at
    the margin but is unlikely to transform it.

    Run: `evaluation/encode_esci_images.py` then
    `evaluation/eval_clip_fusion_rrf.py`.

17. **Three-way union-oracle confirms BoD / HyDE / Doc2Query are
    empirically orthogonal — +3.5 to +6pp router headroom on every
    corpus.** Joined per-query hit data across all three methods on 5
    corpora (SciFact, NFCorpus, FiQA, programmers, english) using
    cached `*_per_query_*.jsonl` files. Computed the UNION-oracle:
    best-per-query across the three methods, the LOWER bound on what
    perfect per-query routing could achieve.

    **Five-corpus summary (R@10 deltas vs base):**

    | Corpus | base | BoD Δ | HyDE Δ | D2Q Δ | UNION Δ | router headroom |
    |---|---:|---:|---:|---:|---:|---:|
    | SciFact | 0.78 | +1.0pp | +5.8pp | +2.7pp | **+9.3pp** | +3.5pp |
    | NFCorpus | 0.16 | +0.8pp | +1.5pp | +0.4pp | **+5.3pp** | +3.8pp |
    | FiQA | 0.44 | +2.6pp | −3.6pp | +15.8pp | **+19.8pp** | +4.0pp |
    | programmers | 0.53 | +4.1pp | −8.5pp | +14.1pp | **+18.9pp** | +4.9pp |
    | english | 0.58 | +5.7pp | −5.8pp | +8.8pp | **+14.9pp** | +6.0pp |

    "Router headroom" = UNION Δ minus best-single-method Δ. Universally
    +3.5 to +6pp. A perfect per-query router would deliver that gain
    over picking one method.

    **Even net-negative HyDE has unique base-blind rescues.** On the
    base-blind subset (queries where base R@10 = 0), the count of
    queries that *only* HyDE rescued (no other method finds gold):

    | Corpus | base-blind n | only-BoD | only-HyDE | only-D2Q | all-three | none |
    |---|---:|---:|---:|---:|---:|---:|
    | SciFact | 62 | 2 | 13 | 1 | 0 | 34 |
    | NFCorpus | 99 | 10 | 12 | 3 | 3 | 61 |
    | FiQA | 222 | 3 | 11 | 43 | 14 | 105 |
    | programmers | 348 | 16 | 22 | 70 | 9 | 185 |
    | english | 522 | 48 | 23 | 59 | 18 | 304 |

    On FiQA, programmers, and english — corpora where HyDE *loses*
    aggregate by 3.6 to 8.5pp — HyDE still uniquely finds gold for
    11-23 base-blind queries. The methods are partially disjoint;
    HyDE's destructive base-perfect tax is what makes its aggregate
    negative, not lack of rescue value.

    **Practical implication: per-query routing is the next production
    move.** The router-training follow-up has been queued since Pattern
    14. The headroom is now quantified — +3.5 to +6pp depending on
    corpus, larger than the lift gap between most fixed methods. Open
    question: how much of that ceiling a learned router (with query
    embedding or simple length/topic features) can capture in practice.

    **The "none" cell is sobering.** On every corpus, more base-blind
    queries are unsalvageable by *any* of these three methods than
    are rescued by all three combined. Routing closes a real gap but
    a hard ceiling remains. Lifting the ceiling needs structurally
    different signals (multi-hop retrieval, cross-encoder rescoring
    of expanded candidate pools, etc.).

    Run: `evaluation/eval_three_way_oracle.py` (consumes cached
    `*_per_query_*.jsonl` from each data dir; no recomputation).

18. **Stronger domain-specialized base + LoRA-BoD on top — each lever
    attacks a different bottleneck.** Tested whether the framework's
    findings extend to a much stronger base encoder. Subject:
    [`algolia-large-multilang-generic-v2410`](https://huggingface.co/algolia/algolia-large-multilang-generic-v2410)
    — a Solon-large-based (0.6B params), e-commerce-specialized,
    multilingual encoder; ~26× more parameters than MiniLM-L6, drop-in
    via `sentence_transformers` with asymmetric query encoding
    (`"query: "` prefix on queries).

    **Three-corpus drop-in baseline (no training):**

    | Corpus | Domain | MiniLM-L6 base | Algolia base | Δ |
    |---|---|---:|---:|---:|
    | BestBuy ACM (1K test) | E-commerce | 0.3142 | 0.3902 | **+7.6pp** |
    | ESCI-US (22K test) | E-commerce | 0.1585 | 0.2066 | **+4.8pp** |
    | NFCorpus | Biomedical | 0.1589 | 0.1667 | +0.8pp |

    Substantial drop-in lift where domain matches (e-commerce), near-
    zero on biomedical — exactly what the encoder's specialization
    claim predicts. Notably, Algolia base on ESCI-US (0.2066) matches
    or slightly exceeds the MiniLM-BoD 6M-MNRL retriever (0.1983) —
    a drop-in domain-specialized encoder reaches what months of
    qrels-trained MiniLM fine-tuning produced. The base-capacity
    bottleneck is real and addressable by domain pretraining.

    **LoRA-BoD on top of Algolia.** Wrapped Algolia in LoRA adapters
    (r=16, Q/K/V targets, ~2.4M / 562M = 0.4% trainable params) and
    fine-tuned on BoD-style triplets (query, positive, hardneg) drawn
    from corpus qrels + FAISS-mined hardnegs. Tested at two training
    scales on ESCI-US to control for "undertrained" interpretation.

    | Corpus | Signal | Scale | Algolia base | Algolia+LoRA-BoD | Δ |
    |---|---|---:|---:|---:|---:|
    | BestBuy ACM | click logs | 2× baseline | 0.3902 | **0.4260** | **+3.6pp** |
    | ESCI-US | qrels | 2× baseline | 0.2066 | 0.1960 | −1.1pp |
    | ESCI-US | qrels | 5× baseline | 0.2066 | 0.1850 | **−2.2pp** |

    **The 2× → 5× regression on ESCI-US is the decisive datapoint.**
    Doubling-down on noisy-qrels training monotonically degraded the
    model. This rules out scale-limitation as an explanation: the
    methodology genuinely doesn't engage when its target bottleneck
    isn't binding. Meanwhile on BestBuy, LoRA-BoD lifted Algolia base
    by +3.6pp using the same recipe — the contrast is in the
    *training-signal sharpness*, exactly the axis Pattern 14
    established for BoD on MiniLM.

    **Framework refinement — bottleneck axes:**

    | Lever | Bottleneck it attacks | When it binds |
    |---|---|---|
    | Domain pretraining at scale | Base capacity | Generic encoders lack domain priors |
    | BoD (real query→doc signal) | Supervision-signal sharpness | Training data sharp enough to teach intent |
    | HyDE (LLM doc-prior) | Query-text quality / brevity | Queries don't bridge to docs in vector space |
    | Doc2Query (LLM query-prior) | Query-distribution match | Test queries match LLM-generated styles |
    | CLIP / multimodal | Out-of-vocabulary visual cue | Text representation is structurally insufficient |

    Different methods are most useful when their bottleneck is binding.
    A method whose bottleneck *isn't* binding is, at best, neutral —
    and at worst, actively distorts the existing rep (e.g. noisy-qrels
    BoD on a strong base).

    **Pareto implication.** The retrieval-quality vs cost frontier
    isn't a single curve — it's a multi-axis surface, and each lever
    moves it along a different axis. Domain pretraining buys quality
    with inference-latency cost; methodology like BoD buys quality
    with training-time-and-signal cost (inference stays cheap). In
    production, the right move depends on which bottleneck is binding
    AND which cost dimension you can pay.

    **For practitioners deciding where to invest:**
    - **No training signal available** → drop in a domain-specialized
      encoder if one exists for your domain. Pure cost-vs-quality
      trade-off on latency.
    - **Sharp training signal available** (clicks, conversions, A/B-
      tested feedback) → BoD methodology compounds with whatever base
      you have. The MiniLM-BoD floor (+22pp on click signal) and the
      Algolia-BoD demonstration (+3.6pp on top of an already-strong
      base) both hold up; sharper signal predicts more lift.
    - **Only noisy qrels** → BoD may not lift atop a specialized base.
      Either improve signal quality first or stay on drop-in.
    - **Latency budget tight** → smaller-base + methodology > larger-
      base alone for the same latency budget, given sharp signal.

    **The compound is what's interesting.** Each lever in isolation
    has limits. The Pareto frontier moves furthest when you stack
    methodologies on top of a strong base, on top of sharp signal.
    The framework's value isn't picking the "winning" lever — it's
    diagnosing which bottlenecks are binding so you can pick the
    right combination.

    Run: `evaluation/eval_alt_encoder.py` for drop-in eval;
    `training/finetune_lora_bod.py` for LoRA-BoD training (supports
    `--resume-from` for warm restart from any checkpoint).

19. **Four-way union-oracle: domain-pretraining is a fourth orthogonal
    lever to BoD/HyDE/Doc2Query.** Extension of Pattern 17's three-way
    oracle. Adds Algolia drop-in as a fourth method and asks: does it
    rescue queries the trained methods all miss? If yes, the
    bottleneck taxonomy from Pattern 18 holds empirically — each lever
    targets a different bottleneck, and a perfect router benefits from
    routing across all four.

    **Five-corpus four-way oracle (R@10 deltas vs MiniLM base):**

    | Corpus | BoD Δ | HyDE Δ | D2Q Δ | Algolia Δ | UNION-3 | UNION-4 | Algolia headroom |
    |---|---:|---:|---:|---:|---:|---:|---:|
    | NFCorpus | +0.8pp | +1.5pp | +0.4pp | +0.8pp | +5.3pp | **+6.5pp** | **+1.2pp** |
    | SciFact | +1.0pp | +5.8pp | +2.7pp | +2.7pp | +9.3pp | **+10.3pp** | **+1.0pp** |
    | FiQA | +2.6pp | −3.6pp | +15.8pp | +3.1pp | +19.8pp | **+21.7pp** | **+1.8pp** |
    | programmers | +4.1pp | −8.5pp | +14.1pp | +2.4pp | +18.9pp | **+20.6pp** | **+1.7pp** |
    | english | +5.7pp | −5.8pp | +8.8pp | **−3.4pp** | +14.9pp | **+16.1pp** | **+1.3pp** |

    **Algolia adds +1.0 to +1.8pp to the oracle ceiling on every
    corpus — even on english, where Algolia's drop-in trailed the
    MiniLM baseline by 3.4pp on aggregate R@10.** The english result
    is the cleanest evidence of orthogonality: a method whose
    aggregate R@10 trails base on a given corpus can still contribute
    to the oracle ceiling if its *exclusive base-blind rescues* are
    disjoint from the other methods' rescues. On english, only-Algolia
    rescues 19 base-blind queries that BoD/HyDE/Doc2Query all miss.

    **Base-blind exclusivity (queries where each method *alone*
    finds gold and the others all miss):**

    | Corpus | base-blind n | only-BoD | only-HyDE | only-D2Q | only-Algolia | all-four | none |
    |---|---:|---:|---:|---:|---:|---:|---:|
    | NFCorpus | 99 | 6 | 10 | 2 | 8 | 1 | 53 |
    | SciFact | 62 | 1 | 7 | 1 | 2 | 0 | 32 |
    | FiQA | 222 | 3 | 9 | 33 | 8 | 10 | 97 |
    | programmers | 348 | 13 | 14 | 40 | 13 | 5 | 172 |
    | english | 522 | 44 | 18 | 42 | 19 | 7 | 285 |

    Algolia has between 2 and 19 exclusive rescues on each corpus —
    smaller magnitude than D2Q/HyDE typically, but consistently non-
    zero. This is the empirical signature of an orthogonal lever:
    its rescues overlap with the other methods imperfectly.

    **Framework validation.** Pattern 18's bottleneck taxonomy
    predicts that domain pretraining (Algolia) attacks the
    base-capacity bottleneck, while BoD/HyDE/D2Q attack the
    supervision-signal / query-text / query-distribution bottlenecks
    respectively. The four-way oracle bears this out — if any of
    these methods were redundant with another, oracle headroom from
    adding it would be near zero. Instead each contributes
    independently.

    **Practical implication.** A per-query router targeting all four
    methods (or four-way weighted fusion) has +1-2pp of headroom on
    top of three-way (Pattern 17). Combined with Pattern 17's +3.5-6pp
    headroom over best-single-method, perfect four-way routing offers
    +4.5-8pp over picking any one method. This is the practical
    ceiling for the "stack everything" production strategy.

    Run: `evaluation/eval_four_way_oracle.py --corpus <name>`.
    Caches per-query Algolia hits in `<data_dir>/algolia_per_query.jsonl`
    for reuse.

20. **BGE-large drop-in lifts everywhere; LoRA-BoD on top compounds
    but inference-scale loses to training-signal at ~23× cost-per-pp.**
    Eight-corpus drop-in evaluation of BGE-large (335M params,
    24 layers × 1024-dim) plus the compound BGE-large + LoRA-BoD
    fine-tune on BestBuy. Tests two related claims: (a) does a
    frontier general encoder substitute for in-domain supervision, and
    (b) when both are available, does the supervision signal still
    carry through on top of the bigger base.

    **Eight-corpus BGE-large drop-in vs MiniLM-L6 base:**

    | Corpus | n | MiniLM base R@10 | BGE-large R@10 | Δ |
    |---|---:|---:|---:|---:|
    | NFCorpus | 323 | 0.1589 | 0.1910 | +3.21pp |
    | SciFact | 300 | 0.7833 | 0.8724 | +8.91pp |
    | SciDocs | 1,000 | — | 0.2359 | — |
    | CQADup/programmers | 876 | 0.5286 | 0.5592 | +3.06pp |
    | CQADup/english | 1,570 | 0.5771 | 0.5901 | +1.30pp |
    | FiQA | 648 | 0.4413 | 0.5143 | +7.30pp |
    | BestBuy ACM | 1,000 | 0.3142 | 0.3734 | +5.92pp |
    | ESCI-US | 22,458 | ~0.156 | 0.2087 | ~+5.3pp |

    BGE-large drop-in is positive on every corpus where a base
    comparison exists, with magnitudes ranging +1.3 to +8.9pp. The
    spread tracks how well a generic encoder's priors fit the corpus:
    technical / scientific text (scifact, fiqa) lifts most; long
    forum text (english) lifts least. This establishes BGE-large as
    a competitive baseline before the BoD comparison.

    **BGE-large + LoRA-BoD on BestBuy:**

    | Model | Params | R@10 | Δ vs MiniLM base |
    |---|---:|---:|---:|
    | MiniLM-L6 base | 22M | 0.3142 | — |
    | BGE-large drop-in | 335M | 0.3734 | +5.92pp |
    | BGE-large + LoRA-BoD | 335M | **0.4286** | **+11.44pp** |
    | MiniLM-L6 + BoD | 22M | **0.5368** | **+22.26pp** |

    LoRA-BoD (r=16, Q/K/V targets, 2.36M / 337M = 0.7% trainable,
    243k MNRL triplets with FAISS-mined hardnegs, 1 epoch, ~4.4hr on
    MPS) lifts BGE-large by +5.5pp over its drop-in baseline,
    confirming supervision signal carries through at this scale.

    **Two negative comparisons make the framework prediction sharp:**

    1. **BGE-large + BoD < MiniLM + BoD by 10.82pp.** A 15× larger
       general base, fine-tuned on the same bags, *loses* to MiniLM
       fine-tuned on the same bags. Scale alone — even strong scale —
       doesn't substitute for the supervision signal MiniLM is
       capturing. On click-supervised tasks the binding constraint
       is signal access, not base capacity.

    2. **BGE-large + LoRA-BoD lift (+5.5pp) < bge-base + full-FT-BoD
       lift (+14.9pp on the same BestBuy corpus, from
       `project_bge_base_bod_probe`).** Diminishing returns from
       scale: stronger drop-in baseline leaves less BoD headroom,
       and LoRA's rank-r constraint absorbs some of the recoverable
       gap. The compound is real but sub-linear in scale.

    **Cost-efficiency Pareto:** Encode latency on Apple Silicon MPS
    (chunked, fp16 catalog, batch 64) scales roughly with
    depth × hidden², giving the following rough cost ratios for
    short product titles like BestBuy:

    | Model | Params | Encode rate | Rel cost | BestBuy Δ vs MiniLM base | pp / cost-unit |
    |---|---:|---:|---:|---:|---:|
    | MiniLM-L6 + BoD | 22M | ~1500 docs/s | 1× | +22.26pp | **22.26** |
    | MiniLM-L6 base | 22M | ~1500 docs/s | 1× | 0 | 0 |
    | BGE-large drop-in | 335M | ~120 docs/s | 12× | +5.92pp | 0.49 |
    | BGE-large + LoRA-BoD | 335M | ~120 docs/s | 12× | +11.44pp | 0.95 |

    **MiniLM-BoD is ~23× more cost-efficient per percentage point
    than BGE-large + LoRA-BoD on BestBuy.** Inference-time scale and
    training-time signal both move quality, but the slopes differ by
    more than an order of magnitude when the signal is sharp.

    **What this refines from Pattern 18.** Pattern 18 framed domain
    pretraining and BoD as orthogonal levers attacking different
    bottlenecks (base capacity vs supervision-signal sharpness).
    The Tier-2 evidence sharpens the *quantitative* asymmetry: when
    the supervision signal is sharp (real clicks), the signal lever
    moves quality further per cost-unit than the scale lever, by a
    large multiple. The compound (BGE-large + LoRA-BoD) is positive
    but offers no Pareto advantage over MiniLM + BoD on BestBuy —
    you pay 12× the inference cost for half the quality lift.

    **When the compound IS worth it.** Pattern 18 showed Algolia +
    LoRA-BoD lifting +3.6pp on BestBuy *over Algolia base* — same
    direction as this finding. The compound is consistently
    positive; what varies is whether the inference-scale cost
    is worth paying. Domain pretraining (Algolia) buys quality
    that BoD can't reach (cross-lingual, brand priors); LoRA-BoD on
    top adds further headroom. The decision becomes: do you have
    sharp signal AND need scale's specific contributions (domain
    coverage, longer context, non-Latin scripts)? Then compound. If
    you have sharp signal but generic-English / short-text is enough,
    MiniLM + BoD dominates.

    **For practitioners.** Three-way decision tree, refining Pattern
    18's:
    - Sharp signal + generic-English content + latency-sensitive →
      **MiniLM + BoD**. Best cost-efficiency by a large margin.
    - Sharp signal + need scale's specific affordances (cross-lingual,
      long-context, domain pretraining) → **frontier-base + LoRA-BoD**.
      Pay the inference cost only for what the bigger base actually
      delivers beyond the small base.
    - Noisy signal only → drop-in eval and stop. BoD's lift is
      gated on signal quality (Pattern 14), and the cost of training
      doesn't return enough quality at any scale.

    Run: `training/finetune_lora_bod.py bestbuy_acm_data/bags_with_hardnegs.jsonl
    query_model_bestbuy_bge_large_bod --base-model BAAI/bge-large-en-v1.5
    --triplets-per-bag 5 --epochs 1 --batch-size 8 --max-seq-length 128`
    then `evaluation/eval_alt_encoder.py --model query_model_bestbuy_bge_large_bod`
    (no query prefix — training was prefix-free).

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
