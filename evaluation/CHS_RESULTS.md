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

   | Corpus | base R@10 | base-blind | rescue | spec tax | **overall Δ** | SCHS |
   |---|---:|---:|---:|---:|---:|---:|
   | BestBuy ACM | 0.556 | 44% | **+24.9pp** | −6.4 | **+17.5pp** | 0.525 |
   | ESCI-Spanish | 0.074 | 67% | **+15.1pp** | −12.9 | **+13.2pp** | 0.45 |
   | ESCI-US (E-only) | 0.215 | 34% | **+6.1pp** | −10.4 | **+3.0pp** | 0.54 |
   | NFCorpus | 0.159 | 31% | **+4.2pp** | −17.9 | **+0.8pp** | 0.38 |

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
