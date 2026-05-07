# Bag-of-Documents Product Search

An implementation of the [bag-of-documents](https://dtunkelang.medium.com/modeling-queries-as-bags-of-documents-b7d79d0916ab) model for e-commerce product search, evaluated against ESCI on a 1.2M Amazon product catalog. Two BoD-trained reranker encoders fused on top of BM25 candidates produce the current shipped state of the art.

- **Blog post**: [Distilling Retrieval Pipelines to a Single Embedding Model](https://dtunkelang.medium.com/distilling-retrieval-pipelines-to-a-single-embedding-model-606f3ecf0c91)
- **Live demo**: [huggingface.co/spaces/dtunkelang/bag-of-documents-demo](https://huggingface.co/spaces/dtunkelang/bag-of-documents-demo)
- **Model and data**: [huggingface.co/datasets/dtunkelang/bag-of-documents](https://huggingface.co/datasets/dtunkelang/bag-of-documents)

## Headline Result

| | R@10 | nDCG@10 | E@1 | E@3 | latency |
|---|---|---|---|---|---|
| Base MiniLM (dense retrieval only) | 15.60% | 0.2648 | 31.50% | 28.52% | — |
| RRF(BM25, base) (non-BoD hybrid retrieval) | 18.62% | 0.3048 | 31.54% | 31.98% | — |
| BM25 alone (bm25s, k1=0.3, b=0.6) | 20.33% | 0.3451 | 40.06% | 36.87% | — |
| BM25 + 3-way ensemble rerank (no spell-correct) | 21.61% | 0.3660 | 42.11% | 39.22% | ~50ms |
| **BM25 + 3-way ensemble rerank + spell-correct (fast SOTA)** | **21.84%** | **0.3698** | **42.53%** | **39.60%** | **~50ms** |
| BM25 top-100 + 3-way + LiYuan CE fusion (medium quality) | 22.33% | 0.3842 | 44.85% | 41.61% | ~400ms-1s MPS / 2-6s CPU |
| BM25 top-50 + sumsim + BGE (bridge tier, no LiYuan) | 23.10% | 0.3979 | 46.89% | 43.10% | ~0.5s MPS / 2.5s CPU |
| **BM25 top-100 + sumsim + LiYuan + BGE 3-way weighted fusion (quality SOTA)** | **23.57%** | **0.4055** | **47.95%** | **43.90%** | **~2.6s MPS / 5-15s CPU** |

22,458-query ESCI test set, R@10 with E+S as relevant, nDCG@10 with E=1.0 / S=0.1 gain. Three discrete SOTA tiers ship — each ~10× the latency of the previous, each adding ~+0.6pp R@10 / ~+2pp E@1.

**Fast SOTA** — pre-BM25 catalog-vocab spell correction → bm25s candidates → three BoD-trained MiniLM encoders → mean-cosine fusion. Sub-100ms wall-clock; +1.51pp R@10 over BM25 alone, +6.24pp over base MiniLM. Spell correction adds **+0.23pp R@10 / +0.42pp E@1** (significant via bootstrap); on the 5.4% of queries that actually get corrected, the lift is **+4.25pp R@10 / +7.82pp E@1**.

**Medium tier** — fast SOTA candidates + the LiYuan ESCI cross-encoder (RoBERTa-base, ESCI-supervised). LiYuan scores and 3-way sumsim are per-query min-max normalized and blended at `w_ce=0.25`. +0.49pp R@10 / +2.32pp E@1 over fast SOTA, at ~10× the latency.

**Quality SOTA** — three-way weighted fusion of (sumsim, LiYuan, [BGE-reranker-v2-m3](https://huggingface.co/BAAI/bge-reranker-v2-m3)) over BM25 top-100, with weights (0.4, 0.2, 0.4). Found via 0.1-grid sweep over all weight tuples summing to 1; this combination strictly Pareto-dominates the original uniform 1/3-1/3-1/3 mean by +0.24pp R@10 / +0.13pp E@1 at zero added latency. BGE-reranker is XLM-RoBERTa-large (~568M params, BEIR-tested), much stronger than LiYuan; LiYuan stays in the fusion because it specializes on ESCI labels and contributes orthogonal top-1 signal even when down-weighted. **+1.24pp R@10 / +3.10pp E@1** over the medium tier; the BGE-reranker addition was the biggest single-experiment lift in the project.

The non-BoD hybrid baseline (RRF) is included to keep the comparison honest. It actually *underperforms* BM25 alone (the base FAISS lane displaces BM25's exact-match top-1 with semantically-similar near-misses; E@1 drops from 40.06% to 31.54%), confirming that on entity-anchored product catalogs, lexical retrieval dominates dense.

The retriever uses [bm25s](https://github.com/xhluca/bm25s) (numpy-backed BM25 with configurable k1/b) at **k1=0.3, b=0.6** — swept against ESCI to optimize for keyword-heavy short Amazon product titles.

The deployable architecture:

0. **Spell correction**: each out-of-catalog-vocab query token is matched against the title vocabulary (~172K tokens) at edit distance ≤ 2 via [pyspellchecker](https://github.com/barrust/pyspell-checker), preserving brand names and model numbers. Sub-millisecond per query.
1. **Retrieval**: bm25s (k1=0.3, b=0.6, Snowball english stemmer + en stopwords) returns top-50 (fast tier) or top-100 (medium / quality tiers) candidates.
2. **Bi-encoder rerank** (all tiers): three BoD-trained encoders score candidates — two trained on bag-derived signals (`query_model_us_full_6m_mnrl`, `query_model_us_qrels_mnrl_hardneg`), one trained on ESCI labels directly (`query_model_us_esci_supervised`). Mean cosine over the three is the "sumsim" stream.
3. **Cross-encoder fusion** (medium + quality tiers):
   - *Medium*: LiYuan CE only; fused with sumsim at `w_ce=0.25`.
   - *Quality*: LiYuan + BGE-reranker-v2-m3 both score, then weighted fusion (sumsim 0.4, LiYuan 0.2, BGE 0.4) of per-query min-max normalized streams.
4. **Output**: top-10 by fused score.

Latencies (per query):
- Fast: ~40-100ms wall-clock; three small forward passes against precomputed product embeddings.
- Medium: + 100 LiYuan (RoBERTa-base) forward passes; ~400ms-1s on MPS/GPU, 2-6s on CPU.
- Quality: + 100 BGE-reranker (XLM-RoBERTa-large) forward passes; ~2.6s on MPS/GPU, 5-15s on CPU. BGE is ~6× slower than LiYuan but adds orthogonal signal worth the cost.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Download product catalog (ESCI 1.2M subset)
python download/download_catalog.py

# Build indexes (FAISS + tantivy)
python indexing/build_index.py

# Compute bags from ESCI queries
python training/compute_bags.py queries.jsonl bags.jsonl --ce-rerank models/esci-cross-encoder

# Fine-tune the query model (use --loss mnrl for BoD reranker training)
python training/finetune_query_model.py bags.jsonl query_model/ --loss mnrl

# Run the demo
python demo.py
```

Or use `run_pipeline.sh` to run the full pipeline end-to-end.

## What is the bag-of-documents model?

A search query is represented as the distribution of its relevant products in embedding space:

- **Centroid** = what the query means (direction in product space)
- **Specificity** = how broad or narrow the query is (spread of the distribution)

A query encoder is then trained on (query text → bag) pairs. The original framing was BoD-as-retriever: train the encoder via cosine-to-centroid distillation, use the resulting model directly with FAISS. The current shipped architecture is BoD-as-reranker: train the encoder via MultipleNegativesRanking on bag members, use it (plus a second encoder trained on hard-negative qrels-derived bags) to rerank lexical retrieval candidates.

**The bags themselves are constructed by:** hybrid retrieval (tantivy keyword + FAISS embedding) → cross-encoder scoring against the query → threshold ≥ 0.3 → top-50 candidates as bag members.

## Architecture progression (Apr–May 2026)

Per-query measurements on the full ESCI test set (22,458 queries with at
least one E or S judgment, against the 1.2M-product ESCI index, K_eval = 10):

| # | Pipeline | R@10 | nDCG@10 | E@1 | E@3 |
|---|---|---|---|---|---|
| A | Base MiniLM | 15.60% | 0.2648 | 31.50% | 28.52% |
| B | 6M-MNRL retriever (BoD) | 18.10% | 0.3090 | 36.16% | 33.25% |
| Z | RRF(BM25, base) retrieval (non-BoD hybrid baseline) | 18.62% | 0.3048 | 31.54% | 31.98% |
| C | Base + ensemble rerank | 19.00% | 0.3238 | 37.81% | 34.92% |
| E | 6M-MNRL + ensemble rerank | 19.83% | 0.3375 | 39.13% | 36.12% |
| H | BM25 alone (bm25s, k1=0.3, b=0.6) | 20.33% | 0.3451 | 40.06% | 36.87% |
| K | BM25 + 2-way ensemble rerank | 21.27% | 0.3588 | 41.12% | 38.27% |
| CC3-50 | BM25 top-50 + 3-way ensemble rerank (ESCI-supervised rerank_G added) | 21.61% | 0.3660 | 42.11% | 39.22% |
| CC3-50 + spell | + catalog-vocab spell correction (fast SOTA) | 21.84% | 0.3698 | 42.53% | 39.60% |
| CC4-100 | + LiYuan CE @ w=0.25 (medium tier) | 22.33% | 0.3842 | 44.85% | 41.61% |
| CC5_no_liyuan_K50 | sumsim + BGE @ 0.5/0.5 over BM25 top-50 (bridge tier) | 23.10% | 0.3979 | 46.89% | 43.10% |
| **CC5-100 (quality SOTA)** | **+ LiYuan @ 0.2: weighted 3-way fusion (sumsim 0.4, LiYuan 0.2, BGE 0.4)** | **23.57%** | **0.4055** | **47.95%** | **43.90%** |

**Read-outs:**

- **The cosine-distilled BoD-as-retriever loses to base on this benchmark.** The original release was measured on a 75K-query construction-set eval; on the canonical 22,458-query ESCI test set, R@10 drops below A. MNRL-trained BoD (B) reverses that — +2.50pp R@10 over base, +4.66pp E@1.
- **Ensemble rerank stacks on top of any retriever.** The two BoD-trained rerankers (6M-MNRL, qrels-hardneg) lift base by +3.40pp R@10 (A→C) and 6M-MNRL by +1.73pp R@10 (B→E).
- **BM25 alone beats the dense rerank stack.** Setup H (bm25s with k1=0.3, b=0.6 — tuned for short keyword-stuffed product titles) hits R@10 20.33% — *above* setup E's 19.83% (6M-MNRL + ensemble rerank). On entity-heavy product queries lexical matching does most of the work; tuning BM25 hyperparameters away from Lucene defaults (which assume long natural-language documents) is what makes lexical retrieval beat a tuned dense rerank stack.
- **Pure non-BoD hybrid retrieval (Z) underperforms BM25 alone.** Setup Z (RRF(BM25, base) retrieval, no rerank) scores R@10 18.62% — *worse* than BM25 alone (H, 20.33%) and dramatically worse on E@1 (31.54% vs 40.06%). Mixing dense candidates into the lexical lane displaces BM25's exact-match top-1 with semantically-similar near-misses. This is the strongest "no BoD" baseline available out of the box, and it loses to BM25 alone — let alone to the BoD rerank stack.
- **The third reranker (rerank_G, ESCI-supervised) is real but small.** K (2-way) → CC3-50 (3-way) is +0.34pp R@10 / +0.99pp E@1. The third encoder is trained on ESCI labels directly (E as positives, I as hard negatives) instead of on bag-derived signals, contributing orthogonal signal.
- **Spell correction was the first query-side win** (CC3-50 → CC3-50 + spell, +0.23pp R@10). Five other query-rewriting probes (PRF, LLM rewriting via Qwen3-4B-4bit, composite waterfall, phrase BM25, phonetic-fallback spell) all *hurt*. Spell correction wins because it's high-precision, low-coverage (5.4% of queries get touched, mostly real typos); the others transformed too many queries and broke correctly-spelled ones.
- **Cross-encoder fusion is the biggest single lift.** CC3-50+spell → CC4-100 (medium) adds +0.49pp R@10 / +2.32pp E@1 by fusing the LiYuan ESCI cross-encoder. CC4-100 → CC5-100 (quality) adds another +1.00pp R@10 / +2.95pp E@1 by adding BGE-reranker-v2-m3 to the fusion (XLM-RoBERTa-large vs LiYuan's RoBERTa-base). All deltas statistically significant (1000-resample paired bootstrap, 95% CIs exclude 0).
- **CE distillation back into a bi-encoder didn't work** at 50K or 375K MarginMSE triplets. The student matches relative margins on training queries but its embedding space doesn't generalize to retrieval ranking; net effect on R@10 was within noise of CC3-100. Production-scale distillation (5M+ pairs) might unlock it but wasn't pursued.
- **Encoder capacity is not the bottleneck for the bi-encoder reranker.** A BGE-base bi-encoder (5× MiniLM params) trained on the same bags via MNRL or cosine-to-centroid loss underperforms MiniLM at every learning rate. Encoder capacity helps only at the cross-encoder stage, where BGE-reranker-v2-m3 (568M params) beats LiYuan (125M) cleanly.

### Where the lift comes from (per-query-bin breakdown)

Binning the 22,458 queries by base MiniLM's per-query R@10 surfaces
the shape of the +6.01pp aggregate (CC3-50 - base) lift. A third of
the queries have zero base-FAISS recall — products dense retrieval
can't see at all. BM25 + ensemble rerank rescues those queries from
zero and contributes roughly half the aggregate lift. On the easiest
5% of queries (where dense already does well), the BM25-anchored
candidates are mildly *worse* than dense — but E@1 is preserved across
every bin. **The architecture wins by rescuing the hard regime, not
by being uniformly better.** Diagnostic: `evaluation/eval_per_query_bins.py`.

### What didn't work (negatives worth recording)

These probes were tried and rejected. Each closes a direction; collectively they shape what the next investment should *not* be.

**Bi-encoder side:**
- **Score-fusion of dense retrievers (F, G, J, AA).** RRF-fusing base + 6M-MNRL retrieval (or base + BM25 + MNRL) contributes nothing additive; the dense lanes share failure modes on entity queries.
- **A categorically stronger bi-encoder base (Y/DD setups, BAAI/bge-base-en-v1.5).** Trained on the same bags via MNRL (Y, lr=2e-5 and lr=1e-4) and via cosine-to-centroid (DD). All worse than MiniLM at every LR. DD1 (BGE+cos alone) scored R@10 16.61% — the worst single-reranker we ever measured. **Encoder capacity is not the bi-encoder bottleneck.** It is, however, the *cross-encoder* bottleneck: BGE-reranker-v2-m3 (568M params) beats LiYuan (125M) by +1.00pp R@10 in the quality SOTA.
- **Supervision-widening probes (EE / FF / GG / HH / II).** rerank_I (E∪S positives) and rerank_J (I∪C negatives) both lose to rerank_G as singles (EE1 19.24%, GG1 19.46% vs CC1 19.68%) and dilute existing ensembles. Adding ESCI label resolution (S, C) introduces intra-class noise MNRL penalizes.
- **CE distillation into MiniLM bi-encoder** at 50K and 375K MarginMSE triplets. Student matches CE relative margins on training queries but its embedding space doesn't generalize to retrieval ranking. Production-scale (5M+ pairs) might unlock it; not pursued.

**Retriever / query side:**
- **K_retrieve sweeps (N40 / N50 / N60 / N75 / N200).** Peak around K=40-50 for the 2-way (CC3-50), K=100 for CC5 (with CE), but the curves are flat within ±0.05pp.
- **Alternative bi-encoder fusion functions.** Sumrank (M, -0.53pp), max (-1.20pp), min (-0.73pp), weighted variants (W40 / W60 / W70). 1/3-1/3-1/3 sumsim averaging is optimal.
- **Default tantivy tokenizer (-0.80pp vs en_stem).** Stemming is doing real work.
- **Per-query routing on cheap features.** Oracle headroom is +1.76pp R@10 / +6.95pp E@1 over K, but A-wins distribute uniformly across base-recall bins. Token-count, has-digit, capitalization, base-FAISS top-1 cosine threshold — all predict no better than uniform.
- **Per-query w_ce routing (CC4 fusion).** Same shape: per-bin optimal w_ce is ~0.30 across all token-count bins; routing buys +0.03pp.
- **PRF expansion** (top-3 BM25 docs, 2 highest-IDF tokens appended). Fires on 99.7% of queries — too aggressive. R@10 -0.51pp.
- **LLM query rewriting** via Qwen3-4B-4bit (mlx_lm) on conversational queries. The model strips signal-bearing tokens. R@10 -0.17pp.
- **Composite query rewriting (LLM > spell > original)**. Worse than spell-only; LLM occupies query slots that spell would have done better.
- **Phonetic-fallback spell correction.** Vowel-stripping skeleton too coarse: `'shall'≡'shell'`, `'punk'≡'pink'`, `'earp'≡'europe'`. R@10 -0.60pp vs spell-only.
- **Phrase / 2-gram BM25.** Naive bigram inclusion lets each bigram's huge IDF dominate scoring (each bigram in 1-3 docs), rankings go random. R@10 0.01%. Real phrase BM25 needs BM25F-style field weighting.
- **BM25F first-token brand split.** Splitting titles into [first-token brand] + [rest] and fusing two BM25 streams underperforms full-title BM25.
- **bm25s scoring methods (lucene/atire/bm25l/bm25+).** All within 0.01pp of robertson at (k1=0.3, b=0.6).

Memory entries in `memory/` keep the per-experiment notes.

## Demo

The demo (`demo.py`) shows two columns side by side. Each column has its own
mode dropdown so any two architectures can be compared on the same query.
Default left = base MiniLM (anchor); default right = the quality SOTA
(CC5-100, sumsim + LiYuan + BGE 3-way mean).

Selectable modes span the three SOTA tiers (fast / medium / quality) plus
historical baselines: MNRL retrieval, base + ensemble rerank, BM25 alone, the
original BoD-as-retriever, and a live "build bag at query time" mode that
simulates the offline bag pipeline.

The Hugging Face Space at [huggingface.co/spaces/dtunkelang/bag-of-documents-demo](https://huggingface.co/spaces/dtunkelang/bag-of-documents-demo)
runs a slimmed 4-mode dropdown (base, BM25, RRF baseline, quality SOTA) on
the same code path. CPU-basic Space hardware makes the quality tier visibly
slow (~5-15s per query); the fast and BM25 tiers stay sub-second.

Precomputed product embeddings (`indexing/precompute_rerank_vecs.py`) keep the
bi-encoder rerankers at sub-100ms. The cross-encoder forward passes (LiYuan
in medium, LiYuan + BGE in quality) dominate quality-tier latency.

A second demo, [`demo_bestbuy.py`](demo_bestbuy.py), shows the BoD lift on the
BestBuy ACM clickthrough dataset side-by-side with off-the-shelf MiniLM. Type
e.g. `ati`, `dvd storage`, `turtlebeach`, or `i pad 2` and watch base return 0–1
clicked products in the top-10 while the BoD-trained model returns 8–10. Build
the artifacts with `download/build_bestbuy_bags.py` (after running
`download/prepare_bestbuy_acm.py` on the manually-downloaded Kaggle archive),
then `python demo_bestbuy.py` serves at `http://localhost:7860`.

## When does BoD generalize? — Cluster Hypothesis Score (CHS)

The cluster-hypothesis frame ("documents relevant to the same query tend to be similar to each other") is the load-bearing assumption behind bag-of-documents. We operationalize it as a runnable metric for any (corpus, encoder) pair so you can predict BoD-readiness on a new dataset *before* investing in the full pipeline.

For a one-shot pre-training readiness report on a new corpus — SCHS, base-difficulty distribution, predicted lift band, and a GO / CONDITIONAL / SKIP verdict — use:

```bash
python evaluation/bod_readiness_report.py \
    --catalog NEW/titles.json \
    --product-ids NEW/product_ids.json \
    --qrels NEW/test_qrels.jsonl --min-relevance 1 \
    --queries NEW/test_queries.jsonl \
    --encoder all-MiniLM-L6-v2 \
    --vecs-cache NEW/base_catalog.vecs.fp16.npy   # speeds up re-runs
```

Or run the metrics individually:

```bash
# CHS on ESCI-US (default)
python evaluation/cluster_hypothesis_score.py

# CHS on a new corpus (qrels + product titles)
python evaluation/cluster_hypothesis_score.py \
    --qrels NEW/test_qrels.jsonl \
    --pids NEW/product_ids.json \
    --titles NEW/titles.json \
    --encoder all-MiniLM-L6-v2

# Compare across many corpora at once (BEIR datasets supported via beir:NAME)
python evaluation/chs_corpus_compare.py \
    --datasets esci_us_strict bestbuy_acm beir:scidocs beir:trec-covid
```

The metric splits into:
- **SCHS** (Simple Cluster Hypothesis Score): how much closer in-bag pairs are to each other than random pairs. Computable on any positives-bearing corpus.
- **HCHS** (Hard Cluster Hypothesis Score): how much closer in-bag pairs are to each other than to within-query labeled negatives. Stronger test; needs explicit hard negatives.

Empirically (see [`evaluation/CHS_RESULTS.md`](evaluation/CHS_RESULTS.md) for the full table): SCHS ≥ 0.50 corresponds to BoD-positive corpora (ESCI-US, BestBuy product search), 0.40-0.50 to weakly-positive (ESCI-Spanish), and < 0.40 to BoD-negative (NFCorpus). The metric also tracks BoD *lift magnitude*, not just success/failure.

**Out-of-sample validation (BestBuy ACM, May 2026).** CHS predicted GREEN for the BestBuy 2012 ACM Hackathon clickthrough dataset (SCHS = 0.525) before any training. We then ran the full BoD pipeline end-to-end — split 60K multi-positive queries 80/20, built 48,516 click-derived bags, fine-tuned MiniLM with MNRL — and measured retrieval against the 12,128-query holdout:

| Model | R@10 | E@1 |
|---|---:|---:|
| `all-MiniLM-L6-v2` (base) | 0.5559 | 0.2538 |
| BoD-trained (this work) | **0.7308** | **0.3718** |
| **Δ** | **+17.49pp** | **+11.80pp** |

This is the largest BoD lift we have measured on any corpus and confirms that CHS rank-orders BoD-readiness across domains. Reproduce the run with [`download/build_bestbuy_bags.py`](download/build_bestbuy_bags.py) → [`download/add_random_hardnegs.py`](download/add_random_hardnegs.py) → [`training/finetune_with_hardnegs.py`](training/finetune_with_hardnegs.py) → [`evaluation/eval_bestbuy_bod.py`](evaluation/eval_bestbuy_bod.py).

Library: [`bagofdocs/cluster_hypothesis.py`](bagofdocs/cluster_hypothesis.py). Tests: [`tests/test_cluster_hypothesis.py`](tests/test_cluster_hypothesis.py) — synthetic-corpus tests that pin down the metric properties (perfect clustering → SCHS≈1; no structure → SCHS≈0; monotone in noise; etc.).

## Repository Layout

| Directory | Contents |
|---|---|
| `bagofdocs/` | Package with shared utilities (`bagofdocs.utils`) and the cluster-hypothesis library (`bagofdocs.cluster_hypothesis`) imported across the codebase |
| `download/` | Catalog and dataset acquisition: `download_catalog.py`, `download_esci_*.py`, `download_nfcorpus.py`, `download_fiqa.py` (auto-download from HuggingFace / ir_datasets); `prepare_bestbuy_acm.py` (preps already-downloaded Kaggle data); `build_bestbuy_bags.py` (build BoD training artifacts from BestBuy click data); `add_random_hardnegs.py` (generic random-hardneg augmentation for any bags.jsonl) |
| `indexing/` | Build search indexes (FAISS, tantivy) and precompute reranker product vectors / BM25 top-K caches |
| `training/` | Bag construction, CE training, query-model fine-tuning |
| `evaluation/` | Eval scripts and per-query / per-bin diagnostics |
| `scripts/` | One-off CLIs (`preflight.py`, `query_index.py`, `push_to_hf.py`) |
| `space_demo/` | HuggingFace Space demo (Gradio) |
| `tests/` | pytest suite |
| `docs/`, `memory/` | Documentation and persistent project notes |

Top-level: `demo.py` (ESCI side-by-side demo) and `demo_bestbuy.py` (BestBuy base vs BoD-trained side-by-side).

### Key Scripts

| Script | Purpose |
|---|---|
| `training/compute_bags.py` | Bag computation: hybrid retrieval → CE scoring → centroid |
| `training/finetune_query_model.py` | Fine-tune sentence transformer on bag centroids (cosine or MNRL loss) |
| `training/train_esci_ce.py` | Train a cross-encoder on ESCI grades; `--label-mode {regression,wide,binary}` |
| `indexing/build_index.py` | Build FAISS HNSW + tantivy indexes from titles, with validation |
| `indexing/build_tantivy.py` | Build a tantivy index from titles.json under a configurable tokenizer |
| `indexing/build_mnrl_hnsw_index.py` | Build the 6M-MNRL HNSW from cached product vectors |
| `indexing/precompute_rerank_vecs.py` | One-shot encode of the catalog under a reranker → cached fp16 numpy |
| `indexing/precompute_bm25_top_k.py` | Precompute BM25 top-K positions for the test queries |
| `indexing/compute_idf.py` | Per-token doc-frequency for IDF combo ranking |
| `download/download_catalog.py` | Download and sample McAuley Lab product catalog |
| `evaluation/eval_mnrl_retriever.py` | Comprehensive ESCI eval — setups A through Y |
| `evaluation/eval_per_query_bins.py` | Per-query-bin breakdown (binned by base R@10) |
| `evaluation/eval_oracle_routing.py` | Oracle + threshold/heuristic per-query routing analysis |
| `evaluation/eval_model.py` | Single-model ESCI label-recall evaluation |
| `evaluation/eval_rerank_full.py` | Full-ESCI rerank eval: base ± rerank, R@10 and nDCG@10 |
| `evaluation/eval_ensemble.py` | RRF / sumrank fusion over saved rerank position arrays |
| `evaluation/eval_ce_es_gap.py` | Measure CE E-vs-S separation (gap, E>S frequency) |
| `evaluation/build_regime_eval.py` | Build a per-regime (easy/mid/hard) eval harness from ESCI test queries |
| `demo.py` | Web demo: per-column mode dropdowns (retrieval / rerank / bag / SOTA hybrid) |
| `scripts/preflight.py` | Pre-run validation (index consistency, disk, memory) |
| `scripts/query_index.py` | CLI for querying the product index |

## Dataset

- **1.2M Amazon products** (the ESCI subset of the McAuley Lab catalog) used for the canonical eval and the shipped demo
- **6M Amazon products** (20% random sample of the 30M [McAuley Lab catalog](https://amazon-reviews-2023.github.io/), all 33 categories) used for the original BoD release and bag construction
- **75K queries** from the [Amazon Shopping Queries Dataset](https://arxiv.org/abs/2206.06588) (ESCI, US locale) for bag construction and as the eval source; 22,458 of those have at least one E-or-S qrel and form the canonical test set
- Evaluated against ESCI relevance judgments (Exact / Substitute / Complement / Irrelevant)

## Original release results (Apr 2026, BoD-as-retriever framing)

The originally-published results, on a 75K-query construction-set eval against
the 6M-product catalog. Kept for historical comparison; the canonical
22,458-query ESCI test results are in the architecture-progression table above.

| Metric | Base MiniLM | Fine-tuned (cosine BoD) |
|---|---|---|
| Cosine sim to centroids | 0.787 | **0.914** |
| Recall@10 | 0.367 | **0.506** |
| ESCI precision | 96.0% | **97.0%** |
| Complement retrieval rate | 14.2% | **7.7%** |

Specificity correctly correlates with query breadth ("laptop" 0.70 < "hp laptop" 0.81 < "hp laptop 16gb ram" 0.84).

## Known Limitations

- Model number sensitivity: "iphone 6" may retrieve iPhone 7/8 products. MiniLM embeddings don't distinguish numeric model identifiers well — but the BM25 retrieval lane in the SOTA architecture handles them via lexical matching.
- ESCI recall is low for all models in absolute terms — an artifact of top-K retrieval against a 1.2M product catalog. Relative metrics (delta over base, nDCG@10, E@1/E@3) are more meaningful for comparing architectures.

## Data and Model Sources

- **Product catalog**: [Amazon Reviews 2023](https://amazon-reviews-2023.github.io/) (McAuley Lab, UCSD; 1996–2023)
- **ESCI relevance judgments**: [Shopping Queries Dataset](https://arxiv.org/abs/2206.06588) (Reddy et al., KDD Cup 2022). ~130K queries with Exact/Substitute/Complement/Irrelevant labels.
- **Cross-encoders** (used in fusion at the medium/quality SOTA tiers and during bag construction):
   - [LiYuan/Amazon-Cup-Cross-Encoder-Regression](https://huggingface.co/LiYuan/Amazon-Cup-Cross-Encoder-Regression) — RoBERTa-base, ESCI-supervised. Used at medium tier and as one component of the quality-tier fusion. Originally used for bag-member CE filtering (threshold 0.3).
   - [BAAI/bge-reranker-v2-m3](https://huggingface.co/BAAI/bge-reranker-v2-m3) — XLM-RoBERTa-large (~568M params), BEIR-tested. Added to the quality SOTA fusion (CC5-100); +1.00pp R@10 / +2.95pp E@1 over CC4-100 (LiYuan-only fusion).
- **Base embedding model**: [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) (Sentence Transformers). Fine-tuned via MNRL on bag-derived signals to produce two of the three bi-encoder rerankers.
- **Spell correction**: [pyspellchecker](https://github.com/barrust/pyspell-checker) over a catalog-derived vocabulary (~172K title tokens, freq ≥ 2) at edit distance ≤ 2.
- **Vector index**: [FAISS](https://github.com/facebookresearch/faiss) (Meta AI) — HNSW for product embeddings (used by historical retrieval modes; not in the SOTA inference path).
- **Keyword index**: [bm25s](https://github.com/xhluca/bm25s) — numpy-backed BM25 with configurable k1/b; the SOTA's retrieval lane at (k1=0.3, b=0.6, Snowball english stemmer + en stopwords). [tantivy](https://github.com/quickwit-oss/tantivy) is kept as a legacy fallback for build/eval reproducibility.

## Acknowledgments

This project is based on work by [Daniel Tunkelang](https://www.linkedin.com/in/dtunkelang/) and [Aritra Mandal](https://www.linkedin.com/in/aritram/).

## License

MIT
