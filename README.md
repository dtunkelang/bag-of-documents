# Bag-of-Documents Product Search

An implementation of the [bag-of-documents](https://dtunkelang.medium.com/modeling-queries-as-bags-of-documents-b7d79d0916ab) model for e-commerce product search, evaluated against ESCI on a 1.2M Amazon product catalog. Two BoD-trained reranker encoders fused on top of BM25 candidates produce the current shipped state of the art.

- **Blog post**: [Distilling Retrieval Pipelines to a Single Embedding Model](https://dtunkelang.medium.com/distilling-retrieval-pipelines-to-a-single-embedding-model-606f3ecf0c91)
- **Live demo**: [huggingface.co/spaces/dtunkelang/bag-of-documents-demo](https://huggingface.co/spaces/dtunkelang/bag-of-documents-demo)
- **Model and data**: [huggingface.co/datasets/dtunkelang/bag-of-documents](https://huggingface.co/datasets/dtunkelang/bag-of-documents)

## Headline Result

| | R@10 | nDCG@10 | E@1 | E@3 |
|---|---|---|---|---|
| Base MiniLM (dense retrieval only) | 15.60% | 0.2648 | 31.50% | 28.52% |
| RRF(BM25, base) (non-BoD hybrid retrieval) | 18.62% | 0.3048 | 31.54% | 31.98% |
| BM25 alone (lexical retrieval only) | 19.50% | 0.3322 | 38.79% | 35.72% |
| **BM25 + ensemble rerank (current SOTA)** | **21.11%** | **0.3566** | **40.87%** | **38.04%** |

22,458-query ESCI test set, R@10 with E+S as relevant, nDCG@10 with E=1.0 / S=0.1 gain. The BoD rerank stack buys **+1.61pp R@10 over BM25 alone**, **+2.49pp over the strongest non-BoD baseline (RRF hybrid retrieval)**, and **+5.51pp over base MiniLM**. No dense retrieval and no HNSW in the inference path.

The non-BoD hybrid baseline is included to make the comparison honest — RRF-fusing BM25 and base MiniLM is the standard "vanilla hybrid retrieval" recipe, available with no training, no fine-tuning, and one extra forward pass per query at inference. It actually *underperforms* BM25 alone on this benchmark (the base FAISS lane displaces BM25's exact-match top-1 with semantically-similar near-misses; E@1 drops from 38.79% to 31.54%), confirming that on entity-anchored product catalogs, lexical retrieval dominates dense.

The deployable architecture is:

1. **Retrieval**: tantivy BM25 (en_stem tokenizer) returns top-100 candidates.
2. **Rerank**: two BoD-trained encoders (`query_model_us_full_6m_mnrl`, `query_model_us_qrels_mnrl_hardneg`) score the candidates.
3. **Fusion**: sumsim — average the two cosine similarities; sort descending; return top-10.

Three small forward passes per query, all over precomputed product embeddings; sub-50ms wall-clock on commodity hardware.

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

## Architecture progression (Apr 2026)

Per-query measurements on the full ESCI test set (22,458 queries with at
least one E or S judgment, against the 1.2M-product ESCI index, K_retrieve
= 100, K_eval = 10):

| # | Pipeline | R@10 | nDCG@10 | E@1 | E@3 |
|---|---|---|---|---|---|
| A | Base MiniLM | 15.60% | 0.2648 | 31.50% | 28.52% |
| B | 6M-MNRL retriever (BoD) | 18.10% | 0.3090 | 36.16% | 33.25% |
| C | Base + ensemble rerank | 19.00% | 0.3238 | 37.81% | 34.92% |
| D | 6M-MNRL + hardneg rerank (single) | 17.53% | 0.2967 | 34.55% | 31.77% |
| E | 6M-MNRL + ensemble rerank | 19.83% | 0.3375 | 39.13% | 36.12% |
| H | BM25 alone (tantivy, en_stem) | 19.50% | 0.3322 | 38.79% | 35.72% |
| Z | RRF(BM25, base) retrieval (non-BoD hybrid baseline) | 18.62% | 0.3048 | 31.54% | 31.98% |
| I | RRF(BM25, MNRL) + ensemble rerank | 20.01% | 0.3394 | 39.19% | 36.22% |
| AA | RRF(BM25, base) + ensemble rerank | 20.43% | 0.3451 | 39.42% | 36.73% |
| **K** | **BM25 + ensemble rerank (no dense retrieval)** | **21.11%** | **0.3566** | **40.87%** | **38.04%** |

**Read-outs:**

- **The cosine-distilled BoD-as-retriever loses to base on this benchmark.** The original release was measured on a 75K-query construction-set eval; on the canonical 22,458-query ESCI test set, R@10 drops below A. MNRL-trained BoD (B) reverses that — +2.50pp R@10 over base, +4.66pp E@1.
- **Ensemble rerank stacks on top of any retriever.** The two BoD-trained rerankers (6M-MNRL, qrels-hardneg) lift base by +3.40pp R@10 (A→C) and 6M-MNRL by +1.73pp R@10 (B→E).
- **BM25 alone is shockingly close to the dense rerank stack.** Setup H (stemmed BM25, no dense, no rerank) hits R@10 19.50% — within rounding of E. On entity-heavy product queries, lexical matching does most of the work.
- **MNRL retrieval is dead weight in the SOTA pipeline.** Setup K (BM25 + ensemble rerank, *no* dense retrieval) scores R@10 21.11% — +1.10pp over setup I, the previous shipped hybrid. Adding MNRL retrieval to the candidate pool *dilutes* BM25's lexically-anchored hits with semantically-near-but-irrelevant ones the rerank can't fully clean up. The same pattern holds when fusing in base MiniLM (AA: 20.43%, -0.68pp from K) or both base + MNRL (J: 20.07%, -1.04pp). BM25-alone candidates remain the best input to the BoD rerank.
- **Pure non-BoD hybrid retrieval (Z) underperforms BM25 alone.** Setup Z (RRF(BM25, base) retrieval, no rerank) scores R@10 18.62% — *worse* than BM25 alone (H, 19.50%) and dramatically worse on E@1 (31.54% vs 38.79%). Mixing dense candidates into the lexical lane displaces BM25's exact-match top-1 with semantically-similar near-misses. This is the strongest "no BoD" baseline available out of the box, and it loses to BM25 alone — let alone to K.
- **The two BoD rerankers are well-matched; adding a third doesn't help.** Single-reranker setups L1 (BM25 + 6M-MNRL alone) and L2 (BM25 + hardneg alone) score 19.67% / 19.20% — both below K. Three-way ensembles with cosine-distilled (R50: -0.4pp) or MNRL-with-different-bag-threshold (S50: ties within rounding) third encoders don't budge the SOTA.

### Where the lift comes from (per-query-bin breakdown)

Binning the 22,458 queries by base MiniLM's per-query R@10 reveals the
+5.51pp aggregate K - base lift is dramatically bimodal:

| bin (base R@10) | n queries | base R@10 | K R@10 | lift | E@1 lift |
|---|---|---|---|---|---|
| 0 (hard regime) | 7,293 (32%) | 0.00% | 8.57% | **+8.57pp** | +13.86pp |
| (0, 0.25] | 10,339 (46%) | 12.86% | 18.07% | +5.21pp | +9.66pp |
| (0.25, 0.50] | 3,594 (16%) | 37.45% | 40.25% | +2.80pp | +2.62pp |
| (0.50, 1.00] (easy) | 1,232 (5%) | 67.10% | 64.99% | **-2.11pp** | +0.08pp |

A third of the queries have zero base-FAISS recall — products dense
retrieval can't surface at all. BM25 + ensemble rerank rescues 8.57% of
them and lifts E@1 from 0% to 13.86%. That single bin contributes +2.74pp
to the +5.51pp aggregate — half the total lift. On the 5% of queries
where dense already does well (R@10 > 0.5), K *loses* 2.11pp to base — but
E@1 is preserved. **The architecture wins by rescuing the hard regime,
not by being uniformly better.** Diagnostic: `evaluation/eval_per_query_bins.py`.

### What didn't work (negatives worth recording)

The local maximum is K. These probes were tried and rejected:

- **Score-fusion of dense retrievers (F, G, J).** RRF-fusing base + 6M-MNRL retrieval contributes nothing additive; both fail on overlapping query types.
- **K_retrieve sweeps (N40 / N50 / N60 / N75 / N200).** Peak around top-50 to top-100; +0.06pp gain over K is within noise.
- **Alternative fusion functions.** Sumrank (M, -0.53pp), max (Tmax, -1.20pp), min (Tmin, -0.73pp), and weighted variants (W40 / W60 / W70, all losing) all underperform 0.5/0.5 sumsim averaging.
- **Min-max normalization before fusion (Q, -0.05pp).** The two rerankers are already similarly calibrated.
- **Default tantivy tokenizer (P, -0.80pp vs en_stem).** Stemming is doing real work.
- **A categorically stronger base (Y\* setups, BAAI/bge-base-en-v1.5).** A BoD reranker trained on the same bags but with BGE-base (5× MiniLM params, 768-dim) underperforms every variant tested. Y1 (BGE alone) is 17.16% — below plain BM25. The BGE training plateaued early at bag-internal recall@10 = 0.40 vs MiniLM's published 0.506. Stronger base ≠ better reranker under the naive bag-MNRL recipe; would need a different recipe (higher LR, hard negatives, or supervised data) to unlock.
- **Per-query routing between A and K.** Oracle headroom is +1.76pp R@10 / +6.95pp E@1, but A-wins are spread uniformly (~24%) across every non-zero base-recall bin. Every simple feature tested (base-FAISS top-1 cosine threshold, query length, digit presence) loses to K-only. Diagnostic: `evaluation/eval_oracle_routing.py`.

Memory entries in `memory/` keep the per-experiment notes.

## Demo

The demo (`demo.py`) shows two columns side by side. Each column has its own
mode dropdown so any two architectures can be compared on the same query.
Default left = base MiniLM, default right = the SOTA (BM25 + ensemble rerank).

Other selectable modes include MNRL retrieval, base + ensemble rerank, BM25-alone,
the original BoD-as-retriever (kept for historical comparison), and a live
"build bag at query time" mode that simulates the offline bag pipeline.

Precomputed product embeddings (`indexing/precompute_rerank_vecs.py`) keep dense
modes at sub-100ms; the BM25 path is faster still.

## Repository Layout

| Directory | Contents |
|---|---|
| `bagofdocs/` | Package with shared utilities (`bagofdocs.utils`) imported across the codebase |
| `download/` | Catalog and dataset acquisition (`download_catalog.py`, `download_esci_*.py`, `download_nfcorpus.py`) |
| `indexing/` | Build search indexes (FAISS, tantivy) and precompute reranker product vectors / BM25 top-K caches |
| `training/` | Bag construction, CE training, query-model fine-tuning |
| `evaluation/` | Eval scripts and per-query / per-bin diagnostics |
| `scripts/` | One-off CLIs (`preflight.py`, `query_index.py`, `push_to_hf.py`) |
| `space_demo/` | HuggingFace Space demo (Gradio) |
| `tests/` | pytest suite |
| `docs/`, `memory/` | Documentation and persistent project notes |

Top-level: `demo.py` (local FastAPI demo, main entry point).

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
- **Cross-encoder**: [LiYuan/Amazon-Cup-Cross-Encoder-Regression](https://huggingface.co/LiYuan/Amazon-Cup-Cross-Encoder-Regression) — RoBERTa-based cross-encoder trained on ESCI data for the KDD Cup 2022 competition. Used for bag member scoring (threshold 0.3).
- **Base embedding model**: [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) (Sentence Transformers). Fine-tuned via MNRL on bags to produce the two rerankers.
- **Vector index**: [FAISS](https://github.com/facebookresearch/faiss) (Meta AI) — HNSW for product embeddings (used by historical retrieval modes; not in the SOTA inference path)
- **Keyword index**: [tantivy](https://github.com/quickwit-oss/tantivy) — Rust-based full-text search; the SOTA's retrieval lane

## Acknowledgments

This project is based on work by [Daniel Tunkelang](https://www.linkedin.com/in/dtunkelang/) and [Aritra Mandal](https://www.linkedin.com/in/aritram/).

## License

MIT
