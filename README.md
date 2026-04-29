# Bag-of-Documents Product Search

An implementation of the [bag-of-documents](https://dtunkelang.medium.com/modeling-queries-as-bags-of-documents-b7d79d0916ab) model for e-commerce product search, built on 6M Amazon products across all 33 categories.

- **Blog post**: [Distilling Retrieval Pipelines to a Single Embedding Model](https://dtunkelang.medium.com/distilling-retrieval-pipelines-to-a-single-embedding-model-606f3ecf0c91)
- **Live demo**: [huggingface.co/spaces/dtunkelang/bag-of-documents-demo](https://huggingface.co/spaces/dtunkelang/bag-of-documents-demo)
- **Model and data**: [huggingface.co/datasets/dtunkelang/bag-of-documents](https://huggingface.co/datasets/dtunkelang/bag-of-documents)

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Download product catalog (20% sample, ~6M products)
python download_catalog.py

# Build indexes (FAISS + tantivy) — takes ~2-3 hours
python build_index.py

# Compute bags from ESCI queries — takes ~19 hours on Apple Silicon
python compute_bags.py queries.jsonl bags.jsonl --ce-rerank models/esci-cross-encoder

# Fine-tune the query model
python finetune_query_model.py bags.jsonl query_model/

# Run the demo
python demo.py

# Run with real-time bag construction (requires cross-encoder)
python demo.py --bag-search
```

Or use `run_pipeline.sh` to run the full pipeline end-to-end.

## What is the bag-of-documents model?

A search query is represented as the distribution of its relevant products in embedding space:
- **Centroid** = what the query means (direction in product space)
- **Specificity** = how broad or narrow the query is (spread of the distribution)

This enables a query model trained on (query text → centroid) pairs to generalize to unseen queries.

## Pipeline

The pipeline is intentionally simple — no hand-crafted heuristics, just retrieval + a learned quality signal:

1. **Hybrid retrieval**: tantivy keyword AND-matching (with relaxation) + FAISS embedding similarity. Both sources are essential — keyword retrieval uniquely contributes ~35% of bag members (brand/model queries), FAISS contributes ~58% (semantic matches), with only ~7% overlap.
2. **Cross-encoder scoring**: ESCI RoBERTa CE scores all candidates; only results scoring >= 0.3 are kept. Uses MPS (Apple Silicon GPU) with batch size 32 for optimal throughput.
3. **Bag construction**: top 50 passing candidates → encode → spherical mean centroid + specificity
4. **Fine-tuning**: train MiniLM to predict bag centroids from query text (cosine distance loss)
5. **Iterative refinement**: rebuild index with fine-tuned model → recompute bags

### Pipeline evolution

Earlier versions included a rule filter (spec/capacity matching) and a 4-signal heuristic relevance scorer (category accessory rate, "for" pattern, accessory nouns, brand match). These were removed after measurement showed:
- **Rule filter**: rejected only 0.1% of candidates (17/28,232 in a 100-query sample)
- **Relevance scorer**: actively *hurt* bag quality — bags built with CE scoring all candidates had mean CE score 0.743 vs 0.591 when the heuristic pre-filtered the CE window. The heuristic was preventing the CE from seeing its best candidates.

The cross-encoder alone produces higher quality bags than the combination of heuristics + CE.

## Key Files

| File | Purpose |
|------|---------|
| `compute_bags.py` | Bag computation: hybrid retrieval → CE scoring → centroid |
| `compute_idf.py` | Per-token doc-frequency over the catalog (used for IDF combo ranking in compute_bags) |
| `build_index.py` | Build FAISS HNSW + tantivy indexes from titles, with validation |
| `build_tantivy.py` | Build a tantivy index from titles.json under a configurable tokenizer (e.g. `en_stem`) |
| `download_catalog.py` | Download and sample McAuley Lab product catalog |
| `finetune_query_model.py` | Fine-tune sentence transformer on bag centroids (cosine or MNRL loss) |
| `train_esci_ce.py` | Train a cross-encoder on ESCI grades; `--label-mode {regression,wide,binary}` |
| `precompute_rerank_vecs.py` | One-shot encode of the catalog under a reranker → cached fp16 numpy |
| `eval_model.py` | Single-model ESCI label-recall evaluation |
| `eval_rerank_full.py` | Full-ESCI rerank eval: base ± rerank, R@10 and nDCG@10 |
| `eval_ensemble.py` | RRF / sumrank fusion over saved rerank position arrays |
| `eval_ce_es_gap.py` | Measure CE E-vs-S separation (gap, E>S frequency) |
| `eval_new_ce.sh` | Post-CE-training validation pipeline |
| `build_regime_eval.py` | Build a per-regime (easy/mid/hard) eval harness from ESCI test queries |
| `demo.py` | Web demo: base + selectable right-column mode (retrieval / rerank / bag) |
| `preflight.py` | Pre-run validation (index consistency, disk, memory) |
| `query_index.py` | CLI for querying the product index |
| `run_pipeline.sh` | Pipeline orchestration |

## Dataset

- **~6M products** across all 33 Amazon categories (20% random sample of the full 30M [McAuley Lab catalog](https://amazon-reviews-2023.github.io/); data collected 1996–2023)
- **75K queries** from the [Amazon Shopping Queries Dataset](https://arxiv.org/abs/2206.06588) (ESCI, US locale) — real Amazon search queries
- Evaluated against ESCI relevance judgments (Exact/Substitute/Complement/Irrelevant labels)

## Results

Broad catalog (6M products, 33 categories, 75K ESCI queries):

| Metric | Base MiniLM | Fine-tuned |
|--------|-------------|------------|
| Cosine sim to centroids | 0.787 | **0.914** |
| Recall@10 | 0.367 | **0.506** |
| ESCI precision | 96.0% | **97.0%** |
| Complement retrieval rate | 14.2% | **7.7%** |

Specificity correctly correlates with query breadth ("laptop" 0.70 < "hp laptop" 0.81 < "hp laptop 16gb ram" 0.84)

### Architecture progression (Apr 2026)

Per-query measurements on the full ESCI test set (22,458 queries with at
least one E or S judgment, against the 1.2M-product ESCI index, K_retrieve
= 100, K_eval = 10, R@10 pools E+S as relevant, nDCG@10 uses E=1.0 / S=0.1
gain):

| # | Pipeline | R@10 | nDCG@10 | E@1 | E@3 |
|---|---|---|---|---|---|
| A | Base MiniLM | 15.60% | 0.2648 | 31.50% | 28.52% |
| B | 6M-MNRL retriever (BoD) | 18.10% | 0.3090 | 36.16% | 33.25% |
| C | Base + ensemble rerank | 19.00% | 0.3238 | 37.81% | 34.92% |
| D | 6M-MNRL + hardneg rerank | 17.53% | 0.2967 | 34.55% | 31.77% |
| E | 6M-MNRL + ensemble rerank | 19.83% | 0.3375 | 39.13% | 36.12% |
| F | RRF(base, MNRL) retrieval | 17.12% | 0.2916 | 34.36% | 31.43% |
| G | RRF(base, MNRL) + ensemble rerank | 19.84% | 0.3375 | 39.14% | 36.12% |
| H | BM25 alone (tantivy, en_stem) | 19.50% | 0.3322 | 38.79% | 35.72% |
| **I** | **RRF(BM25, MNRL) + ensemble rerank** | **20.01%** | **0.3394** | **39.19%** | **36.22%** |
| J | RRF(BM25, base, MNRL) + ensemble rerank | 20.07% | 0.3401 | 39.23% | 36.25% |

Read-outs:

- **The cosine-distilled BoD-as-retriever loses to base on this benchmark.**
  The original release was measured on a 75K-query construction-set eval; on
  the canonical 22,458-query ESCI test set, R@10 drops below A. MNRL-trained
  BoD (B) reverses that — +2.50pp R@10 over base, +4.66pp E@1.
- **Ensemble rerank stacks on top of any retriever.** Same two BoD-trained
  rerankers (6M-MNRL, qrels-hardneg) lift base by +3.40pp R@10 (A→C) and
  6M-MNRL by +1.73pp R@10 (B→E).
- **Setup E reaches +4.23pp R@10 over base with no MiniLM in the inference
  path** — three forward passes per query, all BoD-trained, plus an HNSW
  lookup in BoD-trained product space.
- **BM25 alone is shockingly close to the dense rerank stack.** Setup H
  (stemmed BM25, no dense, no rerank) hits R@10 19.50% — within rounding
  of E. On entity-heavy product queries, lexical matching is doing most
  of the work.
- **Hybrid is the SOTA.** Setup I (RRF-fuses BM25 and 6M-MNRL retrieval,
  then ensemble-reranks) wins every metric. Setup J adds base FAISS to the
  fusion pool — +0.06pp over I, within noise.
- **Pure dense fusion (G) doesn't beat dense-only (E).** Base and 6M-MNRL
  fail on overlapping query types, so RRF-fusing two dense retrievers
  contributes nothing additive after rerank. The hybrid lift in I/J comes
  from BM25's *different* failure mode, not from the number of retrievers.

Eval scripts: `eval_mnrl_retriever.py` (the table above), `precompute_bm25_top_k.py`
(precomputes BM25 top-100 against the en_stem tantivy index for setups H/I/J),
`build_mnrl_hnsw_index.py` (builds the 6M-MNRL HNSW from cached product
embeddings).

## Queries

The query set consists of all 75K US-locale queries from the Amazon ESCI dataset — real search queries from Amazon covering all product categories. No synthetic or curated queries are used.

## Demo

The demo (`demo.py`) shows two columns side by side. Each column has its own
mode dropdown so any two architectures can be compared on the same query.
Default left = base MiniLM, default right = the SOTA hybrid:

- **BM25 + MNRL hybrid + ensemble rerank** (default right) — RRF-fuses BM25
  and 6M-MNRL top-100 candidate sets, then sumsim-reranks with two BoD-trained
  encoders (6M-MNRL + qrels-hardneg). R@10 20.01% on the ESCI test set.
- **MNRL + BoD ensemble rerank** — 6M-MNRL retrieves top-100, the same two
  BoD encoders rerank. R@10 19.83%.
- **Base + BoD ensemble rerank** — same reranker stack on plain MiniLM
  retrieval. R@10 19.00%.
- **BM25 retrieval** — tantivy en_stem alone, no dense, no rerank. R@10
  19.50%.
- **MNRL retrieval (no rerank)** — 6M-MNRL alone. R@10 18.10%.
- **Fine-tuned retrieval (cosine BoD)** — the originally deployed
  BoD-as-retriever (single cosine-distilled model + FAISS); kept for
  historical comparison.
- **Base MiniLM retrieval** (default left) — no fine-tuning, no rerank.
- **Build bag at query time** — simulates the offline bag pipeline live:
  hybrid retrieval → CE scores all candidates → 0.3 threshold → centroid
  → FAISS re-retrieval. Requires `--bag-search` to load the cross-encoder
  at startup. Right-column only.

Precomputed product embeddings (`precompute_rerank_vecs.py`) keep dense
modes at sub-100ms; the BM25 path is faster still.

## Known Limitations

- Model number sensitivity: "iphone 6" may retrieve iPhone 7/8 products. MiniLM embeddings don't distinguish numeric model identifiers well.
- ESCI recall is low for all models — an artifact of top-50 retrieval from 6M products. Precision is the more meaningful metric.

## Data and Model Sources

- **Product catalog**: [Amazon Reviews 2023](https://amazon-reviews-2023.github.io/) (McAuley Lab, UCSD; data collected 1996–2023)
- **ESCI relevance judgments**: [Shopping Queries Dataset](https://arxiv.org/abs/2206.06588) (Reddy et al., KDD Cup 2022). ~130K queries with Exact/Substitute/Complement/Irrelevant labels.
- **Cross-encoder**: [LiYuan/Amazon-Cup-Cross-Encoder-Regression](https://huggingface.co/LiYuan/Amazon-Cup-Cross-Encoder-Regression) — RoBERTa-based cross-encoder trained on ESCI data for the KDD Cup 2022 competition. Used for bag member scoring (threshold 0.3).
- **Base embedding model**: [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) (Sentence Transformers). Fine-tuned on bag centroids to produce the query model.
- **Vector index**: [FAISS](https://github.com/facebookresearch/faiss) (Meta AI) — HNSW index for product embeddings
- **Keyword index**: [tantivy](https://github.com/quickwit-oss/tantivy) — Rust-based full-text search for hybrid retrieval

## Acknowledgments

This project is based on work by [Daniel Tunkelang](https://www.linkedin.com/in/dtunkelang/) and [Aritra Mandal](https://www.linkedin.com/in/aritram/).

## License

MIT
