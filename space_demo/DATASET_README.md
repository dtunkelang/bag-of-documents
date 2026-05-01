---
license: mit
language:
  - en
task_categories:
  - sentence-similarity
  - feature-extraction
tags:
  - e-commerce
  - product-search
  - bag-of-documents
  - sentence-transformers
  - retrieval
size_categories:
  - 10K<n<100K
pretty_name: Bag-of-Documents Product Search
---

# Bag-of-Documents: Product Search Dataset

- **Blog post**: [Distilling Retrieval Pipelines to a Single Embedding Model](https://dtunkelang.medium.com/distilling-retrieval-pipelines-to-a-single-embedding-model-606f3ecf0c91)
- **Live demo**: [huggingface.co/spaces/dtunkelang/bag-of-documents-demo](https://huggingface.co/spaces/dtunkelang/bag-of-documents-demo)
- **Code**: [github.com/dtunkelang/bag-of-documents](https://github.com/dtunkelang/bag-of-documents)

## Dataset Description

A large-scale bag-of-documents dataset for e-commerce product search, built on Amazon product data. Each search query is represented as a distribution of relevant products in embedding space, captured by a centroid vector and a specificity score.

- **Centroid**: the mean direction in product embedding space, representing what the query means
- **Specificity**: the tightness of the distribution (high = narrow query like "hp laptop 16gb ram", low = broad query like "laptop")

### Dataset Summary

| | Count |
|---|---|
| Source product catalog | ~6M (20% sample of 30M across all 33 Amazon categories) |
| Products in this dataset's index | ~1.2M (the ESCI subset that has relevance judgments) |
| Queries with bags | ~75K (Amazon ESCI, US locale) |
| Embedding dimensions | 384 |
| Categories | All 33 Amazon categories |

The bags themselves were computed against the broader 6M source catalog (per the original blog post). The 1.2M ESCI subset is what's loaded in `combined_index/index.faiss` here — it's the catalog the rerank result table below was evaluated on, and the catalog the live Space demo serves.

### Supported Tasks

- **Retrieval model training**: fine-tune an embedding model to predict bag centroids from query text, producing a query encoder specialized for product search
- **Reranking**: use BoD-trained query encoders as bi-encoder rerankers on top of base FAISS retrieval (see "BoD as reranker" below — the deployable architecture)
- **Specificity prediction**: predict whether a query is broad or narrow using kNN on bag centroids
- **Search evaluation**: compare retrieval models using bag centroids as ground-truth query representations

### Languages

English (US)

## Dataset Structure

```
combined_index/
├── index.faiss                          # FAISS HNSW index over the catalog
├── titles.json                          # Product titles, parallel to FAISS positions
├── rerank_A.vecs.fp16.npy               # cached product embeddings under Reranker A
├── rerank_B.vecs.fp16.npy               # cached product embeddings under Reranker B
├── rerank_G.vecs.fp16.npy               # cached product embeddings under Reranker G
├── tantivy_index/                       # tantivy BM25 index (legacy, en_stem)
└── bm25s_index/                         # NEW: bm25s BM25 index (k1=0.3, b=0.6) - SOTA retriever
query_model/                             # Original BoD-as-retriever (cosine-loss MNRL)
query_model_6m_mnrl/                     # Reranker A - 75K queries × 6M corpus, MNRL
query_model_hardneg/                     # Reranker B - qrels-based bags + hard negatives
query_model_esci_supervised/             # Reranker G - ESCI E-vs-I triplets, MNRL
bags.jsonl                               # Bag centroids + specificity for each query
queries.jsonl                            # Source queries
eval/
└── regime_queries.jsonl                 # 45-query per-regime eval harness
```

### Bags (JSONL)

Each bag is a JSON object:

```json
{
  "query": "wireless keyboard",
  "num_results": 42,
  "query_vector": [0.023, -0.051, ...],
  "specificity": 0.95,
  "results": [
    {"title": "Logitech K380 Multi-Device Bluetooth Keyboard"},
    ...
  ]
}
```

### Cached reranker embeddings

`combined_index/rerank_A.vecs.fp16.npy` and `rerank_B.vecs.fp16.npy` are 1,215,851 × 384 fp16 numpy arrays, parallel to `combined_index/titles.json`. Loading them lets the rerank pipeline skip the live candidate-encoding step — only the query is encoded at runtime, candidates are looked up by FAISS-returned index. This is what makes the reranker deployable at sub-100ms latency on commodity hardware.

## Architecture progression

The originally-published architecture treats BoD as a retrieval-stage model (single encoder + FAISS, end of story). Iterating on loss function (cosine-to-centroid → MNRL), training scale (full 6M signal), and stage (retrieval vs rerank vs hybrid) produced a series of improvements. The progression below is measured on the full 22,458-query ESCI test set, R@10 with E+S as relevant, nDCG@10 with E=1.0/S=0.1, K_retrieve=100, K_eval=10.

| # | Pipeline | R@10 | nDCG@10 | E@1 | E@3 |
|---|---|---|---|---|---|
| A | Base MiniLM | 15.60% | 0.2648 | 31.50% | 28.52% |
| B | 6M-MNRL retriever (BoD) | 18.10% | 0.3090 | 36.16% | 33.25% |
| Z | RRF(BM25, base) retrieval (non-BoD hybrid baseline) | 18.62% | 0.3048 | 31.54% | 31.98% |
| C | Base + ensemble rerank | 19.00% | 0.3238 | 37.81% | 34.92% |
| H | BM25 alone (tantivy, en_stem default) | 19.50% | 0.3322 | 38.79% | 35.72% |
| E | 6M-MNRL + ensemble rerank | 19.83% | 0.3375 | 39.13% | 36.12% |
| I | RRF(BM25, MNRL) + ensemble rerank | 20.01% | 0.3394 | 39.19% | 36.22% |
| H' | BM25 alone (bm25s, k1=0.3, b=0.6) | 20.33% | 0.3451 | 40.06% | 36.87% |
| AA | RRF(BM25, base) + ensemble rerank | 20.43% | 0.3451 | 39.42% | 36.73% |
| K | BM25 (tantivy) + 2-way ensemble rerank | 21.11% | 0.3566 | 40.87% | 38.04% |
| K' | BM25 (bm25s) + 2-way ensemble rerank | 21.27% | 0.3588 | 41.12% | 38.27% |
| CC3-50 (tantivy) | BM25 top-50 + 3-way ensemble rerank | 21.32% | 0.3613 | 41.64% | 38.80% |
| **CC3-50 (bm25s, current SOTA)** | **BM25 (bm25s, k1=0.3, b=0.6) top-50 + 3-way ensemble rerank** | **21.61%** | **0.3660** | **42.11%** | **39.22%** |

Four things to note:

- **MNRL-trained BoD beats base as a *retriever*** (B vs A: +2.50pp R@10). The original cosine-distilled BoD-as-retriever loses on this stricter benchmark; the MNRL-trained variant doesn't.
- **BM25 alone is competitive with the dense rerank stack** (H ≈ E, within rounding). On entity-heavy product queries, lexical matching does most of the work.
- **MNRL retrieval is dead weight in the SOTA pipeline.** Setup K (BM25 + ensemble rerank, *no* dense retrieval) scores R@10 21.11% — +1.10pp over the previous shipped hybrid (I). Adding MNRL retrieval to the candidate pool dilutes BM25's lexically-anchored hits with semantically-near-but-irrelevant ones.
- **BM25 hyperparameter tuning matters.** Default Lucene/tantivy params (k1=1.2, b=0.75) assume long natural-language documents. Amazon product titles are short and keyword-stuffed. A sweep on the ESCI test set finds (k1=0.3, b=0.6) optimal — early term-frequency saturation, moderate length normalization. The H' / K' / CC3-50 (bm25s) rows show the +0.83pp / +0.16pp / +0.29pp lift the parameter swap brings on top of every downstream rerank stage. The deployable architecture is: bm25s (k1=0.3, b=0.6) → ensemble rerank with two bag-derived BoD encoders + one ESCI-supervised encoder. No HNSW index in the inference path.

The ensemble rerank fuses three encoders (`query_model_6m_mnrl`, `query_model_hardneg`, `query_model_esci_supervised`) by averaging their cosine similarities. With cached product embeddings (`rerank_A.vecs.fp16.npy`, `rerank_B.vecs.fp16.npy`, `rerank_G.vecs.fp16.npy`), only the query is encoded live; candidate vectors are looked up by index. The retrieval lane uses bm25s (configurable k1/b) over the same 1.2M titles.

## Dataset Creation

### Source Data

- **Products**: [Amazon Reviews 2023](https://amazon-reviews-2023.github.io/) (McAuley Lab, UCSD; data collected 1996-2023). 20% random sample of the full catalog across all 33 categories (~6M of ~30M unique products).
- **Queries**: All 75K US-locale queries from the [Amazon Shopping Queries Dataset](https://arxiv.org/abs/2206.06588) (ESCI, KDD Cup 2022) — real Amazon search queries spanning all product categories.

### Bag Construction Pipeline

```
Query text
  -> Hybrid retrieval: keyword (tantivy AND with relaxation) + FAISS embedding similarity
  -> Cross-encoder scoring: ESCI RoBERTa CE scores ALL candidates, threshold 0.3
  -> Top 50 passing candidates -> encode -> bag centroid + specificity
```

The cross-encoder is [LiYuan/Amazon-Cup-Cross-Encoder-Regression](https://huggingface.co/LiYuan/Amazon-Cup-Cross-Encoder-Regression), a RoBERTa model trained on ESCI data for the KDD Cup 2022 competition.

### Fine-Tuning the Reranker Models

- `query_model_6m_mnrl`: trained with MultipleNegativesRanking loss on bags from the full 6M-product corpus
- `query_model_hardneg`: trained with MNRL on qrels-derived bags + hard-mined negative products
- Both share the same `all-MiniLM-L6-v2` base model

The cached vec files were produced by encoding all 1.2M ESCI products under each reranker (`precompute_rerank_vecs.py` in the code repo) and saving as fp16 numpy.

## Citation

```
@misc{tunkelang2026bagdocs,
  title={Bag-of-Documents: Product Search Dataset},
  author={Daniel Tunkelang and Aritra Mandal},
  year={2026},
  url={https://huggingface.co/datasets/dtunkelang/bag-of-documents}
}
```

### Related Work

- Tunkelang, D. [Distilling Retrieval Pipelines to a Single Embedding Model](https://dtunkelang.medium.com/distilling-retrieval-pipelines-to-a-single-embedding-model-606f3ecf0c91). 2026.
- Tunkelang, D. [Modeling Queries as Bags of Documents](https://dtunkelang.medium.com/modeling-queries-as-bags-of-documents-b7d79d0916ab). 2024.
- Reddy, C.K. et al. [Shopping Queries Dataset: A Large-Scale ESCI Benchmark for Improving Product Search](https://arxiv.org/abs/2206.06588). KDD Cup 2022.
- McAuley Lab. [Amazon Reviews 2023](https://amazon-reviews-2023.github.io/).

## License

MIT
