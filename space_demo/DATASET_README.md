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
| Products | ~6M (20% sample of 30M across all 33 categories) |
| Queries with bags | ~75K (Amazon ESCI, US locale) |
| Embedding dimensions | 384 |
| Categories | All 33 Amazon categories |

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
├── rerank_A.vecs.fp16.npy               # NEW: cached product embeddings under Reranker A
└── rerank_B.vecs.fp16.npy               # NEW: cached product embeddings under Reranker B
query_model/                             # Original BoD-as-retriever (cosine-loss MNRL)
query_model_6m_mnrl/                     # NEW: Reranker A — 75K queries × 6M corpus, MNRL
query_model_hardneg/                     # NEW: Reranker B — qrels-based bags + hard negatives
bags.jsonl                               # Bag centroids + specificity for each query
queries.jsonl                            # Source queries
eval/
└── regime_queries.jsonl                 # NEW: 45-query per-regime eval harness
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

## BoD as reranker

The originally-published architecture treats BoD as a retrieval-stage model (single encoder + FAISS, end of story). After running the full ESCI test set across multiple variants, we found this consistently underperforms base MiniLM on aggregate R@10 / nDCG@10 — but the same trained models become **clean wins as rerankers** on top of base retrieval.

### ESCI test results (22,458 queries, R@10 with E+S relevant, nDCG@10 with E=1.0 / S=0.1)

| Pipeline | R@10 | nDCG@10 |
|---|---|---|
| Base MiniLM only | 15.60% | 0.2648 |
| Base + 6M-MNRL reranker | 17.53% | 0.3000 |
| Base + qrels-hardneg reranker | 17.25% | 0.2920 |
| **Base + sumrank ensemble** | **18.35%** | **0.3139** |

The two rerankers carry orthogonal relevance signal — the +0.8pp ensemble lift over the best individual reranker shows neither is redundant. With cached product embeddings, the ensemble adds only 3 small forward passes per query (base + 2 reranker query encodes); FAISS dominates the remaining latency.

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
