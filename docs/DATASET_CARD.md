# Bag-of-Documents: Product Search Dataset

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
- **Specificity prediction**: predict whether a query is broad or narrow using kNN on bag centroids
- **Search evaluation**: compare retrieval models using bag centroids as ground-truth query representations

### Languages

English (US)

## Dataset Structure

### Bags (JSONL)

Each bag is a JSON object:

```json
{
  "query": "wireless keyboard",
  "num_results": 42,
  "query_vector": [0.023, -0.051, ...],  // 384-dim normalized centroid
  "specificity": 0.95,
  "results": [
    {"title": "Logitech K380 Multi-Device Bluetooth Keyboard"},
    ...
  ]
}
```

### Products (Parquet)

Product titles with category and brand metadata, embedded with fine-tuned all-MiniLM-L6-v2.

### ESCI Evaluation

Cross-referenced with the Amazon Shopping Queries Dataset for external evaluation of retrieval quality.

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

Earlier pipeline versions included a rule filter and heuristic relevance scorer. These were measured and removed after finding the cross-encoder alone produces higher quality bags (mean CE score 0.743 vs 0.591 with heuristic pre-filtering).

### Fine-Tuning

The bag centroids serve as training targets for a query encoder:

- **Base model**: [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
- **Loss**: cosine distance between model output and bag centroid
- **Result**: cosine similarity to ground-truth centroids improved from 0.787 to 0.914

## Considerations

### Known Limitations

- **Model number sensitivity**: "iphone 6" may retrieve iPhone 7/8 products. MiniLM embeddings don't distinguish numeric identifiers well.
- **Category coverage**: current dataset covers a 20% random sample of the full catalog. Scaling to 100% requires more compute (see SCALING.md).
- **ESCI recall**: low for all models because top-50 retrieval from 6M products covers a small fraction of labeled products. Precision is the more meaningful metric.

### Ethical Considerations

- Product data is from a public academic dataset (McAuley Lab) intended for research use
- No user behavior data, personal information, or purchase history is included
- Query-product relevance judgments are from Amazon's public ESCI benchmark

## Citation

If you use this dataset, please cite:

```
@misc{tunkelang2026bagdocs,
  title={Bag-of-Documents: Product Search Dataset},
  author={Daniel Tunkelang and Aritra Mandal},
  year={2026},
  url={https://huggingface.co/datasets/dtunkelang/bag-of-documents}
}
```

### Related Work

- Tunkelang, D. [Modeling Queries as Bags of Documents](https://dtunkelang.medium.com/modeling-queries-as-bags-of-documents-b7d79d0916ab). 2024.
- Reddy, C.K. et al. [Shopping Queries Dataset: A Large-Scale ESCI Benchmark for Improving Product Search](https://arxiv.org/abs/2206.06588). KDD Cup 2022.
- McAuley Lab. [Amazon Reviews 2023](https://amazon-reviews-2023.github.io/).

## License

MIT
