# Bag-of-Documents Product Search: Summary

*Work in progress — April 2026*

## What is this?

An end-to-end implementation of the [bag-of-documents](https://dtunkelang.medium.com/modeling-queries-as-bags-of-documents-b7d79d0916ab) model for e-commerce product search. A search query is represented not as a text embedding but as the distribution of its relevant products in embedding space. The centroid captures what the query *means* in product space, and the spread (specificity) captures how broad or narrow the query is.

## Dataset

- **6M unique Amazon products** across all 33 categories (20% random sample of the full 30M McAuley Lab catalog), sourced from the [Amazon Reviews 2023 dataset](https://amazon-reviews-2023.github.io/) (McAuley Lab, UCSD; data collected 1996–2023)
- **75K queries** from the [Amazon Shopping Queries Dataset](https://arxiv.org/abs/2206.06588) (ESCI, US locale) — real Amazon search queries spanning all product categories
- All products embedded with a fine-tuned all-MiniLM-L6-v2 (384-dim) and indexed in FAISS HNSW

## Pipeline

```
Query text
  → Hybrid retrieval: keyword (tantivy AND with relaxation) + embedding (FAISS)
  → Cross-encoder scoring: ESCI RoBERTa CE scores ALL candidates, threshold 0.3 (MPS GPU, batch 32)
  → Top 50 passing candidates → encode → bag centroid + specificity
  → Fine-tune MiniLM on (query text → bag centroid) pairs
  → Rebuild index with fine-tuned model (iterative refinement)
```

### What was removed and why

Earlier versions included a rule filter and a heuristic relevance scorer. Both were measured and removed:

- **Rule filter** (spec/capacity/wattage matching, junk detection): rejected only 0.1% of candidates in a 100-query sample. The CE handles these cases.
- **Heuristic relevance scorer** (4 signals: category accessory rate, "for" pattern, accessory nouns, brand match): actively hurt quality. It pre-filtered which candidates the CE could see, and the CE produced better bags when scoring all candidates directly (mean CE score 0.743 vs 0.591 with heuristic pre-filtering).

The pipeline is now: retrieval → CE → centroid. No hand-crafted heuristics.

## Key Results

**Broad catalog (6M products, 33 categories, 75K ESCI queries):**
- Cosine similarity to ground truth centroids: 0.787 → **0.914**
- Nearest-neighbor recall@10: 0.367 → **0.506**
- ESCI precision: 96.0% → **97.0%**
- Complement retrieval rate: 14.2% → **7.7%**

**Specificity correlates with query breadth.** "laptop" (0.70) < "hp laptop" (0.81) < "hp laptop 16gb ram" (0.84).

## Known Limitations

- **Model number sensitivity:** "iphone 6" retrieves iPhone 7/8; "macbook air 15 inch" retrieves 13-inch models. MiniLM embeddings don't distinguish numeric identifiers well.
- **20% sample:** Current dataset covers a random 20% of the full 30M product catalog. Some niche categories may be underrepresented.

## Interactive Demo

The web demo (`demo.py`) has two modes:

- **Default** — side-by-side: fine-tuned model vs base MiniLM, pure FAISS retrieval
- **Bag search** (checkbox, requires `--bag-search` flag) — simulates the offline pipeline: hybrid retrieval → CE scores all candidates → threshold 0.3 → centroid → FAISS re-retrieval

## Data and Model Sources

- **Product catalog**: [Amazon Reviews 2023](https://amazon-reviews-2023.github.io/) (McAuley Lab, UCSD; 1996–2023)
- **ESCI relevance judgments**: [Shopping Queries Dataset](https://arxiv.org/abs/2206.06588) (Reddy et al., KDD Cup 2022)
- **Cross-encoder**: [LiYuan/Amazon-Cup-Cross-Encoder-Regression](https://huggingface.co/LiYuan/Amazon-Cup-Cross-Encoder-Regression) — RoBERTa cross-encoder trained on ESCI data
- **Base embedding model**: [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) (Sentence Transformers)
- **Vector index**: [FAISS](https://github.com/facebookresearch/faiss) (Meta AI) — HNSW index
- **Keyword index**: [tantivy](https://github.com/quickwit-oss/tantivy) — Rust full-text search

## Architecture

```
Products → embed with fine-tuned MiniLM → FAISS HNSW index
Queries → hybrid retrieval (keyword + FAISS)
  → ESCI RoBERTa CE scores all candidates, keep those >= 0.3
  → top 50 → encode → bag centroid + specificity
Bags → fine-tune MiniLM (query text → centroid, MSE loss)
Fine-tuned model → rebuild index → iterate
```
