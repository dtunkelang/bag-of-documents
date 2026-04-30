---
title: Bag Of Documents Demo
emoji: 📉
colorFrom: indigo
colorTo: pink
sdk: gradio
sdk_version: 6.12.0
app_file: app.py
pinned: false
---

# Bag-of-Documents Product Search Demo

Compare retrieval / rerank architectures on 1.2M Amazon ESCI products (75K real Amazon search queries). Each column has its own mode dropdown so any two architectures can be compared on the same query.

ESCI 22,458-query R@10 in parens:

- **BM25 + ensemble rerank** (default right, 21.11%) — tantivy BM25 retrieves top-100, two BoD-trained encoders rerank via sumsim fusion. Current SOTA. No HNSW index in the inference path.
- **BM25 + MNRL hybrid + ensemble rerank** (20.01%) — RRF-fuses BM25 and 6M-MNRL retrieval, then ensemble-reranks. Previously shipped; superseded since MNRL retrieval was found to dilute the candidate pool.
- **MNRL + BoD ensemble rerank** (19.83%) — 6M-MNRL retrieves top-100, then ensemble rerank.
- **BM25 retrieval** (19.50%) — tantivy en_stem alone, no dense, no rerank.
- **Base + BoD ensemble rerank** (19.00%) — the same reranker stack on plain MiniLM retrieval.
- **MNRL retrieval (no rerank)** (18.10%) — 6M-MNRL alone.
- **RRF(BM25, base) hybrid retrieval** (18.62%) — non-BoD hybrid baseline (no fine-tuning, no rerank). Underperforms BM25 alone — the dense lane displaces BM25's exact-match top-1 with semantically-similar near-misses.
- **Fine-tuned retrieval (cosine BoD)** — the originally deployed BoD-as-retriever; kept for historical comparison.
- **Base MiniLM retrieval** (default left, 15.60%) — no fine-tuning, no rerank.

The BoD rerank stack buys **+1.61pp R@10 over BM25 alone**, **+2.49pp over the strongest non-BoD baseline (RRF hybrid)**, and **+5.51pp over base MiniLM**. Precomputed product embeddings keep dense modes at sub-100ms; BM25 is faster still.

- **Blog post**: [Distilling Retrieval Pipelines to a Single Embedding Model](https://dtunkelang.medium.com/distilling-retrieval-pipelines-to-a-single-embedding-model-606f3ecf0c91)
- **Model and data**: [huggingface.co/datasets/dtunkelang/bag-of-documents](https://huggingface.co/datasets/dtunkelang/bag-of-documents)
- **Code**: [github.com/dtunkelang/bag-of-documents](https://github.com/dtunkelang/bag-of-documents)
