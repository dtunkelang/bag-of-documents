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

- **BM25 + MNRL hybrid + ensemble rerank** (default right, 20.01%) — RRF-fuses BM25 and 6M-MNRL retrieval, then ensemble-reranks with two BoD-trained encoders. Current SOTA.
- **MNRL + BoD ensemble rerank** (19.83%) — 6M-MNRL retrieves top-100, two BoD models reorder via sumsim fusion.
- **BM25 retrieval** (19.50%) — tantivy en_stem alone, no dense, no rerank. Surprisingly competitive on entity-heavy product queries.
- **Base + BoD ensemble rerank** (19.00%) — the same reranker stack on plain MiniLM retrieval.
- **MNRL retrieval (no rerank)** (18.10%) — 6M-MNRL alone.
- **Fine-tuned retrieval (cosine BoD)** — the originally deployed BoD-as-retriever; kept for historical comparison.
- **Base MiniLM retrieval** (default left, 15.60%) — no fine-tuning, no rerank.

Precomputed product embeddings keep dense modes at sub-100ms; BM25 is faster still.

- **Blog post**: [Distilling Retrieval Pipelines to a Single Embedding Model](https://dtunkelang.medium.com/distilling-retrieval-pipelines-to-a-single-embedding-model-606f3ecf0c91)
- **Model and data**: [huggingface.co/datasets/dtunkelang/bag-of-documents](https://huggingface.co/datasets/dtunkelang/bag-of-documents)
- **Code**: [github.com/dtunkelang/bag-of-documents](https://github.com/dtunkelang/bag-of-documents)
