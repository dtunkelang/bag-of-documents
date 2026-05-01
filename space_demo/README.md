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

Side-by-side comparison of four retrieval architectures on 1.2M Amazon ESCI products. Pick a mode for each column, run a query, see how the rankings differ. ESCI 22,458-query R@10 in parens.

- **Base MiniLM retrieval** (15.60) - dense baseline, no fine-tuning.
- **BM25 retrieval** (20.33) - bm25s with k1=0.3, b=0.6 (tuned for short keyword-stuffed product titles).
- **RRF(BM25, base)** - vanilla hybrid retrieval; on this corpus it actually loses to BM25 alone.
- **BM25 + 3-way ensemble rerank** (SOTA, 21.61) - BM25 top-50 reranked by three BoD-trained MiniLM encoders (sumsim fusion). +6.01pp over base MiniLM, +2.11pp over BM25 alone (tantivy default).

The SOTA pipeline is BM25 → three forward passes against precomputed product embeddings → average cosine, sort, return top-10. No HNSW in the inference path; ~40ms/query on commodity hardware.

- **Blog post**: [Distilling Retrieval Pipelines to a Single Embedding Model](https://dtunkelang.medium.com/distilling-retrieval-pipelines-to-a-single-embedding-model-606f3ecf0c91)
- **Model and data**: [huggingface.co/datasets/dtunkelang/bag-of-documents](https://huggingface.co/datasets/dtunkelang/bag-of-documents)
- **Code**: [github.com/dtunkelang/bag-of-documents](https://github.com/dtunkelang/bag-of-documents)
