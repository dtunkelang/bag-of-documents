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

Side-by-side comparison of five retrieval architectures on 1.2M Amazon ESCI products. Pick a mode for each column, run a query, see how the rankings differ. ESCI 22,458-query R@10 in parens.

- **Base MiniLM retrieval** (15.60) - dense baseline, no fine-tuning.
- **BM25 retrieval** (20.33) - bm25s with k1=0.3, b=0.6 (tuned for short keyword-stuffed product titles).
- **RRF(BM25, base)** - vanilla hybrid retrieval; on this corpus it actually loses to BM25 alone.
- **BM25 + 3-way ensemble rerank** (fast SOTA, 21.61) - BM25 top-50 reranked by three BoD-trained MiniLM encoders (sumsim fusion). ~50ms/query.
- **BM25 + 3-way + CE fusion** (quality SOTA, 22.22; E@1 44.74) - adds the LiYuan ESCI cross-encoder, fused at w_ce=0.25 with the 3-way sumsim via per-query min-max normalization. +0.61pp R@10, +2.63pp E@1 over the fast SOTA. ~1-3s/query on Space CPU.

The fast SOTA does three forward passes against precomputed product embeddings then averages cosine — sub-100ms wall-clock. The quality SOTA adds 50 cross-encoder forward passes (full attention, ESCI-supervised) for a meaningful E@1 lift on near-miss queries.

- **Blog post**: [Distilling Retrieval Pipelines to a Single Embedding Model](https://dtunkelang.medium.com/distilling-retrieval-pipelines-to-a-single-embedding-model-606f3ecf0c91)
- **Model and data**: [huggingface.co/datasets/dtunkelang/bag-of-documents](https://huggingface.co/datasets/dtunkelang/bag-of-documents)
- **Code**: [github.com/dtunkelang/bag-of-documents](https://github.com/dtunkelang/bag-of-documents)
