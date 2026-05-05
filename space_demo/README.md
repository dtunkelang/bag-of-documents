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
- **BM25 + 3-way ensemble rerank + spell-correct** (fast SOTA, 21.84) - catalog-vocab spell correction (pyspellchecker over the ~172K-token title vocabulary) + BM25 top-50 reranked by three BoD-trained MiniLM encoders. Spell correction lifts R@10 +0.23pp / E@1 +0.42pp (sig.) at no latency cost. ~50ms/query.
- **BM25 + sumsim + LiYuan + BGE** (quality SOTA, 23.57; E@1 47.95) - weighted 3-way fusion (sumsim 0.4, LiYuan 0.2, BGE 0.4) of sumsim (3 bi-encoders), the LiYuan ESCI cross-encoder, and BGE-reranker-v2-m3 (568M-param XLM-RoBERTa-large reranker). All three streams are per-query min-max normalized then averaged. +1.49pp R@10, +5.28pp E@1 over the fast SOTA. ~5-15s/query on Space CPU.

The fast SOTA does three forward passes against precomputed product embeddings then averages cosine — sub-100ms wall-clock. The quality SOTA adds 200 cross-encoder forward passes (100 LiYuan + 100 BGE-reranker) over the BM25 top-100, then weighted-averages all three streams (sumsim 0.4, LiYuan 0.2, BGE 0.4) (sumsim, LiYuan, BGE) per-query min-max normalized. The CE forward passes dominate latency on CPU.

- **Blog post**: [Distilling Retrieval Pipelines to a Single Embedding Model](https://dtunkelang.medium.com/distilling-retrieval-pipelines-to-a-single-embedding-model-606f3ecf0c91)
- **Model and data**: [huggingface.co/datasets/dtunkelang/bag-of-documents](https://huggingface.co/datasets/dtunkelang/bag-of-documents)
- **Code**: [github.com/dtunkelang/bag-of-documents](https://github.com/dtunkelang/bag-of-documents)
