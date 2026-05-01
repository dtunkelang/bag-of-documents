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

- **BM25 + 3-way ensemble rerank** (default right, 21.61%) - bm25s (k1=0.3, b=0.6) retrieves top-50, three BoD-trained encoders rerank via sumsim fusion. Current SOTA. The third encoder is trained on ESCI labels directly (E as positives, I as hardnegs); it adds orthogonal signal to the bag-trained pair. Tuning BM25 from Lucene defaults (k1=1.2, b=0.75) to short-keyword-doc-friendly (0.3, 0.6) added +0.29pp on top of the prior tantivy-based SOTA (21.32%).
- **BM25 + 2-way ensemble rerank** (21.27%) - same retrieval, two-encoder rerank. Kept for comparison.
- **BM25 + MNRL hybrid + ensemble rerank** (20.01%) - RRF-fuses BM25 and 6M-MNRL retrieval, then ensemble-reranks. Adding MNRL retrieval to the candidate pool actually hurts.
- **MNRL + BoD ensemble rerank** (19.83%) - 6M-MNRL retrieves top-100, then ensemble rerank.
- **BM25 retrieval** (20.33%) - bm25s (k1=0.3, b=0.6) alone, no dense, no rerank.
- **Base + BoD ensemble rerank** (19.00%) - the same reranker stack on plain MiniLM retrieval.
- **RRF(BM25, base) hybrid retrieval** (18.62%) - non-BoD hybrid baseline. Underperforms BM25 alone.
- **MNRL retrieval (no rerank)** (18.10%) - 6M-MNRL alone.
- **Fine-tuned retrieval (cosine BoD)** - the originally deployed BoD-as-retriever; kept for historical comparison.
- **Base MiniLM retrieval** (default left, 15.60%) - no fine-tuning, no rerank.

The 3-way BoD rerank stack on bm25s candidates buys **+2.11pp R@10 over BM25 alone (tantivy default)**, **+2.99pp over the strongest non-BoD baseline (RRF hybrid)**, and **+6.01pp over base MiniLM**. The 3-way ensemble adds **+0.21pp R@10** over the 2-way by including an ESCI-label-supervised third encoder. The bm25s retriever swap (vs tantivy at default params) adds another **+0.29pp R@10 / +0.47pp E@1** on top — keyword-stuffed Amazon titles want early term-frequency saturation (k1=0.3) and moderate length normalization (b=0.6). Precomputed product embeddings keep dense modes at sub-100ms; BM25 is faster still.

- **Blog post**: [Distilling Retrieval Pipelines to a Single Embedding Model](https://dtunkelang.medium.com/distilling-retrieval-pipelines-to-a-single-embedding-model-606f3ecf0c91)
- **Model and data**: [huggingface.co/datasets/dtunkelang/bag-of-documents](https://huggingface.co/datasets/dtunkelang/bag-of-documents)
- **Code**: [github.com/dtunkelang/bag-of-documents](https://github.com/dtunkelang/bag-of-documents)
