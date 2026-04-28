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

Compare the **base MiniLM** model with two BoD architectures on 1.2M Amazon ESCI products (75K real Amazon search queries):

- **Fine-tuned retrieval** — original BoD-as-retriever: a single fine-tuned query encoder + FAISS, in one shot.
- **Base + BoD ensemble rerank** — base MiniLM retrieves top-100, two BoD-trained query encoders (full-6M MNRL + qrels-hardneg) independently rank candidates, and a sumrank fusion produces the final top-K. **+2.75pp R@10** over base alone on the full ESCI test set (15.60% → 18.35%; nDCG@10 0.2648 → 0.3139). Precomputed product embeddings keep per-query latency sub-100ms.

Pick the right-column mode from the dropdown to compare each BoD architecture against base on the same query.

- **Blog post**: [Distilling Retrieval Pipelines to a Single Embedding Model](https://dtunkelang.medium.com/distilling-retrieval-pipelines-to-a-single-embedding-model-606f3ecf0c91)
- **Model and data**: [huggingface.co/datasets/dtunkelang/bag-of-documents](https://huggingface.co/datasets/dtunkelang/bag-of-documents)
- **Code**: [github.com/dtunkelang/bag-of-documents](https://github.com/dtunkelang/bag-of-documents)
