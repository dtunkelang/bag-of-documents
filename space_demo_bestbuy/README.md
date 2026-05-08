---
title: Bag Of Documents - BestBuy
emoji: 🛒
colorFrom: indigo
colorTo: pink
sdk: gradio
sdk_version: 6.12.0
app_file: app.py
pinned: false
---

# Bag-of-Documents on BestBuy click data

Side-by-side comparison of off-the-shelf MiniLM vs a bag-of-documents-fine-tuned MiniLM on the BestBuy 2012 ACM Hackathon clickthrough dataset.

- **Catalog**: 53,048 products — the subset of the full ~1.2M-SKU BestBuy 2012 catalog that appeared in queries with ≥2 distinct clicked SKUs (so each query has bag-trainable structure). Filtering done in [`download/prepare_bestbuy_acm.py`](https://github.com/dtunkelang/bag-of-documents/blob/main/download/prepare_bestbuy_acm.py).
- **Holdout test set**: 12,128 multi-positive queries with click-derived ground truth.
- **Training**: 48,516 query → clicked-SKU bags, MNRL fine-tuning of `all-MiniLM-L6-v2`.

## Headline result

| Model | R@10 (binary hit-rate) | E@1 |
|---|---:|---:|
| `all-MiniLM-L6-v2` (base) | 0.5559 | 0.2538 |
| BoD-trained (this work) | **0.7308** | **0.3718** |
| **Δ** | **+17.49pp** | **+11.80pp** |

The largest single-corpus BoD lift in the project. CHS predicted GREEN (SCHS=0.525) before training; this is the empirical confirmation.

## How to read the demo

- Type a query (or pick an example). The two columns show top-10 retrieved products under each model.
- For **holdout queries** (queries that appear in the test set), products that were actually clicked for that query in the original 2012 log are highlighted with **✅**, and the summary line above the columns shows how many clicked products each model surfaces in its top-10.
- The example queries are picked to illustrate distinct failure modes of off-the-shelf semantic search:
  - `ati` — brand abbreviation (Radeon graphics cards) base doesn't recognize
  - `dvd storage` — abstract intent (cases/wallets, not DVDs themselves)
  - `turtlebeach` — joined brand name (headsets)
  - `i pad 2` — spaced tokenization (iPad 2)
  - `iphone 4 incase` — InCase brand vs preposition

## Links

- **Code**: [github.com/dtunkelang/bag-of-documents](https://github.com/dtunkelang/bag-of-documents)
- **Companion ESCI demo**: [huggingface.co/spaces/dtunkelang/bag-of-documents-demo](https://huggingface.co/spaces/dtunkelang/bag-of-documents-demo)
- **Source dataset**: [Kaggle ACM SF Chapter Hackathon (2012)](https://www.kaggle.com/competitions/acm-sf-chapter-hackathon-big)
