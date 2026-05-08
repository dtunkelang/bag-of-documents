---
license: cc-by-nc-sa-4.0
language:
- en
tags:
- bag-of-documents
- retrieval
- bestbuy
- click-data
- esci
pretty_name: Bag-of-Documents BestBuy Demo Artifacts
---

# Bag-of-Documents on BestBuy Click Data — Demo Artifacts

Backing data for the [BestBuy BoD demo Space](https://huggingface.co/spaces/dtunkelang/bag-of-documents-bestbuy-demo). The Space pulls these files at startup via `huggingface_hub.snapshot_download`. Available here so anyone can reproduce the demo, replicate the evaluation, or build their own UI on top.

## What's here

| File | Size | Description |
|---|---:|---|
| `titles.json` | 47 MB | 1,274,801 BestBuy product titles (full 2012 catalog from the Kaggle ACM Hackathon archive). |
| `product_ids.json` | 14 MB | Parallel list of SKU strings, same order as `titles.json`. |
| `holdout_queries.jsonl` | 0.6 MB | 12,128 multi-positive holdout queries (20% split of the 60K queries with ≥2 distinct clicked SKUs). |
| `holdout_qrels.jsonl` | 5 MB | 78,635 click-derived (query_id, product_id, relevance=1) rows for the holdout. Used for the demo's ✅ ground-truth highlighting. |
| `base_catalog.vecs.fp16.npy` | 934 MB | All 1,274,801 titles encoded with `all-MiniLM-L6-v2` (base) — `(1274801, 384) float16`. |
| `bod_catalog.vecs.fp16.npy` | 934 MB | All 1,274,801 titles encoded with the BoD-fine-tuned MiniLM (this work) — `(1274801, 384) float16`. |
| `query_model_bestbuy_bod/` | 90 MB | The fine-tuned `sentence-transformers` model. Load with `SentenceTransformer("…/query_model_bestbuy_bod")`. |

## Headline result

12,128-query holdout against the full 1,274,801-product catalog:

| Model | R@10 (binary hit-rate) | E@1 |
|---|---:|---:|
| `all-MiniLM-L6-v2` (base) | 0.3238 | 0.0926 |
| BoD-trained (this work) | **0.5013** | **0.1589** |
| **Δ** | **+17.75pp** | **+6.63pp** |

Largest single-corpus BoD lift in the broader [bag-of-documents project](https://github.com/dtunkelang/bag-of-documents).

## Provenance

- Source: [Kaggle ACM SF Chapter Hackathon (BestBuy clickthrough, 2012)](https://www.kaggle.com/competitions/acm-sf-chapter-hackathon-big). Catalog and click logs are © Best Buy and licensed for the original hackathon. Re-distribution here is for demo / educational use under the spirit of the competition's data-sharing terms; if you're a Best Buy rights-holder and want this taken down, open an issue on the [GitHub repo](https://github.com/dtunkelang/bag-of-documents).
- Pipeline: `download/prepare_bestbuy_acm.py` → `download/build_bestbuy_bags.py` → `training/finetune_with_hardnegs.py` (in the [GitHub repo](https://github.com/dtunkelang/bag-of-documents)). Full-catalog re-encoding via `download/expand_bestbuy_catalog.py`.

## License

Code is MIT (see the GitHub repo). The catalog text and click data inherit Best Buy's original license; this dataset is licensed CC BY-NC-SA 4.0 for everything else (encodings, derived bags, model). Use accordingly.
