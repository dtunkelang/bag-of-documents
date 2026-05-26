---
title: Jobs Search Demo (348K postings, 4 corpora)
emoji: 💼
colorFrom: indigo
colorTo: green
sdk: gradio
sdk_version: 6.12.0
app_file: app.py
pinned: false
---

# Unified Jobs Search: 348K postings across 4 corpora

Side-by-side retriever comparison demo over a unified jobs corpus assembled from four sources: Open-Apply (OAP), LinkedIn (LI), JobStreet (JS), and USAJobs (USA). Pick any two retrievers; type a query; compare top-10 results.

- **Corpus**: 347,900 job postings (titles, locations, employers, salaries, descriptions).
- **Retrievers**:
  - `bm25` — bm25s with English stemming over titles
  - `bge_small` — `BAAI/bge-small-en-v1.5` dense retrieval (local, no API)
  - `te3_cached` — OpenAI `text-embedding-3-large` @ 1024d, with a pre-encoded query cache of ~196k popular roles/locations (no live API calls — falls back to BM25 when the query isn't cached)
  - Hybrids: RRF, cascade (BM25 → dense rerank), and weighted-sum (0.5·BM25 + 0.5·dense)
- **Source tag** (OAP/LI/JS/USA) on each result row shows which corpus it came from.
- **Click "description"** on any result to expand the full posting text.

## Why this exists

Part of the Bag-of-Documents research track ([code](https://github.com/dtunkelang/bag-of-documents)). The jobs corpus is the project's first multi-source, multi-language search testbed; this Space lets colleagues kick the tires on retrieval quality across 4 different jobs feeds with no API setup.

## Architecture notes

- All retrieval runs in-process (BM25 index built at startup; dense catalogs are mmap'd fp16).
- The te3 query cache (~196k pre-encoded queries) makes te3-large retrieval free at query time, but only for queries we anticipated. The cache was built from title vocabulary, role+location combinations, autocomplete-derived expansions, and LLM-synthesized tail queries.
- The 3.5 GB of catalog vectors + metadata live in a companion HF dataset and are snapshot-downloaded at startup (Space file-size limits make this the cleanest split).

## Companion artifacts

- **Code**: [github.com/dtunkelang/bag-of-documents](https://github.com/dtunkelang/bag-of-documents)
- **Companion ESCI demo**: [bag-of-documents-demo](https://huggingface.co/spaces/dtunkelang/bag-of-documents-demo)
- **Companion BestBuy demo**: [bag-of-documents-bestbuy](https://huggingface.co/spaces/dtunkelang/bag-of-documents-bestbuy)
