---
title: Jobs Search Demo
emoji: 🔎
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
license: cc-by-4.0
short_description: Hybrid search over ~197K job postings · RRF(BM25 + e5-small)
---

# Jobs Search Demo

Hybrid job search over **~197,000 postings** from two corpora (OpenApply, USAJobs).

## Retrieval

The default strategy is **RRF(BM25, e5-small-v2)** — reciprocal-rank fusion (k=60)
of a lexical BM25 lane and a dense `intfloat/e5-small-v2` lane (384-dim, asymmetric
query/passage prefixes), each pooled to the top 100.

## Features

- **Faceted search** — role family, seniority, industry, remote mode, location,
  posted date, salary band, and tech stack, with employer-diversity capping.
- **Typeahead + related searches** — a curated query corpus plus e5-embedding
  narrow/lateral role suggestions.
- **More jobs like this** — pivot from any posting to similar roles via a
  re-embedded title + description, RRF-fused and employer-diversified.
- **Match your profile** — paste or upload a resume / LinkedIn PDF to rank jobs by
  fit (3-axis constraint filter).
- **Personalized re-ranking** — a profile re-ranks every query via a candidate-
  prefiltered KNN over the e5 vectors.

## Hosting

Retrieval is served by **Solr 10** running internally on port 8983; only the
FastAPI shim on 7860 is exposed. On cold start the container pulls the prebuilt
index tarball (~0.95 GB) from the companion
[dataset](https://huggingface.co/datasets/dtunkelang/jobs-demo) at
`solr_index/solr_jobs_core.tar`, extracts it, then starts Solr and the shim.
