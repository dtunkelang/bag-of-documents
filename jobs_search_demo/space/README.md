---
title: Job Search Demo
emoji: 🔎
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
license: cc-by-4.0
short_description: Multilingual hybrid search over ~340K job postings
---

# Job Search Demo

Hybrid job search over **~340,000 postings** from 15 sources — ATS crawls
(OpenApply, SmartRecruiters, Workable, Recruitee, Breezy), public/federal feeds
(USAJobs, France Travail, JobTech/Sweden), aggregators (Adzuna, Jooble), and job
boards (Reed, Findwork, The Muse, RemoteOK). Postings span 7 languages
(English, French, Swedish, Spanish, German, Italian, Dutch).

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
  fit (4-axis constraint filter: field, seniority, location, qualification).
- **Personalized re-ranking** — a profile re-ranks every query via a candidate-
  prefiltered KNN over the e5 vectors.
- **Snippets** — each result shows the passage most semantically similar to the
  query (e5 cosine over pre-computed passage vectors), with lexical term highlighting.
- **Multilingual UI** — a masthead language picker localizes the entire interface
  (7 languages: English, French, German, Dutch, Spanish, Swedish, Italian) and
  lightly promotes same-language postings on a blank browse. Defaults to English.
- **View original posting** — results link out to the source posting where the
  URL is available (~91% of the corpus; recovered without re-crawling).

## Hosting

Retrieval is served by **Solr 10** running internally on port 8983; only the
FastAPI shim on 7860 is exposed. On cold start the container pulls the prebuilt
index tarball (~3.4 GB; includes stored passage vectors for snippets) from the companion
[dataset](https://huggingface.co/datasets/dtunkelang/jobs-demo) at
`solr_index/solr_jobs_core.tar`, extracts it, then starts Solr and the shim.
