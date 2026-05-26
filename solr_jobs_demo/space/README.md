---
title: Jobs Search Demo (Solr)
emoji: 🔎
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
license: cc-by-4.0
short_description: 348K jobs · RRF(BM25, bge-small, te3) via Solr 10
---

# Jobs Search Demo — Solr backend

Solr 10 port of the [jobs-demo](https://huggingface.co/spaces/dtunkelang/jobs-demo) default retrieval strategy:
**RRF(BM25, bge-small, te3-large-cached)** over 347,900 jobs across 4 corpora
(jobs_data, LinkedIn, JobStreet, USAJobs).

Same fusion formula (k=60, pool=100) as the in-memory FastAPI demo. Solr's
BM25 differs slightly from `bm25s` due to tokenizer/stopword differences;
empirical overlap@10 with the in-memory demo is ~9.5/10 on a 6-query test set.

Solr lives internally on 8983; only the FastAPI shim on 7860 is exposed.
On cold start the container pulls the prebuilt index tarball
(~2.7 GB) from the companion [dataset](https://huggingface.co/datasets/dtunkelang/jobs-demo)
under `solr_index/`, extracts it, then starts Solr + the shim.
