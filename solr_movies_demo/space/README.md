---
title: Movies Search Demo (Solr)
emoji: 🎬
colorFrom: indigo
colorTo: pink
sdk: docker
app_port: 7860
pinned: false
license: cc-by-4.0
short_description: 1.27M movies/TV · RRF(BM25, bge-small-en-v1.5) via Solr 10
---

# Movies Search Demo — Solr backend

Solr 10 port of the BoD movies/TV demo: **RRF(BM25, bge-small-en-v1.5)** over
1,272,422 titles from IMDb + Wikidata + Wikipedia + MovieLens, with the BGE
KNN lane covering the 199,991 titles that have a Wikipedia lead.

Lanes:

- **BM25** — edismax over `title^6 original_title^4 cast_names^3 director_names^3 genres^2 lead plot^0.5`, log-votes popularity boost.
- **KNN** — `{!knn f=bge_vec topK=...}` over 384-dim bge-small-en-v1.5 cosine vectors.
- **Hybrid** — weighted RRF (k=60, w_bm25=1, w_knn=2) with soft gating that boosts KNN further on high-recall lexical queries.

Facets (`genres`, `decade`, `type`, `has_bag`) use `{!tag/ex}` so each facet
keeps siblings visible while filtered.

Solr lives internally on 8983; only the FastAPI shim on 7860 is exposed.
On cold start the container pulls the prebuilt index tarball from the
companion [dataset](https://huggingface.co/datasets/dtunkelang/movies-demo)
under `solr_index/`, extracts it, then starts Solr + the shim.
