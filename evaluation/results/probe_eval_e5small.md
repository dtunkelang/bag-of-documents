# Probe Set Hit@K Evaluation (102 queries, 3,143 judged candidates)

**Built 2026-05-27.** Hand-curated 102-query probe set across 14 archetypes, judged in-session by Claude Opus 4.7 (label scheme: 0=not relevant, 1=related/marginal, 2=relevant).

## Label distribution

| Label | Count | % |
|---|---:|---:|
| 0 (not relevant) | 368 | 11.7% |
| 1 (related/marginal) | 812 | 25.8% |
| 2 (relevant) | 1,963 | 62.5% |
| **Total** | **3,143** | 100% |

## Overall Hit@K

| Retriever | H@1 strict | H@1 lenient | H@5 strict | H@5 lenient | H@10 strict | H@10 lenient |
|---|---:|---:|---:|---:|---:|---:|
| bge_base | 71.6 | 96.1 | 93.1 | 99.0 | 93.1 | 99.0 |
| bm25 | 57.8 | 83.3 | 81.4 | 96.1 | 85.3 | 98.0 |
| e5_base | 77.5 | 96.1 | 92.2 | 99.0 | 94.1 | 100.0 |
| e5_small | 64.7 | 75.5 | 86.3 | 94.1 | 92.2 | 98.0 |
| te3_large_1024 | 70.6 | 95.1 | 94.1 | 99.0 | 96.1 | 99.0 |

## Per-archetype Hit@10 strict

| Archetype | N | bge_base | bm25 | e5_base | e5_small | te3_large_1024 |
|---|---:|---:|---:|---:|---:|---:|
| ambiguous | 6 | 100.0 | 100.0 | 83.3 | 100.0 | 100.0 |
| domain_role | 8 | 100.0 | 87.5 | 100.0 | 100.0 | 100.0 |
| generic_title | 10 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| geo_conditioned | 10 | 90.0 | 70.0 | 90.0 | 90.0 | 90.0 |
| jobstreet_vocab | 6 | 83.3 | 66.7 | 100.0 | 100.0 | 100.0 |
| long_nl | 6 | 100.0 | 33.3 | 100.0 | 83.3 | 83.3 |
| misspelled_casual | 6 | 83.3 | 66.7 | 83.3 | 66.7 | 100.0 |
| negation | 4 | 50.0 | 50.0 | 75.0 | 50.0 | 75.0 |
| seniority_modified | 8 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| skill_only | 8 | 100.0 | 100.0 | 100.0 | 87.5 | 100.0 |
| skill_stack | 8 | 87.5 | 87.5 | 87.5 | 87.5 | 87.5 |
| specific_title | 10 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| tail_rare | 6 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| usajobs_vocab | 6 | 83.3 | 100.0 | 83.3 | 100.0 | 100.0 |
