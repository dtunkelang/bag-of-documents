# Probe Set Hit@K Evaluation (102 queries, 2,630 judged candidates)

**Built 2026-05-27.** Hand-curated 102-query probe set across 14 archetypes, judged in-session by Claude Opus 4.7 (label scheme: 0=not relevant, 1=related/marginal, 2=relevant).

## Label distribution

| Label | Count | % |
|---|---:|---:|
| 0 (not relevant) | 288 | 11.0% |
| 1 (related/marginal) | 655 | 24.9% |
| 2 (relevant) | 1,687 | 64.1% |
| **Total** | **2,630** | 100% |

## Overall Hit@K

| Retriever | H@1 strict | H@5 strict | H@10 strict | H@1 lenient | H@5 lenient | H@10 lenient |
|---|---:|---:|---:|---:|---:|---:|
| bm25 | 57.8 | 81.4 | 85.3 | 83.3 | 96.1 | 98.0 |
| te3_large_1024 | 70.6 | 94.1 | **96.1** | 95.1 | 99.0 | 99.0 |
| e5_base | **77.5** | 92.2 | 94.1 | **96.1** | 99.0 | **100.0** |

**Headline:** e5_base wins on Hit@1 (77.5 vs te3's 70.6). te3 wins on Hit@10 strict (96.1 vs 94.1). On lenient labels, e5_base reaches 100% Hit@10 — i.e., every probe query has at least one related result in its top-10. This **inverts the synth-eval finding** where te3-large dominated at R@10=70.5%.

## Per-archetype Hit@10 strict

| Archetype | N | bm25 | te3_large | e5_base |
|---|---:|---:|---:|---:|
| ambiguous | 6 | 100.0 | 100.0 | 83.3 |
| domain_role | 8 | 87.5 | 100.0 | 100.0 |
| generic_title | 10 | 100.0 | 100.0 | 100.0 |
| geo_conditioned | 10 | 70.0 | 90.0 | 90.0 |
| jobstreet_vocab | 6 | 66.7 | 100.0 | 100.0 |
| long_nl | 6 | 33.3 | 83.3 | 100.0 |
| misspelled_casual | 6 | 66.7 | 100.0 | 83.3 |
| negation | 4 | 50.0 | 75.0 | 75.0 |
| seniority_modified | 8 | 100.0 | 100.0 | 100.0 |
| skill_only | 8 | 100.0 | 100.0 | 100.0 |
| skill_stack | 8 | 87.5 | 87.5 | 87.5 |
| specific_title | 10 | 100.0 | 100.0 | 100.0 |
| tail_rare | 6 | 100.0 | 100.0 | 100.0 |
| usajobs_vocab | 6 | 100.0 | 100.0 | 83.3 |

**Where BM25 fails hardest:** long_nl (33.3), negation (50.0), jobstreet_vocab/misspelled_casual (66.7), geo_conditioned (70.0). Long natural-language queries and dense vocabulary are dense-retriever territory.

**Where dense retrievers don't break BM25:** generic_title, specific_title, skill_only, seniority_modified, tail_rare — easy queries where lexical alone is sufficient.

**Where e5_base loses to te3:** ambiguous, misspelled_casual, usajobs_vocab (each 83.3 vs 100). e5_base struggles with vocabulary noise that te3 handles cleanly.

## Methodology notes

- Candidates: 3-retriever top-10 dedup → 2,630 (query, doc) pairs, mean 25.8/query.
- Labels emitted in batches of 100 by Claude Opus 4.7 reading title + body[:200] of each candidate.
- For negation queries: label 0 if the negated term IS present in the doc.
- For misspelled queries: treat as intended spelling.
- For ambiguous queries (e.g., "python"): any role primarily requiring that skill = 2.
