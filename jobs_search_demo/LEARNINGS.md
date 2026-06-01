# Jobs demo — learnings

What we know after building the unified-jobs catalog (347.9k postings) + Solr demo + HF Space. Captures the non-obvious stuff a reader can't recover from the code.

## Retrieval

- **Open-weight ranking on the judged probe (Hit@1)**: `e5_base` 77.5 > `bge_base` 71.6 > `te3` 70.6 > `bge_small` (training source). e5-family wins H@1; te3 still wins H@10. Synth-eval head-of-list is inverted vs the judged probe — synth-eval reflects retrieval over the LLM-distilled query distribution, judged probe reflects human-relevance on the real query distribution. **Trust the judged probe for ranking decisions.**
- **e5-small ship**: +6.9pp probe H@1 over bge-small, +4.66pp R@10 over bge-small on synth eval. Same parameter budget, drop-in replacement, no training. Ship-it move.
- **te3-large**: best aggregate R@10 on synth, but only reachable for the ~39k known queries cached in `te3_cache_canonical.json`. For tail queries we serve open-weights.
- **RRF(BM25, e5_small)** = R@10 0.6802 on synth (ties RRF(BM25, e5_base)). Cuts te3 gap to 2.9pp from 5.3pp. **3-way fusion with the te3 cache is untested** — could close more of the gap on the cached lane.
- **BoD-as-retriever collapses on jobs.** bge-small BoD trained on linkedin (its source) still loses badly to base bge-small on cross-corpus probe. Don't ship BoD here.
- **Rocchio PRF**: strictly negative on all 4 retrievers (commit a8a0aab). Don't revisit without a different signal.
- **gte-base**: MPS crash, abandoned.

## Facets

- **Position-weighted facet sort > raw count.** Raw-count sort makes the head dominant on every query; position-weighting (rank within the result list, decayed) makes tail values surface for narrow queries. Same impact-vs-count tradeoff facet UIs have everywhere.
- **Role-modifier normalizer.** Strip `(general|district|area|regional|...|assistant|senior|...)` between an industry anchor and `manager|supervisor|director|coordinator|lead`. Lets a single `restaurant manager` pattern catch the whole `restaurant <modifier> manager` family. Cut role_family `other` by ~1pp on the full corpus.
- **Industry-from-`[Category:]` description prefix.** When the source ATS hands us its category (the `[Category:]` prefix Seek-family feeds use), prefer it over title regex. Flipped construction-engineer SWE/trades split from 96/88 to 25/150 (commit e75cd8d).
- **The `other` residual is mostly correct.** After 4 rounds of audit→pattern→regen, the remaining ~28% `other` is dominated by titles that are genuinely ambiguous without the description context — "manager part time", "consultant part time", "R&D manager", "non-profit developer". These mostly need full-text classification, not more regex.

## Industry labels (slug → industry)

- **Local-embedding classifiers plateau at macro-F1 ~0.21** (TE3, bge, char-tfidf all converge). Coverage past 68% needs **more labels**, not better features. Both 3B/7B local LLMs miss the 80% quality bar.
- **TF-IDF + slug+titles + margin gate > BGE-centroid propagation.** BGE-centroid collapses to a consulting-attractor on this corpus.
- **Self-labeling in-conversation works at small scale.** Tail-2 (1.5k slugs) self-labeled by Claude lifted Solr coverage 68% → 78%. The first 1.5k high-yield slugs are tractable manually; past that, return-on-time drops.
- **Staffing/employment-agency override.** Per-job override on slug detects staffing employers and routes industry away from "staffing" to the inferred destination industry. Necessary because staffing dominates raw slug counts.

## Operational gotchas (jobs-specific; first three generalize)

- **Solr atomic update silently wipes non-stored fields.** Using `_doc_update_:set` on facet fields blew away the (unstored) `title` field. **Use full re-push via `push_docs.py`** — there is no atomic-update path that's safe for this schema. Generalizes to any Solr index with unstored fields.
- **HF upload stall at 99-100%.** `huggingface_hub.upload_file` hangs at 99-100% for 30-45 min. TaskStop + retry the same call finishes in 2-5 min via server-side LFS dedup. Generalizes to any HF dataset/space tarball push.
- **HF Space file path matters from subdirectories.** Relative `.venv` paths fail when invoked from a subdirectory of the repo. Use absolute paths.
- **te3 query cache (~64MB) served locally.** The HF Space can re-hydrate from the dataset tarball, but for dev iteration the cache lives at `unified_jobs/te3_cache_canonical.json` and `_stage/` mirrors it for fast restart.
- **`unify` vs `_with_facets` push_docs.py duplicates.** Two scripts existed; `_with_facets` was renamed canonical (commit 0bc8be2). If you find a `push_docs.py.bak` or similar, it's the retired path.

## Eval

- **Synth-eval (LLM-distilled queries) and judged-probe (human-relevance) disagree at the head of the ranking.** Both are useful, but they answer different questions. Use synth for coarse retriever ranking (R@10 over the full distribution), judged probe for ship-decisions on the head.
- **102-query probe set, ~2,630 candidates labeled.** Pooling adds new candidates whenever a new retriever joins; labeling is human-only (no OpenAI dependency).

## Patterns that generalize to other domains

These hold up beyond jobs. Pull from this section first when building the next Solr demo (movies is the active second corpus).

1. **Position-weighted facet sort.** Any facet UI on top of Solr/Elastic benefits. Sort by `sum(decay(rank_i))` over the result list, not `count`. Head values stop swallowing tail values for narrow queries.
2. **Source-category override before regex.** If the source feed has a structured category column, prefer it over title-derived regex classification — even noisy categories beat regex on ambiguous titles. The regex is the fallback, not the primary.
3. **Audit-driven heuristic refinement.** `audit_other_in_titles.py` queries Solr for phrases that dominantly land in `other` but have a viable non-`other` runner-up. Each audit round picks the top-N by impact, writes patterns, regenerates, re-audits. 4 rounds got us from 31.8% → 27.96% on the held-out corpus. Reusable across any taxonomy + corpus pair.
4. **Modifier normalization.** Strip rank/scope tokens (general/district/area/regional/senior/junior/...) between an industry anchor and a role token. Cuts pattern combinatorics 5-10x and reduces miss rate on the same family.
5. **Solr atomic update is unsafe with unstored fields.** Full re-push for facet refresh. Across any Solr deployment.
6. **HF upload stall + retry.** Stop and retry — server dedup makes the second call fast. Across any HF tarball workflow.
7. **Synth-eval vs judged-eval disagreement.** Build both; trust the judged eval for ship decisions; use synth eval for retriever-shape sanity-check. Generalizes wherever you have an LLM-distilled query distribution and a separate human-judged probe.
8. **Local-embedding classifiers plateau on small label budgets.** Past a few hundred labels, more labels >> better features. True wherever you're building a slug/category classifier with <2k labels and no transformer fine-tune.
9. **In-conversation self-labeling for the top-N tail.** Effective at ~1-2k items, breaks down past that. Use when an LLM API budget isn't available.

## What's queued (post-2026-05-27)

- 3-way RRF (BM25 + e5_small + te3) on the cached-query lane.
- te3-distillation with **listwise** loss (cosine-distill collapsed at R@10 0.03 despite cos=0.79 pointwise).
- Industry tail-3 (lift coverage past 78%).
- Re-judge probe with e5-small in the pool (shipped after probe complete; not in current Hit@K table).
