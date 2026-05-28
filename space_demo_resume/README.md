---
title: Resume to Job Matching (constraint-aware)
emoji: 🧭
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 6.12.0
app_file: app.py
pinned: false
---

# Resume → Job Matching: cosine vs. a 3-axis hard-constraint filter

Browse 6,904 synthetic resumes; click any one to see its top matching jobs **side-by-side** as raw embedding cosine vs. a constraint-aware re-rank.

The point this surfaces: **dense cosine retrieval is constraint-blind.** It happily ranks a senior role #1 for a junior candidate, an on-site Sydney job #1 for a candidate who needs remote, or a role that gates on a credential the candidate doesn't hold. A judge-free re-rank that filters on three hard axes fixes this nearly for free.

- **Corpus**: 347,900 job postings (unified Open-Apply / LinkedIn / JobStreet / USAJobs feed).
- **Resumes**: 6,904 synthetic LinkedIn-style profiles (headline, location, experience, education, skills).
- **Encoder**: `intfloat/e5-base-v2` (resume + job vectors precomputed; no model runs at serve time).

## The three axes

Each candidate job carries `sen` / `loc` / `gate` ✓/✗ badges:

- **sen** — seniority: candidate's level vs. the job's required level (no over- or under-qualification).
- **loc** — location: candidate's location/remote-need vs. the job's location and remote flag.
- **gate** — qualification gates: years of experience, degree, and licenses/certifications the job requires.

The right panel drops every job that violates a hard axis and promotes the highest-cosine survivor. Clearance / work-authorization requirements are shown as warn-chips (stated, but not resume-checkable).

## Why this exists

Part of the Bag-of-Documents research track ([code](https://github.com/dtunkelang/bag-of-documents)). On the validated probe, the 3-axis filter took constraint-correct top-1 from **12.5% → 95.5%** at a negligible cosine cost — the gain is almost entirely from re-ranking jobs already in the candidate pool, not from a stronger encoder.

## Architecture notes

- All matching runs in-process: catalog vectors are mmap'd fp16; resume vectors are precomputed, so **no embedding model is loaded** at query time.
- Constraint parsing (seniority/geo/years/degree/credentials) is judge-free regex/lexicon logic in `resume_match_lib.py`.
- The catalog vectors + 1.5 GB metadata (for description expansion) + resume caches live in a companion HF dataset and are snapshot-downloaded at startup.

## Companion artifacts

- **Code**: [github.com/dtunkelang/bag-of-documents](https://github.com/dtunkelang/bag-of-documents)
- **Companion jobs-search demo**: [bag-of-documents-jobs](https://huggingface.co/spaces/dtunkelang/bag-of-documents-jobs)
