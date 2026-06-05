#!/usr/bin/env python3
"""One-shot full re-push: original fields (title, vectors, metadata) + the
9 facet fields from facets.jsonl. Replaces atomic_update_facets.py which
wiped the unstored title field. The dense vec field (`e5_vec`) carries
e5-small-v2 vectors (384-dim).
"""

import hashlib
import json
import os
import sys
import time
from collections.abc import Iterator

import numpy as np
import requests

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "space"))
from snippet_lib import pack_vecs  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_apply_urls import load_oa_join  # noqa: E402
from build_apply_urls import template as apply_url_template

STAGE = os.environ.get("JOBS_STAGE", "/Users/dtunkelang/bagofdocs/unified_jobs")
FACETS = os.environ.get(
    "JOBS_FACETS", "/Users/dtunkelang/bagofdocs/jobs_search_demo/facets/facets.jsonl"
)
SOLR = os.environ.get("SOLR", "http://localhost:8983")
CORE = os.environ.get("JOBS_CORE", "jobs")
BATCH = 500

FACET_FIELDS = (
    "role_family",
    "seniority",
    "remote_mode",
    "location_country",
    "location_state",
    "location_city",
    "posted_bucket",
    "salary_band_usd_annual",
    "tech_stack",
)

# Incremental-upsert knobs (set by refresh.py --delta). Defaults reproduce the
# original full-rebuild behaviour: clear the core, then push every doc.
NO_CLEAR = os.environ.get("JOBS_NO_CLEAR") == "1"
_POS_FILE = os.environ.get("JOBS_POSITIONS")
ONLY_POSITIONS: set[int] | None = None
if _POS_FILE:
    with open(_POS_FILE) as _pf:
        ONLY_POSITIONS = set(json.load(_pf))


def stable_id(doc_id: str) -> int:
    """52-bit blake2b of the real (corpus-stable) doc id -> a stable integer Solr
    id. Independent of the doc's row position in the unified catalog (which shifts
    daily as postings open/close), so the SAME job keeps the SAME id across runs —
    the precondition for incremental upsert + Xet tar dedup. 52 bits stays under
    JS Number.MAX_SAFE_INTEGER (2**53) so the Space frontend round-trips it, and
    app.py's int(d["id"]) parses it unchanged."""
    h = hashlib.blake2b(doc_id.encode("utf-8"), digest_size=7).digest()
    return int.from_bytes(h, "big") & ((1 << 52) - 1)


def load_facets() -> dict[int, dict]:
    print(f"loading facets from {FACETS} ...", flush=True)
    out: dict[int, dict] = {}
    with open(FACETS) as f:
        for line in f:
            d = json.loads(line)
            idx = d["idx"]
            out[idx] = {k: d[k] for k in FACET_FIELDS}
    print(f"  {len(out):,} facet records", flush=True)
    return out


def stream_docs(facets: dict[int, dict]) -> Iterator[dict]:
    with open(os.path.join(STAGE, "titles.json")) as f:
        titles = json.load(f)
    with open(os.path.join(STAGE, "source_index.json")) as f:
        sources = json.load(f)["sources"]
    dense = np.load(os.path.join(STAGE, "e5_small_catalog.vecs.fp16.npy"), mmap_mode="r")

    # Pre-computed snippet passage vectors (optional): the encoder writes one normalized
    # fp16 row per UNIQUE passage; snippet_doc_rows maps a doc's metadata position to its
    # passage rows. Present => attach the stored snippet_vecs field so the Space picks
    # snippet passages by dot product instead of encoding at query time. Absent => docs
    # ship without it and the Space falls back to a live encode (still correct, slower).
    snip_vecs = None
    snip_rows: dict[str, list[int]] = {}
    snip_path = os.path.join(STAGE, "snippet_passages.vecs.fp16.npy")
    rows_path = os.path.join(STAGE, "snippet_doc_rows.json")
    if os.path.exists(snip_path) and os.path.exists(rows_path):
        snip_vecs = np.load(snip_path, mmap_mode="r")
        with open(rows_path) as f:
            snip_rows = json.load(f)
        print(
            f"  snippet vecs: {snip_vecs.shape[0]:,} unique passages, "
            f"{len(snip_rows):,} docs mapped",
            flush=True,
        )

    # Stable Solr ids from the real (position-independent) doc ids.
    with open(os.path.join(STAGE, "doc_ids.json")) as f:
        solr_ids = [stable_id(str(x)) for x in json.load(f)]
    if len(set(solr_ids)) != len(solr_ids):
        raise SystemExit(f"stable_id collision among {len(solr_ids):,} doc ids — widen digest_size")

    import sys as _sys

    _sys.path.insert(0, os.path.dirname(__file__))
    # Confidence gate: keep hand-labeled seeds + rules, drop the noisy propagation tiers,
    # and require any other propagated label to clear a similarity floor (see
    # industry_filter.py). Loading round2 verbatim is what filled education_higher with
    # ~75% wrong members (you.com, NIST, ...).
    from industry_filter import DEFAULT_SIM_FLOOR, load_overrides, load_slug_industry  # noqa: E402

    industry_csv = os.path.join(STAGE, "slug_industry_labels_round2.csv")
    slug_to_industry: dict[str, str] = {}
    if os.path.exists(industry_csv):
        slug_to_industry = load_slug_industry(industry_csv)
        print(
            f"  loaded {len(slug_to_industry):,} slug -> industry labels "
            f"(confidence-gated: seeds+rules + propagated >= {DEFAULT_SIM_FLOOR:g} sim)",
            flush=True,
        )

    # Hand-curated overrides WIN over the gated propagation: they add trusted labels for
    # gated-out slugs AND correct wrong ones (incl. wrong seeds the similarity floor can't
    # prune). Staffing agencies are still per-job resolved below, so a slug-level override
    # of an agency is harmless -- _resolve_industry overrides it from the job's role_family.
    overrides = load_overrides(os.path.join(STAGE, "slug_industry_overrides.csv"))
    if overrides:
        n_new = sum(1 for s in overrides if s not in slug_to_industry)
        n_fix = sum(
            1 for s, v in overrides.items() if s in slug_to_industry and slug_to_industry[s] != v
        )
        slug_to_industry.update(overrides)
        print(
            f"  applied {len(overrides):,} hand-curated overrides "
            f"({n_new:,} new slugs, {n_fix:,} relabels)",
            flush=True,
        )

    # Per-doc industry override for staffing/employment-agency employers.
    from staffing_override import resolve_industry as _resolve_industry  # noqa: E402

    # Embedding-derived role_family overrides for docs the title heuristics leave
    # in 'other'. Keyed by source doc id (ensemble-gated e5 kNN, ~96% precision;
    # see classify_other_emb.py). Applied over the heuristic role_family below.
    emb_path = os.path.join(os.path.dirname(__file__), "role_family_emb_overrides.json")
    role_emb_override: dict[str, str] = {}
    if os.path.exists(emb_path):
        with open(emb_path) as ef:
            role_emb_override = json.load(ef)
        print(f"  loaded {len(role_emb_override):,} embedding role_family overrides", flush=True)

    # LLM-derived overrides (offline gpt-4o-mini backfill, gpt-4.1-judge allowlisted;
    # see classify_other_llm.py). Separate file so the nightly refresh that regenerates
    # the embedding file never clobbers it. Embedding/dept-agree WINS on conflict.
    llm_path = os.path.join(os.path.dirname(__file__), "role_family_llm_overrides.json")
    if os.path.exists(llm_path):
        with open(llm_path) as lf:
            llm_override = json.load(lf)
        before = len(role_emb_override)
        for k, v in llm_override.items():
            role_emb_override.setdefault(k, v)
        print(
            f"  loaded {len(llm_override):,} LLM role_family overrides "
            f"(+{len(role_emb_override) - before:,} new)",
            flush=True,
        )

    # ROME-derived overrides for France Travail jobs (open-weight, no LLM): every
    # FT offer carries an authoritative ROME 4.0 occupation code assigned by
    # France Travail; we map it straight to a role_family (see classify_other_rome.py
    # / rome_role_family.py). Loaded BEFORE ESCO so the authoritative source code
    # WINS over the lexical ESCO title-match on the (FR) overlap. emb/llm still win
    # (they're English; ROME is FR -> effectively disjoint).
    rome_path = os.path.join(os.path.dirname(__file__), "role_family_rome_overrides.json")
    if os.path.exists(rome_path):
        with open(rome_path) as rf:
            rome_override = json.load(rf)
        before = len(role_emb_override)
        for k, v in rome_override.items():
            role_emb_override.setdefault(k, v)
        print(
            f"  loaded {len(rome_override):,} ROME role_family overrides "
            f"(+{len(role_emb_override) - before:,} new)",
            flush=True,
        )

    # JobTech-derived overrides for Swedish jobs (open-weight, no LLM): every
    # Arbetsförmedlingen ad carries the Swedish occupation taxonomy; its broad
    # occupation_field bucket (stored as `department`) is authoritative, so we map
    # it straight to a role_family (see classify_other_jobtech.py /
    # jobtech_role_family.py). Loaded BEFORE ESCO so the source-assigned field
    # WINS over the lexical Swedish ESCO title-match. emb/llm/rome still win first.
    jobtech_path = os.path.join(os.path.dirname(__file__), "role_family_jobtech_overrides.json")
    if os.path.exists(jobtech_path):
        with open(jobtech_path) as jf:
            jobtech_override = json.load(jf)
        before = len(role_emb_override)
        for k, v in jobtech_override.items():
            role_emb_override.setdefault(k, v)
        print(
            f"  loaded {len(jobtech_override):,} JobTech role_family overrides "
            f"(+{len(role_emb_override) - before:,} new)",
            flush=True,
        )

    # Adzuna-derived overrides (open-weight, no LLM): every Adzuna posting carries
    # a category from Adzuna's own taxonomy, stored as `department`; we map the
    # (localized) label straight to a role_family (see classify_other_adzuna.py /
    # adzuna_role_family.py). Loaded BEFORE ESCO so the source category beats the
    # lexical title-match. emb/llm/rome/jobtech still win first.
    adzuna_path = os.path.join(os.path.dirname(__file__), "role_family_adzuna_overrides.json")
    if os.path.exists(adzuna_path):
        with open(adzuna_path) as af:
            adzuna_override = json.load(af)
        before = len(role_emb_override)
        for k, v in adzuna_override.items():
            role_emb_override.setdefault(k, v)
        print(
            f"  loaded {len(adzuna_override):,} Adzuna role_family overrides "
            f"(+{len(role_emb_override) - before:,} new)",
            flush=True,
        )

    # ESCO-derived overrides for the non-English 'other' residual (open-weight, no
    # LLM): each non-English title is matched to an ESCO occupation in its own
    # language and the occupation's ISCO-08 code maps to a role_family (see
    # classify_other_esco.py / isco_role_family.py). Separate file, regenerated each
    # refresh. emb/llm/rome win on the rare conflict (emb/llm are English; ROME is
    # the authoritative FT code -> ESCO is the lexical fallback).
    esco_path = os.path.join(os.path.dirname(__file__), "role_family_esco_overrides.json")
    if os.path.exists(esco_path):
        with open(esco_path) as sf:
            esco_override = json.load(sf)
        before = len(role_emb_override)
        for k, v in esco_override.items():
            role_emb_override.setdefault(k, v)
        print(
            f"  loaded {len(esco_override):,} ESCO role_family overrides "
            f"(+{len(role_emb_override) - before:,} new)",
            flush=True,
        )

    # Multilingual-embedding ESCO overrides for the non-English 'other' residual
    # (open-weight, no LLM): each non-English title is matched to its nearest ESCO
    # occupation label by multilingual-e5 cosine and the occupation's ISCO-08 code
    # maps to a role_family (see classify_other_esco_emb.py / isco_role_family.py).
    # LOWEST precedence -- loaded last so every authoritative source (ROME/JobTech/
    # Adzuna) and the higher-precision lexical ESCO match win first; the semantic
    # match only fills docs no stronger signal reached. Regenerated each refresh.
    esco_emb_path = os.path.join(os.path.dirname(__file__), "role_family_esco_emb_overrides.json")
    if os.path.exists(esco_emb_path):
        with open(esco_emb_path) as sf:
            esco_emb_override = json.load(sf)
        before = len(role_emb_override)
        for k, v in esco_emb_override.items():
            role_emb_override.setdefault(k, v)
        print(
            f"  loaded {len(esco_emb_override):,} emb-ESCO role_family overrides "
            f"(+{len(role_emb_override) - before:,} new)",
            flush=True,
        )

    # Outbound "view original posting" link. Recovered WITHOUT a re-crawl: the
    # OpenApply (greenhouse/lever/ashby) raw files on disk carry the authoritative
    # apply_url; everything else with a deterministic public-posting URL is
    # reconstructed from the id (see build_apply_urls.py). A record that already
    # carries its own apply_url (the forward path, once the fetchers capture it)
    # wins over both. Tokenised/aggregator redirects (adzuna/jooble/...) stay blank.
    oa_join = load_oa_join()
    print(f"  loaded {len(oa_join):,} OpenApply apply_url join entries", flush=True)

    meta_path = os.path.join(STAGE, "metadata.jsonl")
    with open(meta_path) as mf:
        for i, line in enumerate(mf):
            if ONLY_POSITIONS is not None and i not in ONLY_POSITIONS:
                continue
            rec = json.loads(line)
            title_display = (rec.get("title") or titles[i].split("\n", 1)[0]).strip()
            slug = rec.get("source_slug") or ""
            fac = facets.get(i, {})
            emb_fam = role_emb_override.get(str(rec.get("id")))
            if emb_fam and (fac.get("role_family") or "other") == "other":
                fac = {**fac, "role_family": emb_fam}
            slug_ind = slug_to_industry.get(slug, "unclassified")
            doc_industry = _resolve_industry(
                slug, slug_ind, fac.get("role_family") or "", title_display
            )
            rec_id = str(rec.get("id") or "")
            apply_url = (
                rec.get("apply_url") or oa_join.get(rec_id) or apply_url_template(rec_id) or ""
            )
            doc = {
                "id": str(solr_ids[i]),
                "title": titles[i],  # full title + description for BM25
                "title_display": title_display,
                "employer": slug,
                "industry": doc_industry,
                "locations": rec.get("locations") or [],
                "employment_type": rec.get("employment_type") or "",
                "salary_currency": rec.get("salary_currency") or "",
                "department": rec.get("department") or "",
                "rome_code": rec.get("rome_code") or "",  # France Travail ROME occupation code
                "posted_at": rec.get("posted_at") or "",
                "source_corpus": sources[i],
                "lang": rec.get("lang") or "en",
                "description": rec.get("description") or "",
                "apply_url": apply_url,
                "e5_vec": dense[i].astype(np.float32).tolist(),
            }
            rows = snip_rows.get(str(i)) if snip_vecs is not None else None
            if rows:
                doc["snippet_vecs"] = pack_vecs(snip_vecs[rows])
            if rec.get("salary_min") is not None:
                doc["salary_min"] = float(rec["salary_min"])
            if rec.get("salary_max") is not None:
                doc["salary_max"] = float(rec["salary_max"])

            # Layer in facets
            for k in FACET_FIELDS:
                v = fac.get(k)
                if k == "tech_stack":
                    doc[k] = v or []
                else:
                    doc[k] = v or ""
            yield doc


def main() -> int:
    facets = load_facets()
    if NO_CLEAR:
        print(
            f"incremental: NOT clearing core; pushing {len(ONLY_POSITIONS or []):,} docs",
            flush=True,
        )
    else:
        print("clearing core ...", flush=True)
        requests.post(
            f"{SOLR}/solr/{CORE}/update",
            json={"delete": {"query": "*:*"}},
            params={"commit": "true"},
            timeout=120,
        ).raise_for_status()
    t0 = time.time()
    batch: list[dict] = []
    n = 0
    for doc in stream_docs(facets):
        batch.append(doc)
        if len(batch) >= BATCH:
            r = requests.post(
                f"{SOLR}/solr/{CORE}/update/json/docs",
                json=batch,
                params={"commit": "false"},
                timeout=120,
            )
            r.raise_for_status()
            n += len(batch)
            batch = []
            if n % 10000 == 0:
                rate = n / (time.time() - t0)
                print(f"  {n:,} ({rate:.0f}/s)", flush=True)
    if batch:
        requests.post(
            f"{SOLR}/solr/{CORE}/update/json/docs",
            json=batch,
            params={"commit": "false"},
            timeout=120,
        ).raise_for_status()
        n += len(batch)
    print("committing ...", flush=True)
    requests.get(
        f"{SOLR}/solr/{CORE}/update", params={"commit": "true"}, timeout=300
    ).raise_for_status()
    print(f"done: {n:,} in {time.time() - t0:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
