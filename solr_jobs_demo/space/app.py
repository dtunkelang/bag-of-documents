#!/usr/bin/env python3
"""Solr-backed jobs search demo.

Default retrieval strategy:
  RRF(BM25, e5-small) with RRF_K=60 over top-100 per lane.

Backed by Solr 10 for BM25 + dense vector retrieval. Autocomplete suggestions
come from a curated query corpus (lexical prefix match), with Solr's
titleSuggester as fallback.
Run after push_docs.py has populated the 'jobs' core.
"""

import bisect
import html
import json
import os
import re
import time
from collections import defaultdict
from contextlib import asynccontextmanager

import numpy as np
import requests
import resume_match_lib as L
from fastapi import FastAPI, File, Form, Query, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from maps_svg import US_STATES_SVG, WORLD_SVG

# ===== configuration =====

SOLR = os.environ.get("SOLR", "http://localhost:8983")
CORE = "jobs"
DATASET_REPO = os.environ.get("DATASET_REPO", "dtunkelang/jobs-demo")
DENSE_MODEL = "intfloat/e5-small-v2"
DENSE_QUERY_PREFIX = (
    "query: "  # e5 family requires asymmetric prefixes; catalog encoded with "passage: "
)
RRF_POOL = 100
RRF_K = 60
EMPLOYER_CAP = int(os.environ.get("EMPLOYER_CAP", "3"))

SRC_SHORT = {
    "jobs_data": "OAP",
    "jobs_data_linkedin": "LI",
    "jobs_data_jobstreet": "JS",
    "jobs_data_usajobs": "USA",
}

ABBREV_EXPANSIONS: dict[str, list[str]] = {
    "rn": ["registered nurse"],
    "lpn": ["licensed practical nurse"],
    "np": ["nurse practitioner"],
    "pa": ["physician assistant"],
    "md": ["medical doctor", "doctor"],
    "rd": ["registered dietitian"],
    "rt": ["respiratory therapist", "radiologic technologist"],
    "pt": ["physical therapist"],
    "ot": ["occupational therapist"],
    "cna": ["certified nursing assistant"],
    "emt": ["emergency medical technician"],
    "swe": ["software engineer"],
    "sde": ["software development engineer"],
    "sre": ["site reliability engineer"],
    "dev": ["developer"],
    "qa": ["quality assurance"],
    "ux": ["user experience"],
    "ui": ["user interface"],
    "ml": ["machine learning"],
    "ai": ["artificial intelligence"],
    "ds": ["data scientist"],
    "pm": ["project manager", "product manager"],
    "tpm": ["technical program manager"],
    "ba": ["business analyst"],
    "csm": ["customer success manager"],
    "sdr": ["sales development representative"],
    "bdr": ["business development representative"],
    "ae": ["account executive"],
    "vp": ["vice president"],
    "ceo": ["chief executive officer"],
    "cfo": ["chief financial officer"],
    "cto": ["chief technology officer"],
    "ciso": ["chief information security officer"],
    "hr": ["human resources"],
    "it": ["information technology"],
    "ops": ["operations"],
    "admin": ["administrative", "administrator"],
    "sr": ["senior"],
    "sr.": ["senior"],
    "jr": ["junior"],
    "jr.": ["junior"],
    "mgr": ["manager"],
    "asst": ["assistant"],
    "exec": ["executive"],
    "engr": ["engineer"],
    "eng": ["engineer"],
}

_DIGIT_RUN = re.compile(r"\d{3,}")
_SLUG_ISH = re.compile(r"\b[a-z]+\d+\b")
_BAD_CHARS = re.compile(r"[<>{}@]")
_DOUBLE_SPACE = re.compile(r"\s{2,}")


def _is_clean(q: str) -> bool:
    if len(q) < 2 or len(q) > 60:
        return False
    if _DIGIT_RUN.search(q) or _SLUG_ISH.search(q) or _BAD_CHARS.search(q):
        return False
    return not _DOUBLE_SPACE.search(q)


# ===== resources =====

R: dict = {}


def _download_suggest_cache() -> str:
    """Snapshot the curated query corpus (suggestion strings + tags) from the
    companion HF dataset. The 1024-dim te3 vectors are no longer downloaded —
    these files supply autocomplete suggestions only."""
    from huggingface_hub import snapshot_download

    return snapshot_download(
        repo_id=DATASET_REPO,
        repo_type="dataset",
        allow_patterns=[
            "te3_queries.ids.json",
            "te3_queries.sources.json",
            "te3_cache_canonical.json",
        ],
    )


def load_resources() -> None:
    t0 = time.time()
    print(f"loading {DENSE_MODEL}...", flush=True)
    import torch
    from sentence_transformers import SentenceTransformer

    device = os.environ.get("DENSE_DEVICE") or (
        "mps" if torch.backends.mps.is_available() else "cpu"
    )
    dense_model = SentenceTransformer(DENSE_MODEL, device=device)
    print(f"  dense loaded on {device} in {time.time() - t0:.1f}s", flush=True)

    # Query-context related-search suggester (offline corpus role vocab + e5 embeddings).
    try:
        from suggest_lib import RoleSuggester

        role_suggester = RoleSuggester()
        print(f"  role suggester: {len(role_suggester.phrases)} roles", flush=True)
    except Exception as e:  # files missing -> feature degrades to off, app still serves
        role_suggester = None
        print(f"  role suggester unavailable: {e}", flush=True)

    t0 = time.time()
    print("downloading suggestion corpus from HF dataset...", flush=True)
    cache_dir = _download_suggest_cache()
    print("loading suggestion corpus...", flush=True)
    with open(os.path.join(cache_dir, "te3_queries.ids.json")) as f:
        qids = json.load(f)
    with open(os.path.join(cache_dir, "te3_queries.sources.json")) as f:
        qsrc = json.load(f)
    canonical_path = os.path.join(cache_dir, "te3_cache_canonical.json")
    if os.path.exists(canonical_path):
        with open(canonical_path) as f:
            canonical = json.load(f)
    else:
        canonical = {}

    # Keep the best (highest-priority) source tag per unique query string;
    # the tag drives suggestion tier ordering. No vectors are needed.
    qkey_src: dict[str, str] = {}
    TAG_PRIORITY = {"title": 0, "combo": 1, "head": 2, "tail": 3, "synth": 4}
    for i, q in enumerate(qids):
        k = q.strip().lower()
        tag = qsrc[i]
        cur = qkey_src.get(k)
        if cur is None or TAG_PRIORITY.get(tag, 9) < TAG_PRIORITY.get(cur, 9):
            qkey_src[k] = tag
    print(
        f"  suggestion corpus: {len(qkey_src):,} unique keys in {time.time() - t0:.1f}s",
        flush=True,
    )

    by_tag: dict[str, list[str]] = defaultdict(list)
    for k in qkey_src:
        if not _is_clean(k) or k in canonical:
            continue
        by_tag[qkey_src[k]].append(k)
    for v in by_tag.values():
        v.sort()
    sorted_keys = sorted(qkey_src.keys())

    R.update(
        {
            "dense_model": dense_model,
            "sorted_keys": sorted_keys,
            "tier_keys": dict(by_tag),
            "role_suggester": role_suggester,
        }
    )
    print("ready.", flush=True)


# ===== query encoders =====


def _dense_qv(query: str) -> list[float]:
    text = DENSE_QUERY_PREFIX + query
    qv = R["dense_model"].encode([text], normalize_embeddings=True, show_progress_bar=False)[0]
    return qv.astype(np.float32).tolist()


# ===== solr retrieval lanes =====


def _vec_str(vec: list[float]) -> str:
    return "[" + ",".join(f"{x:.6f}" for x in vec) + "]"


FACET_FIELDS = (
    "role_family",
    "seniority",
    "industry",
    "remote_mode",
    "location_country",
    "location_state",
    "posted_bucket",
    "salary_band_usd_annual",
    "tech_stack",
)


# posted_bucket values are mutually exclusive (each job in exactly one), so a
# "posted in the last N days" filter must OR in every fresher bucket. past_24h was
# previously omitted -> a "Past 7 days" filter silently dropped the freshest jobs.
POSTED_BUCKET_NESTING = {
    "past_24h": ["past_24h"],
    "past_7d": ["past_24h", "past_7d"],
    "past_30d": ["past_24h", "past_7d", "past_30d"],
    "past_90d": ["past_24h", "past_7d", "past_30d", "past_90d"],
    "older": ["older"],
}


def _filter_clauses(filters: dict[str, str | list[str]]) -> list[str]:
    """Build Solr fq= clauses from a {field: value(s)} filter dict. A value may be a
    single string or a list of strings; a list becomes an OR within that field's
    clause (multi-select facet). Clauses AND together across fields. posted_bucket is
    single-select with cumulative nesting. Quotes values to handle spaces/specials."""
    out = []
    for k, v in filters.items():
        if k not in FACET_FIELDS:
            continue
        values = [v] if isinstance(v, str) else list(v)
        values = [x for x in values if x]
        if not values:
            continue
        if k == "posted_bucket":
            members = POSTED_BUCKET_NESTING.get(values[0], [values[0]])
            out.append("posted_bucket:(" + " OR ".join(members) + ")")
            continue
        ors = " OR ".join(f'"{x}"' for x in values)
        out.append(f"{k}:({ors})")
    return out


def _topk_bm25(
    query: str, k: int, filters: dict[str, str] | None = None
) -> list[tuple[int, float]]:
    params: list[tuple[str, str]] = [
        ("q", "{!edismax qf=title v=$user_q}"),
        ("user_q", query),
        ("rows", str(k)),
        ("fl", "id,score"),
    ]
    for clause in _filter_clauses(filters or {}):
        params.append(("fq", clause))
    r = requests.get(f"{SOLR}/solr/{CORE}/select", params=params, timeout=10)
    r.raise_for_status()
    return [(int(d["id"]), float(d["score"])) for d in r.json()["response"]["docs"]]


def _count_bm25(query: str, filters: dict[str, str] | None = None) -> int:
    """numFound for a BM25 title query — used to validate suggested searches against
    the live index (rows=0, no encode, so it's cheap to run for several candidates)."""
    params: list[tuple[str, str]] = [
        ("q", "{!edismax qf=title v=$user_q}"),
        ("user_q", query),
        ("rows", "0"),
    ]
    for clause in _filter_clauses(filters or {}):
        params.append(("fq", clause))
    r = requests.get(f"{SOLR}/solr/{CORE}/select", params=params, timeout=10)
    r.raise_for_status()
    return int(r.json()["response"]["numFound"])


def _topk_knn(
    field: str, qv: list[float], k: int, filters: dict[str, str] | None = None
) -> list[tuple[int, float]]:
    clauses = _filter_clauses(filters or {})
    if clauses:
        # Solr 10 knn preFilter narrows the candidate set BEFORE HNSW traversal.
        # Multiple filters AND together inside the single preFilter.
        pre = " AND ".join(clauses).replace("'", r"\'")
        q = f"{{!knn f={field} topK={k} preFilter='{pre}'}}{_vec_str(qv)}"
    else:
        q = f"{{!knn f={field} topK={k}}}{_vec_str(qv)}"
    r = requests.post(
        f"{SOLR}/solr/{CORE}/select",
        data={"q": q, "rows": k, "fl": "id,score"},
        timeout=15,
    )
    r.raise_for_status()
    return [(int(d["id"]), float(d["score"])) for d in r.json()["response"]["docs"]]


def _knn_over_ids(field: str, qv: list[float], ids: list[int]) -> list[tuple[int, float]]:
    """KNN of `qv` restricted to a fixed candidate id set, returning EVERY candidate
    scored (topK == set size). Used to score the profile fit of a query's own
    candidates — so the profile re-ranks any query, not only ones whose results land
    in the profile's global top-N. e5_vec is stored=false, so a preFiltered KNN is the
    only way to read a doc's profile cosine."""
    if not ids:
        return []
    pre = ("id:(" + " ".join(str(i) for i in ids) + ")").replace("'", r"\'")
    q = f"{{!knn f={field} topK={len(ids)} preFilter='{pre}'}}{_vec_str(qv)}"
    r = requests.post(
        f"{SOLR}/solr/{CORE}/select",
        data={"q": q, "rows": len(ids), "fl": "id,score"},
        timeout=15,
    )
    r.raise_for_status()
    return [(int(d["id"]), float(d["score"])) for d in r.json()["response"]["docs"]]


# ===== RRF fusion + result hydration =====


def _hydrate(ids: list[int], with_facets: bool = False) -> dict[int, dict]:
    """Fetch metadata for a list of doc ids in one Solr call."""
    if not ids:
        return {}
    id_clause = " OR ".join(f'id:"{i}"' for i in ids)
    fl = (
        "id,title_display,employer,locations,employment_type,"
        "salary_min,salary_max,salary_currency,department,posted_at,source_corpus,industry"
    )
    if with_facets:
        fl += "," + ",".join(FACET_FIELDS)
    r = requests.get(
        f"{SOLR}/solr/{CORE}/select",
        params={"q": id_clause, "rows": len(ids), "fl": fl},
        timeout=10,
    )
    r.raise_for_status()
    return {int(d["id"]): d for d in r.json()["response"]["docs"]}


def _fmt_salary(d: dict) -> str:
    lo, hi, cur = d.get("salary_min"), d.get("salary_max"), d.get("salary_currency") or ""
    if lo is None and hi is None:
        return ""
    if lo is not None and hi is not None:
        return f"{cur} {int(lo):,}-{int(hi):,}".strip()
    if hi is not None:
        return f"{cur} up to {int(hi):,}".strip()
    return f"{cur} from {int(lo):,}".strip()


def _make_result(rank: int, score: float, idx: int, hyd: dict) -> dict:
    locs = hyd.get("locations") or []
    title = (hyd.get("title_display") or "").strip()
    if len(title) > 140:
        title = title[:137] + "..."
    return {
        "rank": rank,
        "score": float(score),
        "title": title,
        "idx": idx,
        "source": hyd.get("source_corpus") or "",
        "employer": hyd.get("employer") or "",
        "industry": hyd.get("industry") or "",
        "location": ", ".join(locs[:2]) if locs else "",
        "employment_type": hyd.get("employment_type") or "",
        "salary": _fmt_salary(hyd),
        "department": hyd.get("department") or "",
        "posted": (hyd.get("posted_at") or "")[:10],
    }


def _fused_topk(
    query: str,
    k: int,
    filters: dict[str, str] | None = None,
    pool: int = RRF_POOL,
) -> list[tuple[int, float]]:
    """Run BM25 + e5-small lanes and RRF-fuse to top-k."""
    contrib: dict[int, float] = defaultdict(float)
    for rank, (idx, _) in enumerate(_topk_bm25(query, pool, filters), 1):
        contrib[idx] += 1.0 / (RRF_K + rank)
    # Solr field "e5_vec" holds e5-small-v2 vectors (384-dim, passage: prefix at index time).
    for rank, (idx, _) in enumerate(_topk_knn("e5_vec", _dense_qv(query), pool, filters), 1):
        contrib[idx] += 1.0 / (RRF_K + rank)
    return sorted(contrib.items(), key=lambda x: -x[1])[:k]


EMPLOYER_DOMINANCE = float(os.environ.get("EMPLOYER_DOMINANCE", "0.30"))


def _dominant_employers(items: list[tuple[int, float]], hyd: dict[int, dict]) -> set[str]:
    """Employers whose share of the fused pool >= EMPLOYER_DOMINANCE — these are exempt
    from the per-employer cap (employer-coupled query intent, e.g. 'amazon jobs')."""
    counts: dict[str, int] = defaultdict(int)
    for idx, _ in items:
        emp = (hyd.get(idx, {}).get("employer") or "").strip().lower()
        if emp:
            counts[emp] += 1
    total = sum(counts.values())
    return {e for e, n in counts.items() if total and n / total >= EMPLOYER_DOMINANCE}


def _cap_employers(
    items: list[tuple[int, float]],
    hyd: dict[int, dict],
    k: int,
    filters: dict[str, str] | None,
    dominance_bypass: bool = True,
) -> list[tuple[int, float]]:
    """Cap to EMPLOYER_CAP results per employer for display diversity.
    Cap is bypassed when the user explicitly filtered by employer, OR (when
    dominance_bypass) when one employer dominates the pool (>= EMPLOYER_DOMINANCE
    share), which signals employer-coupled intent (e.g. 'amazon jobs'). The "more
    like this" pivot turns dominance_bypass off — the user clicked a role, not an
    employer, so near-duplicate postings from one shop add little."""
    cap = 0 if (filters and filters.get("employer")) else EMPLOYER_CAP
    if cap <= 0:
        return items[:k]
    exempt = _dominant_employers(items, hyd) if dominance_bypass else set()
    kept: list[tuple[int, float]] = []
    seen: dict[str, int] = {}
    for idx, score in items:
        emp = (hyd.get(idx, {}).get("employer") or "").strip().lower()
        if emp and emp not in exempt and seen.get(emp, 0) >= cap:
            continue
        kept.append((idx, score))
        if emp:
            seen[emp] = seen.get(emp, 0) + 1
        if len(kept) >= k:
            break
    return kept


def search_default(
    query: str,
    k: int = 10,
    filters: dict[str, str] | None = None,
) -> list[dict]:
    """RRF(BM25, e5-small) with optional facet filters, then cap to
    EMPLOYER_CAP results per employer for display diversity."""
    cap = 0 if (filters and filters.get("employer")) else EMPLOYER_CAP
    pool_k = max(k * (cap + 2), k + 20) if cap > 0 else k
    items = _fused_topk(query, pool_k, filters, RRF_POOL)
    hyd = _hydrate([i for i, _ in items])
    items = _cap_employers(items, hyd, k, filters)
    return [_make_result(r + 1, s, i, hyd.get(i, {})) for r, (i, s) in enumerate(items)]


# ===== blank/browse default (no query): recent + low-barrier "minimal skills" =====
# Fires on page load and whenever the query box is empty. posted_at is indexed=false
# so recency can't be a sort — it rides posted_bucket instead; "minimal skill
# requirements" is proxied by seniority (entry/intern/junior favored), the only indexed
# experience signal. Pure ADDITIVE edismax boosts over a match-all base, so fresher and
# lower-barrier jobs float to the top without excluding anything — facet filters still
# apply, and an uploaded profile re-ranks via browse_personalized(). Weights env-tunable.
def _browse_bq() -> list[str]:
    rec = {"past_24h": 8, "past_7d": 5, "past_30d": 3, "past_90d": 1}
    skill = {"entry": 4, "intern": 4, "junior": 2, "not_specified": 0.5}
    rec_w = float(os.environ.get("BROWSE_RECENCY_W", "1.0"))
    skill_w = float(os.environ.get("BROWSE_SKILL_W", "1.0"))
    return [f"posted_bucket:{b}^{w * rec_w:g}" for b, w in rec.items()] + [
        f"seniority:{s}^{w * skill_w:g}" for s, w in skill.items()
    ]


def _browse_topk(
    k: int, filters: dict[str, str | list[str]] | None = None, pool: int | None = None
) -> list[tuple[int, float]]:
    params: list[tuple[str, str]] = [
        ("defType", "edismax"),
        ("q", ""),
        ("q.alt", "*:*"),
        ("rows", str(pool or k)),
        ("fl", "id,score"),
    ]
    for b in _browse_bq():
        params.append(("bq", b))
    for clause in _filter_clauses(filters or {}):
        params.append(("fq", clause))
    r = requests.get(f"{SOLR}/solr/{CORE}/select", params=params, timeout=10)
    r.raise_for_status()
    return [(int(d["id"]), float(d["score"])) for d in r.json()["response"]["docs"]]


def browse_default(k: int = 10, filters: dict[str, str | list[str]] | None = None) -> list[dict]:
    """Default browse: recent + low-barrier jobs, with facet filters applied."""
    cap = 0 if (filters and filters.get("employer")) else EMPLOYER_CAP
    pool_k = max(k * (cap + 2), k + 20) if cap > 0 else k
    items = _browse_topk(max(pool_k, 200), filters)
    hyd = _hydrate([i for i, _ in items])
    # No query intent in a blank browse, so keep employer diversity (no dominance bypass).
    items = _cap_employers(items, hyd, k, filters, dominance_bypass=False)
    return [_make_result(r + 1, s, i, hyd.get(i, {})) for r, (i, s) in enumerate(items)]


# ===== "more jobs like this one": doc -> similar docs =====
# Same RRF(BM25, e5-small) machinery as search_default, but the "query" is a source
# job rather than typed text: the title drives the BM25 lane, and title + the lead of
# the description drives the e5 dense lane. e5_vec is stored=false so we can't read the
# source doc's stored vector back — re-embedding its text at query time is the only way
# to KNN from it, and it lets the same asymmetric "query: " prefix bridge to the indexed
# "passage: " vectors. The source doc is dropped from its own neighbour list.

MLT_DESC_CHARS = 900  # lead of the description fed to the dense encoder (e5-small @ 512 tok)


def _source_doc(idx: int) -> dict | None:
    r = requests.get(
        f"{SOLR}/solr/{CORE}/select",
        params={"q": f'id:"{idx}"', "fl": "id,title_display,description", "rows": 1},
        timeout=10,
    )
    r.raise_for_status()
    docs = r.json()["response"]["docs"]
    return docs[0] if docs else None


def more_like_this(idx: int, k: int = 10) -> dict:
    """Find jobs similar to job `idx` via RRF(BM25-on-title, e5-dense-on-title+lead)."""
    src = _source_doc(idx)
    if src is None:
        return {"source": None, "results": []}
    title = (src.get("title_display") or "").strip()
    desc = (src.get("description") or "").strip()
    seed = (title + ". " + desc)[: len(title) + 2 + MLT_DESC_CHARS] if desc else title
    contrib: dict[int, float] = defaultdict(float)
    # dense lane (+1 pool depth so dropping the source doc leaves a full pool)
    for rank, (i, _) in enumerate(_topk_knn("e5_vec", _dense_qv(seed), RRF_POOL + 1), 1):
        if i != idx:
            contrib[i] += 1.0 / (RRF_K + rank)
    # bm25 lane on the source title
    if title:
        for rank, (i, _) in enumerate(_topk_bm25(title, RRF_POOL + 1), 1):
            if i != idx:
                contrib[i] += 1.0 / (RRF_K + rank)
    ranked = sorted(contrib.items(), key=lambda x: -x[1])
    pool = ranked[: max(k * (EMPLOYER_CAP + 2), k + 20)]
    hyd = _hydrate([i for i, _ in pool])
    capped = _cap_employers(pool, hyd, k, None, dominance_bypass=False)
    results = [_make_result(r + 1, s, i, hyd.get(i, {})) for r, (i, s) in enumerate(capped)]
    return {
        "source": {"idx": idx, "title": _clean_text(title)},
        "results": results,
    }


# ===== personalized search (keyword query re-ranked by an uploaded profile) =====

PROF_WEIGHT = float(os.environ.get("PROF_WEIGHT", "1.0"))


def _personalized_topk(
    query: str,
    qv_profile: list[float],
    k: int,
    filters: dict[str, str] | None = None,
    pool: int = RRF_POOL,
    prof_weight: float = PROF_WEIGHT,
) -> tuple[list[tuple[int, float]], dict[int, float]]:
    """RRF(BM25, e5-small) for the query, then a third lane that re-ranks the query's
    OWN candidates by profile fit. The keyword query still defines what's eligible
    (we never inject off-query jobs), but every candidate is scored against the profile
    — so a profile reshapes essentially any query, including ones far from it (a
    data-engineer profile floats the most data/eng-flavored 'manager' jobs up). Returns
    (ranked, prof_cos) where prof_cos maps idx -> profile cosine for EVERY candidate."""
    contrib: dict[int, float] = defaultdict(float)
    for rank, (idx, _) in enumerate(_topk_bm25(query, pool, filters), 1):
        contrib[idx] += 1.0 / (RRF_K + rank)
    for rank, (idx, _) in enumerate(_topk_knn("e5_vec", _dense_qv(query), pool, filters), 1):
        contrib[idx] += 1.0 / (RRF_K + rank)
    # Rank the query candidates by profile fit and blend that rank in as a third lane.
    prof_hits = _knn_over_ids("e5_vec", qv_profile, list(contrib.keys()))
    prof_cos = {idx: cos for idx, cos in prof_hits}
    for rank, (idx, _) in enumerate(prof_hits, 1):
        contrib[idx] += prof_weight * (1.0 / (RRF_K + rank))
    ranked = sorted(contrib.items(), key=lambda x: -x[1])[:k]
    return ranked, prof_cos


def _make_result_personalized(
    rank: int, score: float, idx: int, d: dict, st: dict, cos: float | None
) -> dict:
    res = _make_result(rank, score, idx, d)
    res["cosine"] = round(float(cos), 4) if cos is not None else None
    res["axes"] = st  # {sen,loc,gate: {ok,reason}, all: bool} for ✓/✗ badges
    return res


def search_personalized(
    query: str,
    r: dict,
    qv_profile: list[float],
    k: int = 10,
    filters: dict[str, str] | None = None,
    hard_filter: bool = False,
) -> list[dict]:
    """Keyword search re-ranked by profile fit. Soft by default (profile-KNN RRF
    boost + per-result 3-axis badges); when hard_filter is set, drop results the
    candidate doesn't qualify for (under-seniority / location / years-degree-cred
    gates), mirroring the profile lane's filtered panel."""
    cap = 0 if (filters and filters.get("employer")) else EMPLOYER_CAP
    ranked, prof_cos = _personalized_topk(query, qv_profile, RRF_POOL, filters)
    ids = [i for i, _ in ranked]
    hyd = _hydrate_for_match(ids)
    exempt = _dominant_employers(ranked, hyd) if cap > 0 else set()
    rows: list[dict] = []
    seen: dict[str, int] = {}
    for idx, score in ranked:
        d = hyd.get(idx)
        if not d:
            continue
        jf = _job_feats_from_solr(d)
        st = L.axis_status(r, jf)
        if hard_filter and not st["all"]:
            continue
        emp = (d.get("employer") or "").strip().lower()
        if cap > 0 and emp and emp not in exempt and seen.get(emp, 0) >= cap:
            continue
        rows.append(_make_result_personalized(len(rows) + 1, score, idx, d, st, prof_cos.get(idx)))
        if emp:
            seen[emp] = seen.get(emp, 0) + 1
        if len(rows) >= k:
            break
    return rows


def browse_personalized(
    r: dict,
    qv_profile: list[float],
    k: int = 10,
    filters: dict[str, str | list[str]] | None = None,
    hard_filter: bool = False,
) -> list[dict]:
    """Blank-query browse personalized to an uploaded profile: rank purely by profile
    fit (e5 KNN over filtered candidates), with the same 3-axis qualification badges as
    keyword personalization. The recency/low-barrier browse boost is replaced by profile
    cosine here — the profile IS the intent when there's no query."""
    cap = 0 if (filters and filters.get("employer")) else EMPLOYER_CAP
    pool = max(PROFILE_POOL, k * (cap + 2), k + 20)
    hits = _topk_knn("e5_vec", qv_profile, pool, filters)  # (idx, cosine), best-first
    hyd = _hydrate_for_match([i for i, _ in hits])
    rows: list[dict] = []
    seen: dict[str, int] = {}
    for idx, cos in hits:
        d = hyd.get(idx)
        if not d:
            continue
        jf = _job_feats_from_solr(d)
        st = L.axis_status(r, jf)
        if hard_filter and not st["all"]:
            continue
        emp = (d.get("employer") or "").strip().lower()
        if cap > 0 and emp and seen.get(emp, 0) >= cap:
            continue
        rows.append(_make_result_personalized(len(rows) + 1, cos, idx, d, st, cos))
        if emp:
            seen[emp] = seen.get(emp, 0) + 1
        if len(rows) >= k:
            break
    return rows


FACET_TAIL_VALUES = {
    "role_family": {"other"},
    "industry": {"unclassified"},
}


def _retrieve_for_facets(
    query: str,
    filters: dict[str, str | list[str]],
    pool: int,
    qv_profile: list[float] | None = None,
) -> list[tuple[int, float]]:
    """The retrieval whose top-`pool` docs we facet over, so facet values always match
    what the user is actually looking at: fused query results when there's a query;
    else a profile-KNN pool when a profile is driving a blank personalized browse;
    else the blank recency-browse pool."""
    if query and query.strip():
        return _fused_topk(query, pool, filters, pool)
    if qv_profile is not None:
        return _topk_knn("e5_vec", qv_profile, pool, filters)
    return _browse_topk(pool, filters, pool)


def _aggregate_facets(items: list[tuple[int, float]]) -> dict[str, list[tuple[str, float]]]:
    """Rank-weighted (1/(rank+1)) value tallies per facet field over `items`, so values
    at the head of the result list dominate ordering. Tail values (role 'other',
    'unclassified') sink to the bottom. Ordinal/static ordering is applied client-side."""
    ids = [i for i, _ in items]
    hyd = _hydrate(ids, with_facets=True)
    weights: dict[str, dict[str, float]] = {f: defaultdict(float) for f in FACET_FIELDS}
    for rank, (i, _s) in enumerate(items):
        w = 1.0 / (rank + 1)
        d = hyd.get(i, {})
        for f in FACET_FIELDS:
            v = d.get(f)
            if v is None or v == "":
                continue
            for vv in v if isinstance(v, list) else [v]:
                if vv:
                    weights[f][vv] += w
    out: dict[str, list[tuple[str, float]]] = {}
    for f in FACET_FIELDS:
        tail = FACET_TAIL_VALUES.get(f, set())
        out[f] = sorted(weights[f].items(), key=lambda x: (x[0] in tail, -x[1]))
    return out


def compute_facets(
    query: str,
    filters: dict[str, str | list[str]] | None = None,
    pool: int = 200,
    qv_profile: list[float] | None = None,
) -> dict[str, list[tuple[str, float]]]:
    """Facet value tallies over the top-`pool` results. Returns {field: [(value, w)]}.
    For multi-select usability, each actively-filtered field's options are recomputed
    against the pool filtered by all OTHER fields — otherwise a field constrained by its
    own selection would only show the chosen values, leaving no way to add OR options.
    When qv_profile is given (personalized blank browse), the pool is the profile-KNN
    set, so facets reflect the profile-ranked results, not the generic recency browse."""
    filters = filters or {}
    out = _aggregate_facets(_retrieve_for_facets(query, filters, pool, qv_profile))
    for f in list(filters):
        if not filters[f]:
            continue
        alt = {k: v for k, v in filters.items() if k != f}
        out[f] = _aggregate_facets(_retrieve_for_facets(query, alt, pool, qv_profile)).get(
            f, out.get(f, [])
        )
    return out


SERVING_MODE = "RRF: BM25 + e5-small [via Solr]"


# ===== autocomplete (server-side, in-process; identical to demo) =====


def _prefix_matches(keys: list[str], prefix: str, limit: int) -> list[str]:
    lo = bisect.bisect_left(keys, prefix)
    out: list[str] = []
    for i in range(lo, len(keys)):
        if not keys[i].startswith(prefix):
            break
        out.append(keys[i])
        if len(out) >= limit:
            break
    return out


def _expand_prefix(prefix: str) -> list[str]:
    parts = prefix.split(" ", 1)
    head = parts[0]
    rest = (" " + parts[1]) if len(parts) > 1 else ""
    out = [prefix]
    if head in ABBREV_EXPANSIONS:
        for exp in ABBREV_EXPANSIONS[head]:
            out.append(exp + rest)
    return out


# ===== FastAPI app =====


@asynccontextmanager
async def lifespan(_app: FastAPI):
    load_resources()
    yield


app = FastAPI(title="Jobs Search Demo (Solr)", lifespan=lifespan)


HTML_PAGE = """<!doctype html>
<html><head><meta charset="utf-8"><title>__PAGE_TITLE__</title>
<style>
body { font-family: -apple-system, system-ui, sans-serif; max-width: 1100px; margin: 30px auto; padding: 0 16px; color: #222; }
h1 { font-size: 1.4em; margin-bottom: 8px; }
.subtle { color: #777; font-size: 0.9em; margin-bottom: 18px; }
.search { display: flex; gap: 8px; margin-bottom: 6px; position: relative; }
.qwrap { flex: 1; position: relative; }
#query { width: 100%; padding: 8px 12px; font-size: 1.05em; border: 1px solid #ccc; border-radius: 4px; box-sizing: border-box; }
#suggest { position: absolute; top: 100%; left: 0; right: 0; background: #fff; border: 1px solid #ccc; border-top: none; border-radius: 0 0 4px 4px; box-shadow: 0 4px 8px rgba(0,0,0,0.08); max-height: 280px; overflow-y: auto; z-index: 100; display: none; }
#suggest .item { padding: 6px 12px; cursor: pointer; font-size: 0.95em; color: #333; }
#suggest .item:hover, #suggest .item.active { background: #eef4fb; }
#suggest .hint { font-size: 0.75em; color: #aaa; margin-left: 8px; }
.badge { display: inline-block; padding: 3px 10px; border-radius: 12px; font-size: 0.82em; margin-bottom: 12px; }
.badge.cached { background: #e8f4ec; color: #186537; border: 1px solid #b9dec5; }
.badge.uncached { background: #f4eee8; color: #6b4a18; border: 1px solid #ddc8a8; }
button { padding: 8px 18px; font-size: 1em; cursor: pointer; border: 1px solid #888; border-radius: 4px; background: #fafafa; }
.results-panel { border: 1px solid #ddd; border-radius: 6px; padding: 10px 14px; background: #fff; }
.result { display: grid; grid-template-columns: 28px 60px 70px 1fr; gap: 10px; padding: 9px 0; border-bottom: 1px dotted #eee; font-size: 0.95em; align-items: start; cursor: pointer; }
.result:hover { background: #fafafa; }
.r-rank { color: #aaa; text-align: right; }
.r-score { color: #555; font-variant-numeric: tabular-nums; }
.r-source { color: #888; font-size: 0.82em; text-transform: uppercase; letter-spacing: 0.5px; }
.r-title { color: #222; word-break: break-word; }
.r-title .t { font-weight: 500; }
.r-title .m { color: #666; font-size: 0.85em; margin-top: 3px; }
.r-title .m2 { color: #888; font-size: 0.8em; margin-top: 2px; font-style: italic; }
.r-title .sep { color: #ccc; padding: 0 6px; }
.detail { grid-column: 4 / 5; margin-top: 8px; padding: 10px 12px; background: #f7f7f9; border-left: 3px solid #c4c4cc; border-radius: 3px; white-space: pre-wrap; color: #333; font-size: 0.88em; line-height: 1.45; max-height: 480px; overflow-y: auto; }
.detail.loading { color: #888; font-style: italic; }
.mlt-pivot { margin-top: 10px; display: inline-block; font-size: 0.85em; font-weight: 600; color: #2b6cb0; cursor: pointer; }
.mlt-pivot:hover { text-decoration: underline; }
.mlt-head .hl { font-size: 1.0em; }
.empty { color: #999; padding: 30px; text-align: center; }
.timing { font-size: 0.8em; color: #888; padding-top: 8px; }
.layout { display: grid; grid-template-columns: 240px 1fr; gap: 18px; }
.facets { font-size: 0.88em; }
.facet { margin-bottom: 14px; padding-bottom: 10px; border-bottom: 1px solid #eee; }
.facet h3 { font-size: 0.82em; text-transform: uppercase; letter-spacing: 0.5px; color: #888; margin: 0 0 6px 0; font-weight: 600; }
.facet .opt { display: flex; justify-content: space-between; padding: 2px 0; cursor: pointer; color: #444; }
.facet .opt:hover { color: #0a5fbf; }
.facet .opt.active { color: #0a5fbf; font-weight: 500; }
.facet .opt .v { flex: 1; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.facet .opt .n { color: #999; font-size: 0.85em; font-variant-numeric: tabular-nums; }
.facet .clear { color: #c33; cursor: pointer; font-size: 0.8em; }
.facet-empty { color: #aaa; font-style: italic; font-size: 0.85em; padding: 6px 0; }
.active-filters { font-size: 0.85em; color: #666; margin-bottom: 8px; }
.active-filters .chip { display: inline-block; background: #eef4fb; color: #0a5fbf; padding: 2px 8px; border-radius: 10px; margin-right: 6px; cursor: pointer; }
.active-filters .chip::after { content: ' ×'; color: #888; }
.ownbox { border: 1px solid #ddd; border-radius: 6px; background: #fff; margin-bottom: 16px; }
.ownbox > summary { padding: 9px 12px; cursor: pointer; font-size: 0.92em; font-weight: 600; color: #2b6cb0; list-style: none; }
.ownbox > summary::-webkit-details-marker { display: none; }
.ownbox > summary::before { content: '\\25b8 '; color: #999; }
.ownbox[open] > summary::before { content: '\\25be '; }
.ownbody { padding: 0 12px 12px; }
#own-text { width: 100%; min-height: 120px; box-sizing: border-box; padding: 8px 10px; font-size: 0.88em; font-family: inherit; border: 1px solid #ddd; border-radius: 5px; resize: vertical; }
.ownrow { display: flex; gap: 10px; align-items: center; margin-top: 9px; flex-wrap: wrap; }
#own-loc { flex: 1; min-width: 180px; padding: 7px 10px; font-size: 0.86em; border: 1px solid #ddd; border-radius: 5px; box-sizing: border-box; }
#own-go { padding: 7px 16px; font-size: 0.88em; background: #2b6cb0; color: #fff; border: 1px solid #2b6cb0; border-radius: 5px; cursor: pointer; }
#own-go:hover { background: #245a96; }
.ownstatus { font-size: 0.82em; color: #b3261e; margin-top: 7px; min-height: 1em; }
.rsum { border: 1px solid #ddd; border-radius: 6px; padding: 12px 14px; background: #fafbfc; margin-bottom: 14px; }
.rsum .nm { font-weight: 600; font-size: 1.05em; }
.rsum .hl { color: #444; margin: 3px 0; }
.rsum .facts { color: #555; font-size: 0.85em; margin-top: 6px; }
.rsum .facts b { color: #222; }
.rsum .back { float: right; font-size: 0.82em; color: #2b6cb0; cursor: pointer; }
.panels { display: grid; grid-template-columns: 1fr 1fr; gap: 14px; }
.panel { border: 1px solid #ddd; border-radius: 6px; background: #fff; }
.panel h3 { margin: 0; padding: 9px 12px; font-size: 0.9em; border-bottom: 1px solid #eee; }
.panel.cos h3 { background: #f4eee8; color: #6b4a18; }
.panel.flt h3 { background: #e8f4ec; color: #186537; }
.panel .note { font-size: 0.78em; color: #999; padding: 6px 12px; border-bottom: 1px dotted #eee; }
.job { padding: 9px 12px; border-bottom: 1px dotted #eee; cursor: pointer; }
.job:hover { background: #fafafa; }
.job .jt { font-weight: 500; font-size: 0.92em; }
.job .jm { color: #777; font-size: 0.8em; margin-top: 2px; }
.job .jm .sep { color: #ccc; padding: 0 5px; }
.job .badges { margin-top: 5px; }
.b { display: inline-block; font-size: 0.72em; padding: 1px 7px; border-radius: 9px; margin-right: 5px; }
.b.ok { background: #e6f4ea; color: #1a7a3a; border: 1px solid #b6dec4; }
.b.bad { background: #fbe9e7; color: #b3261e; border: 1px solid #f0c2bd; }
.b.warn { background: #fff4e5; color: #8a5a00; border: 1px solid #f0d9a8; }
.cos-num { color: #555; font-variant-numeric: tabular-nums; font-size: 0.8em; float: right; }
.jobdetail { margin-top: 7px; padding: 9px 11px; background: #f7f7f9; border-left: 3px solid #c4c4cc; border-radius: 3px; white-space: pre-wrap; color: #333; font-size: 0.84em; line-height: 1.4; max-height: 320px; overflow-y: auto; }
.jobdetail.loading { color: #888; font-style: italic; }
.related { margin-bottom: 14px; }
.related .rel-h { font-size: 0.8em; text-transform: uppercase; letter-spacing: 0.5px; color: #888; margin-bottom: 7px; }
.related .rel-chips { display: flex; flex-wrap: wrap; gap: 8px; }
.sug { background: #eef4fb; color: #0a5fbf; border: 1px solid #cfe0f5; border-radius: 14px; padding: 4px 12px; font-size: 0.86em; cursor: pointer; }
.sug:hover { background: #dde9f8; }
.sug .sug-n { color: #7aa3d0; font-size: 0.82em; margin-left: 4px; }
#personalize-row { margin: 4px 0 12px; font-size: 0.88em; color: #444; }
#personalize-row label { cursor: pointer; }
#personalize-row .pz-name { color: #888; margin-left: 6px; }
.fit { display: inline-block; font-size: 0.72em; padding: 1px 7px; border-radius: 9px; margin-right: 5px; background: #eef0fb; color: #3a45a0; border: 1px solid #cfd3f0; }
/* facet controls: checkbox (multi-select OR) / radio (single-select recency) */
.facet .opt .cbox { flex: 0 0 auto; width: 12px; height: 12px; margin-right: 7px; border: 1px solid #bbb; border-radius: 3px; background: #fff; display: inline-block; position: relative; }
.facet .opt .cbox.radio { border-radius: 50%; }
.facet .opt .cbox.on { background: #2b6cb0; border-color: #2b6cb0; }
.facet .opt .cbox.on::after { content: '✓'; color: #fff; font-size: 9px; line-height: 12px; position: absolute; left: 1px; top: -1px; }
.facet .opt .cbox.radio.on::after { content: ''; left: 3px; top: 3px; width: 6px; height: 6px; background: #fff; border-radius: 50%; }
.facet .moreless { color: #2b6cb0; cursor: pointer; font-size: 0.82em; margin-top: 4px; }
.facet .moreless:hover { text-decoration: underline; }
.facet h3 .map-link { float: right; color: #2b6cb0; cursor: pointer; text-transform: none; letter-spacing: 0; font-weight: 500; font-size: 0.95em; }
.facet h3 .map-link::before { content: '🗺 '; }
.facet h3 .map-link:hover { text-decoration: underline; }
/* map picker modal */
.map-modal { position: fixed; inset: 0; background: rgba(0,0,0,0.4); z-index: 500; display: flex; align-items: center; justify-content: center; }
.map-card { background: #fff; border-radius: 8px; padding: 14px 16px; width: min(760px, 94vw); max-height: 92vh; overflow: auto; box-shadow: 0 8px 30px rgba(0,0,0,0.25); }
.map-head { display: flex; justify-content: space-between; align-items: center; font-weight: 600; font-size: 1.05em; }
.map-head .map-close { cursor: pointer; color: #888; font-size: 1.5em; line-height: 1; padding: 0 4px; }
.map-head .map-close:hover { color: #333; }
.map-hint { color: #888; font-size: 0.82em; margin: 4px 0 8px; }
.map-wrap { width: 100%; }
.map-foot { display: flex; justify-content: space-between; align-items: center; margin-top: 8px; }
.map-foot #map-sel { color: #555; font-size: 0.85em; }
.map-foot .map-done { background: #2b6cb0; color: #fff; border: 1px solid #2b6cb0; border-radius: 5px; padding: 6px 16px; cursor: pointer; }
.map-attr { color: #bbb; font-size: 0.72em; margin-top: 8px; text-align: right; }
.geomap { width: 100%; height: auto; display: block; }
.geomap path, .geomap g { fill: #e8e8ec; stroke: #fff; stroke-width: 0.7; cursor: pointer; transition: fill 0.1s; }
.geomap path:hover, .geomap g:hover path { fill: #cfe0f5; }
.geomap .hasdata, .geomap .hasdata path { fill: #bcd3ef; }
.geomap .sel, .geomap .sel path { fill: #2b6cb0; }
.geomap .sel:hover, .geomap .sel:hover path { fill: #245a96; }
</style></head>
<body>
<h1>__PAGE_TITLE__</h1>
<div class="subtle">__PAGE_SUBTITLE__</div>
<details class="ownbox">
  <summary>Find jobs for yourself &mdash; paste your profile, or upload a .txt / LinkedIn PDF</summary>
  <div class="ownbody">
    <textarea id="own-text" placeholder="Paste your LinkedIn &lsquo;About&rsquo; + experience, or any resume text&hellip;&#10;(LinkedIn URLs can't be fetched server-side, so paste or upload the PDF export: Profile &rarr; Resources &rarr; Save to PDF.)"></textarea>
    <div class="ownrow">
      <input id="own-loc" placeholder="Your location (optional, e.g. 'Boston, MA' &mdash; improves location matching)" autocomplete="off" />
      <input type="file" id="own-file" accept=".txt,.pdf" />
      <button id="own-go">Match my profile</button>
    </div>
    <div id="own-status" class="ownstatus"></div>
  </div>
</details>
<div class="search">
  <div class="qwrap">
    <input id="query" placeholder="e.g. registered nurse" autocomplete="off" />
    <div id="suggest"></div>
  </div>
  <button onclick="runSearch()">Search</button>
</div>
<div id="personalize-row" style="display:none">
  <label><input type="checkbox" id="pz-on"> &#10024; Personalize results to my profile</label>
  <label id="pz-hard-wrap" style="display:none; margin-left:16px"><input type="checkbox" id="pz-hard"> only jobs I qualify for (3-axis filter)</label>
  <span class="pz-name" id="pz-name"></span>
</div>
<div id="badge-row"></div>
<div id="active-filters" class="active-filters"></div>
<div class="layout">
  <div class="facets" id="facets"></div>
  <div class="results-panel">
    <div id="related" class="related"></div>
    <div id="results"><div class="empty">loading recent jobs…</div></div>
  </div>
</div>
<div id="map-modal" class="map-modal" style="display:none">
  <div class="map-card">
    <div class="map-head"><span id="map-title"></span><span class="map-close">&times;</span></div>
    <div class="map-hint">Click regions to toggle filters (multi-select OR) &middot; shaded regions have results in the current view.</div>
    <div id="map-us" class="map-wrap" style="display:none">__US_MAP_SVG__</div>
    <div id="map-world" class="map-wrap" style="display:none">__WORLD_MAP_SVG__</div>
    <div class="map-foot"><span id="map-sel"></span><button class="map-done">Done</button></div>
    <div class="map-attr">US states: WebsiteBeaver (MIT) &middot; world map: simple-world-map (CC BY-SA 3.0)</div>
  </div>
</div>
<script>
const input = document.getElementById('query');
const suggestBox = document.getElementById('suggest');
let suggestItems = [];
let suggestActive = -1;
let suggestTimer = null;
let profile = null;   // parsed profile {r, qv} from /api/match_profile; client-held, re-sent to personalize
let profileSuggestions = [];   // [{q, n}] profile-derived searches; shown in the related slot on a blank personalized browse

function esc(s) { return (s == null ? '' : String(s)).replace(/[&<>"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c])); }
function closeSuggest() { suggestBox.style.display = 'none'; suggestActive = -1; }
function renderSuggest(items) {
  // items are {text}; rendered as plain suggestions (no source badge).
  suggestItems = items.map(s => (typeof s === 'string' ? {text: s} : s));
  if (!suggestItems.length) { closeSuggest(); return; }
  suggestBox.innerHTML = suggestItems.map((s, i) =>
    `<div class="item" data-i="${i}">${esc(s.text)}</div>`
  ).join('');
  suggestBox.style.display = 'block';
  suggestActive = -1;
  suggestBox.querySelectorAll('.item').forEach(el => {
    el.addEventListener('mousedown', e => {
      e.preventDefault();
      input.value = suggestItems[parseInt(el.dataset.i)].text;
      closeSuggest();
      runSearch();
    });
  });
}
async function fetchSuggest() {
  const q = input.value.trim();
  if (!q) { closeSuggest(); return; }
  try {
    const r = await fetch('/api/suggest?q=' + encodeURIComponent(q));
    const d = await r.json();
    renderSuggest(d.suggestions || []);
  } catch (e) { closeSuggest(); }
}
input.addEventListener('input', () => {
  clearTimeout(suggestTimer);
  suggestTimer = setTimeout(fetchSuggest, 90);
});
input.addEventListener('keydown', e => {
  const visible = suggestBox.style.display === 'block';
  if (e.key === 'Enter') {
    if (visible && suggestActive >= 0 && suggestItems[suggestActive]) {
      input.value = suggestItems[suggestActive].text;
    }
    closeSuggest();
    runSearch();
  } else if (e.key === 'ArrowDown' && visible) {
    e.preventDefault();
    suggestActive = Math.min(suggestActive + 1, suggestItems.length - 1);
    suggestBox.querySelectorAll('.item').forEach((el, i) => el.classList.toggle('active', i === suggestActive));
  } else if (e.key === 'ArrowUp' && visible) {
    e.preventDefault();
    suggestActive = Math.max(suggestActive - 1, -1);
    suggestBox.querySelectorAll('.item').forEach((el, i) => el.classList.toggle('active', i === suggestActive));
  } else if (e.key === 'Escape') {
    closeSuggest();
  }
});
input.addEventListener('blur', () => setTimeout(closeSuggest, 120));
const SRC_SHORT = {
  'jobs_data': 'OAP', 'jobs_data_linkedin': 'LI', 'jobs_data_jobstreet': 'JS', 'jobs_data_usajobs': 'USA'
};
function shortSrc(s) { return s == null ? '' : (SRC_SHORT[s] || s); }
function metaLine(r) {
  const parts = [];
  if (r.employer) parts.push(esc(r.employer));
  if (r.industry && r.industry !== 'unclassified') parts.push(esc(facetValueLabel('industry', r.industry)));
  if (r.location) parts.push(esc(r.location));
  if (r.employment_type) parts.push(esc(r.employment_type));
  if (r.salary) parts.push(esc(r.salary));
  if (!parts.length) return '';
  return `<div class="m">${parts.join('<span class="sep">·</span>')}</div>`;
}
function metaLine2(r) {
  const parts = [];
  if (r.department) parts.push(esc(r.department));
  if (r.posted) parts.push('Posted ' + esc(r.posted));
  if (!parts.length) return '';
  return `<div class="m2">${parts.join('<span class="sep">·</span>')}</div>`;
}
async function toggleDetail(idx, container) {
  let existing = container.querySelector('.detail');
  if (existing) { existing.remove(); return; }
  const div = document.createElement('div');
  div.className = 'detail loading';
  div.textContent = 'loading...';
  container.appendChild(div);
  try {
    const r = await fetch('/api/detail?idx=' + idx);
    const data = await r.json();
    div.classList.remove('loading');
    div.textContent = data.description || '(no description)';
    const mlt = document.createElement('div');
    mlt.className = 'mlt-pivot';
    mlt.textContent = '→ More jobs like this one';
    mlt.addEventListener('click', (e) => {
      e.stopPropagation();
      pivotMoreLikeThis(idx, data.title || '');
    });
    div.appendChild(mlt);
  } catch (e) {
    div.classList.remove('loading');
    div.textContent = '(failed to load)';
  }
}

// ===== "more jobs like this one" pivot — loads the similar set into the main pane =====
let lastQuery = '';   // restored by the pivot's "back to search" affordance
async function pivotMoreLikeThis(idx, title) {
  const div = document.getElementById('results');
  document.getElementById('badge-row').innerHTML = '';
  document.getElementById('facets').innerHTML = '';
  document.getElementById('active-filters').innerHTML = '';
  document.getElementById('related').innerHTML = '';
  div.innerHTML = '<div class="empty">finding similar jobs…</div>';
  let d;
  try { d = await fetch('/api/more_like_this?idx=' + idx).then(r => r.json()); }
  catch (e) { div.innerHTML = '<div class="empty">(failed to load similar jobs)</div>'; return; }
  const srcTitle = (d.source && d.source.title) || title;
  if (d.served_with) {
    document.getElementById('badge-row').innerHTML =
      `<span class="badge cached">Served with: ${esc(d.served_with)}</span>`;
  }
  div.innerHTML = `<div class="rsum mlt-head">
      <span class="back">&larr; back to search</span>
      <div class="hl">Jobs similar to: <b>${esc(srcTitle)}</b></div>
      <div class="lc" style="color:#888;font-size:0.85em">RRF(BM25 + e5-small) seeded by this posting</div>
    </div>
    <div id="mlt-list"></div>`;
  div.querySelector('.back').addEventListener('click', () => {
    if (lastQuery) { input.value = lastQuery; runSearch(); } else { clearMatch(); }
  });
  renderResults(div.querySelector('#mlt-list'), d.results, d.ms);
}
function renderResults(div, items, ms) {
  if (!items || !items.length) { div.innerHTML = '<div class="empty">no results</div>'; return; }
  div.innerHTML = '';
  items.forEach(r => {
    const row = document.createElement('div');
    row.className = 'result';
    let fit = '';
    if (r.axes) {
      const cos = (r.cosine != null) ? `<span class="fit" title="profile-to-job embedding similarity">fit ${r.cosine.toFixed(3)}</span>` : '';
      fit = `<div class="badges" style="margin-top:5px">${cos}${badge('sen', r.axes.sen)}${badge('loc', r.axes.loc)}${badge('gate', r.axes.gate)}</div>`;
    }
    row.innerHTML = `<span class="r-rank">${r.rank}</span><span class="r-score">${r.score.toFixed(4)}</span><span class="r-source">${esc(shortSrc(r.source))}</span><span class="r-title"><div class="t">${esc(r.title)}</div>${metaLine(r)}${metaLine2(r)}${fit}</span>`;
    if (r.idx != null && r.idx >= 0) {
      const titleCell = row.querySelector('.r-title');
      row.addEventListener('click', () => toggleDetail(r.idx, titleCell));
    }
    div.appendChild(row);
  });
  if (ms != null) {
    const t = document.createElement('div');
    t.className = 'timing';
    t.textContent = ms + ' ms';
    div.appendChild(t);
  }
}
const FACET_FIELDS = [
  'role_family', 'seniority', 'industry', 'remote_mode',
  'location_country', 'location_state',
  'posted_bucket', 'salary_band_usd_annual', 'tech_stack',
];
const FACET_LABELS = {
  role_family: 'Role family',
  seniority: 'Seniority',
  industry: 'Industry',
  remote_mode: 'Remote mode',
  location_country: 'Country',
  location_state: 'US state',
  posted_bucket: 'Posted',
  salary_band_usd_annual: 'Salary (USD/yr)',
  tech_stack: 'Tech stack',
};
const FACET_VALUE_LABELS = {
  posted_bucket: {
    past_24h: 'Past 24 hours',
    past_7d: 'Past 7 days',
    past_30d: 'Past 30 days',
    past_90d: 'Past 90 days',
    older: 'Older than 90 days',
  },
  seniority: {
    intern: 'Intern', entry: 'Entry level', junior: 'Junior', mid: 'Mid level',
    senior: 'Senior', lead: 'Lead', staff: 'Staff', manager: 'Manager',
    senior_manager: 'Senior manager', director: 'Director', vp: 'VP',
    c_level: 'C-level', not_specified: 'Not specified',
  },
  salary_band_usd_annual: {
    under_50k: 'Under $50k', '50k_75k': '$50k–75k', '75k_100k': '$75k–100k',
    '100k_150k': '$100k–150k', '150k_200k': '$150k–200k', '200k_300k': '$200k–300k',
    '300k_plus': '$300k+', not_specified: 'Not specified',
  },
  remote_mode: {
    on_site: 'On-site', remote: 'Remote', hybrid: 'Hybrid', not_specified: 'Not specified',
  },
  industry: {
    tech_software_internet: 'Software / Internet',
    tech_hardware_semiconductors: 'Hardware / Semiconductors',
    finance_banking: 'Banking',
    finance_fintech: 'Fintech',
    finance_insurance: 'Insurance',
    healthcare_provider: 'Healthcare provider',
    healthcare_pharma_biotech: 'Pharma / Biotech',
    healthcare_devices: 'Medical devices',
    retail_ecommerce: 'Retail / E-commerce',
    consumer_brands: 'Consumer brands',
    media_entertainment: 'Media / Entertainment',
    gaming: 'Gaming',
    automotive: 'Automotive',
    energy_utilities: 'Energy / Utilities',
    public_sector_government: 'Government / Public sector',
    defense_aerospace: 'Defense / Aerospace',
    nonprofit: 'Nonprofit',
    education_higher: 'Higher education',
    education_k12: 'K-12 education',
    consulting_professional_services: 'Consulting / Professional services',
    legal_services: 'Legal',
    real_estate_construction: 'Real estate / Construction',
    agriculture_food_production: 'Agriculture / Food production',
    manufacturing: 'Manufacturing',
    telecommunications: 'Telecom',
    transportation_logistics: 'Transportation / Logistics',
    hospitality_food_service: 'Hospitality / Food service',
    unclassified: 'Unclassified',
  },
};
function facetValueLabel(f, v) {
  return (FACET_VALUE_LABELS[f] && FACET_VALUE_LABELS[f][v]) || v;
}
// Static presentation order for ordinal facets (low->high) + remote_mode.
const ORDINAL_ORDER = {
  seniority: ['intern','entry','junior','mid','senior','lead','staff','manager','senior_manager','director','vp','c_level','not_specified'],
  salary_band_usd_annual: ['under_50k','50k_75k','75k_100k','100k_150k','150k_200k','200k_300k','300k_plus','not_specified'],
  remote_mode: ['on_site','remote','hybrid','not_specified'],
  posted_bucket: ['past_24h','past_7d','past_30d','past_90d','older'],
};
const TOGGLE_FACETS = new Set(['role_family','industry','location_state','location_country','tech_stack']); // More/Less
const MAP_FACETS = { location_state: 'us', location_country: 'world' };  // also offer a map picker
const SINGLE_SELECT = new Set(['posted_bucket']);   // everything else is multi-select OR
const FACET_TOP_N = 8;
const expandedFacets = new Set();
let lastFacets = {};

// activeFilters: field -> array of values (multi-select) | string (posted_bucket).
const activeFilters = {};
function selectedList(f) { const a = activeFilters[f]; return a == null ? [] : (Array.isArray(a) ? a : [a]); }
function isSelected(f, v) { return selectedList(f).includes(v); }
function toggleFilter(f, v) {
  if (SINGLE_SELECT.has(f)) {
    if (activeFilters[f] === v) delete activeFilters[f]; else activeFilters[f] = v;
  } else {
    let a = selectedList(f);
    a = a.includes(v) ? a.filter(x => x !== v) : a.concat([v]);
    if (a.length) activeFilters[f] = a; else delete activeFilters[f];
  }
}

function buildFilterQS() {
  const parts = [];
  for (const [k, v] of Object.entries(activeFilters)) {
    for (const x of (Array.isArray(v) ? v : (v ? [v] : []))) parts.push(`${k}=${encodeURIComponent(x)}`);
  }
  return parts.length ? '&' + parts.join('&') : '';
}
function renderActiveFilters() {
  const row = document.getElementById('active-filters');
  const chips = [];
  for (const f of Object.keys(activeFilters)) {
    for (const v of selectedList(f)) {
      chips.push(`<span class="chip" data-k="${f}" data-v="${esc(v)}">${esc(FACET_LABELS[f] || f)}: ${esc(facetValueLabel(f, v))}</span>`);
    }
  }
  if (!chips.length) { row.innerHTML = ''; return; }
  row.innerHTML = 'Filters: ' + chips.join('');
  row.querySelectorAll('.chip').forEach(el => el.addEventListener('click', () => {
    toggleFilter(el.dataset.k, el.dataset.v);
    runSearch();
  }));
}
// Order a facet's options for display: ordinal facets in fixed low->high order,
// others in backend weight order. Selected values absent from the current pool are
// kept (weight 0) so they remain de-selectable.
function orderedOpts(f, opts) {
  const present = new Map((opts || []).map(([v, w]) => [v, w]));
  selectedList(f).forEach(v => { if (!present.has(v)) present.set(v, 0); });
  if (ORDINAL_ORDER[f]) {
    return ORDINAL_ORDER[f].filter(v => present.has(v)).map(v => [v, present.get(v)]);
  }
  const arr = (opts || []).slice();
  selectedList(f).forEach(v => { if (!arr.some(o => o[0] === v)) arr.push([v, 0]); });
  return arr;
}
function renderFacets(facets) {
  lastFacets = facets || {};
  const root = document.getElementById('facets');
  const parts = [];
  for (const f of FACET_FIELDS) {
    const opts = orderedOpts(f, (facets && facets[f]) || []);
    if (!opts.length) continue;
    const isToggle = TOGGLE_FACETS.has(f);
    const expanded = expandedFacets.has(f);
    // Collapsed toggle facet: show the top N, but always include any selected value
    // that would otherwise be hidden under "More" so the user can see/clear it.
    let shown = opts;
    if (isToggle && !expanded) {
      shown = opts.slice(0, FACET_TOP_N);
      for (const o of opts.slice(FACET_TOP_N)) if (isSelected(f, o[0])) shown.push(o);
    }
    const single = SINGLE_SELECT.has(f);
    let inner = `<h3>${esc(FACET_LABELS[f] || f)}`;
    if (MAP_FACETS[f]) inner += `<span class="map-link" data-mapf="${f}">map</span>`;
    inner += `</h3>`;
    inner += shown.map(([v]) => {
      const on = isSelected(f, v);
      const box = `<span class="cbox${single ? ' radio' : ''}${on ? ' on' : ''}"></span>`;
      return `<div class="opt${on ? ' active' : ''}" data-f="${f}" data-v="${esc(v)}">${box}<span class="v">${esc(facetValueLabel(f, v))}</span></div>`;
    }).join('');
    if (isToggle && opts.length > FACET_TOP_N) {
      const hidden = opts.length - shown.length;
      if (expanded) inner += `<div class="moreless" data-f="${f}">− Less</div>`;
      else if (hidden > 0) inner += `<div class="moreless" data-f="${f}">+ More (${hidden})</div>`;
    }
    parts.push(`<div class="facet">${inner}</div>`);
  }
  root.innerHTML = parts.join('') || '<div class="facet-empty">no facets</div>';
  root.querySelectorAll('.opt').forEach(el => el.addEventListener('click', () => {
    toggleFilter(el.dataset.f, el.dataset.v);
    runSearch();
  }));
  root.querySelectorAll('.moreless').forEach(el => el.addEventListener('click', () => {
    const f = el.dataset.f;
    if (expandedFacets.has(f)) expandedFacets.delete(f); else expandedFacets.add(f);
    renderFacets(lastFacets);
  }));
  root.querySelectorAll('.map-link').forEach(el => el.addEventListener('click', (e) => {
    e.stopPropagation();
    openMap(el.dataset.mapf);
  }));
  if (document.getElementById('map-modal').style.display !== 'none') repaintMaps();
}

// ===== map picker (country / US state) — clicks drive the same activeFilters =====
let mapField = null;
function regionCodeFromEvent(e) {
  let el = e.target;
  while (el && el.tagName && el.tagName.toLowerCase() !== 'svg') {
    if (el.id && /^[A-Za-z]{2}$/.test(el.id)) return el.id.toUpperCase();
    el = el.parentNode;
  }
  return null;
}
function paintMap(svgEl, field, facetVals) {
  if (!svgEl) return;
  const has = new Set((facetVals || []).map(o => o[0]));
  const sel = new Set(selectedList(field));
  svgEl.querySelectorAll('[id]').forEach(el => {
    if (!/^[A-Za-z]{2}$/.test(el.id)) return;
    const code = el.id.toUpperCase();
    el.classList.toggle('hasdata', has.has(code));
    el.classList.toggle('sel', sel.has(code));
  });
}
function repaintMaps() {
  paintMap(document.querySelector('#map-us .geomap'), 'location_state', lastFacets.location_state);
  paintMap(document.querySelector('#map-world .geomap'), 'location_country', lastFacets.location_country);
  if (mapField) {
    const sel = selectedList(mapField);
    document.getElementById('map-sel').textContent = sel.length ? (sel.length + ' selected: ' + sel.join(', ')) : 'none selected';
  }
}
function openMap(field) {
  mapField = field;
  document.getElementById('map-us').style.display = field === 'location_state' ? 'block' : 'none';
  document.getElementById('map-world').style.display = field === 'location_country' ? 'block' : 'none';
  document.getElementById('map-title').textContent = field === 'location_state' ? 'Filter by US state' : 'Filter by country';
  repaintMaps();
  document.getElementById('map-modal').style.display = 'flex';
}
function closeMap() { document.getElementById('map-modal').style.display = 'none'; }
['map-us', 'map-world'].forEach(id => {
  const field = id === 'map-us' ? 'location_state' : 'location_country';
  document.getElementById(id).addEventListener('click', e => {
    const code = regionCodeFromEvent(e);
    if (!code) return;
    toggleFilter(field, code);
    repaintMaps();
    runSearch();
  });
});
document.querySelector('.map-close').addEventListener('click', closeMap);
document.querySelector('.map-done').addEventListener('click', closeMap);
document.getElementById('map-modal').addEventListener('click', e => { if (e.target.id === 'map-modal') closeMap(); });
async function runSearch() {
  const q = input.value.trim();
  lastQuery = q;
  closeSuggest();
  if (profile && document.getElementById('pz-on').checked) { return runPersonalized(q); }
  const div = document.getElementById('results');
  const badgeRow = document.getElementById('badge-row');
  badgeRow.innerHTML = '';
  div.innerHTML = q ? '<div class="empty">searching...</div>' : '<div class="empty">loading recent jobs…</div>';
  renderActiveFilters();
  const qs = buildFilterQS();
  // Fire search + facets in parallel.
  const [searchRes, facetRes] = await Promise.all([
    fetch(`/api/search?q=${encodeURIComponent(q)}${qs}`).then(r => r.json()),
    fetch(`/api/facets?q=${encodeURIComponent(q)}${qs}`).then(r => r.json()),
  ]);
  if (searchRes.served_with) {
    badgeRow.innerHTML = `<span class="badge cached">Served with: ${esc(searchRes.served_with)}</span>`;
  }
  renderResults(div, searchRes.results, searchRes.ms);
  renderFacets(facetRes.facets);
  loadRelated(q);
}

// ===== suggested-searches slot at the top of the results panel =====
// Shared by query-context related searches AND profile-derived suggestions — they
// never co-exist (a profile match overwrites this slot), so one renderer serves both.
function renderRelated(label, items) {
  // items: [{q, n}] — q is the search to run, n the result count shown on the chip.
  const el = document.getElementById('related');
  if (!items || !items.length) { el.innerHTML = ''; return; }
  el.innerHTML = `<div class="rel-h">${esc(label)}</div><div class="rel-chips">`
    + items.map(s => `<span class="sug" data-q="${esc(s.q)}">${esc(s.q)}`
        + `<span class="sug-n">${(s.n || 0).toLocaleString()}</span></span>`).join('')
    + '</div>';
  el.querySelectorAll('.sug').forEach(c => c.addEventListener('click', () => {
    input.value = c.dataset.q;
    runSearch();
  }));
}
// related searches = narrow/lateral role moves for the current query
async function loadRelated(q) {
  document.getElementById('related').innerHTML = '';
  if (!q) return;
  let d;
  try { d = await fetch(`/api/related_searches?q=${encodeURIComponent(q)}`).then(r => r.json()); }
  catch (e) { return; }
  const sugs = (d && d.suggestions) || [];
  renderRelated('Related searches', sugs.map(s => ({ q: s.display, n: s.count })));
}

// ===== personalized keyword search (re-rank the query by the held profile) =====
async function runPersonalized(q) {
  const div = document.getElementById('results');
  const badgeRow = document.getElementById('badge-row');
  badgeRow.innerHTML = '';
  div.innerHTML = '<div class="empty">personalizing to your profile…</div>';
  renderActiveFilters();
  const hard = document.getElementById('pz-hard').checked;
  const body = { q, k: 10, hard_filter: hard, filters: activeFilters, profile };
  // Facets come back inline, computed over the SAME profile-ranked pool as the results
  // (a separate /api/facets call would be profile-blind and mismatch the listing).
  const searchRes = await fetch('/api/search_personalized', {
    method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body),
  }).then(r => r.json());
  if (searchRes.error) { div.innerHTML = '<div class="empty">' + esc(searchRes.error) + '</div>'; return; }
  badgeRow.innerHTML = `<span class="badge cached">Served with: ${esc(searchRes.served_with)}</span>`;
  if (!searchRes.results || !searchRes.results.length) {
    div.innerHTML = hard
      ? '<div class="empty">no jobs match this query that you also qualify for — untick the 3-axis filter to see near-misses</div>'
      : '<div class="empty">no results</div>';
  } else {
    renderResults(div, searchRes.results, searchRes.ms);
  }
  renderFacets(searchRes.facets);
  // On a blank profile-driven browse, surface the profile's suggested searches;
  // on a typed query, surface query-related role moves.
  if (q) loadRelated(q);
  else renderRelated('Suggested searches from your profile', profileSuggestions);
}
function showPersonalize(name) {
  document.getElementById('personalize-row').style.display = 'block';
  document.getElementById('pz-name').textContent =
    (name && name !== '(your profile)') ? '— using ' + name + "'s profile" : '— using your profile';
}
function togglePzHard() {
  const on = document.getElementById('pz-on').checked;
  document.getElementById('pz-hard-wrap').style.display = on ? 'inline' : 'none';
  if (!on) document.getElementById('pz-hard').checked = false;
}
// Re-run on toggle even with a blank query, so the default browse switches between
// recency and profile-ranked (and the 3-axis filter applies) without needing a query.
document.getElementById('pz-on').addEventListener('change', () => { togglePzHard(); runSearch(); });
document.getElementById('pz-hard').addEventListener('change', () => { runSearch(); });

// ===== "find jobs for yourself": profile -> jobs, cosine vs 3-axis filter =====
function badge(name, ax) {
  const cls = ax.ok ? 'ok' : 'bad';
  const mark = ax.ok ? '✓' : '✗';
  const tip = ax.reason ? ' — ' + ax.reason : '';
  return `<span class="b ${cls}" title="${esc(ax.reason)}">${name} ${mark}${ax.ok ? '' : esc(tip)}</span>`;
}
function clearMatch() {
  // leave the profile-match panel and return to the (possibly personalized) browse/search
  input.value = lastQuery || '';
  runSearch();
}
async function matchOwn() {
  const text = document.getElementById('own-text').value.trim();
  const loc = document.getElementById('own-loc').value.trim();
  const file = document.getElementById('own-file').files[0];
  const status = document.getElementById('own-status');
  if (!text && !file) { status.textContent = 'Paste some text or choose a .txt/.pdf file first.'; return; }
  const fd = new FormData();
  fd.append('text', text); fd.append('loc', loc);
  if (file) fd.append('file', file);
  status.textContent = 'matching…';
  document.getElementById('results').innerHTML = '<div class="empty">matching your profile…</div>';
  try {
    const r = await fetch('/api/match_profile', { method: 'POST', body: fd });
    const d = await r.json();
    if (!r.ok || d.error) {
      const msg = d.error || ('error ' + r.status);
      status.textContent = msg;
      document.getElementById('results').innerHTML = '<div class="empty">' + esc(msg) + '</div>';
      return;
    }
    status.textContent = '';
    profile = d.profile || null;
    profileSuggestions = (d.suggestions || []).map(s => ({ q: s.text, n: s.n }));
    if (profile) {
      // Uploading a profile turns the default view into a profile-driven browse:
      // a blank "match my profile" query, ranked by profile fit, with the facet rail
      // and filters fully available. Subsequent typed queries personalize too.
      showPersonalize(d.resume && d.resume.name);
      const pz = document.getElementById('pz-on');
      pz.checked = true;
      togglePzHard();
      const own = document.querySelector('.ownbox');
      if (own) own.open = false;   // collapse the upload panel to reveal results
      input.value = '';
      runSearch();   // -> runPersonalized('') -> browse_personalized + facets
    }
  } catch (e) { status.textContent = 'failed: ' + e; }
}
document.getElementById('own-go').addEventListener('click', matchOwn);
// Blank search runs by default on page load: recent + low-barrier jobs.
runSearch();
</script>
</body></html>
"""


@app.get("/", response_class=HTMLResponse)
def index():
    try:
        n = requests.get(
            f"{SOLR}/solr/{CORE}/select", params={"q": "*:*", "rows": "0"}, timeout=5
        ).json()["response"]["numFound"]
        n_str = f"{n:,}"
    except Exception:
        n_str = "~197,000"
    title = f"Jobs Search Demo: {n_str} postings across 2 corpora (Solr backend)"
    subtitle = (
        f"{n_str} postings (OpenApply + USAJobs) · RRF(BM25 + e5-small) via Solr 10 · "
        "browse recent jobs by default, then narrow with facets (multi-select) or the "
        "country / US-state maps · click a result for the full description · "
        "or paste your profile above to find jobs for yourself (3-axis constraint filter), "
        "get suggested searches, and personalize results"
    )
    return (
        HTML_PAGE.replace("__PAGE_TITLE__", title)
        .replace("__PAGE_SUBTITLE__", subtitle)
        .replace("__US_MAP_SVG__", US_STATES_SVG)
        .replace("__WORLD_MAP_SVG__", WORLD_SVG)
    )


_HTML_TAG_RE = re.compile(r"<[^>]+>")


def _solr_suggest(prefix: str, limit: int) -> list[str]:
    """Call Solr Suggester for the prefix, strip the highlighter tags, and
    return lowercase suggestion strings."""
    try:
        r = requests.get(
            f"{SOLR}/solr/{CORE}/suggest",
            params={
                "suggest": "true",
                "suggest.dictionary": "titleSuggester",
                "suggest.q": prefix,
                "suggest.count": str(limit),
            },
            timeout=2,
        )
        r.raise_for_status()
        sg = r.json().get("suggest", {}).get("titleSuggester", {})
        if not sg:
            return []
        entry = next(iter(sg.values()))
        return [
            _HTML_TAG_RE.sub("", s.get("term", "")).strip().lower()
            for s in entry.get("suggestions", [])
            if s.get("term")
        ]
    except Exception:
        return []


@app.get("/api/suggest")
def api_suggest(q: str = Query(""), limit: int = Query(10)):
    if not q or not R:
        return JSONResponse({"suggestions": []})
    prefix = q.strip().lower()
    prefixes = _expand_prefix(prefix)
    tiers = [
        R["tier_keys"].get("title", []),
        R["tier_keys"].get("combo", []),
        R["tier_keys"].get("head", []),
        R["tier_keys"].get("tail", []),
        R["tier_keys"].get("synth", []),
        R["sorted_keys"],
    ]
    seen: set[str] = set()
    out: list[dict] = []
    for tier in tiers:
        for p in prefixes:
            for k in _prefix_matches(tier, p, limit * 2):
                if k not in seen:
                    seen.add(k)
                    out.append({"text": k})
                    if len(out) >= limit:
                        break
            if len(out) >= limit:
                break
        if len(out) >= limit:
            break
    if len(out) < limit:
        for s in _solr_suggest(prefix, limit - len(out)):
            if s and s not in seen:
                seen.add(s)
                out.append({"text": s})
                if len(out) >= limit:
                    break
    return JSONResponse({"suggestions": out})


@app.get("/api/related_searches")
def api_related_searches(q: str = Query(""), k: int = Query(4)):
    """Related searches for a query: NARROW (software engineer -> ML engineer) or
    LATERAL (-> data engineer) role moves mined from the corpus. NOT synonyms or
    level-only variants (those are redundant / belong to the facet rail). Every
    suggestion is a corpus-grounded role, so it always has results."""
    rs = R.get("role_suggester")
    if not q.strip() or rs is None:
        return JSONResponse({"suggestions": []})
    qv = np.asarray(_dense_qv(q), dtype=np.float32)
    return JSONResponse({"suggestions": rs.suggest(q, qv, k=k)})


def _parse_filters(request: Request) -> dict[str, str | list[str]]:
    """Read facet filters from the query string. A field repeated across params
    (role_family=a&role_family=b) becomes a list -> OR within that field. posted_bucket
    is single-select (cumulative recency), so only its first value is kept."""
    qp = request.query_params
    out: dict[str, str | list[str]] = {}
    for f in FACET_FIELDS:
        vals = [v.strip() for v in qp.getlist(f) if v.strip()]
        if not vals:
            continue
        out[f] = vals[0] if f == "posted_bucket" else vals
    return out


@app.get("/api/search")
def api_search(request: Request, q: str = Query(""), k: int = Query(10)):
    """Keyword search, or — when q is blank — the recent/low-barrier browse default."""
    filters = _parse_filters(request)
    t0 = time.time()
    res = search_default(q, k, filters) if q.strip() else browse_default(k, filters)
    ms = int((time.time() - t0) * 1000)
    return JSONResponse(
        {
            "query": q,
            "retriever": "rrf_bm25_e5" if q.strip() else "browse_recent",
            "served_with": SERVING_MODE if q.strip() else "Browse: recent + low-barrier [via Solr]",
            "filters": filters,
            "results": res,
            "ms": ms,
        }
    )


@app.get("/api/facets")
def api_facets(request: Request, q: str = Query(""), pool: int = Query(200)):
    """Facet counts over the top-`pool` results (fused query results, or the blank-browse
    pool when q is empty) with the same filters the search uses, so counts stay coherent
    with what the user sees."""
    filters = _parse_filters(request)
    t0 = time.time()
    facets = compute_facets(q, filters, pool=pool)
    ms = int((time.time() - t0) * 1000)
    return JSONResponse(
        {
            "query": q,
            "filters": filters,
            "pool": pool,
            "facets": facets,
            "ms": ms,
        }
    )


_WS_RUN = re.compile(r"[ \t]+")
_NL_RUN = re.compile(r"\n{3,}")


def _clean_text(s: str) -> str:
    if not s:
        return ""
    s = html.unescape(s)  # decode literal &nbsp; &amp; etc.
    s = s.replace("\xa0", " ")  # collapse non-breaking spaces
    s = _WS_RUN.sub(" ", s)
    s = _NL_RUN.sub("\n\n", s)
    return s.strip()


@app.get("/api/detail")
def api_detail(idx: int = Query(...)):
    r = requests.get(
        f"{SOLR}/solr/{CORE}/select",
        params={
            "q": f'id:"{idx}"',
            "fl": "id,title_display,description,posted_at,department",
            "rows": 1,
        },
        timeout=10,
    )
    r.raise_for_status()
    docs = r.json()["response"]["docs"]
    if not docs:
        return JSONResponse({"error": "idx not found"}, status_code=404)
    d = docs[0]
    return JSONResponse(
        {
            "idx": idx,
            "title": _clean_text(d.get("title_display") or ""),
            "description": _clean_text(d.get("description") or ""),
            "posted_at": d.get("posted_at") or "",
            "department": d.get("department") or "",
        }
    )


@app.get("/api/more_like_this")
def api_more_like_this(idx: int = Query(...), k: int = Query(10)):
    """Jobs similar to job `idx` — RRF(BM25-on-title, e5-dense-on-title+lead)."""
    t0 = time.time()
    out = more_like_this(idx, k)
    out["ms"] = int((time.time() - t0) * 1000)
    out["served_with"] = SERVING_MODE
    return JSONResponse(out)


# ===== "find jobs for yourself": profile -> jobs with 3-axis constraint filter =====
# Reuses the same e5-small dense lane (Solr KNN over e5_vec). The profile text is
# reduced to its DEMONSTRATED experience via resume_match_lib.query_text (most-recent
# role + Experience section, NOT the aspirational headline / skills sidebar), then the
# 3-axis filter (seniority/location/qualification gates) is applied to the top-K pool.
# job_features are computed LIVE from Solr's stored description/locations/remote_mode —
# no precomputed sidecar. NOTHING the visitor uploads is persisted.

PROFILE_POOL = 50  # candidate pool depth (matches the validated probe)
PROFILE_TOP_N = 10
# Solr stores everything job_features() needs; seniority is derived from the title.
_PROFILE_FL = (
    "id,title_display,description,locations,remote_mode,employer,"
    "posted_at,source_corpus,industry,employment_type,department,"
    "salary_min,salary_max,salary_currency"
)


def _hydrate_for_match(ids: list[int]) -> dict[int, dict]:
    if not ids:
        return {}
    id_clause = " OR ".join(f'id:"{i}"' for i in ids)
    r = requests.get(
        f"{SOLR}/solr/{CORE}/select",
        params={"q": id_clause, "rows": len(ids), "fl": _PROFILE_FL},
        timeout=10,
    )
    r.raise_for_status()
    return {int(d["id"]): d for d in r.json()["response"]["docs"]}


def _job_feats_from_solr(d: dict) -> dict:
    """Adapt a hydrated Solr doc to the dict resume_match_lib.job_features expects.
    remote_mode is the derived facet ('remote'/'on_site'/'hybrid'); job_is_remote also
    falls back to scanning the locations list for 'remote'."""
    return L.job_features(
        {
            "title": d.get("title_display") or "",
            "locations": d.get("locations") or [],
            "remote": "True" if d.get("remote_mode") == "remote" else "False",
            "text": d.get("description") or "",
        }
    )


def _profile_summary(r: dict) -> dict:
    sen = "not stated" if not r.get("seniority_known", True) else L.SENIORITY_LABELS[r["seniority"]]
    return {
        "name": r["name"] or "(your profile)",
        "headline": r["headline"],
        "loc": r["loc"],
        "seniority": sen,
        "years": int(r["years"]) if r["years"] is not None else None,
        "degree": L.DEGREE_LABELS[r["degree"]],
        "creds": [L.CRED_LABELS.get(c, c) for c in r["creds"]],
    }


def _profile_job_brief(idx: int, cos: float, st: dict, d: dict, jf: dict) -> dict:
    locs = d.get("locations") or []
    title = (d.get("title_display") or "").strip()
    return {
        "idx": idx,
        "title": title[:140],
        "employer": d.get("employer") or "",
        "location": ", ".join(locs[:2]) if locs else "",
        "remote": bool(jf["remote"]),
        "seniority": L.SENIORITY_LABELS[jf["sen"]],
        "years_req": jf["years_req"],
        "degree_req": L.DEGREE_LABELS[jf["degree_req"]] if jf["degree_req"] else None,
        "cred_gates": [L.CRED_LABELS.get(c, c) for c in jf["cred_gates"]],
        "clearance": bool(jf["clearance"]),
        "workauth": bool(jf["workauth"]),
        "posted": (d.get("posted_at") or "")[:7],
        "source": SRC_SHORT.get(d.get("source_corpus") or "", d.get("source_corpus") or ""),
        "cosine": round(float(cos), 4),
        "axes": st,
    }


def _run_profile_match(r: dict, qv: list[float]) -> dict:
    """e5-small KNN top-`PROFILE_POOL`, then the 3-axis filter with job_features
    computed live from the hydrated Solr docs."""
    hits = _topk_knn("e5_vec", qv, PROFILE_POOL)  # [(idx, cosine), ...] best-first
    hyd = _hydrate_for_match([i for i, _ in hits])
    cosine_list: list[dict] = []
    filtered_list: list[dict] = []
    filtered_count = 0
    for idx, cos in hits:
        d = hyd.get(idx)
        if not d:
            continue
        jf = _job_feats_from_solr(d)
        st = L.axis_status(r, jf)
        brief = _profile_job_brief(idx, cos, st, d, jf)
        if len(cosine_list) < PROFILE_TOP_N:
            cosine_list.append(brief)
        if st["all"]:
            filtered_count += 1
            if len(filtered_list) < PROFILE_TOP_N:
                filtered_list.append(brief)
    return {
        "resume": _profile_summary(r),
        "pool_n": len(hits),
        "filtered_count": filtered_count,
        "cosine": cosine_list,
        "filtered": filtered_list,
    }


def _pdf_to_text(raw: bytes) -> str:
    import io

    from pypdf import PdfReader

    reader = PdfReader(io.BytesIO(raw))
    return "\n".join((page.extract_text() or "") for page in reader.pages)


_SENIORITY_PREFIX = re.compile(
    r"^(senior|sr\.?|junior|jr\.?|lead|principal|staff|chief|head of|vp(?: of)?|"
    r"vice president(?: of)?|director(?: of)?|associate)\s+",
    re.I,
)
_TITLE_AT = re.compile(
    r"\s+(?:at|@|[-|–—,]).*$", re.I
)  # drop "Engineer at Google" / "Engineer | ..."
_ASPIRATIONAL = re.compile(r"\b(aspiring|seeking|looking for|recent grad)", re.I)


def _suggest_queries(blob: str, r: dict, limit: int = 6) -> list[dict]:
    """Deterministic query suggestions from the parsed profile, validated against the
    live index. Sources (most-specific first): the recent role title, a seniority-
    broadened variant, earlier role titles, and the headline when it reads like a role.
    Each candidate is kept only if BM25 returns at least one job, and tagged with that
    count so the UI can show how many postings it would surface."""
    cands: list[str] = []
    for t in L.role_titles(blob)[:3]:
        t = _TITLE_AT.sub("", t).strip(" -|,")
        if t:
            cands.append(t)
            broad = _SENIORITY_PREFIX.sub("", t).strip()
            if broad and broad.lower() != t.lower():
                cands.append(broad)
    hl = (r.get("headline") or "").strip()
    if hl and 2 <= len(hl) <= 60 and not _ASPIRATIONAL.search(hl) and not L._looks_like_name(hl):
        cands.append(_TITLE_AT.sub("", hl).strip(" -|,"))
    out: list[dict] = []
    seen: set[str] = set()
    for c in cands:
        k = c.lower()
        if not c or k in seen or not _is_clean(k):
            continue
        seen.add(k)
        try:
            n = _count_bm25(c)
        except Exception:
            continue
        if n > 0:
            out.append({"text": c, "n": n})
        if len(out) >= limit:
            break
    return out


@app.post("/api/match_profile")
async def api_match_profile(
    text: str = Form(""),
    loc: str = Form(""),
    file: UploadFile | None = File(None),
):
    """Match an ad-hoc profile (pasted text, an uploaded .txt, or a LinkedIn
    'Save to PDF' export) against the catalog. Nothing is persisted."""
    blob = (text or "").strip()
    if file is not None and file.filename:
        raw = await file.read()
        try:
            if file.filename.lower().endswith(".pdf"):
                blob = _pdf_to_text(raw)
            else:
                blob = raw.decode("utf-8", "ignore")
        except Exception as e:
            return JSONResponse({"error": f"could not read file: {e}"}, status_code=400)
    blob = _clean_text(blob)
    if len(blob) < 30:
        return JSONResponse(
            {"error": "Need more text — paste your profile or upload a .txt / LinkedIn PDF."},
            status_code=400,
        )
    r = L.features_from_text(blob, loc=loc)
    # embed DEMONSTRATED experience (recent role + work history), not the aspirational
    # headline / skills sidebar — query_text isolates that, and BM25 is deliberately NOT
    # used here so the rest of the document can't dilute the most-recent-role emphasis.
    qv = _dense_qv(L.query_text(blob))
    out = _run_profile_match(r, qv)
    # suggested searches (#1) + the parsed profile the client holds and re-sends to
    # personalize subsequent keyword searches (#2). Nothing is persisted server-side.
    out["suggestions"] = _suggest_queries(blob, r)
    out["profile"] = {"r": r, "qv": qv}
    return JSONResponse(out)


@app.post("/api/search_personalized")
async def api_search_personalized(request: Request):
    """Keyword search re-ranked by a client-held profile (from /api/match_profile).
    Stateless: the profile (features + e5 vector) is sent in the body each call."""
    body = await request.json()
    q = (body.get("q") or "").strip()
    prof = body.get("profile") or {}
    r, qv = prof.get("r"), prof.get("qv")
    if not r or not qv:
        return JSONResponse({"error": "no profile loaded"}, status_code=400)
    raw_filters = body.get("filters") or {}
    filters: dict[str, str | list[str]] = {}
    for f in FACET_FIELDS:
        v = raw_filters.get(f)
        if isinstance(v, list):
            vv = [str(x).strip() for x in v if str(x).strip()]
            if vv:
                filters[f] = vv[0] if f == "posted_bucket" else vv
        elif v and str(v).strip():
            filters[f] = str(v).strip()
    k = int(body.get("k") or 10)
    hard = bool(body.get("hard_filter"))
    t0 = time.time()
    # Blank query + profile: rank the whole (filtered) catalog by profile fit.
    res = (
        search_personalized(q, r, qv, k, filters, hard)
        if q
        else browse_personalized(r, qv, k, filters, hard)
    )
    # Facets must reflect the SAME pool the user sees: for a blank personalized browse
    # that's the profile-KNN pool (not the generic recency browse). Returned inline so
    # the client doesn't make a second, profile-blind /api/facets call.
    facets = compute_facets(q, filters, qv_profile=(None if q else qv))
    ms = int((time.time() - t0) * 1000)
    return JSONResponse(
        {
            "query": q,
            "retriever": "rrf_bm25_e5+profile" if q else "browse+profile",
            "served_with": (SERVING_MODE if q else "Browse") + " + profile re-rank",
            "filters": filters,
            "hard_filter": hard,
            "results": res,
            "facets": facets,
            "ms": ms,
        }
    )


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("SHIM_PORT", os.environ.get("PORT", 7860)))
    uvicorn.run(app, host="0.0.0.0", port=port)
