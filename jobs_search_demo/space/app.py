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
import functools
import html
import json
import os
import re
import time
import unicodedata
from collections import defaultdict
from contextlib import asynccontextmanager

import numpy as np
import requests
import resume_match_lib as L
from fastapi import FastAPI, File, Form, Query, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from lang_detect import detect_lang, query_lang_mode
from maps_svg import US_STATES_SVG, WORLD_SVG
from snippet_lib import (
    SNIPPET_LEN,
    SNIPPET_PASSAGE_PREFIX,
    clean_text,
    passages_for,
    unpack_vecs,
)
from suggest_lib import degender_fr

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


def _fold(s: str) -> str:
    """Lowercase + strip diacritics and apostrophes, so an accent-free query
    ('ingenieur') matches an accented suggestion ('ingénieur') — French speakers
    routinely type without accents, and the title suggester is accent-sensitive. German
    ß is mapped to 'ss' (NFKD leaves it intact) so 'strasse' matches 'straße'."""
    nfkd = unicodedata.normalize("NFKD", s.replace("ß", "ss"))
    base = "".join(c for c in nfkd if not unicodedata.combining(c)).lower()
    return base.replace("'", "").replace("’", "")


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

    # French related searches via the ROME taxonomy (build_fr_related.py) — the English
    # e5 suggester ranks French roles by morphology, not meaning, so French gets this
    # grounded mobilite-based lane instead (suggest_lib.FrRelatedSuggester).
    try:
        from suggest_lib import FrRelatedSuggester

        fr_related = FrRelatedSuggester()
        print(f"  fr related: {len(fr_related.label2rome):,} ROME query keys", flush=True)
    except Exception as e:
        fr_related = None
        print(f"  fr related unavailable: {e}", flush=True)

    # German related searches via the ESCO occupation backbone (build_de_related.py) — same
    # rationale as French (e5 ranks German by morphology), but ESCO has no mobilite graph so
    # relatedness is skill-overlap based (suggest_lib.DeRelatedSuggester).
    try:
        from suggest_lib import DeRelatedSuggester

        de_related = DeRelatedSuggester()
        print(f"  de related: {len(de_related.label2uri):,} ESCO query keys", flush=True)
    except Exception as e:
        de_related = None
        print(f"  de related unavailable: {e}", flush=True)

    # Dutch related searches via the ESCO occupation backbone (build_nl_related.py) — same
    # rationale and mechanism as German (e5 ranks Dutch by morphology; ESCO has no mobilite
    # graph so relatedness is skill-overlap based; suggest_lib.NlRelatedSuggester).
    try:
        from suggest_lib import NlRelatedSuggester

        nl_related = NlRelatedSuggester()
        print(f"  nl related: {len(nl_related.label2uri):,} ESCO query keys", flush=True)
    except Exception as e:
        nl_related = None
        print(f"  nl related unavailable: {e}", flush=True)

    # Spanish related searches via the ESCO occupation backbone (build_es_related.py) — same
    # rationale and mechanism as German/Dutch (e5 ranks Spanish by morphology; ESCO has no
    # mobilite graph so relatedness is skill-overlap based; suggest_lib.EsRelatedSuggester).
    try:
        from suggest_lib import EsRelatedSuggester

        es_related = EsRelatedSuggester()
        print(f"  es related: {len(es_related.label2uri):,} ESCO query keys", flush=True)
    except Exception as e:
        es_related = None
        print(f"  es related unavailable: {e}", flush=True)

    # Swedish related searches via the ESCO occupation backbone (build_sv_related.py). Swedish
    # autocomplete shipped earlier; related was deferred (SSYK has no mobility graph). ESCO has
    # 100% Swedish labels, so it now rides the same skill-overlap lane as de/nl/es (no degender
    # — Swedish occupational nouns are gender-neutral). suggest_lib.SvRelatedSuggester.
    try:
        from suggest_lib import SvRelatedSuggester

        sv_related = SvRelatedSuggester()
        print(f"  sv related: {len(sv_related.label2uri):,} ESCO query keys", flush=True)
    except Exception as e:
        sv_related = None
        print(f"  sv related unavailable: {e}", flush=True)

    # Italian related searches via the ESCO occupation backbone (build_it_related.py) — same
    # rationale and mechanism as German/Dutch/Spanish (e5 ranks Italian by morphology; ESCO
    # has no mobilite graph so relatedness is skill-overlap based; ItRelatedSuggester).
    try:
        from suggest_lib import ItRelatedSuggester

        it_related = ItRelatedSuggester()
        print(f"  it related: {len(it_related.label2uri):,} ESCO query keys", flush=True)
    except Exception as e:
        it_related = None
        print(f"  it related unavailable: {e}", flush=True)

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

    # French canonical roles mined from France Travail titles (mine_fr_roles.py) ->
    # a dedicated autocomplete tier. The English query corpus carries no French, so
    # without this French prefixes only hit the (accent-sensitive) Solr title suggester.
    fr_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fr_roles.json")
    fr_roles: list[str] = []
    if os.path.exists(fr_path):
        with open(fr_path) as f:
            fr_roles = [x["text"] for x in json.load(f) if _is_clean(x["text"])]
        by_tag["fr"] = sorted(dict.fromkeys(fr_roles))
    # Swedish canonical roles mined from JobTech titles (mine_sv_roles.py) -> a dedicated
    # autocomplete tier, same rationale as French: the English corpus carries no Swedish,
    # so without this a Swedish prefix ("lära") only hits English keys ("laravel").
    sv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sv_roles.json")
    sv_roles: list[str] = []
    if os.path.exists(sv_path):
        with open(sv_path) as f:
            sv_roles = [x["text"] for x in json.load(f) if _is_clean(x["text"])]
        by_tag["sv"] = sorted(dict.fromkeys(sv_roles))
    # German canonical roles mined from Adzuna Germany titles (mine_de_roles.py) -> a
    # dedicated autocomplete tier, same rationale as French/Swedish: the English corpus
    # carries no German, so without this a German prefix only hits English keys.
    de_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "de_roles.json")
    de_roles: list[str] = []
    if os.path.exists(de_path):
        with open(de_path) as f:
            de_roles = [x["text"] for x in json.load(f) if _is_clean(x["text"])]
        by_tag["de"] = sorted(dict.fromkeys(de_roles))
    # Dutch canonical roles mined from Adzuna Netherlands titles (mine_nl_roles.py) -> a
    # dedicated autocomplete tier, same rationale as French/Swedish/German: the English
    # corpus carries no Dutch, so without this a Dutch prefix only hits English keys.
    nl_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "nl_roles.json")
    nl_roles: list[str] = []
    if os.path.exists(nl_path):
        with open(nl_path) as f:
            nl_roles = [x["text"] for x in json.load(f) if _is_clean(x["text"])]
        by_tag["nl"] = sorted(dict.fromkeys(nl_roles))
    # Spanish canonical roles mined from Adzuna Spain titles (mine_es_roles.py) -> a
    # dedicated autocomplete tier, same rationale as French/Swedish/German/Dutch: the English
    # corpus carries no Spanish, so without this a Spanish prefix only hits English keys.
    es_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "es_roles.json")
    es_roles: list[str] = []
    if os.path.exists(es_path):
        with open(es_path) as f:
            es_roles = [x["text"] for x in json.load(f) if _is_clean(x["text"])]
        by_tag["es"] = sorted(dict.fromkeys(es_roles))
    # Italian canonical roles mined from Adzuna Italy titles (mine_it_roles.py) -> a
    # dedicated autocomplete tier, same rationale as French/Swedish/German/Dutch/Spanish: the
    # English corpus carries no Italian, so without this an Italian prefix only hits English.
    it_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "it_roles.json")
    it_roles: list[str] = []
    if os.path.exists(it_path):
        with open(it_path) as f:
            it_roles = [x["text"] for x in json.load(f) if _is_clean(x["text"])]
        by_tag["it"] = sorted(dict.fromkeys(it_roles))
    # Accent-folded index over every suggestion key (curated corpus + FR + SV + DE + NL + ES
    # + IT roles): folded prefix -> originals, so "ingenieur" matches "ingénieur", "lara" ->
    # "lärare".
    folded_pairs = sorted(
        (_fold(k), k)
        for k in set(sorted_keys)
        | set(fr_roles)
        | set(sv_roles)
        | set(de_roles)
        | set(nl_roles)
        | set(es_roles)
        | set(it_roles)
    )
    folded_keys = [p[0] for p in folded_pairs]
    # Per-tier accent-folded index (folded_key, original), sorted by folded key. Lets
    # autocomplete match an accent-free prefix WITHIN the quality tiers ("electr" ->
    # "électricien" in the fr tier) instead of only via a final alphabetical catch-all
    # pass, where US-geo/company keys ("electric...") would otherwise crowd it out.
    tier_folded = {tag: sorted((_fold(x), x) for x in keys) for tag, keys in by_tag.items()}
    # French roles longest-first, for dictionary-matching a French resume to roles.
    fr_roles_folded = sorted(
        ((_fold(r), r) for r in dict.fromkeys(fr_roles)), key=lambda p: -len(p[0])
    )

    R.update(
        {
            "dense_model": dense_model,
            "sorted_keys": sorted_keys,
            # set form for O(1) "is this a known English query?" membership, used to keep
            # franglais English queries ("data scientist") out of the French ROME lane.
            "query_key_set": set(sorted_keys),
            "tier_keys": dict(by_tag),
            "role_suggester": role_suggester,
            "fr_related": fr_related,
            "de_related": de_related,
            "nl_related": nl_related,
            "es_related": es_related,
            "sv_related": sv_related,
            "it_related": it_related,
            # folded German role keys: lets the related-search router recognise a bare
            # German role ("techniker", "elektriker") that carries no lang-gate signal and
            # may not resolve in ESCO, so it's served by the German lane (empty if no move)
            # rather than the English e5 lane (morphology noise).
            "de_role_keys": {_fold(r) for r in de_roles},
            # folded Dutch role keys: same role for the Dutch lane (bare cognate roles like
            # "monteur"/"verpleegkundige" that carry no lang-gate signal).
            "nl_role_keys": {_fold(r) for r in nl_roles},
            # folded Spanish role keys: same role for the Spanish lane (bare cognate roles
            # like "camarero"/"electricista" that carry no lang-gate signal).
            "es_role_keys": {_fold(r) for r in es_roles},
            # folded Swedish role keys: same role for the Swedish lane (bare roles like
            # "snickare"/"elektriker" that carry no lang-gate signal).
            "sv_role_keys": {_fold(r) for r in sv_roles},
            # folded Italian role keys: same role for the Italian lane (bare cognate roles
            # like "cameriere"/"elettricista" that carry no lang-gate signal).
            "it_role_keys": {_fold(r) for r in it_roles},
            "folded_pairs": folded_pairs,
            "folded_keys": folded_keys,
            "tier_folded": tier_folded,
            "fr_roles_folded": fr_roles_folded,
        }
    )
    print(f"  french roles: {len(fr_roles)}", flush=True)
    print(f"  swedish roles: {len(sv_roles)}", flush=True)
    print(f"  german roles: {len(de_roles)}", flush=True)
    print(f"  dutch roles: {len(nl_roles)}", flush=True)
    print(f"  spanish roles: {len(es_roles)}", flush=True)
    print(f"  italian roles: {len(it_roles)}", flush=True)
    print("ready.", flush=True)


# ===== query encoders =====


@functools.lru_cache(maxsize=4096)
def _encode_query_cached(text: str) -> tuple[float, ...]:
    """Memoize the e5-small query encode. A single user search fans out to
    /api/search + /api/facets + /api/related_searches (and re-fires per pagination
    page), each on the SAME query string — without this they'd each re-run the
    model. Keyed on the already-prefixed text; returns an immutable tuple so the
    shared cache entry can't be mutated by a caller."""
    qv = R["dense_model"].encode([text], normalize_embeddings=True, show_progress_bar=False)[0]
    return tuple(qv.astype(np.float32).tolist())


def _dense_qv(query: str) -> list[float]:
    return list(_encode_query_cached(DENSE_QUERY_PREFIX + query))


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
    "lang",
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


def _apply_lang_gate(query: str, filters: dict[str, str | list[str]]) -> None:
    """Confident-French query-language gate (in place). The index is ~33% French
    (France Travail) under an English-only encoder; English queries already pick up
    only ~5% French docs (low harm), but a confidently-French query should be scoped
    to French inventory. Detection is asymmetric (see lang_detect.query_lang_mode):
    only an unmistakably-French (or Swedish from JobTech, German from Adzuna DE, Dutch from
    Adzuna NL, Spanish from Adzuna ES, or Italian from Adzuna IT) query flips the gate, so a
    short ambiguous query never strands a user. We setdefault so an explicit user `lang`
    facet selection wins."""
    if not (query and query.strip()):
        return
    mode = query_lang_mode(query)
    if mode in ("fr", "sv", "de", "nl", "es", "it"):
        filters.setdefault("lang", mode)


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


def _prof_vecs(qv, qvs=None) -> list[list[float]]:
    """Profile vector list for max-sim matching: the multi-vector `qvs` when present
    (full lead-sharpened vector + a dense specialization vector), else just `qv`.
    Backward-compatible with old clients that only hold a single `qv`."""
    vs = [v for v in (qvs or []) if v]
    return vs or [qv]


def _max_prof_cos(vecs: list[list[float]], ids: list[int]) -> dict[int, float]:
    """Max profile cosine per candidate id over all profile vectors (max-sim): a
    specialist scores high if ANY facet of their profile fits, so a long generic
    experience centroid no longer washes out the on-specialty signal."""
    best: dict[int, float] = {}
    for v in vecs:
        for idx, cos in _knn_over_ids("e5_vec", v, ids):
            if cos > best.get(idx, -2.0):
                best[idx] = cos
    return best


def _topk_knn_multi(vecs: list[list[float]], k: int, filters=None) -> list[tuple[int, float]]:
    """Candidate pool ranked by MAX profile cosine across `vecs`. Unions each vector's
    top-k KNN, then rescores the union by max-sim so a doc near ANY profile facet ranks."""
    if len(vecs) == 1:
        return _topk_knn("e5_vec", vecs[0], k, filters)
    pool: set[int] = set()
    for v in vecs:
        pool.update(idx for idx, _ in _topk_knn("e5_vec", v, k, filters))
    best = _max_prof_cos(vecs, list(pool))
    return sorted(best.items(), key=lambda x: -x[1])[:k]


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
        "snippet": "",  # filled in by _attach_snippets at the endpoint layer
    }


# ===== result snippets =====
# description is stored but indexed=false (a single un-tokenized string token), so Solr
# can't highlight it server-side without a reindex. Instead we pick the best passage in
# Python over just the handful of DISPLAYED results. Selection is SEMANTIC: every
# candidate passage is encoded with the same e5-small model used for retrieval and the
# one whose embedding is closest to the query vector wins — so a relevant passage surfaces
# even when it shares no literal words with the query (the win lexical selection couldn't
# get). Highlighting stays lexical on top: query terms that *do* appear in the chosen
# passage are wrapped in <em> as a bonus, but they no longer decide which passage shows.
# Passage vectors are PRE-COMPUTED at index time and stored in the Solr `snippet_vecs`
# field, so the serve-time cost is just dot products (no per-query encode). A doc lacking
# stored vecs (pre-backfill / a fresh delta posting) falls back to a live batched encode,
# and an encode failure degrades to the old lexical (most-query-terms) selection. Passage
# segmentation + the fp16 vector codec live in snippet_lib so offline and serve-time can't
# drift (SNIPPET_LEN / PASSAGES_PER_DOC / passages_for / unpack_vecs imported from there).
_SNIPPET_STOP = {
    "the",
    "and",
    "for",
    "with",
    "you",
    "your",
    "our",
    "are",
    "job",
    "jobs",
    "role",
    "roles",
    "work",
    "will",
    "this",
    "that",
    "from",
    "have",
    "all",
    "who",
}
_SNIP_TOKEN = re.compile(r"[A-Za-z0-9][A-Za-z0-9+#.\-]*")
_SNIP_SENT = re.compile(r"(?<=[.!?])\s+|\n+")


def _snippet_terms(query: str) -> list[str]:
    """Lowercased content tokens of the query, deduped, used for scoring + highlighting."""
    out: list[str] = []
    for w in _SNIP_TOKEN.findall(query.lower()):
        if len(w) > 1 and w not in _SNIPPET_STOP and w not in out:
            out.append(w)
    return out


def _term_hit(word_lc: str, term: str) -> bool:
    # exact, or the query term is a prefix of the doc word (cheap stem: python->pythonic,
    # engineer->engineering). Prefix only for terms >= 4 chars, to avoid noisy matches.
    return word_lc == term or (len(term) >= 4 and word_lc.startswith(term))


def _distinct_hits(text: str, terms: list[str]) -> int:
    words = {w.lower() for w in _SNIP_TOKEN.findall(text)}
    return sum(1 for t in terms if any(_term_hit(w, t) for w in words))


def _lead(text: str) -> str:
    if len(text) <= SNIPPET_LEN:
        return text
    cut = text[:SNIPPET_LEN]
    sp = cut.rfind(" ")
    if sp > SNIPPET_LEN * 0.6:
        cut = cut[:sp]
    return cut.rstrip() + "…"


def _window(text: str, terms: list[str]) -> str:
    """Trim a long winning sentence to a window around its first matched token."""
    if len(text) <= SNIPPET_LEN:
        return text
    pos = 0
    for m in _SNIP_TOKEN.finditer(text):
        if any(_term_hit(m.group(0).lower(), t) for t in terms):
            pos = m.start()
            break
    start = max(0, pos - 50)
    seg = text[start : start + SNIPPET_LEN]
    if start > 0:
        seg = "…" + seg.lstrip()
    if start + SNIPPET_LEN < len(text):
        seg = seg.rstrip() + "…"
    return seg


def _highlight(text: str, terms: list[str]) -> str:
    """HTML-escape `text` and wrap matched word tokens in <em>. Returns safe HTML: only
    <em> tags are introduced, every other character is escaped."""
    out: list[str] = []
    last = 0
    for m in _SNIP_TOKEN.finditer(text):
        word = m.group(0)
        if any(_term_hit(word.lower(), t) for t in terms):
            # keep internal punctuation (node.js, c#) but leave a trailing sentence
            # period/comma outside the <em> so the highlight ends on the word.
            core = word.rstrip(".,;:!?")
            trail = word[len(core) :]
            out.append(html.escape(text[last : m.start()]))
            out.append("<em>" + html.escape(core) + "</em>" + html.escape(trail))
            last = m.end()
    out.append(html.escape(text[last:]))
    return "".join(out)


def _snippet_for(description: str, terms: list[str]) -> str:
    """Best-passage snippet (safe HTML) for one description given query terms."""
    text = _clean_text(description)
    if not text:
        return ""
    if not terms:
        return html.escape(_lead(text))
    best, best_score = "", 0
    for s in _SNIP_SENT.split(text):
        s = s.strip()
        if not s:
            continue
        score = _distinct_hits(s, terms)
        if score > best_score:
            best, best_score = s, score
    if best_score == 0:
        return html.escape(_lead(text))
    return _highlight(_window(best, terms), terms)


# Passage vectors are deterministic for a given passage string, so cache them across
# queries and pagination pages: the same job's passages recur on page 2, on "more like
# this", and on a re-typed query. Only cache-miss passages hit the model, in one batched
# encode call. Bounded so a long session can't grow it without limit.
_PASSAGE_VEC_CACHE: dict[str, np.ndarray] = {}
_PASSAGE_CACHE_MAX = 20000


def _encode_passages(passages: list[str]) -> dict[str, np.ndarray]:
    """Return {passage: unit vector} for every passage, encoding only cache misses in a
    single batched e5 call. Normalized, so cosine to the query vector is a plain dot."""
    miss = [p for p in dict.fromkeys(passages) if p not in _PASSAGE_VEC_CACHE]
    if miss:
        vecs = R["dense_model"].encode(
            [SNIPPET_PASSAGE_PREFIX + p for p in miss],
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        if len(_PASSAGE_VEC_CACHE) + len(miss) > _PASSAGE_CACHE_MAX:
            _PASSAGE_VEC_CACHE.clear()
        for p, v in zip(miss, vecs):
            _PASSAGE_VEC_CACHE[p] = np.asarray(v, dtype=np.float32)
    return {p: _PASSAGE_VEC_CACHE[p] for p in passages}


def _resolve_passage_vecs(
    doc_passages: dict[int, list[str]], vecs_b64: dict[int, str]
) -> dict[int, np.ndarray]:
    """Per doc, return a (n_passages, dim) vector matrix. Prefer the stored snippet_vecs
    (zero encode); a doc with no/stale stored vecs (count != passage count) has its
    passages queued and batch-encoded live in one call. Docs with no passages are absent."""
    resolved: dict[int, np.ndarray] = {}
    need: list[str] = []
    for i, ps in doc_passages.items():
        if not ps:
            continue
        b64 = vecs_b64.get(i)
        if b64:
            try:
                v = unpack_vecs(b64)
                if v.shape[0] == len(ps):
                    resolved[i] = v
                    continue
            except Exception:
                pass  # corrupt/stale -> live encode below
        need.extend(ps)
    if need:
        enc = _encode_passages(need)
        for i, ps in doc_passages.items():
            if ps and i not in resolved:
                resolved[i] = np.vstack([enc[p] for p in ps])
    return resolved


# Visible-match gate: if the semantically-best passage contains no query term but another
# passage that DOES sits within this cosine of it, prefer the visible-match passage so the
# snippet shows why the job matched. eps-bounded so the gate never trades away a real
# relevance gap for a cosmetic highlight (offline bake-off: vs pure-semantic this lifts
# highlight coverage 96%->100% at ~0 cosine regret; RRF/weighted blends over-correct).
SNIPPET_GATE_EPS = 0.03


def _semantic_snippets(
    query: str, terms: list[str], raw: dict[int, str], vecs_b64: dict[int, str]
) -> dict[int, str]:
    """Pick each doc's snippet by embedding similarity, then apply the visible-match gate:
    re-derive the candidate passages, pair each with its (stored or live-encoded) vector,
    take the passage closest to the query vector — but if that passage has no query term
    while a term-containing one sits within SNIPPET_GATE_EPS cosine, prefer the latter so
    the snippet visibly shows the match. Reduces to pure-semantic when no near-tie visible
    match exists. Lexical term hits in the winner are highlighted either way."""
    doc_passages = {i: passages_for(t) for i, t in raw.items()}
    pvecs = _resolve_passage_vecs(doc_passages, vecs_b64)
    qv = np.asarray(_dense_qv(query), dtype=np.float32)
    out: dict[int, str] = {}
    for i, ps in doc_passages.items():
        if not ps:
            cleaned = clean_text(raw[i])
            out[i] = html.escape(_lead(cleaned)) if cleaned else ""
            continue
        sims = pvecs[i] @ qv
        pick = int(np.argmax(sims))
        if terms and _distinct_hits(ps[pick], terms) == 0:
            floor = float(sims[pick]) - SNIPPET_GATE_EPS
            hit_cands = [
                j for j in range(len(ps)) if sims[j] >= floor and _distinct_hits(ps[j], terms) > 0
            ]
            if hit_cands:
                pick = max(hit_cands, key=lambda j: sims[j])
        out[i] = _highlight(_window(ps[pick], terms), terms)
    return out


def _snippets(query: str, ids: list[int]) -> dict[int, str]:
    """Fetch descriptions + stored passage vectors for the displayed ids in one Solr call
    and build a snippet for each. With a query, selection is semantic (best passage by e5
    cosine, dot against the stored snippet_vecs) with an eps-bounded visible-match gate
    (see _semantic_snippets) and lexical <em> highlighting layered on; blank query
    (seed/browse) shows the description lead. Falls back to lexical most-terms selection if
    the semantic path raises."""
    if not ids:
        return {}
    id_clause = " OR ".join(f'id:"{i}"' for i in ids)
    r = requests.get(
        f"{SOLR}/solr/{CORE}/select",
        params={"q": id_clause, "rows": len(ids), "fl": "id,description,snippet_vecs"},
        timeout=10,
    )
    r.raise_for_status()
    docs = r.json()["response"]["docs"]
    raw = {int(d["id"]): (d.get("description") or "") for d in docs}
    if not query.strip():
        return {i: (html.escape(_lead(clean_text(t))) if t else "") for i, t in raw.items()}
    vecs_b64 = {int(d["id"]): (d.get("snippet_vecs") or "") for d in docs}
    terms = _snippet_terms(query)
    try:
        return _semantic_snippets(query, terms, raw, vecs_b64)
    except Exception as e:  # model/encode hiccup -> lexical selection still serves a snippet
        print(f"semantic snippet fallback ({e}); using lexical selection", flush=True)
        return {i: _snippet_for(t, terms) for i, t in raw.items()}


def _attach_snippets(res: list[dict], query: str) -> None:
    """Attach a `snippet` to each result row in place (one Solr fetch for the page)."""
    if not res:
        return
    snips = _snippets(query, [row["idx"] for row in res if row.get("idx", -1) >= 0])
    for row in res:
        row["snippet"] = snips.get(row.get("idx"), "")


class QSpec:
    """A retrieval intent that the whole pipeline (search, facets, pagination,
    personalization) operates on, so a typed query and a "more jobs like this" seed
    travel the SAME code path. Two flavours:
      * typed text  -> bm25_text == dense_text == the query, no exclusion.
      * seed job     -> bm25_text = the seed title (crisp lexical anchor), dense_text =
        title + description lead (semantic intent), and `exclude` drops the seed itself
        from its own neighbour list. e5_vec is stored=false, so the seed text is
        re-embedded at query time — the same asymmetric "query: " prefix bridges to the
        indexed "passage: " vectors, exactly as a typed query does.
    A seed sets `exclude`; that also signals the employer-dominance bypass should be OFF
    (the user clicked a role, not an employer)."""

    __slots__ = ("bm25_text", "dense_text", "exclude")

    def __init__(self, bm25_text: str, dense_text: str, exclude: int | None = None):
        self.bm25_text = bm25_text or ""
        self.dense_text = dense_text or ""
        self.exclude = exclude

    @property
    def active(self) -> bool:
        return bool(self.bm25_text.strip() or self.dense_text.strip())

    @property
    def is_seed(self) -> bool:
        return self.exclude is not None


def qspec_text(q: str) -> QSpec:
    q = (q or "").strip()
    return QSpec(q, q)


def _fused_topk(
    spec: QSpec,
    k: int,
    filters: dict[str, str] | None = None,
    pool: int = RRF_POOL,
) -> list[tuple[int, float]]:
    """Run BM25 + e5-small lanes for a QSpec and RRF-fuse to top-k. When the spec
    excludes a doc (a seed), each lane pulls one extra so dropping the seed still leaves
    a full pool."""
    depth = pool + (1 if spec.exclude is not None else 0)
    contrib: dict[int, float] = defaultdict(float)
    if spec.bm25_text.strip():
        for rank, (idx, _) in enumerate(_topk_bm25(spec.bm25_text, depth, filters), 1):
            if idx != spec.exclude:
                contrib[idx] += 1.0 / (RRF_K + rank)
    # Solr field "e5_vec" holds e5-small-v2 vectors (384-dim, passage: prefix at index time).
    if spec.dense_text.strip():
        for rank, (idx, _) in enumerate(
            _topk_knn("e5_vec", _dense_qv(spec.dense_text), depth, filters), 1
        ):
            if idx != spec.exclude:
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


# "More jobs like this" is a similarity read, not a navigational filter — so a seed
# does NOT get the per-employer diversity cap. Instead we only collapse literal reprints:
# the exact same req (one employer, same normalized title) posted across many locations,
# which would otherwise fill the page with identical rows. SEED_EMPLOYER_CAP (default 0 =
# uncapped) can still bound any single employer if a softer limit is ever wanted.
SEED_EMPLOYER_CAP = int(os.environ.get("SEED_EMPLOYER_CAP", "0"))
_TITLE_NORM = re.compile(r"[^a-z0-9]+")
_TOK = re.compile(r"[a-z0-9]+")
_SEG_SEP = re.compile(r"\s*[-–—]\s*")  # ATS titles delimit geo prefixes with hyphen/dash

# US state codes + names — leading title segments matching one of these (or a token from
# the job's own location) are stripped before the reprint check, so "GA - Atlanta - RN
# Case Manager" and "PA - Philadelphia - RN Case Manager" collapse to one role.
_STATE_CODES = (  # noqa: SIM905 — a space-split string reads better than 51 quoted items
    "AL AK AZ AR CA CO CT DE DC FL GA HI ID IL IN IA KS KY LA ME MD MA MI MN MS MO "
    "MT NE NV NH NJ NM NY NC ND OH OK OR PA RI SC SD TN TX UT VT VA WA WV WI WY"
).split()
_US_STATES = frozenset(
    [c.lower() for c in _STATE_CODES]
    + [
        "alabama",
        "alaska",
        "arizona",
        "arkansas",
        "california",
        "colorado",
        "connecticut",
        "delaware",
        "florida",
        "georgia",
        "hawaii",
        "idaho",
        "illinois",
        "indiana",
        "iowa",
        "kansas",
        "kentucky",
        "louisiana",
        "maine",
        "maryland",
        "massachusetts",
        "michigan",
        "minnesota",
        "mississippi",
        "missouri",
        "montana",
        "nebraska",
        "nevada",
        "new hampshire",
        "new jersey",
        "new mexico",
        "new york",
        "north carolina",
        "north dakota",
        "ohio",
        "oklahoma",
        "oregon",
        "pennsylvania",
        "rhode island",
        "south carolina",
        "south dakota",
        "tennessee",
        "texas",
        "utah",
        "vermont",
        "virginia",
        "washington",
        "west virginia",
        "wisconsin",
        "wyoming",
        "district of columbia",
        "remote",
    ]
)


def _norm_title(t: str) -> str:
    return _TITLE_NORM.sub(" ", (t or "").lower()).strip()


def _loc_tokens(d: dict) -> set[str]:
    """Lowercased word tokens from a doc's location strings (e.g. 'Atlanta, GA' ->
    {'atlanta','ga'}) — corroborating evidence that a leading title segment is geo."""
    toks: set[str] = set()
    for loc in d.get("locations") or []:
        toks.update(_TOK.findall((loc or "").lower()))
    return toks


def _strip_geo_prefix(title: str, loc_tokens: set[str]) -> str:
    """Drop leading 'STATE - [City -]' location prefixes (a common ATS title convention)
    so the same role posted across locations collapses in the reprint check. A leading
    segment is stripped only if it's a US state code/name OR shares a token with the
    job's own location; the scan stops at the first non-geo segment, so a legitimate
    title like 'TX - Senior Manager - Clinical Quality' keeps 'Senior Manager - …'. The
    final segment is never stripped, so a fully-geo title can't vanish."""
    parts = _SEG_SEP.split(title.strip())
    i = 0
    while i < len(parts) - 1:
        seg = parts[i].strip().lower()
        seg_toks = _TOK.findall(seg)
        if seg in _US_STATES or (
            loc_tokens and seg_toks and any(t in loc_tokens for t in seg_toks)
        ):
            i += 1
        else:
            break
    return " - ".join(parts[i:]).strip()


def _reprint_key(d: dict) -> tuple[str, str]:
    """(employer, geo-stripped normalized title) — the identity used to collapse the same
    req reposted across locations within one employer's seed neighbourhood."""
    emp = (d.get("employer") or "").strip().lower()
    title = _strip_geo_prefix(d.get("title_display") or "", _loc_tokens(d))
    return (emp, _norm_title(title))


def _diversify_seed(
    items: list[tuple[int, float]], hyd: dict[int, dict], k: int
) -> list[tuple[int, float]]:
    """Seed diversification: keep similarity order, but drop reprints (same employer +
    same geo-stripped title) so the same role reposted across cities doesn't crowd out
    genuinely distinct similar jobs. No per-employer cap unless SEED_EMPLOYER_CAP > 0."""
    kept: list[tuple[int, float]] = []
    seen_reprint: set[tuple[str, str]] = set()
    emp_count: dict[str, int] = defaultdict(int)
    for idx, score in items:
        d = hyd.get(idx, {})
        emp = (d.get("employer") or "").strip().lower()
        key = _reprint_key(d)
        if emp and key in seen_reprint:
            continue
        if SEED_EMPLOYER_CAP > 0 and emp and emp_count[emp] >= SEED_EMPLOYER_CAP:
            continue
        kept.append((idx, score))
        if emp:
            seen_reprint.add(key)
            emp_count[emp] += 1
        if len(kept) >= k:
            break
    return kept


def search_default(
    spec: QSpec,
    k: int = 10,
    filters: dict[str, str] | None = None,
    offset: int = 0,
) -> list[dict]:
    """RRF(BM25, e5-small) for a QSpec (typed query or seed job) with optional facet
    filters, then cap to EMPLOYER_CAP results per employer for display diversity.
    `offset` paginates: the employer cap is applied across the full ranked list, then the
    [offset, offset+k] window is returned, so paging is stable (page 2 never repeats a
    page-1 row). A seed disables the dominance bypass — near-duplicate postings from one
    shop add little when the user picked a role, not an employer."""
    cap = 0 if (filters and filters.get("employer")) else EMPLOYER_CAP
    need = offset + k
    pool_k = max(need * (cap + 2), need + 20) if cap > 0 else need
    # A seed keeps a deeper fused pool so reprint collapsing still fills the page.
    if spec.is_seed:
        pool_k = max(pool_k, RRF_POOL)
    items = _fused_topk(spec, pool_k, filters, max(RRF_POOL, pool_k))
    hyd = _hydrate([i for i, _ in items])
    if spec.is_seed:
        items = _diversify_seed(items, hyd, need)[offset : offset + k]
    else:
        items = _cap_employers(items, hyd, need, filters)[offset : offset + k]
    return [_make_result(offset + r + 1, s, i, hyd.get(i, {})) for r, (i, s) in enumerate(items)]


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


def browse_default(
    k: int = 10, filters: dict[str, str | list[str]] | None = None, offset: int = 0
) -> list[dict]:
    """Default browse: recent + low-barrier jobs, with facet filters applied.
    `offset` paginates the same way as search_default."""
    cap = 0 if (filters and filters.get("employer")) else EMPLOYER_CAP
    need = offset + k
    pool_k = max(need * (cap + 2), need + 20) if cap > 0 else need
    items = _browse_topk(max(pool_k, 200), filters)
    hyd = _hydrate([i for i, _ in items])
    # No query intent in a blank browse, so keep employer diversity (no dominance bypass).
    items = _cap_employers(items, hyd, need, filters, dominance_bypass=False)[offset : offset + k]
    return [_make_result(offset + r + 1, s, i, hyd.get(i, {})) for r, (i, s) in enumerate(items)]


# ===== "more jobs like this one": a seed job becomes a QSpec =====
# A seed job is just an alternate query SOURCE: the title drives the BM25 lane (a crisp
# lexical anchor), and title + the lead of the description drives the e5 dense lane. From
# there it rides the identical search_default / compute_facets / pagination /
# personalization pipeline as a typed query — so a seed search gets the full facet rail,
# filters, paging, and profile re-rank for free. e5_vec is stored=false, so the seed text
# is re-embedded at query time (the asymmetric "query: " prefix bridges to the indexed
# "passage: " vectors). The seed is dropped from its own neighbour list via QSpec.exclude.

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


def qspec_seed(idx: int) -> QSpec | None:
    """Build the QSpec for a "more jobs like this" seed: BM25 on the seed title, dense on
    title + description lead, excluding the seed itself. Returns None if idx isn't found."""
    src = _source_doc(idx)
    if src is None:
        return None
    title = (src.get("title_display") or "").strip()
    desc = (src.get("description") or "").strip()
    dense = (title + ". " + desc)[: len(title) + 2 + MLT_DESC_CHARS] if desc else title
    return QSpec(bm25_text=title, dense_text=dense or title, exclude=idx)


def seed_title(idx: int) -> str:
    """Cleaned display title for a seed job (used to label the seed in the UI)."""
    src = _source_doc(idx)
    return _clean_text((src.get("title_display") or "").strip()) if src else ""


# ===== personalized search (keyword query re-ranked by an uploaded profile) =====

PROF_WEIGHT = float(os.environ.get("PROF_WEIGHT", "1.0"))

# Personalized results should not surface the seeker's OWN current/recent employer —
# being shown jobs at the company you already work at (or just left) is noise in a "jobs
# for you" feed. We drop the most-recent SELF_EMPLOYER_MAX employers parsed from the
# profile, except when the user has explicitly filtered to that employer (they asked).
SELF_EMPLOYER_MAX = int(os.environ.get("SELF_EMPLOYER_MAX", "3"))


def _self_employers(r: dict, filters: dict | None = None) -> list[str]:
    """The seeker's recent employer names to suppress — empty when the user explicitly
    filtered by employer (an explicit ask overrides the self-employer suppression)."""
    if filters and filters.get("employer"):
        return []
    emps = r.get("employers") or []
    return emps[:SELF_EMPLOYER_MAX] if SELF_EMPLOYER_MAX > 0 else []


def _is_self_employer(emp: str, self_emps: list[str]) -> bool:
    return bool(emp) and any(L.same_employer(emp, se) for se in self_emps)


def _personalized_topk(
    spec: QSpec,
    qv_profile: list[float],
    k: int,
    filters: dict[str, str] | None = None,
    pool: int = RRF_POOL,
    prof_weight: float = PROF_WEIGHT,
    qvs: list[list[float]] | None = None,
) -> tuple[list[tuple[int, float]], dict[int, float]]:
    """RRF(BM25, e5-small) for the QSpec (typed query OR seed job), then a third lane
    that re-ranks the candidates by profile fit. The query/seed still defines what's
    eligible (we never inject off-query jobs), but every candidate is scored against the
    profile — so a profile reshapes essentially any query, including ones far from it (a
    data-engineer profile floats the most data/eng-flavored 'manager' jobs up). Returns
    (ranked, prof_cos) where prof_cos maps idx -> profile cosine for EVERY candidate."""
    contrib: dict[int, float] = defaultdict(float)
    if spec.bm25_text.strip():
        for rank, (idx, _) in enumerate(_topk_bm25(spec.bm25_text, pool, filters), 1):
            if idx != spec.exclude:
                contrib[idx] += 1.0 / (RRF_K + rank)
    if spec.dense_text.strip():
        for rank, (idx, _) in enumerate(
            _topk_knn("e5_vec", _dense_qv(spec.dense_text), pool, filters), 1
        ):
            if idx != spec.exclude:
                contrib[idx] += 1.0 / (RRF_K + rank)
    # Rank the query candidates by profile fit and blend that rank in as a third lane.
    prof_cos = _max_prof_cos(_prof_vecs(qv_profile, qvs), list(contrib.keys()))
    for rank, (idx, _c) in enumerate(sorted(prof_cos.items(), key=lambda x: -x[1]), 1):
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
    spec: QSpec,
    r: dict,
    qv_profile: list[float],
    k: int = 10,
    filters: dict[str, str] | None = None,
    hard_filter: bool = False,
    qvs: list[list[float]] | None = None,
) -> list[dict]:
    """Keyword/seed search re-ranked by profile fit. Soft by default (profile-KNN RRF
    boost + per-result 3-axis badges); when hard_filter is set, drop results the
    candidate doesn't qualify for (under-seniority / location / years-degree-cred
    gates), mirroring the profile lane's filtered panel. A seed drops the employer
    diversity cap in favour of reprint collapsing (similarity, not filtering), matching
    the non-personalized seed path."""
    cap = 0 if (filters and filters.get("employer") or spec.is_seed) else EMPLOYER_CAP
    ranked, prof_cos = _personalized_topk(spec, qv_profile, RRF_POOL, filters, qvs=qvs)
    ids = [i for i, _ in ranked]
    hyd = _hydrate_for_match(ids)
    exempt = _dominant_employers(ranked, hyd) if cap > 0 else set()
    self_emps = _self_employers(r, filters)
    rows: list[dict] = []
    seen: dict[str, int] = {}
    seen_reprint: set[tuple[str, str]] = set()
    for idx, score in ranked:
        d = hyd.get(idx)
        if not d:
            continue
        if _is_self_employer(d.get("employer") or "", self_emps):
            continue  # don't surface the seeker's own current/recent employer
        jf = _job_feats_from_solr(d)
        st = L.axis_status(r, jf)
        if hard_filter and not st["all"]:
            continue
        emp = (d.get("employer") or "").strip().lower()
        if spec.is_seed:
            key = _reprint_key(d)
            if emp and key in seen_reprint:
                continue
        elif cap > 0 and emp and emp not in exempt and seen.get(emp, 0) >= cap:
            continue
        rows.append(_make_result_personalized(len(rows) + 1, score, idx, d, st, prof_cos.get(idx)))
        if emp:
            seen[emp] = seen.get(emp, 0) + 1
            if spec.is_seed:
                seen_reprint.add(key)
        if len(rows) >= k:
            break
    return rows


def browse_personalized(
    r: dict,
    qv_profile: list[float],
    k: int = 10,
    filters: dict[str, str | list[str]] | None = None,
    hard_filter: bool = False,
    qvs: list[list[float]] | None = None,
) -> list[dict]:
    """Blank-query browse personalized to an uploaded profile: rank purely by profile
    fit (e5 KNN over filtered candidates), with the same 3-axis qualification badges as
    keyword personalization. The recency/low-barrier browse boost is replaced by profile
    cosine here — the profile IS the intent when there's no query."""
    cap = 0 if (filters and filters.get("employer")) else EMPLOYER_CAP
    pool = max(PROFILE_POOL, k * (cap + 2), k + 20)
    hits = _topk_knn_multi(_prof_vecs(qv_profile, qvs), pool, filters)  # (idx, max-sim cos)
    hyd = _hydrate_for_match([i for i, _ in hits])
    self_emps = _self_employers(r, filters)
    rows: list[dict] = []
    seen: dict[str, int] = {}
    for idx, cos in hits:
        d = hyd.get(idx)
        if not d:
            continue
        if _is_self_employer(d.get("employer") or "", self_emps):
            continue  # don't surface the seeker's own current/recent employer
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


# Facet rank decay: weight a doc's facet contributions by 1/(rank+1)**FACET_DECAY_POW.
# A steep (>1) exponent makes the VISIBLE head of the list dominate facet ordering, while
# the long tail still contributes enough to surface values that aren't on the first page.
# At 1.0 this degraded to near-volume-weighting (harmonic tail mass ~ ln(pool) swamped the
# head); 2.0 puts ~95% of the weight in the first page. Env-tunable.
FACET_DECAY_POW = float(os.environ.get("FACET_DECAY_POW", "2.0"))


def _facet_pool(
    spec: QSpec,
    filters: dict[str, str | list[str]],
    pool: int,
    qv_profile: list[float] | None = None,
) -> tuple[list[tuple[int, float]], dict[int, dict]]:
    """The employer-capped, ranked list we facet over — the SAME list the user pages
    through, so facet ordering reconciles with the visible results rather than being
    driven by a deeper, uncapped pool. Fused results when there's a query/seed; else a
    profile-KNN pool when a profile drives a blank personalized browse; else the blank
    recency-browse pool. A seed disables the employer dominance bypass. Returns
    (capped_items, hydration)."""
    is_seed = spec.active and spec.is_seed
    if spec.active:
        items = _fused_topk(spec, pool, filters, pool)
    elif qv_profile is not None:
        items = _topk_knn("e5_vec", qv_profile, pool, filters)
    else:
        items = _browse_topk(pool, filters, pool)
    hyd = _hydrate([i for i, _ in items], with_facets=True)
    # Facet over the SAME post-processed pool the user pages through: seeds get reprint
    # collapsing (no employer cap, to read as similarity), everything else gets the same
    # per-employer cap as search_default/browse_default so one shop posting many
    # near-identical roles can't dominate the facet tally any more than it dominates the page.
    if is_seed:
        items = _diversify_seed(items, hyd, pool)
    else:
        items = _cap_employers(items, hyd, pool, filters)
    return items, hyd


def _aggregate_facets(
    items: list[tuple[int, float]], hyd: dict[int, dict]
) -> dict[str, list[tuple[str, float]]]:
    """Rank-weighted value tallies per facet field over `items`, weighted by
    1/(rank+1)**FACET_DECAY_POW so the head of the result list dominates ordering while
    tail docs still fill in values absent from the first page. Tail values (role 'other',
    'unclassified') sink to the bottom. Ordinal/static ordering is applied client-side."""
    weights: dict[str, dict[str, float]] = {f: defaultdict(float) for f in FACET_FIELDS}
    for rank, (i, _s) in enumerate(items):
        w = 1.0 / (rank + 1) ** FACET_DECAY_POW
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


def _native_facet_options(
    field: str, query: str, filters: dict[str, str | list[str]]
) -> list[tuple[str, float]]:
    """Every value of `field` present in the matching set, via Solr facet.field
    (rows=0), scoped to the same keyword query + filters the user sees but blind to
    relevance/recency ranking. Use for navigational facets whose full value ladder
    should always be offered, even when a boosted top-`pool` would only surface one."""
    params: list[tuple[str, str]] = [
        ("rows", "0"),
        ("facet", "true"),
        ("facet.field", field),
        ("facet.mincount", "1"),
    ]
    if query and query.strip():
        params += [("q", "{!edismax qf=title v=$user_q}"), ("user_q", query)]
    else:
        params.append(("q", "*:*"))
    for clause in _filter_clauses(filters or {}):
        params.append(("fq", clause))
    r = requests.get(f"{SOLR}/solr/{CORE}/select", params=params, timeout=10)
    r.raise_for_status()
    flat = r.json().get("facet_counts", {}).get("facet_fields", {}).get(field, [])
    # facet_fields is a flat [val, count, val, count, ...] list.
    return [(flat[i], float(flat[i + 1])) for i in range(0, len(flat), 2) if flat[i + 1] > 0]


def compute_facets(
    spec: QSpec,
    filters: dict[str, str | list[str]] | None = None,
    pool: int = 200,
    qv_profile: list[float] | None = None,
) -> dict[str, list[tuple[str, float]]]:
    """Facet value tallies over the top-`pool` results for a QSpec (typed query or seed
    job). Returns {field: [(value, w)]}. For multi-select usability, each
    actively-filtered field's options are recomputed against the pool filtered by all
    OTHER fields — otherwise a field constrained by its own selection would only show the
    chosen values, leaving no way to add OR options. When qv_profile is given
    (personalized blank browse), the pool is the profile-KNN set, so facets reflect the
    profile-ranked results, not the generic recency browse."""
    filters = filters or {}
    out = _aggregate_facets(*_facet_pool(spec, filters, pool, qv_profile))
    for f in list(filters):
        if not filters[f]:
            continue
        alt = {k: v for k, v in filters.items() if k != f}
        out[f] = _aggregate_facets(*_facet_pool(spec, alt, pool, qv_profile)).get(f, out.get(f, []))
    # posted_bucket is a navigational time ladder, not a relevance read: on a blank
    # browse the recency boost (past_24h^8) makes the top-`pool` almost entirely
    # past_24h, so a pool-derived tally would offer that single option. Pull its full
    # value set from native Solr faceting over the same query+filters (minus its own
    # selection, so all rungs stay clickable) so every present bucket shows. Counts
    # aren't rendered for facets, so the only thing that matters here is the value set.
    pb_alt = {k: v for k, v in filters.items() if k != "posted_bucket"}
    pb_opts = _native_facet_options("posted_bucket", spec.bm25_text, pb_alt)
    if pb_opts:
        out["posted_bucket"] = pb_opts
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


def _prefix_matches_folded(pairs: list[tuple[str, str]], fprefix: str, limit: int) -> list[str]:
    """Accent-insensitive prefix match within a tier: `pairs` is (folded_key, original)
    sorted by folded_key; returns the originals whose folded form starts with `fprefix`."""
    lo = bisect.bisect_left(pairs, (fprefix,))
    out: list[str] = []
    for i in range(lo, len(pairs)):
        if not pairs[i][0].startswith(fprefix):
            break
        out.append(pairs[i][1])
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


app = FastAPI(title="Jobs Search Demo", lifespan=lifespan)


HTML_PAGE = """<!doctype html>
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>__PAGE_TITLE__</title>
<style>
:root {
  --ink: #1b1f29; --muted: #687081; --faint: #9aa1b0;
  --brand: #3257d6; --brand-ink: #2742a8; --brand-2: #6a44d8;
  --brand-tint: #eef1fd; --brand-tint-bd: #d7defb;
  --ok: #157a45; --ok-tint: #e9f6ee; --ok-bd: #bfe3cd;
  --warn: #8a5a00; --warn-tint: #fff4e0; --warn-bd: #f0d9a8;
  --bad: #c0392b; --bad-tint: #fdeceb; --bad-bd: #f3c7c1;
  --bg: #f6f7fb; --surface: #ffffff; --surface-2: #f8f9fc;
  --border: #e6e8f0; --border-soft: #eef0f6;
  --r: 12px; --r-md: 10px; --r-sm: 7px; --pill: 999px;
  --shadow: 0 1px 2px rgba(22,26,45,.04), 0 6px 20px rgba(22,26,45,.07);
  --shadow-sm: 0 1px 3px rgba(22,26,45,.08);
  --ring: 0 0 0 3px rgba(50,87,214,.18);
  --grad: linear-gradient(135deg, var(--brand), var(--brand-2));
}
* { box-sizing: border-box; }
body { font-family: "Inter", -apple-system, BlinkMacSystemFont, "Segoe UI", system-ui, sans-serif; max-width: 1100px; margin: 0 auto; padding: 28px 16px 56px; color: var(--ink); background: var(--bg); -webkit-font-smoothing: antialiased; line-height: 1.5; }
a { color: var(--brand); }

/* ===== masthead ===== */
.masthead { margin-bottom: 22px; }
h1 { font-size: 1.9em; line-height: 1.1; letter-spacing: -0.02em; font-weight: 700; margin: 0 0 6px; color: var(--ink); }
h1 .acc { background: var(--grad); -webkit-background-clip: text; background-clip: text; color: transparent; }
.tagline { font-size: 1.02em; color: #3b4150; margin-bottom: 4px; }
.tagline b { color: var(--brand-ink); font-weight: 700; }
.meta { color: var(--faint); font-size: 0.82em; }
.subtle { color: var(--muted); font-size: 0.9em; margin-bottom: 18px; }

/* ===== search ===== */
.search { display: flex; gap: 10px; margin-bottom: 8px; position: relative; }
.qwrap { flex: 1; position: relative; }
#query { width: 100%; padding: 13px 16px; font-size: 1.06em; border: 1px solid var(--border); border-radius: var(--r); box-sizing: border-box; background: var(--surface); box-shadow: var(--shadow-sm); transition: border-color .15s, box-shadow .15s; }
#query:focus { outline: none; border-color: var(--brand); box-shadow: var(--ring); }
#query::placeholder { color: var(--faint); }
.search > button { padding: 0 24px; font-size: 1em; font-weight: 600; cursor: pointer; border: none; border-radius: var(--r); background: var(--grad); color: #fff; box-shadow: var(--shadow-sm); transition: transform .12s, box-shadow .12s, filter .12s; }
.search > button:hover { filter: brightness(1.05); box-shadow: var(--shadow); transform: translateY(-1px); }
.search > button:active { transform: translateY(0); }
#suggest { position: absolute; top: calc(100% + 4px); left: 0; right: 0; background: var(--surface); border: 1px solid var(--border); border-radius: var(--r-md); box-shadow: var(--shadow); max-height: 300px; overflow-y: auto; z-index: 100; display: none; padding: 4px; }
#suggest .item { padding: 8px 12px; cursor: pointer; font-size: 0.95em; color: #2c313d; border-radius: var(--r-sm); }
#suggest .item:hover, #suggest .item.active { background: var(--brand-tint); color: var(--brand-ink); }
#suggest .hint { font-size: 0.75em; color: var(--faint); margin-left: 8px; }

.badge { display: inline-block; padding: 3px 11px; border-radius: var(--pill); font-size: 0.82em; margin-bottom: 12px; font-weight: 500; }
.badge.cached { background: var(--ok-tint); color: var(--ok); border: 1px solid var(--ok-bd); }
.badge.uncached { background: var(--warn-tint); color: var(--warn); border: 1px solid var(--warn-bd); }
button { padding: 8px 18px; font-size: 1em; cursor: pointer; border: 1px solid var(--border); border-radius: var(--r-sm); background: var(--surface); color: var(--ink); transition: background .12s, border-color .12s; }
button:hover { background: var(--surface-2); border-color: #d3d7e2; }

/* ===== results ===== */
.results-panel { border: 1px solid var(--border); border-radius: var(--r); padding: 6px 16px; background: var(--surface); box-shadow: var(--shadow); }
.result { display: grid; grid-template-columns: 28px 70px 1fr; gap: 10px; padding: 12px 8px; margin: 0 -8px; border-bottom: 1px solid var(--border-soft); font-size: 0.95em; align-items: start; cursor: pointer; border-radius: var(--r-sm); transition: background .12s; }
.result:hover { background: var(--surface-2); }
.r-rank { color: var(--faint); text-align: right; font-variant-numeric: tabular-nums; }
.r-source { color: var(--muted); font-size: 0.82em; text-transform: uppercase; letter-spacing: 0.5px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.r-title { color: var(--ink); word-break: break-word; }
.r-title .t { font-weight: 600; }
.r-title .m { color: var(--muted); font-size: 0.85em; margin-top: 3px; }
.r-title .m2 { color: var(--faint); font-size: 0.8em; margin-top: 2px; font-style: italic; }
.r-title .r-snip { color: #4a5160; font-size: 0.85em; line-height: 1.45; margin-top: 4px; }
.r-title .r-snip em { font-style: normal; font-weight: 600; background: #fff1c2; padding: 0 2px; border-radius: 3px; }
.r-title .sep { color: #cdd2dd; padding: 0 6px; }
.detail { grid-column: 4 / 5; margin-top: 8px; padding: 12px 14px; background: var(--surface-2); border-left: 3px solid var(--brand); border-radius: var(--r-sm); white-space: pre-wrap; color: #3a4150; font-size: 0.88em; line-height: 1.5; max-height: 480px; overflow-y: auto; }
.detail.loading { color: var(--muted); font-style: italic; }
.mlt-pivot { margin-top: 10px; display: inline-block; font-size: 0.85em; font-weight: 600; color: var(--brand); cursor: pointer; }
.mlt-pivot:hover { text-decoration: underline; }
.empty { color: var(--faint); padding: 36px; text-align: center; }
.empty .clearlink { color: var(--brand); cursor: pointer; text-decoration: underline; }
.timing { font-size: 0.8em; color: var(--faint); padding-top: 8px; }

/* ===== layout / facets ===== */
.layout { display: grid; grid-template-columns: 240px 1fr; gap: 22px; }
.facets { font-size: 0.88em; }
.facet { margin-bottom: 16px; padding-bottom: 12px; border-bottom: 1px solid var(--border-soft); }
.facet h3 { font-size: 0.74em; text-transform: uppercase; letter-spacing: 0.6px; color: var(--faint); margin: 0 0 8px 0; font-weight: 700; }
.facet .opt { display: flex; justify-content: space-between; padding: 3px 0; cursor: pointer; color: #4a5160; }
.facet .opt:hover { color: var(--brand); }
.facet .opt.active { color: var(--brand); font-weight: 600; }
.facet .opt .v { flex: 1; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.facet .opt .n { color: var(--faint); font-size: 0.85em; font-variant-numeric: tabular-nums; }
.facet .clear { color: var(--bad); cursor: pointer; font-size: 0.8em; }
.facet-empty { color: var(--faint); font-style: italic; font-size: 0.85em; padding: 6px 0; }
.active-filters { font-size: 0.85em; color: var(--muted); margin-bottom: 8px; }
.active-filters .chip { display: inline-block; background: var(--brand-tint); color: var(--brand-ink); padding: 3px 10px; border-radius: var(--pill); margin-right: 6px; cursor: pointer; border: 1px solid var(--brand-tint-bd); }
.active-filters .chip::after { content: ' ×'; color: var(--muted); }
.seed-banner { margin-bottom: 8px; }
.seed-chip { display: inline-flex; align-items: center; gap: 8px; background: var(--ok-tint); color: var(--ok); border: 1px solid var(--ok-bd); border-radius: var(--pill); padding: 5px 13px; font-size: 0.88em; }
.seed-chip .seed-x { cursor: pointer; color: #4a9c6a; font-weight: 700; }
.seed-chip .seed-x:hover { color: var(--ok); }

/* ===== profile box ===== */
.ownbox { border: 1px solid var(--border); border-radius: var(--r); background: var(--surface); margin-bottom: 16px; box-shadow: var(--shadow-sm); }
.ownbox > summary { padding: 12px 14px; cursor: pointer; font-size: 0.92em; font-weight: 600; color: var(--brand); list-style: none; }
.ownbox > summary::-webkit-details-marker { display: none; }
.ownbox > summary::before { content: '\\25b8 '; color: var(--faint); }
.ownbox[open] > summary::before { content: '\\25be '; }
.ownbody { padding: 0 14px 14px; }
#own-text { width: 100%; min-height: 120px; box-sizing: border-box; padding: 10px 12px; font-size: 0.88em; font-family: inherit; border: 1px solid var(--border); border-radius: var(--r-sm); resize: vertical; background: var(--surface-2); }
#own-text:focus { outline: none; border-color: var(--brand); box-shadow: var(--ring); background: var(--surface); }
.ownrow { display: flex; gap: 10px; align-items: center; margin-top: 10px; flex-wrap: wrap; }
#own-loc { flex: 1; min-width: 180px; padding: 9px 12px; font-size: 0.86em; border: 1px solid var(--border); border-radius: var(--r-sm); box-sizing: border-box; }
#own-loc:focus { outline: none; border-color: var(--brand); box-shadow: var(--ring); }
#own-go { padding: 9px 18px; font-size: 0.88em; font-weight: 600; background: var(--grad); color: #fff; border: none; border-radius: var(--r-sm); cursor: pointer; transition: filter .12s, transform .12s; }
#own-go:hover { filter: brightness(1.05); transform: translateY(-1px); }
.ownstatus { font-size: 0.82em; color: var(--bad); margin-top: 8px; min-height: 1em; }

/* ===== profile result panels ===== */
.panels { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
.panel { border: 1px solid var(--border); border-radius: var(--r); background: var(--surface); box-shadow: var(--shadow-sm); overflow: hidden; }
.panel h3 { margin: 0; padding: 10px 14px; font-size: 0.9em; border-bottom: 1px solid var(--border-soft); }
.panel.cos h3 { background: var(--warn-tint); color: var(--warn); }
.panel.flt h3 { background: var(--ok-tint); color: var(--ok); }
.panel .note { font-size: 0.78em; color: var(--faint); padding: 7px 14px; border-bottom: 1px solid var(--border-soft); }
.job { padding: 10px 14px; border-bottom: 1px solid var(--border-soft); cursor: pointer; transition: background .12s; }
.job:hover { background: var(--surface-2); }
.job .jt { font-weight: 600; font-size: 0.92em; }
.job .jm { color: var(--muted); font-size: 0.8em; margin-top: 2px; }
.job .jm .sep { color: #cdd2dd; padding: 0 5px; }
.job .badges { margin-top: 6px; }
.b { display: inline-block; font-size: 0.72em; padding: 2px 8px; border-radius: var(--pill); margin-right: 5px; font-weight: 500; }
.b.ok { background: var(--ok-tint); color: var(--ok); border: 1px solid var(--ok-bd); }
.b.bad { background: var(--bad-tint); color: var(--bad); border: 1px solid var(--bad-bd); }
.b.warn { background: var(--warn-tint); color: var(--warn); border: 1px solid var(--warn-bd); }
.cos-num { color: var(--muted); font-variant-numeric: tabular-nums; font-size: 0.8em; float: right; }
.jobdetail { margin-top: 8px; padding: 10px 12px; background: var(--surface-2); border-left: 3px solid var(--brand); border-radius: var(--r-sm); white-space: pre-wrap; color: #3a4150; font-size: 0.84em; line-height: 1.45; max-height: 320px; overflow-y: auto; }
.jobdetail.loading { color: var(--muted); font-style: italic; }

/* ===== related / suggestion chips ===== */
.related { margin-bottom: 16px; }
.related .rel-h { font-size: 0.74em; text-transform: uppercase; letter-spacing: 0.6px; color: var(--faint); margin-bottom: 8px; font-weight: 700; }
.related .rel-chips { display: flex; flex-wrap: wrap; gap: 8px; }
.sug { background: var(--brand-tint); color: var(--brand-ink); border: 1px solid var(--brand-tint-bd); border-radius: var(--pill); padding: 5px 13px; font-size: 0.86em; cursor: pointer; transition: background .12s, transform .12s; }
.sug:hover { background: #e2e7fb; transform: translateY(-1px); }
.sug .sug-n { color: var(--brand); opacity: .7; font-size: 0.82em; margin-left: 4px; }
#personalize-row { margin: 6px 0 14px; font-size: 0.88em; color: #4a5160; }
#personalize-row label { cursor: pointer; }
#personalize-row .pz-name { color: var(--muted); margin-left: 6px; }
.fit { display: inline-block; font-size: 0.72em; padding: 2px 8px; border-radius: var(--pill); margin-right: 5px; background: var(--brand-tint); color: var(--brand-ink); border: 1px solid var(--brand-tint-bd); font-weight: 500; }

/* facet controls: checkbox (multi-select OR) / radio (single-select recency) */
.facet .opt .cbox { flex: 0 0 auto; width: 14px; height: 14px; margin-right: 8px; border: 1.5px solid #c2c8d6; border-radius: 4px; background: var(--surface); display: inline-block; position: relative; transition: background .12s, border-color .12s; }
.facet .opt .cbox.radio { border-radius: 50%; }
.facet .opt .cbox.on { background: var(--brand); border-color: var(--brand); }
.facet .opt .cbox.on::after { content: '✓'; color: #fff; font-size: 10px; line-height: 14px; position: absolute; left: 1px; top: -1px; }
.facet .opt .cbox.radio.on::after { content: ''; left: 4px; top: 4px; width: 6px; height: 6px; background: #fff; border-radius: 50%; }
.facet .moreless { color: var(--brand); cursor: pointer; font-size: 0.82em; margin-top: 4px; }
.facet .moreless:hover { text-decoration: underline; }
.facet h3 .map-link { float: right; color: var(--brand); cursor: pointer; text-transform: none; letter-spacing: 0; font-weight: 600; font-size: 0.95em; }
.facet h3 .map-link::before { content: '🗺 '; }
.facet h3 .map-link:hover { text-decoration: underline; }

/* map picker modal */
.map-modal { position: fixed; inset: 0; background: rgba(20,24,40,0.45); z-index: 500; display: flex; align-items: center; justify-content: center; backdrop-filter: blur(2px); }
.map-card { background: var(--surface); border-radius: var(--r); padding: 16px 18px; width: min(760px, 94vw); max-height: 92vh; overflow: auto; box-shadow: 0 12px 40px rgba(20,24,40,0.3); }
.map-head { display: flex; justify-content: space-between; align-items: center; font-weight: 700; font-size: 1.05em; }
.map-head .map-close { cursor: pointer; color: var(--muted); font-size: 1.5em; line-height: 1; padding: 0 4px; }
.map-head .map-close:hover { color: var(--ink); }
.map-hint { color: var(--muted); font-size: 0.82em; margin: 4px 0 8px; }
.map-wrap { width: 100%; }
.map-foot { display: flex; justify-content: space-between; align-items: center; margin-top: 8px; }
.map-foot #map-sel { color: var(--muted); font-size: 0.85em; }
.map-foot .map-done { background: var(--grad); color: #fff; border: none; border-radius: var(--r-sm); padding: 8px 18px; cursor: pointer; font-weight: 600; }
.map-attr { color: var(--faint); font-size: 0.72em; margin-top: 8px; text-align: right; }
.geomap { width: 100%; height: auto; display: block; }
.geomap path, .geomap g { fill: #e6e8f0; stroke: #fff; stroke-width: 0.7; cursor: pointer; transition: fill 0.1s; }
.geomap path:hover, .geomap g:hover path { fill: var(--brand-tint-bd); }
.geomap .hasdata, .geomap .hasdata path { fill: #c2cdf5; }
.geomap .sel, .geomap .sel path { fill: var(--brand); }
.geomap .sel:hover, .geomap .sel:hover path { fill: var(--brand-ink); }

/* ===== pagination ===== */
.pager { display: flex; align-items: center; justify-content: center; gap: 16px; padding: 16px 0 4px; }
.pager button { padding: 7px 16px; font-size: 0.9em; }
.pager button[disabled] { opacity: 0.4; cursor: default; }
.pg-info { color: var(--muted); font-size: 0.85em; font-variant-numeric: tabular-nums; }

/* ===== responsive / mobile ===== */
.facet-toggle { display: none; }
@media (max-width: 760px) {
  body { padding: 16px 12px 40px; }
  h1 { font-size: 1.5em; }
  .tagline { font-size: 0.95em; }
  .meta { font-size: 0.78em; }
  .subtle { font-size: 0.84em; margin-bottom: 12px; }
  /* iOS zooms the page when focusing an input < 16px; pin form fields to 16px. */
  #query, #own-text, #own-loc { font-size: 16px; }
  .search > button { padding: 0 18px; }
  /* single-column: results first, facet rail collapsed by default. */
  .layout { grid-template-columns: 1fr; gap: 14px; }
  .facets { display: none; }
  .layout.show-facets .facets { display: block; order: -1; margin-bottom: 4px; }
  .facet-toggle { display: inline-block; margin-bottom: 10px; padding: 8px 16px; font-size: 0.9em; border-radius: var(--r-sm); }
  .facet .opt { padding: 9px 0; }   /* larger tap targets */
  .facet { margin-bottom: 12px; }
  /* result rows: drop the rank/score debug columns, let the title own the width. */
  .result { grid-template-columns: 1fr; gap: 3px; padding: 12px 8px; }
  .r-rank { display: none; }
  .r-source { font-size: 0.72em; }
  .detail { grid-column: 1 / -1; max-height: 60vh; }
  /* profile cos-vs-filter panels stack instead of sitting two-up. */
  .panels { grid-template-columns: 1fr; }
  .jobdetail { max-height: 50vh; }
}
</style></head>
<body>
<header class="masthead">
<h1>Jobs Search <span class="acc">Demo</span></h1>
<div class="tagline">__PAGE_SUBTITLE__</div>
<div class="meta">__PAGE_META__</div>
</header>
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
<div id="seed-banner" class="seed-banner"></div>
<div id="active-filters" class="active-filters"></div>
<button id="facet-toggle" class="facet-toggle" onclick="toggleFacets()">&#9776; Filters</button>
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
let seedJob = null;   // {idx, title} when searching by a "more jobs like this" seed instead of typed keywords
let lastSeedQs = '';  // '&seed=<idx>' appended to /api/search & /api/facets while a seed is active

function esc(s) { return (s == null ? '' : String(s)).replace(/[&<>"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c])); }
function toggleFacets() { document.querySelector('.layout').classList.toggle('show-facets'); }
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
  'jobs_data': 'OAP', 'jobs_data_usajobs': 'USA', 'jobs_data_adzuna': 'ADZ', 'jobs_data_ats_extra': 'ATS',
  'jobs_data_francetravail': 'FT', 'jobs_data_jooble': 'JBL', 'jobs_data_smartrecruiters': 'SR',
  'jobs_data_reed': 'REED', 'jobs_data_findwork': 'FW', 'jobs_data_workable': 'WRK',
  'jobs_data_themuse': 'MUSE', 'jobs_data_remoteok': 'ROK', 'jobs_data_breezy': 'BRZ',
  'jobs_data_recruitee': 'RCT'
};
const SRC_FULL = {
  'jobs_data': 'OpenApply (ATS crawl)', 'jobs_data_usajobs': 'USAJobs (federal)',
  'jobs_data_adzuna': 'Adzuna (aggregator)', 'jobs_data_ats_extra': 'Extra-ATS poller',
  'jobs_data_francetravail': 'France Travail (FR public)', 'jobs_data_jooble': 'Jooble (aggregator)',
  'jobs_data_smartrecruiters': 'SmartRecruiters (ATS)', 'jobs_data_reed': 'Reed (UK board)',
  'jobs_data_findwork': 'Findwork (board)', 'jobs_data_workable': 'Workable (ATS)',
  'jobs_data_themuse': 'The Muse (board)', 'jobs_data_remoteok': 'RemoteOK (remote board)',
  'jobs_data_breezy': 'Breezy HR (ATS)', 'jobs_data_recruitee': 'Recruitee (ATS)'
};
function shortSrc(s) { return s == null ? '' : (SRC_SHORT[s] || s); }
function srcFull(s) { return s == null ? '' : (SRC_FULL[s] || s); }
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

// ===== "more jobs like this one" — re-seeds the normal search with this job =====
// A seed behaves exactly like a typed query: same RRF retrieval, facet rail, filters,
// pagination, and profile re-rank. The seed and the keyword box are mutually exclusive —
// seeding clears the query box, and typing a query clears the seed (see runSearch).
let lastQuery = '';   // last typed query (used by clearMatch to return to it)
function pivotMoreLikeThis(idx, title) {
  seedJob = { idx, title };
  input.value = '';     // mutual exclusion: a seed replaces the keyword query
  closeSuggest();
  window.scrollTo({ top: 0, behavior: 'smooth' });
  runSearch();
}
function renderSeedBanner() {
  const el = document.getElementById('seed-banner');
  if (!seedJob || input.value.trim()) { el.innerHTML = ''; return; }
  el.innerHTML = `<span class="seed-chip">&rarr; Jobs like: <b>${esc(seedJob.title)}</b>`
    + `<span class="seed-x" title="clear seed">&times;</span></span>`;
  el.querySelector('.seed-x').addEventListener('click', () => {
    seedJob = null; input.value = ''; runSearch();
  });
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
      fit = `<div class="badges" style="margin-top:5px">${cos}${badge('sen', r.axes.sen)}${badge('loc', r.axes.loc)}${badge('gate', r.axes.gate)}${r.axes.field ? badge('field', r.axes.field) : ''}</div>`;
    }
    row.innerHTML = `<span class="r-rank">${r.rank}</span><span class="r-source" title="${esc(srcFull(r.source))}">${esc(shortSrc(r.source))}</span><span class="r-title"><div class="t">${esc(r.title)}</div>${metaLine(r)}${metaLine2(r)}${r.snippet ? `<div class="r-snip">${r.snippet}</div>` : ''}${fit}</span>`;
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
  'posted_bucket', 'salary_band_usd_annual', 'tech_stack', 'lang',
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
  lang: 'Language',
};
const FACET_VALUE_LABELS = {
  lang: { en: 'English', fr: 'French', sv: 'Swedish', de: 'German', nl: 'Dutch', es: 'Spanish', it: 'Italian' },
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
  role_family: {
    software_engineering: 'Software engineering',
    data_engineering: 'Data engineering',
    data_science_ml: 'Data science',
    data_analytics: 'Analytics / BI',
    ai_ml: 'AI / ML',
    ai_data_annotation: 'AI data annotation',
    devops_sre_infra: 'DevOps / SRE / Infra',
    security: 'Security',
    design_ux: 'Design / UX',
    product_management: 'Product management',
    project_program_management: 'Project / Program mgmt',
    marketing: 'Marketing',
    sales: 'Sales',
    customer_success_support: 'Customer success / Support',
    operations_admin: 'Operations / Admin',
    finance_accounting: 'Finance / Accounting',
    legal: 'Legal',
    hr_people_ops: 'HR / People ops',
    healthcare_clinical: 'Healthcare — clinical',
    healthcare_allied: 'Healthcare — allied',
    healthcare_admin: 'Healthcare — admin',
    education_teaching: 'Education / Teaching',
    skilled_trades_construction: 'Skilled trades / Construction',
    transportation_logistics: 'Transportation / Logistics',
    food_service_hospitality: 'Food service / Hospitality',
    retail: 'Retail',
    creative_content: 'Creative / Content',
    research_academic: 'Research / Academic',
    manufacturing_production: 'Manufacturing / Production',
    public_safety: 'Public safety',
    nonprofit_social_services: 'Nonprofit / Social services',
    consulting_strategy: 'Consulting / Strategy',
    other: 'Other',
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
// ===== pagination for the main search/browse list =====
// Facets/related are page-independent (computed over the whole pool), so paging only
// re-fetches results. Any new query or filter change goes back through runSearch, which
// resets to page 1; the Prev/Next pager moves the window without touching facets.
let resultsOffset = 0;
const PAGE_SIZE = 10;
let lastSearchQs = '';

function renderPager(div, count) {
  const hasPrev = resultsOffset > 0;
  const hasNext = count >= PAGE_SIZE;   // a full page implies there may be more
  if (!hasPrev && !hasNext) return;
  const from = count ? resultsOffset + 1 : resultsOffset;
  const bar = document.createElement('div');
  bar.className = 'pager';
  bar.innerHTML =
    `<button class="pg-prev"${hasPrev ? '' : ' disabled'}>&lsaquo; Prev</button>`
    + `<span class="pg-info">${from}&ndash;${resultsOffset + count}</span>`
    + `<button class="pg-next"${hasNext ? '' : ' disabled'}>Next &rsaquo;</button>`;
  div.appendChild(bar);
  if (hasPrev) bar.querySelector('.pg-prev').addEventListener('click', () => changePage(-1));
  if (hasNext) bar.querySelector('.pg-next').addEventListener('click', () => changePage(1));
}

async function fetchResultsPage(q) {
  const div = document.getElementById('results');
  div.innerHTML = q ? '<div class="empty">searching...</div>' : '<div class="empty">loading recent jobs…</div>';
  const searchRes = await fetch(
    `/api/search?q=${encodeURIComponent(q)}&start=${resultsOffset}${lastSeedQs}${lastSearchQs}`
  ).then(r => r.json());
  if (searchRes.served_with) {
    document.getElementById('badge-row').innerHTML =
      `<span class="badge cached">Served with: ${esc(searchRes.served_with)}</span>`;
  }
  renderResults(div, searchRes.results, searchRes.ms);
  renderPager(div, (searchRes.results || []).length);
}

function changePage(delta) {
  resultsOffset = Math.max(0, resultsOffset + delta * PAGE_SIZE);
  fetchResultsPage(lastQuery);
  window.scrollTo({ top: 0, behavior: 'smooth' });
}

async function runSearch() {
  const q = input.value.trim();
  if (q) seedJob = null;        // typing a query overrides any active seed (mutual exclusion)
  lastQuery = q;
  lastSeedQs = (seedJob && !q) ? ('&seed=' + seedJob.idx) : '';
  renderSeedBanner();
  closeSuggest();
  if (profile && document.getElementById('pz-on').checked) { return runPersonalized(q); }
  document.getElementById('badge-row').innerHTML = '';
  resultsOffset = 0;            // new query/filter context — back to page 1
  renderActiveFilters();
  lastSearchQs = buildFilterQS();
  // Facets are page-independent, so fetch them once alongside the first page.
  const facetP = fetch(`/api/facets?q=${encodeURIComponent(q)}${lastSeedQs}${lastSearchQs}`).then(r => r.json());
  await fetchResultsPage(q);
  renderFacets((await facetP).facets);
  // Related searches need a text anchor — use the typed query, or the seed's title.
  loadRelated(q || (seedJob ? seedJob.title : ''));
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
  const seedIdx = (seedJob && !q) ? seedJob.idx : null;
  const body = { q, seed: seedIdx, k: 10, hard_filter: hard, filters: activeFilters, profile };
  // Facets come back inline, computed over the SAME profile-ranked pool as the results
  // (a separate /api/facets call would be profile-blind and mismatch the listing).
  const searchRes = await fetch('/api/search_personalized', {
    method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body),
  }).then(r => r.json());
  if (searchRes.error) { div.innerHTML = '<div class="empty">' + esc(searchRes.error) + '</div>'; return; }
  badgeRow.innerHTML = `<span class="badge cached">Served with: ${esc(searchRes.served_with)}</span>`;
  if (!searchRes.results || !searchRes.results.length) {
    const hasFilters = Object.keys(activeFilters).length > 0;
    let msg;
    if (hard) {
      msg = 'no jobs match this query that you also qualify for — untick the 3-axis filter to see near-misses';
    } else if (hasFilters) {
      msg = 'No jobs match these filters. <span id="clear-filters-link" class="clearlink">Clear all filters</span> to broaden your search.';
    } else {
      msg = 'no results';
    }
    div.innerHTML = '<div class="empty">' + msg + '</div>';
    const cl = document.getElementById('clear-filters-link');
    if (cl) cl.addEventListener('click', () => {
      for (const f of Object.keys(activeFilters)) delete activeFilters[f];
      runSearch();
    });
  } else {
    renderResults(div, searchRes.results, searchRes.ms);
  }
  renderFacets(searchRes.facets);
  // Typed query -> query-related role moves; seed -> moves around the seed's title;
  // blank profile-driven browse -> the profile's suggested searches.
  if (q) loadRelated(q);
  else if (seedJob) loadRelated(seedJob.title);
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
        n_str = "~300,000"
    title = f"Jobs Search Demo: {n_str} postings"
    subtitle = f"Semantic + lexical search across {n_str} live job postings."
    meta = (
        "14 sources · RRF(BM25 + e5-small) · browse by default, refine with facets "
        "& maps, or match your own profile for personalized results"
    )
    return (
        HTML_PAGE.replace("__PAGE_TITLE__", title)
        .replace("__PAGE_SUBTITLE__", subtitle)
        .replace("__PAGE_META__", meta)
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
    # Tagged tiers (title > combo > head > tail > synth) rank by source quality; sorted_keys
    # is the catch-all fallback (it also carries strings the tagged tiers deliberately
    # excluded), so it's consulted only to fill out the list.
    tier_order = ("title", "fr", "sv", "de", "nl", "es", "it", "combo", "head", "tail", "synth")
    # Gather the whole candidate pool first (best/lowest tier index per unique string),
    # THEN rank — so a bare stem in a low tier ("product manager" is tagged synth) isn't
    # truncated before it can rank. Matching is accent-insensitive WITHIN each tier (the
    # prefix and the tier keys are folded), so "electr"/"électr" both reach the fr-tier
    # "électricien" instead of losing to US-geo/company keys in the alphabetical catch-all
    # below. Shorter suggestions sort above the longer ones that extend them (string
    # length ascending), then by source tier, then alphabetically.
    fprefixes = [_fold(p) for p in prefixes]
    best_tier: dict[str, int] = {}
    for ti, tag in enumerate(tier_order):
        fl = R["tier_folded"].get(tag, [])
        for fp in fprefixes:
            for k in _prefix_matches_folded(fl, fp, limit * 4):
                if k not in best_tier or ti < best_tier[k]:
                    best_tier[k] = ti
    pool = sorted(best_tier, key=lambda k: (len(k), best_tier[k], k))
    seen: set[str] = set(best_tier)
    out: list[dict] = [{"text": k} for k in pool[:limit]]
    if len(out) < limit:  # fall back to the catch-all tier, same stem-first ordering
        fb = {k for p in prefixes for k in _prefix_matches(R["sorted_keys"], p, limit * 4)}
        for k in sorted(fb - seen, key=lambda k: (len(k), k)):
            seen.add(k)
            out.append({"text": k})
            if len(out) >= limit:
                break
    if len(out) < limit:  # accent-insensitive pass: "ingenieur" -> "ingénieur"
        fp = _fold(prefix)
        fkeys = R.get("folded_keys", [])
        fpairs = R.get("folded_pairs", [])
        for i in range(bisect.bisect_left(fkeys, fp), len(fkeys)):
            if not fkeys[i].startswith(fp):
                break
            orig = fpairs[i][1]
            if orig not in seen:
                seen.add(orig)
                out.append({"text": orig})
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
    if not q.strip():
        return JSONResponse({"suggestions": []})
    # High-confidence language routing (checked BEFORE the cognate _resolve chain below).
    # Two unambiguous signals win immediately: (1) the lang gate fired for a specific
    # language; (2) the query is a corpus-grounded role in EXACTLY ONE non-English lane.
    # Without this, an Italian (or other) loanword that also resolves in an earlier Romance
    # taxonomy gets hijacked: "pizzaiolo" resolves in French ROME, so the French branch
    # claimed it and returned French neighbours. The unique-grounding check keeps a query
    # that is a mined role in only one corpus in that corpus's lane, while a true cross-
    # lane cognate (grounded in 2+) or a known English query falls through unchanged.
    _LANES = [
        ("fr", R.get("fr_related"), None),
        ("de", R.get("de_related"), "de_role_keys"),
        ("nl", R.get("nl_related"), "nl_role_keys"),
        ("es", R.get("es_related"), "es_role_keys"),
        ("sv", R.get("sv_related"), "sv_role_keys"),
        ("it", R.get("it_related"), "it_role_keys"),
    ]
    _mode = query_lang_mode(q)
    for _nm, _lane, _ in _LANES:
        if _lane is not None and _mode == _nm:
            return JSONResponse({"suggestions": _lane.suggest(q, k=k)})
    _qf = _fold(q.strip().lower())
    if q.strip().lower() not in R.get("query_key_set", set()):
        _grounded = [
            _lane
            for _nm, _lane, _rk in _LANES
            if _lane is not None and _rk and _qf in R.get(_rk, set())
        ]
        if len(_grounded) == 1:
            return JSONResponse({"suggestions": _grounded[0].suggest(q, k=k)})
    fr = R.get("fr_related")
    # Route to the grounded ROME lane when the query is French. The e5-small-v2 suggester
    # clusters French by morphology, not meaning ("développeur" -> "educateur"), so French
    # gets France-Travail-validated career moves instead (query -> ROME -> mobilite -> a
    # corpus-mined French role per related ROME). French is signalled by EITHER the
    # high-precision lang gate (diacritic/function word) OR the query resolving to a real
    # France-Travail appellation -- the latter catches bare cognate roles the gate misses
    # ("pharmacien", "agent d'entretien", "avocat"), which the English lane otherwise
    # answers with wrong-language neighbours ("Pharmacy Technician"). The corpus-membership
    # guard keeps franglais English queries that are ALSO ROME appellations ("data
    # scientist", "product manager") in the English lane.
    if fr is not None and (
        query_lang_mode(q) == "fr"
        or (q.strip().lower() not in R.get("query_key_set", set()) and fr._resolve(q) is not None)
    ):
        return JSONResponse({"suggestions": fr.suggest(q, k=k)})
    # German rides the same pattern on the ESCO backbone (DeRelatedSuggester): the gate, or
    # a bare German cognate role ("elektriker", "krankenpfleger") that resolves to a real
    # ESCO occupation but isn't a known English query (the corpus-membership guard keeps
    # franglais English in the English lane). Crucially, once we've decided a query is
    # German we serve the ESCO lane EVEN IF it yields nothing (return empty) rather than
    # falling through to the English e5 lane — that lane only produces morphology noise on
    # German, and ESCO's German vocabulary is thinner than ROME's so empty is common.
    de = R.get("de_related")
    if de is not None:
        qstrip = q.strip().lower()
        is_german = query_lang_mode(q) == "de" or (
            qstrip not in R.get("query_key_set", set())
            and (_fold(qstrip) in R.get("de_role_keys", set()) or de._resolve(q) is not None)
        )
        if is_german:
            return JSONResponse({"suggestions": de.suggest(q, k=k)})
    # Dutch rides the same pattern on the ESCO backbone (NlRelatedSuggester): the gate, or
    # a bare Dutch cognate role ("monteur", "verpleegkundige") that resolves to a real ESCO
    # occupation but isn't a known English query. As with German, once a query is judged
    # Dutch we serve the ESCO lane EVEN IF empty rather than falling through to the English
    # e5 lane (morphology noise on Dutch).
    nl = R.get("nl_related")
    if nl is not None:
        qstrip = q.strip().lower()
        is_dutch = query_lang_mode(q) == "nl" or (
            qstrip not in R.get("query_key_set", set())
            and (_fold(qstrip) in R.get("nl_role_keys", set()) or nl._resolve(q) is not None)
        )
        if is_dutch:
            return JSONResponse({"suggestions": nl.suggest(q, k=k)})
    # Spanish rides the same pattern on the ESCO backbone (EsRelatedSuggester): the gate, or
    # a bare Spanish cognate role ("camarero", "electricista") that resolves to a real ESCO
    # occupation but isn't a known English query. As with German/Dutch, once a query is
    # judged Spanish we serve the ESCO lane EVEN IF empty rather than falling through to the
    # English e5 lane (morphology noise on Spanish).
    es = R.get("es_related")
    if es is not None:
        qstrip = q.strip().lower()
        is_spanish = query_lang_mode(q) == "es" or (
            qstrip not in R.get("query_key_set", set())
            and (_fold(qstrip) in R.get("es_role_keys", set()) or es._resolve(q) is not None)
        )
        if is_spanish:
            return JSONResponse({"suggestions": es.suggest(q, k=k)})
    # Swedish rides the same ESCO backbone (SvRelatedSuggester): the gate, or a bare Swedish
    # role ("snickare", "undersköterska") that resolves to a real ESCO occupation but isn't a
    # known English query. As with the other ESCO lanes, once a query is judged Swedish we
    # serve the ESCO lane EVEN IF empty rather than falling through to the English e5 lane
    # (morphology noise on Swedish). A shared cognate ("elektriker") is claimed by the German
    # lane above by precedence, which is fine — German serves it from the same ESCO backbone.
    sv = R.get("sv_related")
    if sv is not None:
        qstrip = q.strip().lower()
        is_swedish = query_lang_mode(q) == "sv" or (
            qstrip not in R.get("query_key_set", set())
            and (_fold(qstrip) in R.get("sv_role_keys", set()) or sv._resolve(q) is not None)
        )
        if is_swedish:
            return JSONResponse({"suggestions": sv.suggest(q, k=k)})
    # Italian rides the same pattern on the ESCO backbone (ItRelatedSuggester): the gate, or
    # a bare Italian cognate role ("cameriere", "elettricista") that resolves to a real ESCO
    # occupation but isn't a known English query. As with the other ESCO lanes, once a query
    # is judged Italian we serve the ESCO lane EVEN IF empty rather than falling through to
    # the English e5 lane (morphology noise on Italian).
    it = R.get("it_related")
    if it is not None:
        qstrip = q.strip().lower()
        is_italian = query_lang_mode(q) == "it" or (
            qstrip not in R.get("query_key_set", set())
            and (_fold(qstrip) in R.get("it_role_keys", set()) or it._resolve(q) is not None)
        )
        if is_italian:
            return JSONResponse({"suggestions": it.suggest(q, k=k)})
    # English / ambiguous: the English e5 lane first.
    rs = R.get("role_suggester")
    en = rs.suggest(q, np.asarray(_dense_qv(q), dtype=np.float32), k=k) if rs is not None else []
    # A French role the checks above missed and the English lane can't answer: last-ditch
    # ROME fallback (only fires if the query resolves to a real French appellation).
    if not en and fr is not None:
        return JSONResponse({"suggestions": fr.suggest(q, k=k)})
    return JSONResponse({"suggestions": en})


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
def api_search(
    request: Request,
    q: str = Query(""),
    seed: int | None = Query(None),
    k: int = Query(10),
    start: int = Query(0),
):
    """Keyword search, "more jobs like this" when `seed` (a job idx) is given, or — when
    both are blank — the recent/low-barrier browse default. A typed query takes
    precedence over a seed (the two are mutually exclusive in the UI). `start` is the
    pagination offset (0-based) into the employer-capped ranked list."""
    filters = _parse_filters(request)
    _apply_lang_gate(q, filters)
    start = max(0, start)
    spec = qspec_text(q) if q.strip() else (qspec_seed(seed) if seed is not None else None)
    t0 = time.time()
    if spec is not None and spec.active:
        res = search_default(spec, k, filters, start)
        retriever = "rrf_bm25_e5_seed" if spec.is_seed else "rrf_bm25_e5"
        served = SERVING_MODE + (" — seeded by a job" if spec.is_seed else "")
    else:
        res = browse_default(k, filters, start)
        retriever = "browse_recent"
        served = "Browse: recent + low-barrier [via Solr]"
    # Highlight the typed query in each result's passage; seed/browse get a plain lead.
    _attach_snippets(res, q if (spec is not None and spec.active and not spec.is_seed) else "")
    ms = int((time.time() - t0) * 1000)
    return JSONResponse(
        {
            "query": q,
            "seed": seed,
            "retriever": retriever,
            "served_with": served,
            "filters": filters,
            "start": start,
            "results": res,
            "ms": ms,
        }
    )


@app.get("/api/facets")
def api_facets(
    request: Request, q: str = Query(""), seed: int | None = Query(None), pool: int = Query(200)
):
    """Facet counts over the top-`pool` results (fused query/seed results, or the
    blank-browse pool when both are empty) with the same filters the search uses, so
    counts stay coherent with what the user sees."""
    filters = _parse_filters(request)
    _apply_lang_gate(q, filters)
    spec = qspec_text(q) if q.strip() else (qspec_seed(seed) if seed is not None else None)
    t0 = time.time()
    facets = compute_facets(spec or qspec_text(""), filters, pool=pool)
    ms = int((time.time() - t0) * 1000)
    return JSONResponse(
        {
            "query": q,
            "seed": seed,
            "filters": filters,
            "pool": pool,
            "facets": facets,
            "ms": ms,
        }
    )


def _clean_text(s: str) -> str:
    """Decode entities + collapse whitespace. Thin alias for snippet_lib.clean_text so
    the live snippet text matches the text the offline encoder split + embedded."""
    return clean_text(s)


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
    "posted_at,source_corpus,industry,employment_type,department,role_family,"
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
            "role_family": d.get("role_family") or "other",
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
        "source": d.get("source_corpus") or "",  # raw; the client maps it to a short code + tooltip
        "cosine": round(float(cos), 4),
        "axes": st,
    }


def _run_profile_match(r: dict, qv: list[float], qvs: list[list[float]] | None = None) -> dict:
    """e5-small KNN top-`PROFILE_POOL` (max-sim over the profile vectors), then the
    3-axis filter with job_features computed live from the hydrated Solr docs."""
    hits = _topk_knn_multi(_prof_vecs(qv, qvs), PROFILE_POOL)  # [(idx, cosine), ...] best
    hyd = _hydrate_for_match([i for i, _ in hits])
    self_emps = _self_employers(r)
    cosine_list: list[dict] = []
    filtered_list: list[dict] = []
    filtered_count = 0
    for idx, cos in hits:
        d = hyd.get(idx)
        if not d:
            continue
        if _is_self_employer(d.get("employer") or "", self_emps):
            continue  # don't surface the seeker's own current/recent employer
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
_DANGLING_PAREN = re.compile(r"\s*\([^()]*$")


def _close_parens(s: str) -> str:
    """Drop a dangling unclosed parenthetical so a personalized suggestion never shows as
    'Applied Researcher (MTS'. _TITLE_AT cuts a title at the first comma/dash, which can
    land INSIDE a parenthetical ('Applied Researcher (MTS, NLP)' -> '...(MTS'); PDF
    extraction can also drop the ')'. Either way, strip the open '(...' tail."""
    if s.count("(") > s.count(")"):
        s = _DANGLING_PAREN.sub("", s)
    return s.strip(" -|,")


# Folded single-token French "roles" that recur inside common phrases and so produce
# spurious resume matches (e.g. "charge" from "prise en charge"). Qualified multi-token
# variants ("chargé de recrutement") are unaffected.
_FR_RESUME_STOP = {"charge"}

# Coarse profile_field() bucket -> a friendlier scope noun for student suggestions when no
# explicit major parsed ("tech internship" reads worse than "software internship"; "cs" here
# means customer-success, not computer-science, so spell it out).
_FIELD_SCOPE = {
    "tech": "software",
    "cs": "customer service",
    "hr": "human resources",
    "product": "product management",
}


def _study_role_form(scope: str) -> str:
    """Turn a field-of-study noun into a role noun for the 'junior/graduate X' variants:
    'mechanical engineering' -> 'mechanical engineer'. Other fields pass through (the caller
    BM25-validates, so a non-role form that matches nothing is dropped)."""
    return scope[:-3] if scope.endswith("engineering") else scope  # 'engineering' -> 'engineer'


def _student_queries(major: str, field: str | None) -> list[str]:
    """Internship + entry-level suggested searches for a student / recent grad, scoped to
    the parsed major when available else the coarse profile field. Over-generates in
    most-specific-first order (Title-cased for display alongside role-title suggestions); the
    caller dedups, BM25-validates, and truncates, so 'Mechanical Engineering Internship'
    survives while a field that surfaces no postings silently drops."""
    out: list[str] = []
    scope = major or _FIELD_SCOPE.get(field or "", field or "")
    if scope:
        s = scope.title()
        role = _study_role_form(scope).title()
        out += [
            f"{s} Internship",
            f"{s} Intern",
            f"Entry Level {s}",
            f"Junior {role}",
            f"Graduate {role}",
            s,
        ]
    out += ["Internship", "Entry Level"]  # field-agnostic fallback, always grounded
    return out


def _suggest_queries(blob: str, r: dict, limit: int = 6) -> list[dict]:
    """Deterministic query suggestions from the parsed profile, validated against the
    live index. Sources (most-specific first): the recent role title, a seniority-
    broadened variant, earlier role titles, and the headline when it reads like a role.
    Each candidate is kept only if BM25 returns at least one job, and tagged with that
    count so the UI can show how many postings it would surface."""
    cands: list[str] = []
    lang, prob = detect_lang(blob)
    fr_folded = R.get("fr_roles_folded") or []
    if lang == "fr" and prob >= 0.5 and fr_folded:
        # The resume parser keys off English section headers/layouts, so role_titles()
        # yields nothing for a French CV. Instead dictionary-match the resume against the
        # mined French role vocab (longest-first so "aide-soignant" beats "aide"); every
        # hit is a corpus-grounded role, then BM25-validated below like any other.
        # Degender the blob so feminine CVs ("infirmière", "vendeuse") match the masculine
        # vocab -- without this an "infirmière" resume matched nothing and fell through to
        # the generic "chargé" picked out of "prise en charge".
        fb = degender_fr(_fold(blob))
        for folded_role, role in fr_folded:
            # generic tokens that recur INSIDE French phrases, not as standalone roles
            # here ("charge" <- "prise en charge"); the qualified forms ("chargé de ...")
            # are multi-token and unaffected.
            if folded_role in _FR_RESUME_STOP:
                continue
            if re.search(r"\b" + re.escape(folded_role) + r"\b", fb):
                cands.append(role)
            if len(cands) >= limit * 2:
                break
    else:
        # Student / recent-grad lane: a CV with little/no work history has no concrete role
        # title to suggest from, so lead with internship + entry-level searches scoped to the
        # field of study (parsed major, else coarse field). These are most-specific-first, so
        # after BM25 validation they fill the slots ahead of any role-title fallback below.
        if L.is_student(blob, r.get("degree"), r.get("years"), r.get("seniority")):
            cands.extend(_student_queries(L.field_of_study(blob), r.get("field")))
        for t in L.role_titles(blob)[:3]:
            t = _close_parens(_TITLE_AT.sub("", t).strip(" -|,"))
            if t:
                cands.append(t)
                broad = _SENIORITY_PREFIX.sub("", t).strip()
                if broad and broad.lower() != t.lower():
                    cands.append(broad)
        hl = (r.get("headline") or "").strip()
        if (
            hl
            and 2 <= len(hl) <= 60
            and not _ASPIRATIONAL.search(hl)
            and not L._looks_like_name(hl)
        ):
            cands.append(_close_parens(_TITLE_AT.sub("", hl).strip(" -|,")))
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
    # Second profile vector: a dense title/headline specialization signal. Matching is
    # max-sim over both, so a specialist isn't washed out by a long generic centroid.
    spec_txt = L.specialization_text(blob)
    qvs = [qv] + ([_dense_qv(spec_txt)] if spec_txt.strip() else [])
    out = _run_profile_match(r, qv, qvs)
    # suggested searches (#1) + the parsed profile the client holds and re-sends to
    # personalize subsequent keyword searches (#2). Nothing is persisted server-side.
    out["suggestions"] = _suggest_queries(blob, r)
    out["profile"] = {"r": r, "qv": qv, "qvs": qvs}
    return JSONResponse(out)


@app.post("/api/search_personalized")
async def api_search_personalized(request: Request):
    """Keyword search re-ranked by a client-held profile (from /api/match_profile).
    Stateless: the profile (features + e5 vector) is sent in the body each call."""
    body = await request.json()
    q = (body.get("q") or "").strip()
    seed = body.get("seed")
    prof = body.get("profile") or {}
    r, qv = prof.get("r"), prof.get("qv")
    if not r or not qv:
        return JSONResponse({"error": "no profile loaded"}, status_code=400)
    qvs = _prof_vecs(qv, prof.get("qvs"))  # multi-vector when the client holds it
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
    _apply_lang_gate(q, filters)
    k = int(body.get("k") or 10)
    hard = bool(body.get("hard_filter"))
    spec = qspec_text(q) if q else (qspec_seed(int(seed)) if seed is not None else None)
    t0 = time.time()
    # A query or seed defines eligibility (re-ranked by profile fit); a blank browse
    # ranks the whole (filtered) catalog by profile fit.
    if spec is not None and spec.active:
        res = search_personalized(spec, r, qv, k, filters, hard, qvs=qvs)
        served = (
            SERVING_MODE + (" — seeded by a job" if spec.is_seed else "") + " + profile re-rank"
        )
        retriever = "rrf_bm25_e5_seed+profile" if spec.is_seed else "rrf_bm25_e5+profile"
    else:
        res = browse_personalized(r, qv, k, filters, hard, qvs=qvs)
        served = "Browse + profile re-rank"
        retriever = "browse+profile"
    # Facets must reflect the SAME pool the user sees: for a query/seed that's the fused
    # pool (profile-blind ranking is fine here), for a blank personalized browse it's the
    # profile-KNN pool. Returned inline so the client doesn't make a second,
    # profile-blind /api/facets call.
    facets = compute_facets(
        spec or qspec_text(""), filters, qv_profile=(None if (spec and spec.active) else qv)
    )
    _attach_snippets(res, q if (spec is not None and spec.active and not spec.is_seed) else "")
    ms = int((time.time() - t0) * 1000)
    return JSONResponse(
        {
            "query": q,
            "seed": seed,
            "retriever": retriever,
            "served_with": served,
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
