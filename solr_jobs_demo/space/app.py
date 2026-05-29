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
from fastapi import FastAPI, File, Form, Query, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse

import resume_match_lib as L

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


POSTED_BUCKET_NESTING = {
    "past_7d": ["past_7d"],
    "past_30d": ["past_7d", "past_30d"],
    "past_90d": ["past_7d", "past_30d", "past_90d"],
    "older": ["older"],
}


def _filter_clauses(filters: dict[str, str]) -> list[str]:
    """Build Solr fq= clauses from a {field: value} filter dict.
    Skips empty values; quotes the value to handle spaces/special chars."""
    out = []
    for k, v in filters.items():
        if not v or k not in FACET_FIELDS:
            continue
        if k == "posted_bucket" and v in POSTED_BUCKET_NESTING:
            members = POSTED_BUCKET_NESTING[v]
            out.append("posted_bucket:(" + " OR ".join(members) + ")")
            continue
        # tech_stack is multi-value; we filter on a single chosen tech.
        out.append(f'{k}:"{v}"')
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
        "posted": (hyd.get("posted_at") or "")[:7],
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


def search_default(
    query: str,
    k: int = 10,
    filters: dict[str, str] | None = None,
) -> list[dict]:
    """RRF(BM25, e5-small) with optional facet filters, then cap to
    EMPLOYER_CAP results per employer for display diversity.
    Cap is bypassed when the user explicitly filtered by employer, OR when
    one employer dominates the unfiltered pool (>= EMPLOYER_DOMINANCE share),
    which signals employer-coupled query intent (e.g. 'amazon jobs')."""
    cap = 0 if (filters and filters.get("employer")) else EMPLOYER_CAP
    pool_k = max(k * (cap + 2), k + 20) if cap > 0 else k
    items = _fused_topk(query, pool_k, filters, RRF_POOL)
    ids = [i for i, _ in items]
    hyd = _hydrate(ids)
    if cap > 0:
        emp_counts: dict[str, int] = defaultdict(int)
        for idx, _ in items:
            emp = (hyd.get(idx, {}).get("employer") or "").strip().lower()
            if emp:
                emp_counts[emp] += 1
        total = sum(emp_counts.values())
        exempt = {e for e, n in emp_counts.items() if total and n / total >= EMPLOYER_DOMINANCE}
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
        items = kept
    else:
        items = items[:k]
    return [_make_result(r + 1, s, i, hyd.get(i, {})) for r, (i, s) in enumerate(items)]


POSTED_BUCKET_ORDER = ["past_7d", "past_30d", "past_90d", "older"]

FACET_TAIL_VALUES = {
    "role_family": {"other"},
    "industry": {"unclassified"},
}


def compute_facets(
    query: str,
    filters: dict[str, str] | None = None,
    pool: int = 200,
) -> dict[str, list[tuple[str, int]]]:
    """Aggregate facet values over the top-`pool` RRF-fused docs (with filters
    applied at retrieval), weighted by 1/(rank+1) so values at the head of the
    result list dominate ordering. Returns {field: [(value, weight), ...]}."""
    items = _fused_topk(query, pool, filters, pool)
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
            if isinstance(v, list):
                for vv in v:
                    weights[f][vv] += w
            else:
                weights[f][v] += w
    out: dict[str, list[tuple[str, float]]] = {}
    for f in FACET_FIELDS:
        if f == "posted_bucket":
            present = weights[f]
            out[f] = [(b, present[b]) for b in POSTED_BUCKET_ORDER if b in present]
        else:
            tail = FACET_TAIL_VALUES.get(f, set())
            out[f] = sorted(weights[f].items(), key=lambda x: (x[0] in tail, -x[1]))
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
<div id="badge-row"></div>
<div id="active-filters" class="active-filters"></div>
<div class="layout">
  <div class="facets" id="facets"></div>
  <div class="results-panel">
    <div id="results"><div class="empty">type a query — RRF(BM25 + e5-small) via Solr</div></div>
  </div>
</div>
<script>
const input = document.getElementById('query');
const suggestBox = document.getElementById('suggest');
let suggestItems = [];
let suggestActive = -1;
let suggestTimer = null;

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
  } catch (e) {
    div.classList.remove('loading');
    div.textContent = '(failed to load)';
  }
}
function renderResults(div, items, ms) {
  if (!items || !items.length) { div.innerHTML = '<div class="empty">no results</div>'; return; }
  div.innerHTML = '';
  items.forEach(r => {
    const row = document.createElement('div');
    row.className = 'result';
    row.innerHTML = `<span class="r-rank">${r.rank}</span><span class="r-score">${r.score.toFixed(4)}</span><span class="r-source">${esc(shortSrc(r.source))}</span><span class="r-title"><div class="t">${esc(r.title)}</div>${metaLine(r)}${metaLine2(r)}</span>`;
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
    past_7d: 'Past 7 days',
    past_30d: 'Past 30 days',
    past_90d: 'Past 90 days',
    older: 'Older than 90 days',
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
const activeFilters = {};   // field -> value

function buildFilterQS() {
  const parts = [];
  for (const [k, v] of Object.entries(activeFilters)) {
    if (v) parts.push(`${k}=${encodeURIComponent(v)}`);
  }
  return parts.length ? '&' + parts.join('&') : '';
}
function renderActiveFilters() {
  const row = document.getElementById('active-filters');
  const keys = Object.keys(activeFilters).filter(k => activeFilters[k]);
  if (!keys.length) { row.innerHTML = ''; return; }
  row.innerHTML = 'Filters: ' + keys.map(k =>
    `<span class="chip" data-k="${k}">${esc(FACET_LABELS[k] || k)}: ${esc(facetValueLabel(k, activeFilters[k]))}</span>`
  ).join('');
  row.querySelectorAll('.chip').forEach(el => el.addEventListener('click', () => {
    delete activeFilters[el.dataset.k];
    runSearch();
  }));
}
function renderFacets(facets) {
  const root = document.getElementById('facets');
  const parts = [];
  for (const f of FACET_FIELDS) {
    const opts = (facets && facets[f]) || [];
    if (!opts.length && !activeFilters[f]) continue;
    let inner = `<h3>${esc(FACET_LABELS[f] || f)}</h3>`;
    if (activeFilters[f]) {
      inner += `<div class="clear" data-f="${f}">clear ${esc(facetValueLabel(f, activeFilters[f]))}</div>`;
    }
    if (!opts.length) {
      inner += '<div class="facet-empty">(no values)</div>';
    } else {
      inner += opts.slice(0, 8).map(([v, _n]) =>
        `<div class="opt${activeFilters[f] === v ? ' active' : ''}" data-f="${f}" data-v="${esc(v)}"><span class="v">${esc(facetValueLabel(f, v))}</span></div>`
      ).join('');
    }
    parts.push(`<div class="facet">${inner}</div>`);
  }
  root.innerHTML = parts.join('') || '<div class="facet-empty">no facets yet — run a search</div>';
  root.querySelectorAll('.opt').forEach(el => el.addEventListener('click', () => {
    const f = el.dataset.f, v = el.dataset.v;
    activeFilters[f] = activeFilters[f] === v ? '' : v;
    if (!activeFilters[f]) delete activeFilters[f];
    runSearch();
  }));
  root.querySelectorAll('.clear').forEach(el => el.addEventListener('click', () => {
    delete activeFilters[el.dataset.f];
    runSearch();
  }));
}
async function runSearch() {
  const q = input.value.trim();
  if (!q) return;
  closeSuggest();
  const div = document.getElementById('results');
  const badgeRow = document.getElementById('badge-row');
  badgeRow.innerHTML = '';
  div.innerHTML = '<div class="empty">searching...</div>';
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
}

// ===== "find jobs for yourself": profile -> jobs, cosine vs 3-axis filter =====
function badge(name, ax) {
  const cls = ax.ok ? 'ok' : 'bad';
  const mark = ax.ok ? '✓' : '✗';
  const tip = ax.reason ? ' — ' + ax.reason : '';
  return `<span class="b ${cls}" title="${esc(ax.reason)}">${name} ${mark}${ax.ok ? '' : esc(tip)}</span>`;
}
function matchJobRow(j) {
  const m = [];
  m.push(j.remote ? '🌐 remote' : esc(j.location || '(no location)'));
  if (j.employer) m.push(esc(j.employer));
  m.push('level: ' + esc(j.seniority));
  if (j.years_req != null) m.push('needs ' + j.years_req + ' yrs');
  if (j.degree_req) m.push('needs ' + esc(j.degree_req));
  if (j.cred_gates && j.cred_gates.length) m.push('needs ' + j.cred_gates.map(esc).join(', '));
  const extra = [];
  if (j.clearance) extra.push('<span class="b warn" title="security clearance stated (not resume-checkable)">clearance</span>');
  if (j.workauth) extra.push('<span class="b warn" title="work-authorization stated (not resume-checkable)">work-auth</span>');
  const row = document.createElement('div');
  row.className = 'job';
  row.innerHTML = `<span class="cos-num">cos ${j.cosine.toFixed(3)}</span>
    <div class="jt">${esc(j.title)}</div>
    <div class="jm">${m.join('<span class="sep">&middot;</span>')}</div>
    <div class="badges">${badge('sen', j.axes.sen)}${badge('loc', j.axes.loc)}${badge('gate', j.axes.gate)}${extra.join('')}</div>`;
  if (j.idx != null && j.idx >= 0) row.addEventListener('click', () => toggleDetail(j.idx, row));
  return row;
}
function clearMatch() {
  document.getElementById('results').innerHTML =
    '<div class="empty">type a query — RRF(BM25 + e5-small) via Solr</div>';
}
function renderMatch(d) {
  document.getElementById('badge-row').innerHTML = '';
  document.getElementById('facets').innerHTML = '';
  document.getElementById('active-filters').innerHTML = '';
  const box = document.getElementById('results');
  const rs = d.resume;
  const facts = [];
  facts.push('level: <b>' + esc(rs.seniority) + '</b>');
  if (rs.years != null) facts.push('experience: <b>' + rs.years + ' yrs</b>');
  facts.push('degree: <b>' + esc(rs.degree) + '</b>');
  if (rs.creds && rs.creds.length) facts.push('creds: <b>' + rs.creds.map(esc).join(', ') + '</b>');
  const note = d.filtered_count < d.pool_n
    ? `${d.filtered_count} of top-${d.pool_n} pass all 3 axes`
    : `all top-${d.pool_n} pass`;
  box.innerHTML = `
    <div class="rsum">
      <span class="back" onclick="clearMatch()">&larr; back to search</span>
      <div class="nm">${esc(rs.name)}</div>
      <div class="hl">${esc(rs.headline)}</div>
      <div class="lc" style="color:#888;font-size:0.85em">${esc(rs.loc)}</div>
      <div class="facts">${facts.join('<span style="color:#ccc;padding:0 6px">&middot;</span>')}</div>
    </div>
    <div class="panels">
      <div class="panel cos">
        <h3>Raw cosine (constraint-blind)</h3>
        <div class="note">nearest jobs by embedding similarity &mdash; ignores hard constraints</div>
        <div id="cos-list"></div>
      </div>
      <div class="panel flt">
        <h3>3-axis constraint filter</h3>
        <div class="note">${note} &middot; best-cosine survivor first${d.filtered_count === 0 ? ' (none qualified &mdash; cosine top-1 fallback)' : ''}</div>
        <div id="flt-list"></div>
      </div>
    </div>`;
  const cosList = box.querySelector('#cos-list');
  (d.cosine || []).forEach(j => cosList.appendChild(matchJobRow(j)));
  if (!d.cosine || !d.cosine.length) cosList.innerHTML = '<div class="empty">none</div>';
  const fltList = box.querySelector('#flt-list');
  const fl = (d.filtered && d.filtered.length) ? d.filtered : (d.cosine || []).slice(0, 1);
  fl.forEach(j => fltList.appendChild(matchJobRow(j)));
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
    renderMatch(d);
  } catch (e) { status.textContent = 'failed: ' + e; }
}
document.getElementById('own-go').addEventListener('click', matchOwn);
</script>
</body></html>
"""


@app.get("/", response_class=HTMLResponse)
def index():
    title = "Jobs Search Demo: 348K postings across 4 corpora (Solr backend)"
    subtitle = (
        "347,900 postings (jobs_data + LinkedIn + JobStreet + USAJobs) · "
        "RRF(BM25 + e5-small) via Solr 10 · "
        "click a result for the full description · "
        "or paste your profile above to find jobs for yourself (3-axis constraint filter)"
    )
    return HTML_PAGE.replace("__PAGE_TITLE__", title).replace("__PAGE_SUBTITLE__", subtitle)


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


def _parse_filters(request_qp: dict) -> dict[str, str]:
    return {
        f: (request_qp.get(f) or "").strip()
        for f in FACET_FIELDS
        if (request_qp.get(f) or "").strip()
    }


@app.get("/api/search")
def api_search(request: Request, q: str = Query(...), k: int = Query(10)):
    qp = dict(request.query_params)
    filters = _parse_filters(qp)
    t0 = time.time()
    res = search_default(q, k, filters)
    ms = int((time.time() - t0) * 1000)
    return JSONResponse(
        {
            "query": q,
            "retriever": "rrf_bm25_e5",
            "served_with": SERVING_MODE,
            "filters": filters,
            "results": res,
            "ms": ms,
        }
    )


@app.get("/api/facets")
def api_facets(request: Request, q: str = Query(...), pool: int = Query(200)):
    """Facet counts over the top-`pool` fused results (with the same filters
    that the search uses). Aggregating over the fused set rather than the BM25
    match set keeps facet counts coherent with what the user sees."""
    qp = dict(request.query_params)
    filters = _parse_filters(qp)
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
    "posted_at,source_corpus,industry,employment_type,"
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
    return JSONResponse(_run_profile_match(r, qv))


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("SHIM_PORT", os.environ.get("PORT", 7860)))
    uvicorn.run(app, host="0.0.0.0", port=port)
