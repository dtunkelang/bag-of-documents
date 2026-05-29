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
from fastapi import FastAPI, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse

# ===== configuration =====

SOLR = os.environ.get("SOLR", "http://localhost:8983")
CORE = "jobs"
STAGE = "/Users/dtunkelang/bagofdocs/space_demo_jobs/_stage"  # te3_queries.* live here
UNIFIED = "/Users/dtunkelang/bagofdocs/unified_jobs"  # te3_cache_canonical.json
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


def load_resources() -> None:
    t0 = time.time()
    print(f"loading {DENSE_MODEL}...", flush=True)
    import torch
    from sentence_transformers import SentenceTransformer

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    dense_model = SentenceTransformer(DENSE_MODEL, device=device)
    print(f"  dense loaded on {device} in {time.time() - t0:.1f}s", flush=True)

    t0 = time.time()
    ids_path = os.path.join(STAGE, "te3_queries.ids.json")
    if os.path.exists(ids_path):
        print("loading suggestion corpus...", flush=True)
        with open(ids_path) as f:
            qids = json.load(f)
        with open(os.path.join(STAGE, "te3_queries.sources.json")) as f:
            qsrc = json.load(f)
        canonical_path = os.path.join(UNIFIED, "te3_cache_canonical.json")
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
    else:
        # Suggestion corpus absent locally — fall back to Solr titleSuggester only.
        print(f"  suggestion corpus absent ({ids_path}); Solr titles only.", flush=True)
        by_tag = defaultdict(list)
        sorted_keys: list[str] = []

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
</style></head>
<body>
<h1>__PAGE_TITLE__</h1>
<div class="subtle">__PAGE_SUBTITLE__</div>
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
</script>
</body></html>
"""


@app.get("/", response_class=HTMLResponse)
def index():
    title = "Jobs Search Demo: 348K postings across 4 corpora (Solr backend)"
    subtitle = (
        "347,900 postings (jobs_data + LinkedIn + JobStreet + USAJobs) · "
        "RRF(BM25 + e5-small) via Solr 10 · "
        "click a result for the full description"
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


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=7864)
