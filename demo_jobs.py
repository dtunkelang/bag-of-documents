#!/usr/bin/env python3
"""Side-by-side jobs retrieval demo: pick a retriever per column.

Retrievers:
  - bm25            : bm25s over titles.json
  - base_minilm     : sentence-transformers/all-MiniLM-L6-v2
  - bod_jobs        : BoD-fine-tuned MiniLM (query_model_jobs_bod/)
  - rrf_*, cascade_*, wsum_* : BM25 + dense hybrids

All retrieval runs locally; no query-time API calls.

Usage:
    python demo_jobs.py                 # http://localhost:7861
    python demo_jobs.py --port 8080
"""

import argparse
import json
import os
import time
from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse, JSONResponse

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "jobs_data")

RETRIEVERS = {
    "bm25": "BM25 (bm25s, English stemmer)",
    "base_minilm": "Base MiniLM (all-MiniLM-L6-v2)",
    "me5_small": "multilingual-e5-small (intfloat)",
    "bod_jobs": "BoD-jobs (MiniLM fine-tuned on 10K bags)",
    "rrf_bm25_base": "RRF(BM25, base MiniLM)",
    "rrf_bm25_bod": "RRF(BM25, BoD-jobs)",
    "rrf_bm25_me5": "RRF(BM25, me5-small)",
    "cascade_bm25_base": "Cascade: BM25 top-100 → base MiniLM rerank",
    "cascade_bm25_bod": "Cascade: BM25 top-100 → BoD-jobs rerank",
    "cascade_bm25_me5": "Cascade: BM25 top-100 → me5-small rerank",
    "wsum_bm25_base": "Weighted: 0.5·BM25_norm + 0.5·base_minilm",
    "wsum_bm25_bod": "Weighted: 0.5·BM25_norm + 0.5·BoD-jobs",
    "wsum_bm25_me5": "Weighted: 0.5·BM25_norm + 0.5·me5-small",
}

RRF_POOL = 100  # top-K per retriever to fuse
RRF_K = 60  # RRF dampening constant (standard)
CASCADE_POOL = 100  # BM25 candidates passed to the dense reranker
WSUM_ALPHA = 0.5  # weight on BM25 in weighted-sum fusion

# Per-model query prefix (catalog vecs were encoded with the matching doc prefix).
MODEL_QUERY_PREFIX = {
    "me5_small": "query: ",
}

RESOURCES: dict = {}


def load_resources():
    print("loading data...", flush=True)
    with open(os.path.join(DATA_DIR, "titles.json")) as f:
        titles = json.load(f)
    print(f"  titles: {len(titles):,}", flush=True)

    cats = {}
    for key, fname in [
        ("base_minilm", "base_catalog.vecs.fp16.npy"),
        ("bod_jobs", "jobs_bod_catalog.vecs.fp16.npy"),
        ("me5_small", "me5_small_catalog.vecs.fp16.npy"),
    ]:
        path = os.path.join(DATA_DIR, fname)
        if os.path.exists(path):
            cats[key] = np.load(path, mmap_mode="r")
            print(f"  {key}: {cats[key].shape} {cats[key].dtype}", flush=True)
        else:
            print(f"  {key}: MISSING ({path})", flush=True)

    from sentence_transformers import SentenceTransformer

    device = "mps"
    print(f"loading sentence-transformers models on {device}...", flush=True)
    st_models = {
        "base_minilm": SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=device),
        "bod_jobs": SentenceTransformer(
            os.path.join(SCRIPT_DIR, "query_model_jobs_bod"), device=device
        ),
        "me5_small": SentenceTransformer("intfloat/multilingual-e5-small", device=device),
    }

    print("building bm25s index over titles (one-shot)...", flush=True)
    import bm25s
    from Stemmer import Stemmer

    stemmer = Stemmer("english")
    t0 = time.time()
    title_tok = bm25s.tokenize(titles, stopwords="en", stemmer=stemmer, show_progress=False)
    bm25_idx = bm25s.BM25(k1=0.9, b=0.4)
    bm25_idx.index(title_tok, show_progress=False)
    print(f"  bm25s indexed in {time.time() - t0:.1f}s", flush=True)

    RESOURCES.update(
        {
            "titles": titles,
            "catalogs": cats,
            "st_models": st_models,
            "bm25_idx": bm25_idx,
            "bm25_stemmer": stemmer,
        }
    )
    print("ready.", flush=True)


def search_st(query: str, model_key: str, k: int = 10):
    model = RESOURCES["st_models"][model_key]
    catalog = RESOURCES["catalogs"][model_key]
    prefix = MODEL_QUERY_PREFIX.get(model_key, "")
    qv = model.encode([prefix + query], normalize_embeddings=True, show_progress_bar=False)[0]
    sims = catalog.astype(np.float32) @ qv.astype(np.float32)
    top = np.argpartition(-sims, k)[:k]
    top = top[np.argsort(-sims[top])]
    return [
        {"rank": i + 1, "score": float(sims[idx]), "title": RESOURCES["titles"][int(idx)]}
        for i, idx in enumerate(top)
    ]


def search_bm25(query: str, k: int = 10):
    import bm25s

    qtok = bm25s.tokenize(
        [query], stopwords="en", stemmer=RESOURCES["bm25_stemmer"], show_progress=False
    )
    idx, scores = RESOURCES["bm25_idx"].retrieve(qtok, k=k, show_progress=False)
    out = []
    for i, (di, s) in enumerate(zip(idx[0], scores[0])):
        out.append({"rank": i + 1, "score": float(s), "title": RESOURCES["titles"][int(di)]})
    return out


def _topk_indices_st(query: str, model_key: str, k: int) -> list[int]:
    model = RESOURCES["st_models"][model_key]
    catalog = RESOURCES["catalogs"][model_key]
    prefix = MODEL_QUERY_PREFIX.get(model_key, "")
    qv = model.encode([prefix + query], normalize_embeddings=True, show_progress_bar=False)[0]
    sims = catalog.astype(np.float32) @ qv.astype(np.float32)
    top = np.argpartition(-sims, k)[:k]
    return list(top[np.argsort(-sims[top])])


def _topk_indices_bm25(query: str, k: int) -> list[int]:
    import bm25s

    qtok = bm25s.tokenize(
        [query], stopwords="en", stemmer=RESOURCES["bm25_stemmer"], show_progress=False
    )
    idx, _ = RESOURCES["bm25_idx"].retrieve(qtok, k=k, show_progress=False)
    return [int(i) for i in idx[0]]


def _bm25_idx_score(query: str, k: int):
    """BM25 top-k with both indices and scores."""
    import bm25s

    qtok = bm25s.tokenize(
        [query], stopwords="en", stemmer=RESOURCES["bm25_stemmer"], show_progress=False
    )
    idx, scores = RESOURCES["bm25_idx"].retrieve(qtok, k=k, show_progress=False)
    return [(int(i), float(s)) for i, s in zip(idx[0], scores[0])]


def _dense_cos(query: str, model_key: str, doc_indices: list[int]) -> np.ndarray:
    """Cosine of dense query vec against a fixed candidate index set."""
    model = RESOURCES["st_models"][model_key]
    catalog = RESOURCES["catalogs"][model_key]
    prefix = MODEL_QUERY_PREFIX.get(model_key, "")
    qv = model.encode([prefix + query], normalize_embeddings=True, show_progress_bar=False)[
        0
    ].astype(np.float32)
    sub = catalog[doc_indices].astype(np.float32)
    return sub @ qv


def search_cascade(query: str, dense_key: str, k: int = 10):
    """BM25 top-CASCADE_POOL → reorder by dense cosine on the pool."""
    pool = _bm25_idx_score(query, CASCADE_POOL)
    if not pool:
        return []
    doc_indices = [i for i, _ in pool]
    cos = _dense_cos(query, dense_key, doc_indices)
    order = np.argsort(-cos)
    out = []
    for rank, j in enumerate(order[:k], start=1):
        out.append(
            {
                "rank": rank,
                "score": float(cos[j]),
                "title": RESOURCES["titles"][int(doc_indices[j])],
            }
        )
    return out


def search_wsum(query: str, dense_key: str, k: int = 10, alpha: float = WSUM_ALPHA):
    """α·BM25_norm + (1-α)·cos over union of BM25 + dense pools (min-max within pool)."""
    bm25_pool = _bm25_idx_score(query, RRF_POOL)
    dense_top = _topk_indices_st(query, dense_key, RRF_POOL)
    bm25_score = {i: s for i, s in bm25_pool}
    candidate_idx = sorted(set(bm25_score) | set(int(i) for i in dense_top))
    if not candidate_idx:
        return []
    cos = _dense_cos(query, dense_key, candidate_idx)
    bm = np.array([bm25_score.get(int(i), 0.0) for i in candidate_idx], dtype=np.float32)

    # min-max normalize within the candidate set
    def _mm(v):
        lo, hi = float(v.min()), float(v.max())
        return (v - lo) / max(hi - lo, 1e-9)

    fused = alpha * _mm(bm) + (1 - alpha) * _mm(cos)
    order = np.argsort(-fused)
    out = []
    for rank, j in enumerate(order[:k], start=1):
        out.append(
            {
                "rank": rank,
                "score": float(fused[j]),
                "title": RESOURCES["titles"][int(candidate_idx[j])],
            }
        )
    return out


def search_rrf(query: str, parts: list[str], k: int = 10):
    """Reciprocal Rank Fusion across the given retriever keys."""
    contrib: dict[int, float] = {}
    for r in parts:
        try:
            if r == "bm25":
                topk = _topk_indices_bm25(query, RRF_POOL)
            elif r in ("base_minilm", "bod_jobs"):
                topk = _topk_indices_st(query, r, RRF_POOL)
            else:
                continue
        except Exception:
            continue
        for rank, idx in enumerate(topk, start=1):
            contrib[int(idx)] = contrib.get(int(idx), 0.0) + 1.0 / (RRF_K + rank)
    items = sorted(contrib.items(), key=lambda x: -x[1])[:k]
    return [
        {"rank": i + 1, "score": float(s), "title": RESOURCES["titles"][int(idx)]}
        for i, (idx, s) in enumerate(items)
    ]


def search_one(query: str, retriever: str, k: int = 10):
    if retriever == "bm25":
        return search_bm25(query, k)
    if retriever in ("base_minilm", "bod_jobs", "me5_small"):
        return search_st(query, retriever, k)
    if retriever == "rrf_bm25_base":
        return search_rrf(query, ["bm25", "base_minilm"], k)
    if retriever == "rrf_bm25_bod":
        return search_rrf(query, ["bm25", "bod_jobs"], k)
    if retriever == "rrf_bm25_me5":
        return search_rrf(query, ["bm25", "me5_small"], k)
    if retriever == "cascade_bm25_base":
        return search_cascade(query, "base_minilm", k)
    if retriever == "cascade_bm25_bod":
        return search_cascade(query, "bod_jobs", k)
    if retriever == "cascade_bm25_me5":
        return search_cascade(query, "me5_small", k)
    if retriever == "wsum_bm25_base":
        return search_wsum(query, "base_minilm", k)
    if retriever == "wsum_bm25_bod":
        return search_wsum(query, "bod_jobs", k)
    if retriever == "wsum_bm25_me5":
        return search_wsum(query, "me5_small", k)
    return [{"rank": 0, "score": 0.0, "title": f"(unknown retriever {retriever})"}]


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_resources()
    yield


app = FastAPI(title="BoD-Jobs Demo", lifespan=lifespan)


HTML_PAGE = """<!doctype html>
<html><head><meta charset="utf-8"><title>BoD-Jobs Demo</title>
<style>
body { font-family: -apple-system, system-ui, sans-serif; max-width: 1500px; margin: 30px auto; padding: 0 16px; color: #222; }
h1 { font-size: 1.4em; margin-bottom: 8px; }
.subtle { color: #777; font-size: 0.9em; margin-bottom: 18px; }
.search { display: flex; gap: 8px; margin-bottom: 14px; }
#query { flex: 1; padding: 8px 12px; font-size: 1.05em; border: 1px solid #ccc; border-radius: 4px; }
button { padding: 8px 18px; font-size: 1em; cursor: pointer; border: 1px solid #888; border-radius: 4px; background: #fafafa; }
.columns { display: flex; gap: 16px; }
.col { flex: 1; border: 1px solid #ddd; border-radius: 6px; padding: 10px 12px; background: #fff; min-width: 0; }
.col-head { display: flex; gap: 8px; align-items: center; padding-bottom: 8px; border-bottom: 1px solid #eee; }
select { padding: 4px 6px; font-size: 0.92em; flex: 1; }
.result { display: grid; grid-template-columns: 28px 60px 1fr; gap: 8px; padding: 5px 0; border-bottom: 1px dotted #eee; font-size: 0.92em; }
.r-rank { color: #aaa; text-align: right; }
.r-score { color: #555; font-variant-numeric: tabular-nums; }
.r-title { color: #222; word-break: break-word; max-height: 4.5em; overflow: hidden; }
.empty { color: #999; padding: 30px; text-align: center; }
.timing { font-size: 0.8em; color: #888; padding-top: 6px; }
</style></head>
<body>
<h1>BoD-Jobs: side-by-side retrieval demo</h1>
<div class="subtle">100K job postings · 12 retrievers (4 single + 8 hybrid) · all local, no API calls · no qrels marking yet</div>
<div class="search">
  <input id="query" placeholder="e.g. senior software engineer remote python" />
  <button onclick="runSearch()">Search</button>
</div>
<div class="columns">
  <div class="col">
    <div class="col-head">
      <select id="left-retriever">__OPTIONS_LEFT__</select>
    </div>
    <div id="left-results"><div class="empty">type a query</div></div>
  </div>
  <div class="col">
    <div class="col-head">
      <select id="right-retriever">__OPTIONS_RIGHT__</select>
    </div>
    <div id="right-results"><div class="empty">type a query</div></div>
  </div>
</div>
<script>
const input = document.getElementById('query');
input.addEventListener('keydown', e => { if (e.key === 'Enter') runSearch(); });
function esc(s) { return s.replace(/[&<>"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c])); }
function renderResults(div, items, ms) {
  if (!items || !items.length) { div.innerHTML = '<div class="empty">no results</div>'; return; }
  let html = items.map(r =>
    `<div class="result"><span class="r-rank">${r.rank}</span><span class="r-score">${r.score.toFixed(3)}</span><span class="r-title">${esc(r.title)}</span></div>`
  ).join('');
  if (ms != null) html += `<div class="timing">${ms} ms</div>`;
  div.innerHTML = html;
}
async function runSearch() {
  const q = input.value.trim();
  if (!q) return;
  const left = document.getElementById('left-retriever').value;
  const right = document.getElementById('right-retriever').value;
  const ldiv = document.getElementById('left-results');
  const rdiv = document.getElementById('right-results');
  ldiv.innerHTML = '<div class="empty">searching...</div>';
  rdiv.innerHTML = '<div class="empty">searching...</div>';
  const r = await fetch(`/api/search?q=${encodeURIComponent(q)}&left=${left}&right=${right}`);
  const data = await r.json();
  renderResults(ldiv, data.left.results, data.left.ms);
  renderResults(rdiv, data.right.results, data.right.ms);
}
</script>
</body></html>
"""


def render_options(default_key: str) -> str:
    out = []
    for k, label in RETRIEVERS.items():
        sel = " selected" if k == default_key else ""
        out.append(f'<option value="{k}"{sel}>{label}</option>')
    return "".join(out)


@app.get("/", response_class=HTMLResponse)
def index():
    return HTML_PAGE.replace("__OPTIONS_LEFT__", render_options("base_minilm")).replace(
        "__OPTIONS_RIGHT__", render_options("cascade_bm25_bod")
    )


@app.get("/api/search")
def api_search(
    q: str = Query(...),
    left: str = Query("base_minilm"),
    right: str = Query("bod_jobs"),
    k: int = Query(10),
):
    t0 = time.time()
    left_res = search_one(q, left, k)
    left_ms = int((time.time() - t0) * 1000)
    t1 = time.time()
    right_res = search_one(q, right, k)
    right_ms = int((time.time() - t1) * 1000)
    return JSONResponse(
        {
            "query": q,
            "left": {"retriever": left, "results": left_res, "ms": left_ms},
            "right": {"retriever": right, "results": right_res, "ms": right_ms},
        }
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=7861)
    args = ap.parse_args()
    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port)
