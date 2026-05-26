#!/usr/bin/env python3
"""HF Space: jobs search demo (FastAPI + server-side autocomplete).

This is a port of the FastAPI demo at demo_jobs.py --unified, adapted to pull
artifacts from a companion HF dataset at startup. We deliberately avoid Gradio
on this Space: the rich HTML/JS UI (live autocomplete, source-corpus tags,
click-to-expand descriptions) is what the local demo already does well; Gradio
adds Svelte reactivity overhead at the 30k+ autocomplete-choice scale.

HF Spaces with sdk: gradio just runs `python app.py`; it does NOT require
the app to be a Gradio app. We bind uvicorn ourselves.
"""

import bisect
import json
import os
import re
import time
from collections import defaultdict
from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse, JSONResponse
from huggingface_hub import snapshot_download

DATASET_REPO = "dtunkelang/jobs-demo"
BGE_MODEL = "BAAI/bge-small-en-v1.5"

RRF_POOL = 100
RRF_K = 60
CASCADE_POOL = 100
WSUM_ALPHA = 0.5

RETRIEVERS: dict[str, str] = {
    "bm25": "BM25 (bm25s, English stemmer)",
    "bge_small": "bge-small-en-v1.5 (local)",
    "te3_cached": "te3-large @ 1024 (cached queries only)",
    "rrf_bm25_bge": "RRF(BM25, bge-small)",
    "rrf_bm25_bge_te3": "RRF(BM25, bge-small, te3-large) — te3 used when cached, skipped otherwise",
    "rrf_bm25_te3": "RRF(BM25, te3-large) — falls back to BM25 if uncached",
    "cascade_bm25_bge": "Cascade: BM25 top-100 → bge-small rerank",
    "cascade_bm25_te3": "Cascade: BM25 top-100 → te3 rerank — cached only",
    "wsum_bm25_bge": "Weighted: 0.5·BM25_norm + 0.5·bge-small",
}

SRC_SHORT = {
    "jobs_data": "OAP",
    "jobs_data_linkedin": "LI",
    "jobs_data_jobstreet": "JS",
    "jobs_data_usajobs": "USA",
}

_DIGIT_RUN = re.compile(r"\d{3,}")
_SLUG_ISH = re.compile(r"\b[a-z]+\d+\b")
_BAD_CHARS = re.compile(r"[<>{}@]")
_DOUBLE_SPACE = re.compile(r"\s{2,}")
_WS_RUN = re.compile(r"[ \t]+")
_NL_RUN = re.compile(r"\n{3,}")


def _clean_text(s: str) -> str:
    """Decode literal &nbsp;/&amp;/etc. in source text and collapse whitespace."""
    if not s:
        return ""
    import html as _html

    s = _html.unescape(s)
    s = s.replace("\xa0", " ")
    s = _WS_RUN.sub(" ", s)
    s = _NL_RUN.sub("\n\n", s)
    return s.strip()


# Abbrev → expansion map. Lookup is on the *leading token* of the user's query.
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


def _is_clean(q: str) -> bool:
    if not q or len(q) < 2 or len(q.split()) > 7:
        return False
    if _DIGIT_RUN.search(q) or _SLUG_ISH.search(q) or _BAD_CHARS.search(q):
        return False
    return not _DOUBLE_SPACE.search(q)


R: dict = {}


def download_data() -> str:
    local = os.environ.get("LOCAL_DATA_DIR")
    if local:
        print(f"using LOCAL_DATA_DIR={local}", flush=True)
        return local
    print(f"snapshot_download from {DATASET_REPO}...", flush=True)
    return snapshot_download(
        repo_id=DATASET_REPO,
        repo_type="dataset",
        allow_patterns=[
            "titles.json",
            "source_index.json",
            "metadata.jsonl",
            "bge_catalog.vecs.fp16.npy",
            "te3_catalog.vecs.fp16.npy",
            "te3_queries.vecs.fp16.npy",
            "te3_queries.ids.json",
            "te3_queries.sources.json",
            "te3_cache_canonical.json",
            "bm25_index/*",
        ],
    )


def load_resources(data_dir: str) -> None:
    print(f"loading data from {data_dir}...", flush=True)
    t0 = time.time()
    with open(os.path.join(data_dir, "titles.json")) as f:
        titles = json.load(f)
    print(f"  titles: {len(titles):,} in {time.time() - t0:.1f}s", flush=True)

    with open(os.path.join(data_dir, "source_index.json")) as f:
        src_idx = json.load(f)
    sources = src_idx["sources"]
    if len(sources) != len(titles):
        raise SystemExit("source_index/titles length mismatch")

    t0 = time.time()
    slim_meta: list[dict] = []
    meta_offsets: list[int] = []
    meta_path = os.path.join(data_dir, "metadata.jsonl")
    with open(meta_path, "rb") as f:
        while True:
            offset = f.tell()
            line = f.readline()
            if not line:
                break
            meta_offsets.append(offset)
            rec = json.loads(line)
            slim_meta.append(
                {
                    "title": rec.get("title") or "",
                    "employer": rec.get("source_slug") or "",
                    "locations": rec.get("locations") or [],
                    "employment_type": rec.get("employment_type") or "",
                    "salary_min": rec.get("salary_min"),
                    "salary_max": rec.get("salary_max"),
                    "salary_currency": rec.get("salary_currency") or "",
                    "department": rec.get("department") or "",
                    "posted_at": rec.get("posted_at") or "",
                }
            )
    print(f"  slim_meta + offsets: {len(slim_meta):,} in {time.time() - t0:.1f}s", flush=True)

    catalogs = {}
    for key, fname in [
        ("bge_small", "bge_catalog.vecs.fp16.npy"),
        ("te3_cached", "te3_catalog.vecs.fp16.npy"),
    ]:
        catalogs[key] = np.load(os.path.join(data_dir, fname), mmap_mode="r")
        print(f"  {key}: {catalogs[key].shape} {catalogs[key].dtype}", flush=True)

    print(f"loading {BGE_MODEL} (CPU)...", flush=True)
    import torch
    from sentence_transformers import SentenceTransformer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    bge_model = SentenceTransformer(BGE_MODEL, device=device)

    t0 = time.time()
    print("loading te3 query cache...", flush=True)
    qvecs = np.load(os.path.join(data_dir, "te3_queries.vecs.fp16.npy"))
    with open(os.path.join(data_dir, "te3_queries.ids.json")) as f:
        qids = json.load(f)
    with open(os.path.join(data_dir, "te3_queries.sources.json")) as f:
        qsrc = json.load(f)
    canonical_path = os.path.join(data_dir, "te3_cache_canonical.json")
    if os.path.exists(canonical_path):
        with open(canonical_path) as f:
            canonical = json.load(f)
    else:
        canonical = {}

    qindex: dict[str, int] = {}
    qkey_src: dict[str, str] = {}
    # Priority for which tag wins on duplicate keys (lower = higher priority).
    TAG_PRIORITY = {"title": 0, "combo": 1, "head": 2, "tail": 3, "synth": 4}
    for i, q in enumerate(qids):
        k = q.strip().lower()
        tag = qsrc[i]
        cur = qkey_src.get(k)
        if cur is None or TAG_PRIORITY.get(tag, 9) < TAG_PRIORITY.get(cur, 9):
            qindex[k] = i
            qkey_src[k] = tag
    print(
        f"  te3 query cache: {qvecs.shape[0]:,} rows, {len(qindex):,} unique keys "
        f"({len(canonical):,} canonical aliases) in {time.time() - t0:.1f}s",
        flush=True,
    )

    # Tiered clean keys for server-side autocomplete (sorted lists for bisect).
    by_tag: dict[str, list[str]] = defaultdict(list)
    for k in qindex:
        if not _is_clean(k) or k in canonical:
            continue
        by_tag[qkey_src[k]].append(k)
    for v in by_tag.values():
        v.sort()
    sorted_keys = sorted(qindex.keys())
    print(
        "  autocomplete tiers: "
        + ", ".join(
            f"{t}={len(by_tag.get(t, [])):,}" for t in ("title", "combo", "head", "tail", "synth")
        ),
        flush=True,
    )

    import bm25s
    from Stemmer import Stemmer

    stemmer = Stemmer("english")
    bm25_dir = os.path.join(data_dir, "bm25_index")
    if os.path.isdir(bm25_dir) and os.path.exists(os.path.join(bm25_dir, "vocab.index.json")):
        t0 = time.time()
        print(f"loading pre-built bm25s index from {bm25_dir}...", flush=True)
        bm25_idx = bm25s.BM25.load(bm25_dir, mmap=True)
        print(f"  bm25s loaded in {time.time() - t0:.1f}s", flush=True)
    else:
        print("building bm25s index (no pre-built copy found)...", flush=True)
        t0 = time.time()
        title_tok = bm25s.tokenize(titles, stopwords="en", stemmer=stemmer, show_progress=False)
        bm25_idx = bm25s.BM25(k1=0.9, b=0.4)
        bm25_idx.index(title_tok, show_progress=False)
        print(f"  bm25s indexed in {time.time() - t0:.1f}s", flush=True)

    R.update(
        {
            "titles": titles,
            "sources": sources,
            "slim_meta": slim_meta,
            "meta_offsets": meta_offsets,
            "meta_path": meta_path,
            "catalogs": catalogs,
            "bge_model": bge_model,
            "qvecs": qvecs,
            "qindex": qindex,
            "qkey_src": qkey_src,
            "canonical": canonical,
            "sorted_keys": sorted_keys,
            "tier_keys": dict(by_tag),
            "bm25_idx": bm25_idx,
            "bm25_stemmer": stemmer,
        }
    )
    print("ready.", flush=True)


# ===== search functions =====


def _fmt_salary(meta: dict) -> str:
    lo, hi, cur = meta.get("salary_min"), meta.get("salary_max"), meta.get("salary_currency") or ""
    if lo is None and hi is None:
        return ""
    if lo is not None and hi is not None:
        return f"{cur} {int(lo):,}-{int(hi):,}".strip()
    if hi is not None:
        return f"{cur} up to {int(hi):,}".strip()
    return f"{cur} from {int(lo):,}".strip()


def _make_result(rank: int, score: float, idx: int) -> dict:
    i = int(idx)
    meta = R["slim_meta"][i]
    title_raw = R["titles"][i]
    title = (meta.get("title") or title_raw.split("\n", 1)[0]).strip()
    if len(title) > 140:
        title = title[:137] + "..."
    locs = meta.get("locations") or []
    return {
        "rank": rank,
        "score": float(score),
        "title": title,
        "idx": i,
        "source": R["sources"][i],
        "employer": meta.get("employer") or "",
        "location": ", ".join(locs[:2]) if locs else "",
        "employment_type": meta.get("employment_type") or "",
        "salary": _fmt_salary(meta),
        "department": meta.get("department") or "",
        "posted": (meta.get("posted_at") or "")[:7],
    }


def _te3_qv(query: str):
    k = query.strip().lower()
    if k in R["qindex"]:
        return R["qvecs"][R["qindex"][k]]
    return None


def _is_cached(q: str) -> bool:
    return q.strip().lower() in R["qindex"]


def _topk_bm25(query: str, k: int):
    import bm25s

    qtok = bm25s.tokenize([query], stopwords="en", stemmer=R["bm25_stemmer"], show_progress=False)
    idx, scores = R["bm25_idx"].retrieve(qtok, k=k, show_progress=False)
    return [(int(i), float(s)) for i, s in zip(idx[0], scores[0])]


def _bge_qv(query: str):
    qv = R["bge_model"].encode([query], normalize_embeddings=True, show_progress_bar=False)[0]
    return qv.astype(np.float32)


def _topk_dense(query: str, model_key: str, k: int):
    qv = _bge_qv(query) if model_key == "bge_small" else _te3_qv(query)
    if qv is None:
        return None
    catalog = R["catalogs"][model_key]
    sims = catalog.astype(np.float32) @ qv.astype(np.float32)
    top = np.argpartition(-sims, k)[:k]
    top = top[np.argsort(-sims[top])]
    return [(int(i), float(sims[i])) for i in top]


def _dense_cos(qv, doc_indices, model_key: str):
    sub = R["catalogs"][model_key][doc_indices].astype(np.float32)
    return sub @ qv.astype(np.float32)


def search_bm25(query: str, k: int):
    return [_make_result(r + 1, s, i) for r, (i, s) in enumerate(_topk_bm25(query, k))]


def search_dense(query: str, model_key: str, k: int):
    hits = _topk_dense(query, model_key, k)
    if hits is None:
        return [
            {
                "rank": 0,
                "score": 0.0,
                "title": "(query not in te3 cache — try BM25 or bge-small)",
                "idx": -1,
                "source": "",
                "employer": "",
                "location": "",
                "employment_type": "",
                "salary": "",
                "department": "",
                "posted": "",
            }
        ]
    return [_make_result(r + 1, s, i) for r, (i, s) in enumerate(hits)]


def search_rrf(query: str, parts: list[str], k: int):
    contrib: dict[int, float] = defaultdict(float)
    for r in parts:
        if r == "bm25":
            hits = _topk_bm25(query, RRF_POOL)
            topk = [i for i, _ in hits]
        else:
            hits = _topk_dense(query, r, RRF_POOL)
            if hits is None:
                continue
            topk = [i for i, _ in hits]
        for rank, idx in enumerate(topk, 1):
            contrib[idx] += 1.0 / (RRF_K + rank)
    items = sorted(contrib.items(), key=lambda x: -x[1])[:k]
    return [_make_result(r + 1, s, i) for r, (i, s) in enumerate(items)]


def search_cascade(query: str, dense_key: str, k: int):
    pool = _topk_bm25(query, CASCADE_POOL)
    if not pool:
        return []
    doc_indices = [i for i, _ in pool]
    if dense_key == "te3_cached":
        qv = _te3_qv(query)
        if qv is None:
            return [
                {
                    "rank": 0,
                    "score": 0.0,
                    "title": "(query not in te3 cache — try cascade_bm25_bge)",
                    "idx": -1,
                    "source": "",
                    "employer": "",
                    "location": "",
                    "employment_type": "",
                    "salary": "",
                    "department": "",
                    "posted": "",
                }
            ]
    else:
        qv = _bge_qv(query)
    cos = _dense_cos(qv, doc_indices, dense_key)
    order = np.argsort(-cos)
    return [_make_result(r + 1, float(cos[j]), doc_indices[j]) for r, j in enumerate(order[:k])]


def search_wsum(query: str, dense_key: str, k: int, alpha: float = WSUM_ALPHA):
    bm25_pool = _topk_bm25(query, RRF_POOL)
    dense_pool = _topk_dense(query, dense_key, RRF_POOL)
    if dense_pool is None:
        return [
            {
                "rank": 0,
                "score": 0.0,
                "title": "(query not in te3 cache — try wsum_bm25_bge)",
                "idx": -1,
                "source": "",
                "employer": "",
                "location": "",
                "employment_type": "",
                "salary": "",
                "department": "",
                "posted": "",
            }
        ]
    bm25_score = {i: s for i, s in bm25_pool}
    candidates = sorted(set(bm25_score) | {i for i, _ in dense_pool})
    qv = _bge_qv(query) if dense_key == "bge_small" else _te3_qv(query)
    cos = _dense_cos(qv, candidates, dense_key)
    bm = np.array([bm25_score.get(i, 0.0) for i in candidates], dtype=np.float32)

    def _mm(v):
        lo, hi = float(v.min()), float(v.max())
        return (v - lo) / max(hi - lo, 1e-9)

    fused = alpha * _mm(bm) + (1 - alpha) * _mm(cos)
    order = np.argsort(-fused)
    return [_make_result(r + 1, float(fused[j]), candidates[j]) for r, j in enumerate(order[:k])]


def search_one(query: str, retriever: str, k: int = 10):
    if retriever == "bm25":
        return search_bm25(query, k)
    if retriever in ("bge_small", "te3_cached"):
        return search_dense(query, retriever, k)
    if retriever == "rrf_bm25_bge":
        return search_rrf(query, ["bm25", "bge_small"], k)
    if retriever == "rrf_bm25_te3":
        return search_rrf(query, ["bm25", "te3_cached"], k)
    if retriever == "rrf_bm25_bge_te3":
        return search_rrf(query, ["bm25", "bge_small", "te3_cached"], k)
    if retriever == "cascade_bm25_bge":
        return search_cascade(query, "bge_small", k)
    if retriever == "cascade_bm25_te3":
        return search_cascade(query, "te3_cached", k)
    if retriever == "wsum_bm25_bge":
        return search_wsum(query, "bge_small", k)
    return [
        {
            "rank": 0,
            "score": 0.0,
            "title": f"(unknown retriever {retriever})",
            "idx": -1,
            "source": "",
            "employer": "",
            "location": "",
            "employment_type": "",
            "salary": "",
            "department": "",
            "posted": "",
        }
    ]


def _serving_mode(query: str, retriever: str) -> str:
    hit = _is_cached(query)
    if retriever == "bm25":
        return "BM25 (lexical)"
    if retriever == "bge_small":
        return "bge-small-en-v1.5 (local dense)"
    if retriever == "te3_cached":
        return "te3-large @ 1024 (cache hit)" if hit else "te3 cache miss"
    if retriever == "rrf_bm25_bge":
        return "RRF: BM25 + bge-small"
    if retriever == "rrf_bm25_bge_te3":
        return (
            "RRF: BM25 + bge-small + te3-large (cache hit)"
            if hit
            else "RRF: BM25 + bge-small (te3 cache miss, silently skipped)"
        )
    if retriever == "rrf_bm25_te3":
        return (
            "RRF: BM25 + te3-large (cache hit)"
            if hit
            else "BM25 only (te3 cache miss, silently skipped)"
        )
    if retriever == "cascade_bm25_bge":
        return "Cascade: BM25 → bge-small rerank"
    if retriever == "cascade_bm25_te3":
        return "Cascade: BM25 → te3 rerank (cache hit)" if hit else "te3 cache miss"
    if retriever == "wsum_bm25_bge":
        return "Weighted sum: 0.5·BM25 + 0.5·bge-small"
    return retriever


# ===== autocomplete helpers =====


def _prefix_matches(keys: list[str], prefix: str, limit: int) -> list[str]:
    lo = bisect.bisect_left(keys, prefix)
    out = []
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
    data_dir = download_data()
    load_resources(data_dir)
    yield


app = FastAPI(title="Jobs Search Demo", lifespan=lifespan)


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
.controls { display: flex; gap: 8px; align-items: center; margin-bottom: 14px; }
.controls label { color: #666; font-size: 0.9em; }
select { padding: 5px 8px; font-size: 0.95em; flex: 1; border: 1px solid #ccc; border-radius: 4px; }
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
<div class="controls">
  <label for="retriever">Retriever:</label>
  <select id="retriever">__OPTIONS__</select>
</div>
<div id="badge-row"></div>
<div class="results-panel">
  <div id="results"><div class="empty">type a query (~196k queries autocompleted from the te3 cache)</div></div>
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
  suggestItems = items;
  if (!items.length) { closeSuggest(); return; }
  suggestBox.innerHTML = items.map((s, i) =>
    `<div class="item" data-i="${i}">${esc(s)}<span class="hint">cached · te3</span></div>`
  ).join('');
  suggestBox.style.display = 'block';
  suggestActive = -1;
  suggestBox.querySelectorAll('.item').forEach(el => {
    el.addEventListener('mousedown', e => {
      e.preventDefault();
      input.value = items[parseInt(el.dataset.i)];
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
      input.value = suggestItems[suggestActive];
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
document.getElementById('retriever').addEventListener('change', runSearch);
const SRC_SHORT = {
  'jobs_data': 'OAP', 'jobs_data_linkedin': 'LI', 'jobs_data_jobstreet': 'JS', 'jobs_data_usajobs': 'USA'
};
function shortSrc(s) { return s == null ? '' : (SRC_SHORT[s] || s); }
function metaLine(r) {
  const parts = [];
  if (r.employer) parts.push(esc(r.employer));
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
    row.innerHTML = `<span class="r-rank">${r.rank}</span><span class="r-score">${r.score.toFixed(3)}</span><span class="r-source">${esc(shortSrc(r.source))}</span><span class="r-title"><div class="t">${esc(r.title)}</div>${metaLine(r)}${metaLine2(r)}</span>`;
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
async function runSearch() {
  const q = input.value.trim();
  if (!q) return;
  closeSuggest();
  const retriever = document.getElementById('retriever').value;
  const div = document.getElementById('results');
  const badgeRow = document.getElementById('badge-row');
  badgeRow.innerHTML = '';
  div.innerHTML = '<div class="empty">searching...</div>';
  const r = await fetch(`/api/search?q=${encodeURIComponent(q)}&retriever=${retriever}`);
  const data = await r.json();
  if (data.served_with) {
    const cls = data.cached ? 'cached' : 'uncached';
    badgeRow.innerHTML = `<span class="badge ${cls}">Served with: ${esc(data.served_with)}</span>`;
  }
  renderResults(div, data.results, data.ms);
}
</script>
</body></html>
"""


def render_options(default_key: str) -> str:
    return "".join(
        f'<option value="{k}"{" selected" if k == default_key else ""}>{label}</option>'
        for k, label in RETRIEVERS.items()
    )


@app.get("/", response_class=HTMLResponse)
def index():
    title = "Jobs Search Demo: 348K postings across 4 corpora"
    subtitle = (
        "347,900 postings (jobs_data + LinkedIn + JobStreet + USAJobs) · "
        "click a result to see the full description · tag shows source corpus"
    )
    default = "rrf_bm25_bge_te3"
    return (
        HTML_PAGE.replace("__PAGE_TITLE__", title)
        .replace("__PAGE_SUBTITLE__", subtitle)
        .replace("__OPTIONS__", render_options(default))
    )


@app.get("/api/suggest")
def api_suggest(q: str = Query(""), limit: int = Query(10)):
    if not q or not R:
        return JSONResponse({"suggestions": []})
    prefix = q.strip().lower()
    prefixes = _expand_prefix(prefix)
    # Tier order: title → combo → head → tail → synth → any-key fallback.
    tiers = [
        R["tier_keys"].get("title", []),
        R["tier_keys"].get("combo", []),
        R["tier_keys"].get("head", []),
        R["tier_keys"].get("tail", []),
        R["tier_keys"].get("synth", []),
        R["sorted_keys"],
    ]
    seen: set[str] = set()
    suggestions: list[str] = []
    for tier in tiers:
        for p in prefixes:
            for k in _prefix_matches(tier, p, limit * 2):
                if k not in seen:
                    seen.add(k)
                    suggestions.append(k)
                    if len(suggestions) >= limit:
                        break
            if len(suggestions) >= limit:
                break
        if len(suggestions) >= limit:
            break
    return JSONResponse({"suggestions": suggestions})


@app.get("/api/search")
def api_search(q: str = Query(...), retriever: str = Query("rrf_bm25_bge_te3"), k: int = Query(10)):
    t0 = time.time()
    res = search_one(q, retriever, k)
    ms = int((time.time() - t0) * 1000)
    return JSONResponse(
        {
            "query": q,
            "retriever": retriever,
            "served_with": _serving_mode(q, retriever),
            "cached": _is_cached(q),
            "results": res,
            "ms": ms,
        }
    )


@app.get("/api/detail")
def api_detail(idx: int = Query(...)):
    offsets = R["meta_offsets"]
    if idx < 0 or idx >= len(offsets):
        return JSONResponse({"error": "idx out of range"}, status_code=404)
    with open(R["meta_path"], "rb") as f:
        f.seek(offsets[idx])
        line = f.readline()
    rec = json.loads(line)
    return JSONResponse(
        {
            "idx": idx,
            "title": _clean_text(rec.get("title") or ""),
            "description": _clean_text(rec.get("description") or ""),
            "posted_at": rec.get("posted_at") or "",
            "department": rec.get("department") or "",
            "id": rec.get("id") or "",
        }
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 7860)))
