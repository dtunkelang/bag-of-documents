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
import bisect
import html
import json
import os
import re
import time
from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse, JSONResponse

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "jobs_data")
UNIFIED = False  # set by --unified flag

RETRIEVERS_SINGLE = {
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

RETRIEVERS_UNIFIED = {
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

RETRIEVERS = RETRIEVERS_SINGLE  # rebound after parsing args

RRF_POOL = 100  # top-K per retriever to fuse
RRF_K = 60  # RRF dampening constant (standard)
CASCADE_POOL = 100  # BM25 candidates passed to the dense reranker
WSUM_ALPHA = 0.5  # weight on BM25 in weighted-sum fusion

# Per-model query prefix (catalog vecs were encoded with the matching doc prefix).
MODEL_QUERY_PREFIX = {
    "me5_small": "query: ",
}

RESOURCES: dict = {}


_DIGIT_RUN = re.compile(r"\d{3,}")
_SLUG_ISH = re.compile(r"\b[a-z]+\d+\b")
_BAD_CHARS = re.compile(r"[<>{}@]")
_DOUBLE_SPACE = re.compile(r"\s{2,}")

# Abbrev → expansion map. Lookup is on the *leading token* of the user's query.
# (Single-direction: typing the short form expands; typing the long form does not contract.)
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
    """Return True if a cached query is clean enough to suggest in autocomplete."""
    if not q or len(q) < 2:
        return False
    if len(q.split()) > 7:
        return False
    if _DIGIT_RUN.search(q):
        return False
    if _SLUG_ISH.search(q):
        return False
    if _BAD_CHARS.search(q):
        return False
    return not _DOUBLE_SPACE.search(q)


def _load_te3_query_cache():
    """Concat train+eval te3 query vecs across 4 corpora + augmentations into one (query → row) cache.

    Each unique key is tagged with `source` (aug | synth) and `clean` (quality flag) for autocomplete filtering.
    """
    sources: list[tuple[str, str, str]] = []
    for d in ["jobs_data", "jobs_data_linkedin", "jobs_data_jobstreet", "jobs_data_usajobs"]:
        for split in ["train", "eval"]:
            sources.append(
                (
                    os.path.join(SCRIPT_DIR, d, f"{split}_queries_te3_1024.vecs.fp16.npy"),
                    os.path.join(SCRIPT_DIR, d, f"{split}_queries_te3_1024.ids.json"),
                    "synth",
                )
            )
    for stem in ("aug_titles", "aug_combos", "head_torso", "head_torso2"):
        sources.append(
            (
                os.path.join(SCRIPT_DIR, "unified_jobs", f"{stem}_te3_1024.vecs.fp16.npy"),
                os.path.join(SCRIPT_DIR, "unified_jobs", f"{stem}_te3_1024.ids.json"),
                "aug",
            )
        )

    parts = []
    queries: list[str] = []
    src_per_row: list[str] = []
    for vec_path, id_path, tag in sources:
        if not (os.path.exists(vec_path) and os.path.exists(id_path)):
            print(f"  te3 cache MISSING: {os.path.basename(vec_path)}", flush=True)
            continue
        v = np.load(vec_path)
        with open(id_path) as f:
            ids = json.load(f)
        if v.shape[0] != len(ids):
            raise SystemExit(f"size mismatch in {vec_path}")
        parts.append(v)
        queries.extend(ids)
        src_per_row.extend([tag] * len(ids))
    vecs = np.concatenate(parts, axis=0) if parts else np.zeros((0, 1024), dtype=np.float16)

    # Build index; prefer 'aug' tag on duplicates (aug overwrites synth)
    index: dict[str, int] = {}
    key_source: dict[str, str] = {}
    for i, q in enumerate(queries):
        k = q.strip().lower()
        tag = src_per_row[i]
        if k not in index or (tag == "aug" and key_source.get(k) == "synth"):
            index[k] = i
            key_source[k] = tag

    # Load canonical map if present (non-canonical key → canonical key); hide non-canon from autocomplete.
    canonical_path = os.path.join(SCRIPT_DIR, "unified_jobs", "te3_cache_canonical.json")
    canonical: dict[str, str] = {}
    if os.path.exists(canonical_path):
        with open(canonical_path) as f:
            canonical = json.load(f)
        print(f"  loaded canonical map: {len(canonical):,} non-canonical aliases", flush=True)

    sorted_keys = sorted(index.keys())

    def _canonical_visible(k: str) -> bool:
        return k not in canonical

    clean_aug = sorted(
        k for k in index if key_source[k] == "aug" and _is_clean(k) and _canonical_visible(k)
    )
    clean_synth = sorted(
        k for k in index if key_source[k] == "synth" and _is_clean(k) and _canonical_visible(k)
    )
    print(
        f"  te3 query cache: {vecs.shape[0]:,} rows, {len(index):,} unique keys "
        f"(aug clean: {len(clean_aug):,}, synth clean: {len(clean_synth):,})",
        flush=True,
    )
    return {
        "vecs": vecs,
        "index": index,
        "sorted_keys": sorted_keys,
        "key_source": key_source,
        "canonical": canonical,
        "sorted_clean_aug": clean_aug,
        "sorted_clean_synth": clean_synth,
    }


def load_resources():
    print(f"loading data from {DATA_DIR} (unified={UNIFIED})...", flush=True)
    with open(os.path.join(DATA_DIR, "titles.json")) as f:
        titles = json.load(f)
    print(f"  titles: {len(titles):,}", flush=True)

    sources: list[str] | None = None
    slim_meta: list[dict] | None = None
    meta_offsets: list[int] | None = None
    if UNIFIED:
        with open(os.path.join(DATA_DIR, "source_index.json")) as f:
            src_idx = json.load(f)
        sources = src_idx["sources"]
        if len(sources) != len(titles):
            raise SystemExit(f"source_index length {len(sources)} != titles length {len(titles)}")

        print("  loading slim metadata + byte offsets...", flush=True)
        slim_meta = []
        meta_offsets = []
        t0 = time.time()
        meta_path = os.path.join(DATA_DIR, "metadata.jsonl")
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
                        "remote": rec.get("remote"),
                        "salary_min": rec.get("salary_min"),
                        "salary_max": rec.get("salary_max"),
                        "salary_currency": rec.get("salary_currency") or "",
                        "department": rec.get("department") or "",
                        "posted_at": rec.get("posted_at") or "",
                    }
                )
        print(
            f"    slim_meta: {len(slim_meta):,} rows + offsets in {time.time() - t0:.1f}s",
            flush=True,
        )
        if len(slim_meta) != len(titles):
            raise SystemExit(f"slim_meta length {len(slim_meta)} != titles length {len(titles)}")

    cats = {}
    if UNIFIED:
        catalog_files = [
            ("bge_small", "bge_catalog.vecs.fp16.npy"),
            ("te3_cached", "te3_catalog.vecs.fp16.npy"),
        ]
    else:
        catalog_files = [
            ("base_minilm", "base_catalog.vecs.fp16.npy"),
            ("bod_jobs", "jobs_bod_catalog.vecs.fp16.npy"),
            ("me5_small", "me5_small_catalog.vecs.fp16.npy"),
        ]
    for key, fname in catalog_files:
        path = os.path.join(DATA_DIR, fname)
        if os.path.exists(path):
            cats[key] = np.load(path, mmap_mode="r")
            print(f"  {key}: {cats[key].shape} {cats[key].dtype}", flush=True)
        else:
            print(f"  {key}: MISSING ({path})", flush=True)

    from sentence_transformers import SentenceTransformer

    device = "mps"
    print(f"loading sentence-transformers models on {device}...", flush=True)
    if UNIFIED:
        st_models = {
            "bge_small": SentenceTransformer("BAAI/bge-small-en-v1.5", device=device),
        }
    else:
        st_models = {
            "base_minilm": SentenceTransformer(
                "sentence-transformers/all-MiniLM-L6-v2", device=device
            ),
            "bod_jobs": SentenceTransformer(
                os.path.join(SCRIPT_DIR, "query_model_jobs_bod"), device=device
            ),
            "me5_small": SentenceTransformer("intfloat/multilingual-e5-small", device=device),
        }

    te3_query_cache = _load_te3_query_cache() if UNIFIED else None

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
            "sources": sources,
            "slim_meta": slim_meta,
            "meta_offsets": meta_offsets if UNIFIED else None,
            "meta_path": os.path.join(DATA_DIR, "metadata.jsonl") if UNIFIED else None,
            "catalogs": cats,
            "st_models": st_models,
            "te3_query_cache": te3_query_cache,
            "bm25_idx": bm25_idx,
            "bm25_stemmer": stemmer,
        }
    )
    print("ready.", flush=True)


def _fmt_salary(meta: dict) -> str:
    lo, hi, cur = meta.get("salary_min"), meta.get("salary_max"), meta.get("salary_currency") or ""
    if lo is None and hi is None:
        return ""
    if lo is not None and hi is not None:
        return f"{cur} {int(lo):,}-{int(hi):,}".strip()
    if hi is not None:
        return f"{cur} up to {int(hi):,}".strip()
    return f"{cur} from {int(lo):,}".strip()


def _fmt_posted(iso: str) -> str:
    if not iso:
        return ""
    # Compact "YYYY-MM" for tag display
    return iso[:7] if len(iso) >= 7 else iso


def _make_result(rank: int, score: float, idx: int) -> dict:
    i = int(idx)
    meta = RESOURCES.get("slim_meta")
    title_raw = RESOURCES["titles"][i]
    # Some corpora concat title + "\n\n" + description into titles.json — prefer the clean title from slim_meta.
    if meta is not None and meta[i].get("title"):
        title = meta[i]["title"]
    else:
        title = title_raw.split("\n", 1)[0]
    title = title.strip()
    if len(title) > 140:
        title = title[:137] + "..."
    out = {"rank": rank, "score": float(score), "title": title, "idx": i}
    src = RESOURCES.get("sources")
    if src is not None:
        out["source"] = src[i]
    if meta is not None:
        m = meta[i]
        locs = m.get("locations") or []
        out["employer"] = m.get("employer") or ""
        out["location"] = ", ".join(locs[:2]) if locs else ""
        out["employment_type"] = m.get("employment_type") or ""
        out["salary"] = _fmt_salary(m)
        out["department"] = m.get("department") or ""
        out["posted"] = _fmt_posted(m.get("posted_at") or "")
    return out


def _te3_cached_qv(query: str):
    cache = RESOURCES.get("te3_query_cache")
    if cache is None:
        return None
    return (
        cache["vecs"][cache["index"][query.strip().lower()]]
        if query.strip().lower() in cache["index"]
        else None
    )


def _topk_indices_te3_cached(query: str, k: int) -> list[int] | None:
    qv = _te3_cached_qv(query)
    if qv is None:
        return None
    catalog = RESOURCES["catalogs"]["te3_cached"]
    sims = catalog.astype(np.float32) @ qv.astype(np.float32)
    top = np.argpartition(-sims, k)[:k]
    return list(top[np.argsort(-sims[top])])


def search_te3_cached(query: str, k: int = 10):
    qv = _te3_cached_qv(query)
    if qv is None:
        return [
            {
                "rank": 0,
                "score": 0.0,
                "title": "(query not in te3 cache — try BM25 or bge-small)",
                "source": "",
            }
        ]
    catalog = RESOURCES["catalogs"]["te3_cached"]
    sims = catalog.astype(np.float32) @ qv.astype(np.float32)
    top = np.argpartition(-sims, k)[:k]
    top = top[np.argsort(-sims[top])]
    return [_make_result(i + 1, sims[idx], idx) for i, idx in enumerate(top)]


def search_st(query: str, model_key: str, k: int = 10):
    model = RESOURCES["st_models"][model_key]
    catalog = RESOURCES["catalogs"][model_key]
    prefix = MODEL_QUERY_PREFIX.get(model_key, "")
    qv = model.encode([prefix + query], normalize_embeddings=True, show_progress_bar=False)[0]
    sims = catalog.astype(np.float32) @ qv.astype(np.float32)
    top = np.argpartition(-sims, k)[:k]
    top = top[np.argsort(-sims[top])]
    return [_make_result(i + 1, sims[idx], idx) for i, idx in enumerate(top)]


def search_bm25(query: str, k: int = 10):
    import bm25s

    qtok = bm25s.tokenize(
        [query], stopwords="en", stemmer=RESOURCES["bm25_stemmer"], show_progress=False
    )
    idx, scores = RESOURCES["bm25_idx"].retrieve(qtok, k=k, show_progress=False)
    return [_make_result(i + 1, s, di) for i, (di, s) in enumerate(zip(idx[0], scores[0]))]


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


def _dense_cos_cached(doc_indices: list[int], qv: np.ndarray) -> np.ndarray:
    catalog = RESOURCES["catalogs"]["te3_cached"]
    sub = catalog[doc_indices].astype(np.float32)
    return sub @ qv.astype(np.float32)


def search_cascade(query: str, dense_key: str, k: int = 10):
    """BM25 top-CASCADE_POOL → reorder by dense cosine on the pool."""
    pool = _bm25_idx_score(query, CASCADE_POOL)
    if not pool:
        return []
    doc_indices = [i for i, _ in pool]
    if dense_key == "te3_cached":
        qv = _te3_cached_qv(query)
        if qv is None:
            return [
                {
                    "rank": 0,
                    "score": 0.0,
                    "title": "(query not in te3 cache — try cascade_bm25_bge)",
                    "source": "",
                }
            ]
        cos = _dense_cos_cached(doc_indices, qv)
    else:
        cos = _dense_cos(query, dense_key, doc_indices)
    order = np.argsort(-cos)
    return [_make_result(rank, cos[j], doc_indices[j]) for rank, j in enumerate(order[:k], start=1)]


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
    return [
        _make_result(rank, fused[j], candidate_idx[j]) for rank, j in enumerate(order[:k], start=1)
    ]


_RRF_ST_KEYS = {"base_minilm", "bod_jobs", "me5_small", "bge_small"}


def search_rrf(query: str, parts: list[str], k: int = 10):
    """Reciprocal Rank Fusion across the given retriever keys."""
    contrib: dict[int, float] = {}
    for r in parts:
        try:
            if r == "bm25":
                topk = _topk_indices_bm25(query, RRF_POOL)
            elif r == "te3_cached":
                topk = _topk_indices_te3_cached(query, RRF_POOL)
                if topk is None:
                    # Cache miss: silently skip te3's contribution so the fusion still serves.
                    continue
            elif r in _RRF_ST_KEYS:
                topk = _topk_indices_st(query, r, RRF_POOL)
            else:
                continue
        except Exception:
            continue
        for rank, idx in enumerate(topk, start=1):
            contrib[int(idx)] = contrib.get(int(idx), 0.0) + 1.0 / (RRF_K + rank)
    items = sorted(contrib.items(), key=lambda x: -x[1])[:k]
    return [_make_result(i + 1, s, idx) for i, (idx, s) in enumerate(items)]


def search_one(query: str, retriever: str, k: int = 10):
    if retriever == "bm25":
        return search_bm25(query, k)
    if retriever == "te3_cached":
        return search_te3_cached(query, k)
    if retriever in ("base_minilm", "bod_jobs", "me5_small", "bge_small"):
        return search_st(query, retriever, k)
    if retriever == "rrf_bm25_base":
        return search_rrf(query, ["bm25", "base_minilm"], k)
    if retriever == "rrf_bm25_bod":
        return search_rrf(query, ["bm25", "bod_jobs"], k)
    if retriever == "rrf_bm25_me5":
        return search_rrf(query, ["bm25", "me5_small"], k)
    if retriever == "rrf_bm25_bge":
        return search_rrf(query, ["bm25", "bge_small"], k)
    if retriever == "rrf_bm25_bge_te3":
        return search_rrf(query, ["bm25", "bge_small", "te3_cached"], k)
    if retriever == "rrf_bm25_te3":
        return search_rrf(query, ["bm25", "te3_cached"], k)
    if retriever == "cascade_bm25_base":
        return search_cascade(query, "base_minilm", k)
    if retriever == "cascade_bm25_bod":
        return search_cascade(query, "bod_jobs", k)
    if retriever == "cascade_bm25_me5":
        return search_cascade(query, "me5_small", k)
    if retriever == "cascade_bm25_bge":
        return search_cascade(query, "bge_small", k)
    if retriever == "cascade_bm25_te3":
        return search_cascade(query, "te3_cached", k)
    if retriever == "wsum_bm25_base":
        return search_wsum(query, "base_minilm", k)
    if retriever == "wsum_bm25_bod":
        return search_wsum(query, "bod_jobs", k)
    if retriever == "wsum_bm25_me5":
        return search_wsum(query, "me5_small", k)
    if retriever == "wsum_bm25_bge":
        return search_wsum(query, "bge_small", k)
    return [{"rank": 0, "score": 0.0, "title": f"(unknown retriever {retriever})"}]


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_resources()
    yield


app = FastAPI(title="BoD-Jobs Demo", lifespan=lifespan)


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
  <div id="results"><div class="empty">type a query (~39k queries autocompleted from the te3 cache)</div></div>
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
    if (r.idx != null) {
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
    out = []
    for k, label in RETRIEVERS.items():
        sel = " selected" if k == default_key else ""
        out.append(f'<option value="{k}"{sel}>{label}</option>')
    return "".join(out)


@app.get("/", response_class=HTMLResponse)
def index():
    if UNIFIED:
        title = "Unified Jobs Search: 348K postings across 4 corpora"
        subtitle = (
            "347,900 postings (jobs_data + LinkedIn + JobStreet + USAJobs) · "
            "click a result to see the full description · tag shows source corpus"
        )
        default = "rrf_bm25_bge_te3"
    else:
        title = "BoD-Jobs Demo"
        subtitle = (
            "100K job postings · 12 retrievers · all local, no API calls · no qrels marking yet"
        )
        default = "rrf_bm25_bod"
    return (
        HTML_PAGE.replace("__PAGE_TITLE__", title)
        .replace("__PAGE_SUBTITLE__", subtitle)
        .replace("__OPTIONS__", render_options(default))
    )


def _is_cached(q: str) -> bool:
    cache = RESOURCES.get("te3_query_cache")
    return bool(cache and q.strip().lower() in cache["index"])


def _serving_mode(query: str, retriever: str) -> str:
    """Human-readable badge: which models actually contribute to this query's results."""
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
    """If the user's first whole token (or full prefix) is a known abbrev, expand it."""
    parts = prefix.split(" ", 1)
    head = parts[0]
    rest = (" " + parts[1]) if len(parts) > 1 else ""
    out = [prefix]
    if head in ABBREV_EXPANSIONS:
        for exp in ABBREV_EXPANSIONS[head]:
            out.append(exp + rest)
    return out


@app.get("/api/suggest")
def api_suggest(q: str = Query(""), limit: int = Query(10)):
    cache = RESOURCES.get("te3_query_cache")
    if not cache or not q:
        return JSONResponse({"suggestions": []})
    prefix = q.strip().lower()
    prefixes = _expand_prefix(prefix)
    # Tier 1: clean aug; tier 2: clean synth; tier 3: anything left. Across all expanded prefixes.
    seen: set[str] = set()
    suggestions: list[str] = []
    for tier in (cache["sorted_clean_aug"], cache["sorted_clean_synth"], cache["sorted_keys"]):
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
def api_search(
    q: str = Query(...),
    retriever: str = Query(None),
    left: str = Query(None),
    right: str = Query(None),
    k: int = Query(10),
):
    # Single-retriever mode (current UI)
    if retriever is not None:
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
    # Legacy side-by-side params (kept for backward-compat callers)
    left_key = left or "base_minilm"
    right_key = right or "bod_jobs"
    t0 = time.time()
    left_res = search_one(q, left_key, k)
    left_ms = int((time.time() - t0) * 1000)
    t1 = time.time()
    right_res = search_one(q, right_key, k)
    right_ms = int((time.time() - t1) * 1000)
    return JSONResponse(
        {
            "query": q,
            "left": {"retriever": left_key, "results": left_res, "ms": left_ms},
            "right": {"retriever": right_key, "results": right_res, "ms": right_ms},
        }
    )


_WS_RUN = re.compile(r"[ \t]+")
_NL_RUN = re.compile(r"\n{3,}")


def _clean_text(s: str) -> str:
    """Decode literal &nbsp;/&amp;/etc. in source text and collapse whitespace."""
    if not s:
        return ""
    s = html.unescape(s)
    s = s.replace("\xa0", " ")
    s = _WS_RUN.sub(" ", s)
    s = _NL_RUN.sub("\n\n", s)
    return s.strip()


@app.get("/api/detail")
def api_detail(idx: int = Query(...)):
    offsets = RESOURCES.get("meta_offsets")
    meta_path = RESOURCES.get("meta_path")
    if offsets is None or meta_path is None:
        return JSONResponse({"error": "detail unavailable (non-unified mode)"}, status_code=400)
    if idx < 0 or idx >= len(offsets):
        return JSONResponse({"error": "idx out of range"}, status_code=404)
    with open(meta_path, "rb") as f:
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
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=None)
    ap.add_argument(
        "--unified",
        action="store_true",
        help="Use unified_jobs/ (348k docs, BM25+bge+te3-cached retrievers)",
    )
    args = ap.parse_args()
    if args.unified:
        UNIFIED = True
        DATA_DIR = os.path.join(SCRIPT_DIR, "unified_jobs")
        RETRIEVERS = RETRIEVERS_UNIFIED
    port = args.port if args.port is not None else (7862 if args.unified else 7861)
    import uvicorn

    uvicorn.run(app, host=args.host, port=port)
