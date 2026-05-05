#!/usr/bin/env python3
"""Gradio demo for bag-of-documents product search (HuggingFace Space version).

Three views, side-by-side with base MiniLM:
  - "Fine-tuned retrieval": the original BoD-as-retriever query model
    (single-pass FAISS).
  - "Base + ensemble rerank": base MiniLM retrieves top-100, two BoD
    models (6M MNRL, qrels-hardneg) independently rank candidates, and
    sumrank fusion produces the final top-K. Uses precomputed product
    embeddings from the dataset for sub-100ms latency.

Resources are downloaded from the dataset repo at first startup and
cached on the Space's persistent storage thereafter.
"""

import json
import os
import re
import time

import bm25s
import faiss
import gradio as gr
import numpy as np
import Stemmer
import tantivy
from huggingface_hub import snapshot_download
from sentence_transformers import SentenceTransformer

DATASET_REPO = "dtunkelang/bag-of-documents"
BASE_MODEL_NAME = "all-MiniLM-L6-v2"


def download_data():
    """Pull required files from the dataset repo."""
    print("Downloading data from HuggingFace dataset...")
    local_dir = snapshot_download(
        repo_id=DATASET_REPO,
        repo_type="dataset",
        allow_patterns=[
            "combined_index/index.faiss",
            "combined_index/rerank_A.index.faiss",
            "combined_index/titles.json",
            "combined_index/rerank_A.vecs.fp16.npy",
            "combined_index/rerank_B.vecs.fp16.npy",
            "combined_index/rerank_G.vecs.fp16.npy",
            "combined_index/tantivy_index/*",
            "combined_index/bm25s_index/*",
            "combined_index/spell_vocab.json",
            "bags.jsonl",
            "query_model/*",
            "query_model_6m_mnrl/*",
            "query_model_hardneg/*",
            "query_model_esci_supervised/*",
        ],
    )
    print(f"  Downloaded to {local_dir}")
    return local_dir


def load_resources(data_dir):
    print("Loading resources...")

    index = faiss.read_index(os.path.join(data_dir, "combined_index", "index.faiss"))
    print(f"  Base MiniLM product index: {index.ntotal:,} products")

    mnrl_index_path = os.path.join(data_dir, "combined_index", "rerank_A.index.faiss")
    mnrl_index = None
    if os.path.exists(mnrl_index_path):
        mnrl_index = faiss.read_index(mnrl_index_path)
        print(f"  6M-MNRL product index: {mnrl_index.ntotal:,} products")
    else:
        print("  6M-MNRL product index: missing — MNRL retrieval modes will be unavailable")

    with open(os.path.join(data_dir, "combined_index", "titles.json")) as f:
        titles = json.load(f)

    base_model = SentenceTransformer(BASE_MODEL_NAME)
    print(f"  Base model: {BASE_MODEL_NAME}")

    retrieval_model = SentenceTransformer(os.path.join(data_dir, "query_model"))
    print("  Retrieval model: query_model")

    rerank_a = SentenceTransformer(os.path.join(data_dir, "query_model_6m_mnrl"))
    rerank_b = SentenceTransformer(os.path.join(data_dir, "query_model_hardneg"))
    rerank_g = None
    rerank_g_path = os.path.join(data_dir, "query_model_esci_supervised")
    if os.path.isdir(rerank_g_path) and os.listdir(rerank_g_path):
        rerank_g = SentenceTransformer(rerank_g_path)
        print("  Rerankers: 6M MNRL + qrels-hardneg + ESCI-supervised")
    else:
        print("  Rerankers: 6M MNRL + qrels-hardneg (3-way mode unavailable)")

    rerank_a_vecs = np.load(os.path.join(data_dir, "combined_index", "rerank_A.vecs.fp16.npy"))
    rerank_b_vecs = np.load(os.path.join(data_dir, "combined_index", "rerank_B.vecs.fp16.npy"))
    rerank_g_vecs = None
    rerank_g_vecs_path = os.path.join(data_dir, "combined_index", "rerank_G.vecs.fp16.npy")
    if os.path.exists(rerank_g_vecs_path):
        rerank_g_vecs = np.load(rerank_g_vecs_path)
    print(f"  Cached vecs: A={rerank_a_vecs.shape}, B={rerank_b_vecs.shape}", end="")
    if rerank_g_vecs is not None:
        print(f", G={rerank_g_vecs.shape}", end="")
    print()

    bag_queries, bag_centroids, bag_specs = [], [], []
    with open(os.path.join(data_dir, "bags.jsonl")) as f:
        for line in f:
            bag = json.loads(line)
            if bag["num_results"] < 2:
                continue
            bag_queries.append(bag["query"])
            bag_centroids.append(bag["query_vector"])
            bag_specs.append(bag["specificity"])
    bag_matrix = np.array(bag_centroids, dtype=np.float32)
    bag_matrix /= np.maximum(np.linalg.norm(bag_matrix, axis=1, keepdims=True), 1e-8)
    bag_specs = np.array(bag_specs)
    print(f"  Bag centroids: {len(bag_queries)} bags")

    # bm25s BM25 index (k1=0.3, b=0.6, optimized — see eval_bm25_sweep.py).
    # Preferred over tantivy when present.
    bm25s_path = os.path.join(data_dir, "combined_index", "bm25s_index")
    bm25s_idx = None
    bm25s_stemmer = None
    if os.path.isdir(bm25s_path) and os.listdir(bm25s_path):
        bm25s_idx = bm25s.BM25.load(bm25s_path, mmap=False)
        bm25s_stemmer = Stemmer.Stemmer("english")
        print("  bm25s BM25 index loaded (k1=0.3, b=0.6)")

    # Catalog-vocab spell corrector. Pre-BM25 query rewriting catches typos
    # like 'tredmills' -> 'treadmills', 'inpjone' -> 'iphone', without breaking
    # in-vocab brand or model tokens. +0.23pp R@10 / +0.42pp E@1 (significant)
    # on the ESCI test set; +4.25pp R@10 on the 5.4% of queries that get
    # corrected.
    spell = None
    spell_vocab_set = None
    spell_vocab_path = os.path.join(data_dir, "combined_index", "spell_vocab.json")
    if os.path.exists(spell_vocab_path):
        try:
            from spellchecker import SpellChecker

            spell = SpellChecker(language=None, distance=2)
            # Use load_dictionary on the JSON directly. The `.add(word, freq)`
            # loop is O(n^2) (each call rebuilds the internal letter set over
            # the whole dictionary) and OOMs on high-frequency tokens like
            # 'for' (freq 453K) — the loop creates a [word]*freq list per call.
            spell.word_frequency.load_dictionary(spell_vocab_path)
            spell_vocab_set = set(spell.word_frequency.dictionary.keys())
            print(f"  spell corrector loaded ({len(spell_vocab_set):,} catalog tokens)")
        except Exception as e:
            print(f"  spell corrector unavailable: {e}")

    # Tantivy BM25 index — legacy fallback when bm25s isn't available.
    tantivy_path = os.path.join(data_dir, "combined_index", "tantivy_index")
    tantivy_searcher = None
    tantivy_idx = None
    if os.path.isdir(tantivy_path) and os.listdir(tantivy_path):
        schema_builder = tantivy.SchemaBuilder()
        schema_builder.add_text_field("title", stored=True, tokenizer_name="en_stem")
        schema = schema_builder.build()
        tv_index = tantivy.Index(schema, path=tantivy_path)
        tv_index.reload()
        tantivy_searcher = tv_index.searcher()
        tantivy_idx = tv_index
        print(f"  Tantivy BM25 index (fallback): {tantivy_searcher.num_docs:,} docs")
    elif bm25s_idx is None:
        print("  No BM25 index loaded — BM25 + hybrid modes will be unavailable")

    # Title -> first FAISS position map for hybrid (BM25 returns titles; RRF
    # fusion needs positions parallel to the dense index).
    title_to_pos = {}
    for i, t in enumerate(titles):
        if t not in title_to_pos:
            title_to_pos[t] = i

    # Quality-tier cross-encoders (CC5 = sumsim + LiYuan + BGE 3-way mean).
    # On CPU-basic Space hardware these dominate latency (~5-15s/query
    # combined for 100 candidates); CC5 is the opt-in quality mode
    # alongside the fast CC3-50.
    ce_model = None
    bge_model = None
    try:
        from sentence_transformers import CrossEncoder

        ce_model = CrossEncoder("LiYuan/Amazon-Cup-Cross-Encoder-Regression")
        print("  LiYuan ESCI cross-encoder loaded")
    except Exception as e:
        print(f"  LiYuan unavailable: {e}")
    try:
        from sentence_transformers import CrossEncoder

        bge_model = CrossEncoder("BAAI/bge-reranker-v2-m3")
        print("  BGE-reranker-v2-m3 loaded")
    except Exception as e:
        print(f"  BGE-reranker unavailable: {e}")

    return {
        "index": index,
        "mnrl_index": mnrl_index,
        "titles": titles,
        "base_model": base_model,
        "retrieval_model": retrieval_model,
        "rerank_a": rerank_a,
        "rerank_b": rerank_b,
        "rerank_g": rerank_g,
        "rerank_a_vecs": rerank_a_vecs,
        "rerank_b_vecs": rerank_b_vecs,
        "rerank_g_vecs": rerank_g_vecs,
        "ce_model": ce_model,
        "bge_model": bge_model,
        "bag_queries": bag_queries,
        "bag_matrix": bag_matrix,
        "bag_specs": bag_specs,
        "tantivy_searcher": tantivy_searcher,
        "tantivy_index": tantivy_idx,
        "bm25s_index": bm25s_idx,
        "bm25s_stemmer": bm25s_stemmer,
        "spell": spell,
        "spell_vocab_set": spell_vocab_set,
        "title_to_pos": title_to_pos,
    }


def _faiss_dist_to_sim(d, metric):
    """FAISS distance → cosine similarity for normalized vectors.

    METRIC_INNER_PRODUCT returns inner product == cosine directly.
    METRIC_L2 returns L2 squared, so cosine = 1 - L2²/2.
    """
    if metric == faiss.METRIC_INNER_PRODUCT:
        return float(d)
    return float(1 - d / 2)


def _dedup_by_title(items, k):
    """Keep first occurrence of each title; assumes items is descending by score."""
    out, seen = [], set()
    for title, score in items:
        if title in seen:
            continue
        seen.add(title)
        out.append((title, score))
        if len(out) >= k:
            break
    return out


def base_top_k(query, R, k, oversample=3):
    vec = R["base_model"].encode(query, normalize_embeddings=True)
    vec = np.asarray(vec, dtype=np.float32).reshape(1, -1)
    faiss.normalize_L2(vec)
    try:
        R["index"].hnsw.efSearch = 128
    except Exception:
        pass
    metric = R["index"].metric_type
    # Oversample so dedup still leaves k unique results when titles repeat.
    D, I = R["index"].search(vec, k * oversample)
    raw = []
    for d, i in zip(D[0], I[0]):
        if i < 0:
            continue
        raw.append((R["titles"][i], round(_faiss_dist_to_sim(d, metric), 4)))
    raw.sort(key=lambda x: -x[1])
    return _dedup_by_title(raw, k), D, I, metric


def retrieval_top_k(query, R, k, oversample=3):
    vec = R["retrieval_model"].encode(query, normalize_embeddings=True)
    vec = np.asarray(vec, dtype=np.float32).reshape(1, -1)
    faiss.normalize_L2(vec)
    try:
        R["index"].hnsw.efSearch = 128
    except Exception:
        pass
    metric = R["index"].metric_type
    D, I = R["index"].search(vec, k * oversample)
    raw = []
    for d, i in zip(D[0], I[0]):
        if i < 0:
            continue
        raw.append((R["titles"][i], round(_faiss_dist_to_sim(d, metric), 4)))
    raw.sort(key=lambda x: -x[1])
    return _dedup_by_title(raw, k)


def mnrl_top_k(query, R, k, oversample=3):
    """6M-MNRL retrieval (no rerank) — encode with rerank_a, search MNRL FAISS."""
    if R.get("mnrl_index") is None:
        # Fall back to base if MNRL index not available.
        return base_top_k(query, R, k, oversample)[0]
    vec = R["rerank_a"].encode(query, normalize_embeddings=True)
    vec = np.asarray(vec, dtype=np.float32).reshape(1, -1)
    faiss.normalize_L2(vec)
    try:
        R["mnrl_index"].hnsw.efSearch = 128
    except Exception:
        pass
    metric = R["mnrl_index"].metric_type
    D, I = R["mnrl_index"].search(vec, k * oversample)
    raw = []
    for d, i in zip(D[0], I[0]):
        if i < 0:
            continue
        raw.append((R["titles"][i], round(_faiss_dist_to_sim(d, metric), 4)))
    raw.sort(key=lambda x: -x[1])
    return _dedup_by_title(raw, k)


def _ensemble_rerank_from_positions(query, R, valid_positions, k_top):
    """Sumsim ensemble rerank over the given candidate index positions."""
    if not valid_positions:
        return []
    qa = np.asarray(R["rerank_a"].encode(query, normalize_embeddings=True), dtype=np.float32)
    qb = np.asarray(R["rerank_b"].encode(query, normalize_embeddings=True), dtype=np.float32)
    cv_a = R["rerank_a_vecs"][valid_positions].astype(np.float32)
    cv_b = R["rerank_b_vecs"][valid_positions].astype(np.float32)
    avg_sim = (cv_a @ qa + cv_b @ qb) / 2
    order = np.argsort(-avg_sim)
    raw = [(R["titles"][valid_positions[int(idx)]], round(float(avg_sim[idx]), 4)) for idx in order]
    return _dedup_by_title(raw, k_top)


def ensemble_rerank_top_k(query, R, k_top, k_retrieve=100):
    """Base FAISS top-K_retrieve → sumsim ensemble rerank → top-K_top.

    +3.40pp R@10 over base on the 22,458-query ESCI test set (15.60% → 19.00%).
    Superseded as production default by bm25_rerank_top_k.
    """
    _, _, I, _ = base_top_k(query, R, k_retrieve)
    valid = [int(i) for i in I[0] if i >= 0]
    return _ensemble_rerank_from_positions(query, R, valid, k_top)


def mnrl_rerank_top_k(query, R, k_top, k_retrieve=100):
    """6M-MNRL retrieval top-K_retrieve → sumsim ensemble rerank → top-K_top.

    +4.23pp R@10 over base on the 22,458-query ESCI test set
    (15.60% → 19.83%), +0.0727 nDCG@10. Superseded as production default by
    bm25_rerank_top_k (R@10 21.11%). Falls back to base+ensemble if the
    MNRL FAISS index isn't loaded.
    """
    if R.get("mnrl_index") is None:
        return ensemble_rerank_top_k(query, R, k_top, k_retrieve)
    vec = R["rerank_a"].encode(query, normalize_embeddings=True)
    vec = np.asarray(vec, dtype=np.float32).reshape(1, -1)
    faiss.normalize_L2(vec)
    try:
        R["mnrl_index"].hnsw.efSearch = 128
    except Exception:
        pass
    _, I = R["mnrl_index"].search(vec, k_retrieve)
    valid = [int(i) for i in I[0] if i >= 0]
    return _ensemble_rerank_from_positions(query, R, valid, k_top)


_BM25_PUNCT_RE = re.compile(r"[^\w\s]")
_TOK_RE = re.compile(r"[a-z0-9]+")


def _spell_correct(query, R):
    """Catalog-vocab spell correction (pyspellchecker, distance 2)."""
    spell = R.get("spell")
    vocab_set = R.get("spell_vocab_set")
    if spell is None or vocab_set is None:
        return query
    tokens = _TOK_RE.findall(query.lower())
    if not tokens:
        return query
    out = []
    for tok in tokens:
        if tok in vocab_set or len(tok) <= 2 or tok.isdigit() or any(c.isdigit() for c in tok):
            out.append(tok)
            continue
        candidates = spell.candidates(tok) or set()
        best = None
        best_freq = -1
        for c in candidates:
            if c not in vocab_set:
                continue
            f = spell.word_frequency[c]
            if f > best_freq:
                best_freq = f
                best = c
        out.append(best if best and best != tok else tok)
    return " ".join(out)


def _safe_parse_bm25(tv_index, query):
    try:
        return tv_index.parse_query(query, ["title"])
    except ValueError:
        cleaned = _BM25_PUNCT_RE.sub(" ", query).strip()
        if not cleaned:
            return None
        try:
            return tv_index.parse_query(cleaned, ["title"])
        except ValueError:
            return None


def _bm25s_top_positions(query, R, k):
    """Top-k FAISS positions from bm25s retrieval (k1=0.3, b=0.6).

    Catalog-vocab spell-corrects the query first, if a corrector is loaded.
    """
    idx = R.get("bm25s_index")
    stemmer = R.get("bm25s_stemmer")
    if idx is None or stemmer is None:
        return []
    corrected = _spell_correct(query, R)
    qt = bm25s.tokenize([corrected], stopwords="en", stemmer=stemmer, show_progress=False)
    if not qt.ids or not qt.ids[0]:
        return []
    results, _ = idx.retrieve(qt, k=k, show_progress=False)
    return [int(p) for p in results[0]]


def _bm25_top_positions(query, R, k):
    """Top-k FAISS positions from BM25 retrieval. Prefers bm25s if loaded
    (optimized k1=0.3, b=0.6); falls back to tantivy."""
    if R.get("bm25s_index") is not None:
        return _bm25s_top_positions(query, R, k)
    tv_idx = R.get("tantivy_index")
    tv_searcher = R.get("tantivy_searcher")
    title_to_pos = R.get("title_to_pos") or {}
    if tv_idx is None or tv_searcher is None:
        return []
    parsed = _safe_parse_bm25(tv_idx, query)
    if parsed is None:
        return []
    hits = tv_searcher.search(parsed, limit=k * 2).hits
    positions, seen = [], set()
    for _, addr in hits:
        title = tv_searcher.doc(addr)["title"][0]
        pos = title_to_pos.get(title)
        if pos is None or pos in seen:
            continue
        seen.add(pos)
        positions.append(pos)
        if len(positions) >= k:
            break
    return positions


def bm25_top_k(query, R, k, oversample=3):
    """BM25 retrieval — no dense, no rerank. Prefers bm25s (k1=0.3, b=0.6).

    On the 22,458-query ESCI test set: bm25s scores R@10 20.32% / E@1 40.08%
    (vs tantivy default 19.50% / 38.79%). The optimized k1/b lift the
    retriever side meaningfully.
    """
    titles = R.get("titles") or []
    bm25s_idx = R.get("bm25s_index")
    bm25s_stemmer = R.get("bm25s_stemmer")
    if bm25s_idx is not None and bm25s_stemmer is not None:
        corrected = _spell_correct(query, R)
        qt = bm25s.tokenize([corrected], stopwords="en", stemmer=bm25s_stemmer, show_progress=False)
        if not qt.ids or not qt.ids[0]:
            return []
        results, scores = bm25s_idx.retrieve(qt, k=k * oversample, show_progress=False)
        raw = []
        for pos, sc in zip(results[0], scores[0]):
            pos = int(pos)
            title = titles[pos] if 0 <= pos < len(titles) else ""
            if not title:
                continue
            raw.append((title, round(float(sc), 4)))
        return _dedup_by_title(raw, k)
    tv_idx = R.get("tantivy_index")
    tv_searcher = R.get("tantivy_searcher")
    if tv_idx is None or tv_searcher is None:
        return []
    parsed = _safe_parse_bm25(tv_idx, query)
    if parsed is None:
        return []
    hits = tv_searcher.search(parsed, limit=k * oversample).hits
    raw = []
    for score, addr in hits:
        title = tv_searcher.doc(addr)["title"][0]
        raw.append((title, round(float(score), 4)))
    return _dedup_by_title(raw, k)


def bm25_rerank_top_k(query, R, k_top, k_retrieve=100):
    """BM25 top-K_retrieve -> ensemble rerank with two BoD models. R@10 21.11%.

    Falls back to mnrl_rerank_top_k if the BM25 index isn't loaded.
    """
    positions = _bm25_top_positions(query, R, k_retrieve)
    if not positions:
        return mnrl_rerank_top_k(query, R, k_top, k_retrieve)
    return _ensemble_rerank_from_positions(query, R, positions, k_top)


def bm25_3way_rerank_top_k(query, R, k_top, k_retrieve=50):
    """Setup CC3-50: BM25 top-K_retrieve -> 3-way ensemble rerank. ESCI SOTA.

    Three reranker encoders fused via sumsim (mean cosine):
      rerank_a: query_model_6m_mnrl     (BoD, MNRL on full-6M bags)
      rerank_b: query_model_hardneg     (BoD, qrels-derived bags + hardnegs)
      rerank_g: query_model_esci_supervised (MNRL on ESCI E-as-positive,
                                             I-as-hardneg triplets - no
                                             bag construction)

    R@10 21.32% on the 22,458-query ESCI test set, +0.21pp over the
    2-way ensemble (K). E@1 41.64%, +0.77pp over K. Default k_retrieve=50;
    100 also works (21.30%). Falls back to bm25_rerank_top_k if rerank_G
    is unavailable.
    """
    if R.get("rerank_g") is None or R.get("rerank_g_vecs") is None:
        return bm25_rerank_top_k(query, R, k_top, k_retrieve)
    positions = _bm25_top_positions(query, R, k_retrieve)
    if not positions:
        return mnrl_rerank_top_k(query, R, k_top, k_retrieve)
    qa = np.asarray(R["rerank_a"].encode(query, normalize_embeddings=True), dtype=np.float32)
    qb = np.asarray(R["rerank_b"].encode(query, normalize_embeddings=True), dtype=np.float32)
    qg = np.asarray(R["rerank_g"].encode(query, normalize_embeddings=True), dtype=np.float32)
    cv_a = R["rerank_a_vecs"][positions].astype(np.float32)
    cv_b = R["rerank_b_vecs"][positions].astype(np.float32)
    cv_g = R["rerank_g_vecs"][positions].astype(np.float32)
    avg_sim = (cv_a @ qa + cv_b @ qb + cv_g @ qg) / 3
    order = np.argsort(-avg_sim)
    raw = [(R["titles"][positions[int(idx)]], round(float(avg_sim[idx]), 4)) for idx in order]
    return _dedup_by_title(raw, k_top)


def bm25_3way_ce_rerank_top_k(query, R, k_top, k_retrieve=100):
    """Setup CC5-100 - quality SOTA. bm25s top-K -> weighted 3-way fusion of
    sumsim (3 bi-encoders) + LiYuan CE + BGE-reranker-v2-m3, all per-query
    min-max normalized. Weights (sumsim 0.4, liyuan 0.2, bge 0.4) found via
    grid sweep — strictly Pareto-dominates uniform 1/3-1/3-1/3 on both metrics.

    Full ESCI test (22,458 queries): R@10 23.57%, E@1 47.95%.
    vs uniform-mean baseline: +0.24pp R@10, +0.13pp E@1 at zero added latency.

    Latency: ~2.6s on MPS, 5-15s on CPU. BGE-reranker-v2-m3 (XLM-RoBERTa-
    large, 568M params) is ~6x slower than LiYuan but adds orthogonal signal.

    Falls back to CC3 if no cross-encoder is loaded.
    """
    ce = R.get("ce_model")
    bge = R.get("bge_model")
    if (ce is None and bge is None) or R.get("rerank_g") is None or R.get("rerank_g_vecs") is None:
        return bm25_3way_rerank_top_k(query, R, k_top, k_retrieve)
    positions = _bm25_top_positions(query, R, k_retrieve)
    if not positions:
        return mnrl_rerank_top_k(query, R, k_top, k_retrieve)
    qa = np.asarray(R["rerank_a"].encode(query, normalize_embeddings=True), dtype=np.float32)
    qb = np.asarray(R["rerank_b"].encode(query, normalize_embeddings=True), dtype=np.float32)
    qg = np.asarray(R["rerank_g"].encode(query, normalize_embeddings=True), dtype=np.float32)
    cv_a = R["rerank_a_vecs"][positions].astype(np.float32)
    cv_b = R["rerank_b_vecs"][positions].astype(np.float32)
    cv_g = R["rerank_g_vecs"][positions].astype(np.float32)
    sumsim = (cv_a @ qa + cv_b @ qb + cv_g @ qg) / 3

    titles = R["titles"]
    pairs = [(query, titles[p]) for p in positions]

    def _minmax(x):
        lo, hi = float(x.min()), float(x.max())
        return (x - lo) / max(hi - lo, 1e-8)

    # Weighted 3-way fusion: sumsim 0.4, LiYuan 0.2, BGE 0.4. Strictly Pareto-
    # dominates uniform 1/3-1/3-1/3 on both R@10 and E@1.
    weighted = [(_minmax(sumsim), 0.4)]
    if ce is not None:
        ce_scores = np.asarray(
            ce.predict(pairs, batch_size=32, show_progress_bar=False), dtype=np.float32
        )
        weighted.append((_minmax(ce_scores), 0.2))
    if bge is not None:
        bge_scores = np.asarray(
            bge.predict(pairs, batch_size=16, show_progress_bar=False), dtype=np.float32
        )
        weighted.append((_minmax(bge_scores), 0.4))

    total_w = sum(w for _, w in weighted)
    fused = sum(w * c for c, w in weighted) / total_w
    order = np.argsort(-fused)
    raw = [(titles[positions[int(idx)]], round(float(fused[idx]), 4)) for idx in order]
    return _dedup_by_title(raw, k_top)


def _bm25_base_rrf_positions(query, R, k_retrieve):
    """RRF positions from BM25 + base FAISS top-K_retrieve (non-BoD hybrid)."""
    bm25_positions = _bm25_top_positions(query, R, k_retrieve)
    base_positions = []
    if R.get("base_model") is not None and R.get("index") is not None:
        vec = R["base_model"].encode(query, normalize_embeddings=True)
        vec = np.asarray(vec, dtype=np.float32).reshape(1, -1)
        faiss.normalize_L2(vec)
        try:
            R["index"].hnsw.efSearch = 128
        except Exception:
            pass
        _, I = R["index"].search(vec, k_retrieve)
        base_positions = [int(p) for p in I[0] if p >= 0]
    rrf_c = 60
    rrf = {}
    for source in (bm25_positions, base_positions):
        for rank, p in enumerate(source):
            rrf[p] = rrf.get(p, 0.0) + 1.0 / (rank + 1 + rrf_c)
    fused = sorted(rrf.items(), key=lambda kv: -kv[1])[:k_retrieve]
    return [p for p, _ in fused]


def bm25_base_rrf_top_k(query, R, k, k_retrieve=100):
    """Setup Z: RRF(BM25, base) retrieval, no rerank. Non-BoD hybrid baseline.

    R@10 18.62% on the 22,458-query ESCI test set — *worse* than BM25 alone
    (19.50%); the base FAISS lane displaces BM25's exact-match top-1 with
    semantically-similar near-misses.
    """
    titles = R["titles"]
    positions = _bm25_base_rrf_positions(query, R, k_retrieve)
    raw = [(titles[p], round(1.0 / (rank + 1), 4)) for rank, p in enumerate(positions)]
    return _dedup_by_title(raw, k)


def bm25_base_rerank_top_k(query, R, k_top, k_retrieve=100):
    """Setup AA: RRF(BM25, base) candidates → ensemble rerank.

    R@10 20.43% — better than I (20.01%) and J (20.07%) but still loses
    -0.68pp to K (BM25-only candidates + ensemble rerank).
    """
    positions = _bm25_base_rrf_positions(query, R, k_retrieve)
    if not positions:
        return bm25_base_rrf_top_k(query, R, k_top, k_retrieve)
    return _ensemble_rerank_from_positions(query, R, positions, k_top)


def hybrid_rerank_top_k(query, R, k_top, k_retrieve=100):
    """RRF(BM25, 6M-MNRL) → ensemble rerank. Previously shipped at R@10 20.01%.

    Retained for comparison: bm25_rerank_top_k (no MNRL retrieval) is the
    current shipped default at R@10 21.11%. Falls back to mnrl_rerank_top_k
    if the BM25 index isn't loaded.
    """
    bm25_positions = _bm25_top_positions(query, R, k_retrieve)
    if not bm25_positions:
        return mnrl_rerank_top_k(query, R, k_top, k_retrieve)

    mnrl_positions = []
    if R.get("mnrl_index") is not None:
        vec = R["rerank_a"].encode(query, normalize_embeddings=True)
        vec = np.asarray(vec, dtype=np.float32).reshape(1, -1)
        faiss.normalize_L2(vec)
        try:
            R["mnrl_index"].hnsw.efSearch = 128
        except Exception:
            pass
        _, I = R["mnrl_index"].search(vec, k_retrieve)
        mnrl_positions = [int(p) for p in I[0] if p >= 0]

    rrf_c = 60
    rrf = {}
    for source in (bm25_positions, mnrl_positions):
        for rank, p in enumerate(source):
            rrf[p] = rrf.get(p, 0.0) + 1.0 / (rank + 1 + rrf_c)
    fused = sorted(rrf.items(), key=lambda kv: -kv[1])[:k_retrieve]
    positions = [p for p, _ in fused]
    return _ensemble_rerank_from_positions(query, R, positions, k_top)


def predict_specificity(query, R, k=5):
    if R["bag_matrix"] is None:
        return None, []
    vec = R["retrieval_model"].encode(query, normalize_embeddings=True)
    vec = np.asarray(vec, dtype=np.float32).reshape(1, -1)
    faiss.normalize_L2(vec)
    sims = (vec @ R["bag_matrix"].T).flatten()
    top = np.argsort(-sims)[:k]
    neighbors = [
        (R["bag_queries"][i], round(float(R["bag_specs"][i]), 3), round(float(sims[i]), 3))
        for i in top
    ]
    weights = sims[top]
    spec = float(np.average(R["bag_specs"][top], weights=weights))
    return round(spec, 3), neighbors


def format_results(results, header, latency_ms=None):
    if not results:
        return f"<h3>{header}</h3><p>No results</p>"
    nw = "white-space:nowrap;padding:2px 6px"
    rows = "".join(
        f"<tr><td style='{nw};text-align:center'>{i + 1}</td>"
        f"<td style='{nw};text-align:center'>{score:.3f}</td>"
        f"<td style='padding:2px 6px'>{title[:150]}</td></tr>"
        for i, (title, score) in enumerate(results)
    )
    sub = f" <small style='color:#888'>({latency_ms:.0f} ms)</small>" if latency_ms else ""
    return (
        f"<h3>{header}{sub}</h3>"
        "<table style='width:100%;font-size:14px;border-collapse:collapse'>"
        f"<tr style='border-bottom:2px solid #ccc'>"
        f"<th style='{nw}'>#</th><th style='{nw}'>Sim</th>"
        f"<th style='padding:2px 6px'>Title</th></tr>" + rows + "</table>"
    )


def format_neighbors(neighbors):
    if not neighbors:
        return ""
    lines = ["| Similarity | Specificity | Query |", "|-----------|------------|-------|"]
    for q, spec, sim in neighbors:
        lines.append(f"| {sim:.3f} | {spec:.3f} | {q} |")
    return "\n".join(lines)


# Load everything once at module load
data_dir = download_data()
R = load_resources(data_dir)


MODE_LABELS = [
    "BM25 + sumsim + LiYuan + BGE quality SOTA (R@10 23.57, E@1 47.95, ~5-15s CPU)",
    "BM25 + 3-way ensemble rerank (fast SOTA, R@10 21.61, ~50ms)",
    "BM25 retrieval (R@10 20.33)",
    "RRF(BM25, base) retrieval - non-BoD hybrid baseline",
    "Base MiniLM retrieval (R@10 15.60)",
]


def _results_for_mode(mode, query, R, k):
    if mode == "BM25 + sumsim + LiYuan + BGE quality SOTA (R@10 23.57, E@1 47.95, ~5-15s CPU)":
        return bm25_3way_ce_rerank_top_k(query, R, k_top=k)
    if mode == "BM25 + 3-way ensemble rerank (fast SOTA, R@10 21.61, ~50ms)":
        return bm25_3way_rerank_top_k(query, R, k_top=k)
    if mode == "BM25 retrieval (R@10 20.33)":
        return bm25_top_k(query, R, k)
    if mode == "RRF(BM25, base) retrieval - non-BoD hybrid baseline":
        return bm25_base_rrf_top_k(query, R, k=k)
    # Base MiniLM retrieval (default left)
    base_results, *_ = base_top_k(query, R, k)
    return base_results


def search_fn(query, k, left_mode, right_mode):
    if not query.strip():
        return "", "", "", ""
    k = int(k)

    spec, neighbors = predict_specificity(query, R)
    spec_text = f"**Specificity:** {spec:.3f}" if spec is not None else ""

    t0 = time.perf_counter()
    left_results = _results_for_mode(left_mode, query, R, k)
    left_latency = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    right_results = _results_for_mode(right_mode, query, R, k)
    right_latency = (time.perf_counter() - t0) * 1000

    left_html = format_results(left_results, left_mode, left_latency)
    right_html = format_results(right_results, right_mode, right_latency)
    return left_html, right_html, spec_text, format_neighbors(neighbors)


with gr.Blocks(title="Bag-of-Documents Search") as demo:
    gr.Markdown(
        "# Bag-of-Documents Product Search\n"
        "Side-by-side comparison of five retrieval architectures on 1.2M Amazon "
        "ESCI products. Pick a mode for each column, run a query, see how the "
        "rankings differ. ESCI 22,458-query R@10 in parens.\n\n"
        "* **Base MiniLM retrieval** (15.60%) — dense baseline, no fine-tuning.\n"
        "* **BM25 retrieval** (20.33%) — bm25s with k1=0.3, b=0.6, tuned for "
        "short keyword-stuffed product titles.\n"
        "* **RRF(BM25, base)** (non-BoD hybrid baseline) — vanilla hybrid retrieval; "
        "on this corpus it actually loses to BM25 alone.\n"
        "* **BM25 + 3-way ensemble rerank** — fast SOTA (21.61%), ~50ms/query. "
        "BM25 top-50 reranked by three BoD-trained MiniLM encoders via sumsim fusion.\n"
        "* **BM25 + sumsim + LiYuan + BGE** — quality SOTA (R@10 23.57%, E@1 47.95%), "
        "~5-15s/query on Space CPU. Equal-weight 3-way fusion of sumsim, the "
        "LiYuan ESCI cross-encoder, and BGE-reranker-v2-m3 (568M-param XLM-RoBERTa-"
        "large reranker), each min-max normalized over the BM25 top-100. "
        "+1.49pp R@10, +5.28pp E@1 over the fast SOTA.\n\n"
        "The fast SOTA does three forward passes against precomputed product "
        "embeddings then averages cosine — sub-100ms wall-clock. The quality SOTA "
        "adds 50 cross-encoder forward passes (full attention, ESCI-supervised) "
        "for a meaningful E@1 lift on near-miss queries."
    )

    with gr.Row():
        query_input = gr.Textbox(
            label="Search query", placeholder="e.g. tom ford lipstick", scale=4
        )
        k_input = gr.Number(label="k", value=10, minimum=1, maximum=50, scale=1)
        search_btn = gr.Button("Search", scale=1)

    with gr.Row():
        left_mode_input = gr.Dropdown(
            choices=MODE_LABELS,
            value="BM25 retrieval (R@10 20.33)",
            label="Left-column mode",
        )
        right_mode_input = gr.Dropdown(
            choices=MODE_LABELS,
            value="BM25 + 3-way ensemble rerank (fast SOTA, R@10 21.61, ~50ms)",
            label="Right-column mode",
        )

    spec_output = gr.Markdown()
    with gr.Row():
        left_output = gr.Markdown()
        right_output = gr.Markdown()
    neighbor_output = gr.Markdown(label="Similar queries")

    search_btn.click(
        search_fn,
        [query_input, k_input, left_mode_input, right_mode_input],
        [left_output, right_output, spec_output, neighbor_output],
    )
    query_input.submit(
        search_fn,
        [query_input, k_input, left_mode_input, right_mode_input],
        [left_output, right_output, spec_output, neighbor_output],
    )

demo.launch()
