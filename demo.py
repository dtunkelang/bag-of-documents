#!/usr/bin/env python3
"""
Web-based search demo using bag-of-documents retrieval.

Demonstrates:
- Side-by-side comparison: fine-tuned model vs base MiniLM
- Specificity prediction via kNN on bag centroids
- Optional bag-search mode: hybrid retrieval → CE scoring → centroid
  → re-retrieve (simulates offline pipeline)

Usage:
    python demo.py
    python demo.py --bag-search   # enable real-time bag construction
    python demo.py --port 8080
"""

import argparse
import json
import os

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def load_resources():
    """Load all models, index, and data at startup."""
    print("Loading resources...")

    # Product index
    index_dir = os.path.join(SCRIPT_DIR, "combined_index")
    hnsw_path = os.path.join(index_dir, "index.faiss")
    index = faiss.read_index(hnsw_path)
    print(f"  Product index: {index.ntotal} products (HNSW, {os.path.basename(index_dir)})")
    with open(os.path.join(index_dir, "titles.json")) as f:
        titles = json.load(f)

    # Retrieval model
    base_model_name = os.environ.get("BASE_MODEL", "all-MiniLM-L6-v2")
    ret_path = os.path.join(SCRIPT_DIR, "retrieval_model")
    if os.path.exists(os.path.join(ret_path, "config.json")):
        retrieval_model = SentenceTransformer(ret_path)
        print(f"  Retrieval model: {ret_path}")
    else:
        retrieval_model = SentenceTransformer(base_model_name)
        print(f"  Retrieval model: {base_model_name} (no fine-tuned model found)")

    # Base model for comparison
    base_model = SentenceTransformer(base_model_name)
    print(f"  Base model: {base_model_name}")

    # Bag centroids for kNN specificity prediction
    bags_path = os.path.join(SCRIPT_DIR, "bags.jsonl")
    bag_queries = []
    bag_centroids = []
    bag_specificities = []
    if os.path.exists(bags_path):
        with open(bags_path) as f:
            for line in f:
                bag = json.loads(line)
                if bag["num_results"] < 2:
                    continue
                bag_queries.append(bag["query"])
                bag_centroids.append(bag["query_vector"])
                bag_specificities.append(bag["specificity"])

    bag_matrix = np.array(bag_centroids, dtype=np.float32) if bag_centroids else None
    if bag_matrix is not None:
        norms = np.linalg.norm(bag_matrix, axis=1, keepdims=True)
        bag_matrix = bag_matrix / np.maximum(norms, 1e-8)
    bag_specs = np.array(bag_specificities) if bag_specificities else None
    print(f"  Bag centroids: {len(bag_queries)} bags")

    # Tantivy index for hybrid retrieval
    tantivy_searcher = None
    tantivy_idx = None
    tantivy_path = os.path.join(index_dir, "tantivy_index")
    if os.path.exists(tantivy_path):
        import tantivy

        schema_builder = tantivy.SchemaBuilder()
        schema_builder.add_text_field("title", stored=True)
        schema = schema_builder.build()
        tv_index = tantivy.Index(schema, path=tantivy_path)
        tv_index.reload()
        tantivy_searcher = tv_index.searcher()
        tantivy_idx = tv_index
        print("  Tantivy index loaded")

    # Cross-encoder for bag search mode
    ce_model = None
    if os.environ.get("LOAD_BAG_SEARCH") == "1":
        ce_path = os.path.join(SCRIPT_DIR, "models", "esci-cross-encoder")
        if os.path.exists(ce_path):
            from sentence_transformers import CrossEncoder

            ce_model = CrossEncoder(ce_path)
            print("  ESCI cross-encoder loaded")

    return {
        "index": index,
        "titles": titles,
        "retrieval_model": retrieval_model,
        "base_model": base_model,
        "bag_queries": bag_queries,
        "bag_matrix": bag_matrix,
        "bag_specs": bag_specs,
        "tantivy_searcher": tantivy_searcher,
        "tantivy_index": tantivy_idx,
        "ce_model": ce_model,
    }


def search_products(query, resources, model_key="retrieval_model", k=50, ef_search=128):
    """Search products by query text."""
    model = resources[model_key]
    index = resources["index"]
    titles = resources["titles"]

    vec = model.encode(query, normalize_embeddings=True)
    vec = np.array(vec, dtype=np.float32).reshape(1, -1)
    faiss.normalize_L2(vec)

    # HNSW uses L2 distance (score = 1 - dist/2)
    try:
        index.hnsw.efSearch = ef_search
        is_hnsw = True
    except Exception:
        is_hnsw = False
    D, I = index.search(vec, k)

    results = []
    for dist, idx in zip(D[0], I[0]):
        if idx < 0:
            continue
        sim = float(1 - dist / 2) if is_hnsw else float(dist)
        results.append({"title": titles[idx], "score": round(sim, 4)})

    # Deduplicate by title (keep highest score) and sort descending
    seen = {}
    for r in results:
        if r["title"] not in seen or r["score"] > seen[r["title"]]["score"]:
            seen[r["title"]] = r
    results = sorted(seen.values(), key=lambda r: -r["score"])

    return results


def predict_specificity(query, resources, k=5):
    """Predict query specificity via kNN on bag centroids."""
    if resources["bag_matrix"] is None:
        return None, []

    model = resources["retrieval_model"]
    vec = model.encode(query, normalize_embeddings=True)
    vec = np.array(vec, dtype=np.float32).reshape(1, -1)
    faiss.normalize_L2(vec)

    sims = (vec @ resources["bag_matrix"].T).flatten()
    top_k = np.argsort(-sims)[:k]

    neighbors = []
    for i in top_k:
        neighbors.append(
            {
                "query": resources["bag_queries"][i],
                "specificity": round(float(resources["bag_specs"][i]), 3),
                "similarity": round(float(sims[i]), 3),
            }
        )

    weights = sims[top_k]
    specificity = float(np.average(resources["bag_specs"][top_k], weights=weights))

    return round(specificity, 3), neighbors


# --- FastAPI app ---

from contextlib import asynccontextmanager

from fastapi import FastAPI, Query

resources = None


@asynccontextmanager
async def lifespan(app):
    global resources
    resources = load_resources()
    yield


app = FastAPI(title="Bag-of-Documents Search Demo", lifespan=lifespan)


@app.get("/")
async def home():
    from fastapi.responses import HTMLResponse

    return HTMLResponse(HTML_PAGE)


@app.get("/api/bag_search")
async def api_bag_search(
    q: str = Query(..., description="Search query"),
    k: int = Query(50, description="Number of results (also scales candidate retrieval)"),
):
    """Real-time bag-of-documents search (simulates offline pipeline):
    1. Hybrid retrieval: keyword (tantivy AND) + embedding (FAISS)
    2. Cross-encoder scores all candidates, threshold at --ce-threshold
    3. Passing products form a bag → compute centroid
    4. Re-retrieve using the centroid vector
    """
    from utils import generate_keyword_combos, tokenize_query

    tv_searcher = resources.get("tantivy_searcher")
    tv_index_obj = resources.get("tantivy_index")
    all_titles = resources.get("titles") or []

    # Step 1a: Keyword retrieval (tantivy AND with relaxation)
    words = tokenize_query(q)
    seen_titles = set()
    raw_candidates = []

    if words and tv_searcher and tv_index_obj:
        for n_required, combos in generate_keyword_combos(words):
            for combo in combos:
                try:
                    parsed = tv_index_obj.parse_query(" AND ".join(combo), ["title"])
                    results = tv_searcher.search(parsed, limit=k * 4)
                except Exception:
                    continue

                if results.count < 3 and n_required > 1:
                    continue

                for _score, addr in results.hits:
                    title = tv_searcher.doc(addr)["title"][0]
                    if title not in seen_titles:
                        seen_titles.add(title)
                        raw_candidates.append(title)

                if raw_candidates:
                    break
            if raw_candidates:
                break

    # Step 1b: FAISS embedding retrieval (complements keyword search)
    base_model = resources.get("base_model")
    index = resources.get("index")
    if base_model and index:
        q_vec = base_model.encode(q, normalize_embeddings=True)
        q_vec = np.array(q_vec, dtype=np.float32).reshape(1, -1)
        faiss.normalize_L2(q_vec)
        D, I = index.search(q_vec, k * 4)
        for _dist, idx in zip(D[0], I[0]):
            if idx >= 0:
                title = all_titles[idx]
                if title not in seen_titles:
                    seen_titles.add(title)
                    raw_candidates.append(title)

    n_candidates = len(raw_candidates)

    # Step 2: CE score all candidates (matches offline pipeline)
    ce_model = resources.get("ce_model")
    if ce_model and raw_candidates:
        ce_pairs = [(q, t) for t in raw_candidates]
        ce_scores = ce_model.predict(ce_pairs)
        scored = sorted(zip(ce_scores, raw_candidates), key=lambda x: -x[0])
        ce_threshold = float(os.environ.get("CE_THRESHOLD", "0.3"))
        relevant_titles = [t for s, t in scored if s >= ce_threshold][:k]
    else:
        relevant_titles = raw_candidates[:k]

    # Step 3: Build bag centroid from all relevant products
    if relevant_titles:
        retrieval_model = resources["retrieval_model"]
        rel_vectors = retrieval_model.encode(relevant_titles, normalize_embeddings=True)
        centroid = rel_vectors.mean(axis=0)
        centroid = centroid / np.linalg.norm(centroid)
        specificity = float(np.mean([centroid @ v for v in rel_vectors]))

        # Step 4: Re-retrieve using centroid
        centroid_query = np.array(centroid, dtype=np.float32).reshape(1, -1)
        faiss.normalize_L2(centroid_query)
        D, I = index.search(centroid_query, k)
        bag_results = []
        for dist, idx in zip(D[0], I[0]):
            if idx >= 0:
                bag_results.append(
                    {
                        "title": all_titles[idx],
                        "score": round(float(1 - dist / 2), 4),
                    }
                )
    else:
        bag_results = []
        specificity = 0
        centroid = None

    # Get similar queries from bag centroids
    _, neighbors = predict_specificity(q, resources)

    # Base model results for comparison (right column), annotated with CE pass/fail
    relevant_set = set(relevant_titles)
    base_results = search_products(q, resources, "base_model", k=k)
    for r in base_results:
        r["ce_pass"] = r["title"] in relevant_set

    return {
        "query": q,
        "step1_candidates": n_candidates,
        "step2_relevant": len(relevant_titles),
        "step3_specificity": round(specificity, 3) if specificity else None,
        "step4_bag_results": bag_results,
        "base_results": base_results,
        "nearest_queries": neighbors,
    }


@app.get("/api/search")
async def api_search(
    q: str = Query(..., description="Search query"),
    k: int = Query(50, description="Number of results (also scales candidate retrieval)"),
):
    specificity, neighbors = predict_specificity(q, resources)

    # Retrieve results with both models
    results = search_products(q, resources, "retrieval_model", k=k)
    base_results = search_products(q, resources, "base_model", k=k)

    return {
        "query": q,
        "specificity": specificity,
        "results": results,
        "nearest_queries": neighbors,
        "base_results": base_results,
    }


HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Bag-of-Documents Search Demo</title>
<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
       max-width: 1100px; margin: 0 auto; padding: 20px; color: #1a1a1a;
       background: #fafafa; }
h1 { font-size: 1.4em; margin-bottom: 4px; }
.subtitle { color: #666; font-size: 0.9em; margin-bottom: 20px; }
.search-box { display: flex; gap: 8px; margin-bottom: 16px; }
.search-box input[type="text"] { flex: 1; padding: 10px 14px; font-size: 16px;
    border: 2px solid #ddd; border-radius: 8px; outline: none; }
.search-box input[type="text"]:focus { border-color: #4a90d9; }
.search-box button { padding: 10px 20px; font-size: 14px; background: #4a90d9;
    color: white; border: none; border-radius: 8px; cursor: pointer; }
.search-box button:hover { background: #357abd; }
.meta { display: flex; gap: 24px; margin-bottom: 16px; padding: 12px 16px;
    background: white; border-radius: 8px; border: 1px solid #e0e0e0;
    font-size: 0.85em; }
.meta-item { display: flex; flex-direction: column; }
.meta-label { color: #888; font-size: 0.8em; text-transform: uppercase; letter-spacing: 0.5px; }
.meta-value { font-weight: 600; font-size: 1.1em; }
.spec-bar { width: 120px; height: 8px; background: #e0e0e0; border-radius: 4px;
    margin-top: 4px; overflow: hidden; }
.spec-fill { height: 100%; border-radius: 4px; transition: width 0.3s; }
.columns { display: flex; gap: 20px; }
.column { flex: 1; }
.column-header { font-size: 0.85em; font-weight: 600; color: #555;
    padding-bottom: 8px; border-bottom: 2px solid #e0e0e0; margin-bottom: 8px; }
.result { padding: 8px 12px; border-bottom: 1px solid #f0f0f0;
    display: flex; align-items: baseline; gap: 8px; font-size: 0.9em; }
.result:hover { background: #f5f8ff; }
.result-rank { color: #aaa; font-size: 0.8em; min-width: 24px; }
.result-score { color: #4a90d9; font-weight: 600; font-size: 0.8em;
    min-width: 48px; font-family: monospace; }
.result-title { flex: 1; line-height: 1.3; }
.neighbors { margin-top: 16px; padding: 12px 16px; background: white;
    border-radius: 8px; border: 1px solid #e0e0e0; }
.neighbors h3 { font-size: 0.85em; color: #888; margin-bottom: 8px; }
.neighbor { font-size: 0.85em; padding: 3px 0; display: flex; gap: 8px; }
.neighbor-sim { color: #4a90d9; font-family: monospace; min-width: 42px; }
.neighbor-spec { color: #888; font-family: monospace; min-width: 42px; }
.neighbor-query { color: #333; }
.empty { text-align: center; color: #888; padding: 40px; font-style: italic; }
.loading { text-align: center; color: #888; padding: 20px; }
</style>
</head>
<body>

<h1>Bag-of-Documents Search</h1>
<p class="subtitle">6M Amazon products (1996&ndash;2023, McAuley Lab) &middot; 75K queries &middot; Fine-tuned MiniLM vs base MiniLM</p>

<div class="search-box">
    <input type="text" id="query" placeholder="Search products..." autofocus>
    <button onclick="runSearch()">Search</button>
    <label style="font-size:0.85em; display:flex; align-items:center; gap:4px; white-space:nowrap">
        k=<input type="number" id="k-value" value="50" min="1" max="200" style="width:50px; padding:2px 4px;">
    </label>
    <label style="font-size:0.85em; display:flex; align-items:center; gap:4px; white-space:nowrap">
        <input type="checkbox" id="bag-mode"> Build bag at query time
    </label>
</div>

<div id="meta-section" style="display:none">
    <div class="meta">
        <div class="meta-item">
            <span class="meta-label">Specificity</span>
            <span class="meta-value" id="specificity">—</span>
            <div class="spec-bar"><div class="spec-fill" id="spec-fill"></div></div>
        </div>
    </div>
</div>

<div class="columns" id="results-section" style="display:none">
    <div class="column" id="main-column">
        <div class="column-header" id="main-header">Fine-tuned model</div>
        <div id="results"></div>
    </div>
    <div class="column" id="base-column" style="display:none">
        <div class="column-header" id="base-header">Base model (MiniLM)</div>
        <div id="base-results"></div>
    </div>
</div>

<div id="neighbors-section" class="neighbors" style="display:none">
    <h3>Similar queries (from bag centroids)</h3>
    <div id="neighbors"></div>
</div>

<div id="empty" class="empty">Type a query to search</div>

<script>
const input = document.getElementById('query');
input.addEventListener('keydown', e => { if (e.key === 'Enter') runSearch(); });

function runSearch() {
    if (document.getElementById('bag-mode').checked) {
        doBagSearch();
    } else {
        doSearch();
    }
}

async function doSearch() {
    const q = input.value.trim();
    if (!q) return;

    document.getElementById('empty').style.display = 'none';
    document.getElementById('results-section').style.display = 'flex';
    document.getElementById('main-header').textContent = 'Fine-tuned model';
    document.getElementById('base-header').textContent = 'Base model (MiniLM)';
    document.getElementById('results').innerHTML = '<div class="loading">Searching...</div>';

    const k = parseInt(document.getElementById('k-value').value) || 50;
    const url = `/api/search?q=${encodeURIComponent(q)}&k=${k}`;
    const resp = await fetch(url);
    const data = await resp.json();

    // Specificity
    if (data.specificity != null) {
        document.getElementById('meta-section').style.display = 'block';
        document.getElementById('specificity').textContent = data.specificity.toFixed(3);
        const specPct = Math.max(0, Math.min(100, (data.specificity - 0.5) / 0.5 * 100));
        const fill = document.getElementById('spec-fill');
        fill.style.width = specPct + '%';
        fill.style.background = specPct > 70 ? '#4caf50' : specPct > 40 ? '#ff9800' : '#f44336';
    } else {
        document.getElementById('meta-section').style.display = 'none';
    }

    // Fine-tuned results
    const resultsDiv = document.getElementById('results');
    if (data.results.length === 0) {
        resultsDiv.innerHTML = '<div class="empty">No results</div>';
    } else {
        resultsDiv.innerHTML = data.results.map((r, i) =>
            `<div class="result">
                <span class="result-rank">${i + 1}</span>
                <span class="result-score">${r.score.toFixed(3)}</span>
                <span class="result-title">${escapeHtml(r.title)}</span>
            </div>`
        ).join('');
    }

    // Base model results
    const baseCol = document.getElementById('base-column');
    baseCol.style.display = 'block';
    const baseDiv = document.getElementById('base-results');
    baseDiv.innerHTML = data.base_results.map((r, i) =>
        `<div class="result">
            <span class="result-rank">${i + 1}</span>
            <span class="result-score">${r.score.toFixed(3)}</span>
            <span class="result-title">${escapeHtml(r.title)}</span>
        </div>`
    ).join('');

    // Similar queries
    renderNeighbors(data.nearest_queries);
}

function renderNeighbors(neighbors) {
    const section = document.getElementById('neighbors-section');
    const div = document.getElementById('neighbors');
    if (neighbors && neighbors.length > 0) {
        section.style.display = 'block';
        div.innerHTML = neighbors.map(n =>
            `<div class="neighbor">
                <span class="neighbor-sim">${n.similarity.toFixed(3)}</span>
                <span class="neighbor-spec">s=${n.specificity.toFixed(2)}</span>
                <span class="neighbor-query">${escapeHtml(n.query)}</span>
            </div>`
        ).join('');
    } else {
        section.style.display = 'none';
    }
}

async function doBagSearch() {
    const q = input.value.trim();
    if (!q) return;

    document.getElementById('empty').style.display = 'none';
    document.getElementById('meta-section').style.display = 'block';
    document.getElementById('results-section').style.display = 'flex';

    const mainHeader = document.getElementById('main-header');
    mainHeader.textContent = 'Building bag...';
    document.getElementById('results').innerHTML = '<div class="loading">Step 1: Hybrid retrieval (keyword + embedding)...<br>Step 2: Cross-encoder scoring...<br>Step 3: Computing bag centroid...<br>Step 4: Re-retrieving with centroid (FAISS)...</div>';

    const baseCol = document.getElementById('base-column');
    baseCol.style.display = 'block';
    document.getElementById('base-results').innerHTML = '<div class="loading">Scoring candidates...</div>';

    const k = parseInt(document.getElementById('k-value').value) || 50;
    const url = `/api/bag_search?q=${encodeURIComponent(q)}&k=${k}`;
    const resp = await fetch(url);
    const data = await resp.json();

    if (data.error) {
        document.getElementById('results').innerHTML = `<div class="empty">${data.error}<br>Restart with: python demo.py --bag-search</div>`;
        return;
    }

    // Meta
    document.getElementById('specificity').textContent = data.step3_specificity ? data.step3_specificity.toFixed(3) : '—';
    const specPct = data.step3_specificity ? Math.max(0, Math.min(100, (data.step3_specificity - 0.5) / 0.5 * 100)) : 0;
    const fill = document.getElementById('spec-fill');
    fill.style.width = specPct + '%';
    fill.style.background = specPct > 70 ? '#4caf50' : specPct > 40 ? '#ff9800' : '#f44336';

    // Bag results (re-retrieved with centroid)
    mainHeader.textContent = `Centroid re-retrieval (${data.step2_relevant} bag members → centroid → FAISS top-${data.step4_bag_results.length})`;
    const resultsDiv = document.getElementById('results');
    if (data.step4_bag_results.length === 0) {
        resultsDiv.innerHTML = '<div class="empty">No relevant products found</div>';
    } else {
        resultsDiv.innerHTML = data.step4_bag_results.map((r, i) =>
            `<div class="result">
                <span class="result-rank">${i + 1}</span>
                <span class="result-score">${r.score.toFixed(3)}</span>
                <span class="result-title">${escapeHtml(r.title)}</span>
            </div>`
        ).join('');
    }

    // Base model results, highlighted if CE approved
    const baseDiv = document.getElementById('base-results');
    baseDiv.innerHTML = data.base_results.map((r, i) => {
        const bg = r.ce_pass ? 'background:#e8f5e9' : '';
        return `<div class="result" style="${bg}">
            <span class="result-rank">${i + 1}</span>
            <span class="result-score">${r.score.toFixed(3)}</span>
            <span class="result-title">${r.ce_pass ? '&check; ' : ''}${escapeHtml(r.title)}</span>
        </div>`;
    }).join('');
    // Similar queries
    renderNeighbors(data.nearest_queries);
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}
</script>

</body>
</html>"""


def main():
    parser = argparse.ArgumentParser(description="Search demo server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--bag-search",
        action="store_true",
        help="Load resources for real-time bag-of-documents search",
    )
    parser.add_argument(
        "--ce-threshold",
        type=float,
        default=0.3,
        help="CE score threshold for bag search mode (default: 0.3)",
    )
    parser.add_argument(
        "--base-model",
        default="all-MiniLM-L6-v2",
        help="Base embedding model (default: all-MiniLM-L6-v2)",
    )
    args = parser.parse_args()

    if args.bag_search:
        os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # suppress Intel MKL duplicate lib warning
        os.environ["LOAD_BAG_SEARCH"] = "1"
    os.environ["CE_THRESHOLD"] = str(args.ce_threshold)
    os.environ["BASE_MODEL"] = args.base_model

    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
