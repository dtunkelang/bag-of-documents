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
import time

import faiss
import gradio as gr
import numpy as np
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
            "combined_index/titles.json",
            "combined_index/rerank_A.vecs.fp16.npy",
            "combined_index/rerank_B.vecs.fp16.npy",
            "bags.jsonl",
            "query_model/*",
            "query_model_6m_mnrl/*",
            "query_model_hardneg/*",
        ],
    )
    print(f"  Downloaded to {local_dir}")
    return local_dir


def load_resources(data_dir):
    print("Loading resources...")

    index = faiss.read_index(os.path.join(data_dir, "combined_index", "index.faiss"))
    print(f"  Product index: {index.ntotal:,} products")

    with open(os.path.join(data_dir, "combined_index", "titles.json")) as f:
        titles = json.load(f)

    base_model = SentenceTransformer(BASE_MODEL_NAME)
    print(f"  Base model: {BASE_MODEL_NAME}")

    retrieval_model = SentenceTransformer(os.path.join(data_dir, "query_model"))
    print("  Retrieval model: query_model")

    rerank_a = SentenceTransformer(os.path.join(data_dir, "query_model_6m_mnrl"))
    rerank_b = SentenceTransformer(os.path.join(data_dir, "query_model_hardneg"))
    print("  Rerankers: 6M MNRL + qrels-hardneg")

    rerank_a_vecs = np.load(os.path.join(data_dir, "combined_index", "rerank_A.vecs.fp16.npy"))
    rerank_b_vecs = np.load(os.path.join(data_dir, "combined_index", "rerank_B.vecs.fp16.npy"))
    print(f"  Cached vecs: {rerank_a_vecs.shape} fp16 each")

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

    return {
        "index": index,
        "titles": titles,
        "base_model": base_model,
        "retrieval_model": retrieval_model,
        "rerank_a": rerank_a,
        "rerank_b": rerank_b,
        "rerank_a_vecs": rerank_a_vecs,
        "rerank_b_vecs": rerank_b_vecs,
        "bag_queries": bag_queries,
        "bag_matrix": bag_matrix,
        "bag_specs": bag_specs,
    }


def base_top_k(query, R, k):
    vec = R["base_model"].encode(query, normalize_embeddings=True)
    vec = np.asarray(vec, dtype=np.float32).reshape(1, -1)
    faiss.normalize_L2(vec)
    try:
        R["index"].hnsw.efSearch = 128
        is_hnsw = True
    except Exception:
        is_hnsw = False
    D, I = R["index"].search(vec, k)
    out = []
    for d, i in zip(D[0], I[0]):
        if i < 0:
            continue
        sim = float(1 - d / 2) if is_hnsw else float(d)
        out.append((R["titles"][i], round(sim, 4)))
    return out, D, I, is_hnsw


def retrieval_top_k(query, R, k):
    vec = R["retrieval_model"].encode(query, normalize_embeddings=True)
    vec = np.asarray(vec, dtype=np.float32).reshape(1, -1)
    faiss.normalize_L2(vec)
    try:
        R["index"].hnsw.efSearch = 128
        is_hnsw = True
    except Exception:
        is_hnsw = False
    D, I = R["index"].search(vec, k)
    out = []
    for d, i in zip(D[0], I[0]):
        if i < 0:
            continue
        sim = float(1 - d / 2) if is_hnsw else float(d)
        out.append((R["titles"][i], round(sim, 4)))
    return out


def ensemble_rerank_top_k(query, R, k_top, k_retrieve=100):
    """Base FAISS top-K_retrieve, sumrank-fuse two rerankers, return top-K_top.

    Uses precomputed cached vecs — only the query is encoded live.
    """
    base_results, D, I, is_hnsw = base_top_k(query, R, k_retrieve)
    valid = [int(i) for i in I[0] if i >= 0]
    if not valid:
        return []

    qa = np.asarray(R["rerank_a"].encode(query, normalize_embeddings=True), dtype=np.float32)
    qb = np.asarray(R["rerank_b"].encode(query, normalize_embeddings=True), dtype=np.float32)

    cv_a = R["rerank_a_vecs"][valid].astype(np.float32)
    cv_b = R["rerank_b_vecs"][valid].astype(np.float32)

    sims_a = cv_a @ qa
    sims_b = cv_b @ qb
    rank_a = np.argsort(np.argsort(-sims_a)) + 1
    rank_b = np.argsort(np.argsort(-sims_b)) + 1
    fused = rank_a + rank_b
    order = np.argsort(fused)[:k_top]

    out = []
    for idx in order:
        pos = valid[int(idx)]
        # Reciprocal-of-fused-rank: 1.0 if both rerankers placed this candidate at #1,
        # decays toward 0 as ranks fall. Monotonic-decreasing in the sumrank order so
        # the Sim column always descends in rerank mode.
        rerank_score = 2.0 / float(fused[idx])
        out.append((R["titles"][pos], round(rerank_score, 4)))
    return out


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
        f"<td style='padding:2px 6px'>{title[:80]}</td></tr>"
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


def search_fn(query, k, mode):
    if not query.strip():
        return "", "", "", ""
    k = int(k)

    spec, neighbors = predict_specificity(query, R)
    spec_text = f"**Specificity:** {spec:.3f}" if spec is not None else ""

    t0 = time.perf_counter()
    base_results, *_ = base_top_k(query, R, k)
    base_latency = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    if mode == "Base + BoD ensemble rerank":
        right_results = ensemble_rerank_top_k(query, R, k_top=k)
        right_label = "Base + BoD ensemble rerank"
    else:
        right_results = retrieval_top_k(query, R, k)
        right_label = "Fine-tuned retrieval (BoD)"
    right_latency = (time.perf_counter() - t0) * 1000

    base_html = format_results(base_results, "Base MiniLM", base_latency)
    right_html = format_results(right_results, right_label, right_latency)
    return right_html, base_html, spec_text, format_neighbors(neighbors)


with gr.Blocks(title="Bag-of-Documents Search") as demo:
    gr.Markdown(
        "# Bag-of-Documents Product Search\n"
        "Comparing the **base MiniLM** model with two BoD architectures on "
        "1.2M ESCI products (75K Amazon search queries).\n\n"
        "* **Fine-tuned retrieval** — original BoD-as-retriever (single fine-tuned model + FAISS).\n"
        "* **Base + BoD ensemble rerank** — base FAISS top-100, then two BoD models reorder "
        "via sumrank fusion. +2.75pp R@10 over base on the full ESCI test set "
        "(15.60% → 18.35%); precomputed product embeddings keep latency sub-100ms."
    )

    with gr.Row():
        query_input = gr.Textbox(
            label="Search query", placeholder="e.g. tom ford lipstick", scale=4
        )
        k_input = gr.Number(label="k", value=10, minimum=1, maximum=50, scale=1)
        mode_input = gr.Dropdown(
            choices=["Base + BoD ensemble rerank", "Fine-tuned retrieval"],
            value="Base + BoD ensemble rerank",
            label="Right-column mode",
            scale=2,
        )
        search_btn = gr.Button("Search", scale=1)

    spec_output = gr.Markdown()
    with gr.Row():
        right_output = gr.Markdown()
        base_output = gr.Markdown()
    neighbor_output = gr.Markdown(label="Similar queries")

    search_btn.click(
        search_fn,
        [query_input, k_input, mode_input],
        [right_output, base_output, spec_output, neighbor_output],
    )
    query_input.submit(
        search_fn,
        [query_input, k_input, mode_input],
        [right_output, base_output, spec_output, neighbor_output],
    )

demo.launch()
