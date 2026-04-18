#!/usr/bin/env python3
"""
Gradio demo for bag-of-documents product search.

Designed for HuggingFace Spaces. Provides:
- Side-by-side comparison: fine-tuned model vs base MiniLM
- Specificity prediction via kNN on bag centroids
- Similar queries from bag centroids

Usage:
    python app.py
"""

import argparse
import json
import os

import faiss
import gradio as gr
import numpy as np
from sentence_transformers import SentenceTransformer

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def load_resources():
    """Load all models, index, and data at startup."""
    print("Loading resources...")

    index_dir = os.path.join(SCRIPT_DIR, "combined_index")
    index = faiss.read_index(os.path.join(index_dir, "index.faiss"))
    print(f"  Product index: {index.ntotal} products")
    with open(os.path.join(index_dir, "titles.json")) as f:
        titles = json.load(f)

    base_model_name = os.environ.get("BASE_MODEL", "all-MiniLM-L6-v2")
    ret_path = os.path.join(SCRIPT_DIR, "retrieval_model")
    if os.path.exists(os.path.join(ret_path, "config.json")):
        retrieval_model = SentenceTransformer(ret_path)
        print(f"  Retrieval model: {ret_path}")
    else:
        retrieval_model = SentenceTransformer(base_model_name)
        print(f"  Retrieval model: {base_model_name} (no fine-tuned model found)")

    base_model = SentenceTransformer(base_model_name)
    print(f"  Base model: {base_model_name}")

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

    return {
        "index": index,
        "titles": titles,
        "retrieval_model": retrieval_model,
        "base_model": base_model,
        "bag_queries": bag_queries,
        "bag_matrix": bag_matrix,
        "bag_specs": bag_specs,
    }


def search_products(query, resources, model_key="retrieval_model", k=50, ef_search=128):
    """Search products by query text."""
    model = resources[model_key]
    index = resources["index"]
    titles = resources["titles"]

    vec = model.encode(query, normalize_embeddings=True)
    vec = np.array(vec, dtype=np.float32).reshape(1, -1)
    faiss.normalize_L2(vec)

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
        results.append((titles[idx], round(sim, 4)))

    # Deduplicate
    seen = {}
    for title, score in results:
        if title not in seen or score > seen[title]:
            seen[title] = score
    return sorted(seen.items(), key=lambda x: -x[1])


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
            (
                resources["bag_queries"][i],
                round(float(resources["bag_specs"][i]), 3),
                round(float(sims[i]), 3),
            )
        )

    weights = sims[top_k]
    specificity = float(np.average(resources["bag_specs"][top_k], weights=weights))
    return round(specificity, 3), neighbors


def format_results(results, header):
    """Format results as an HTML table with a header."""
    if not results:
        return "No results"
    nw = "white-space:nowrap;padding:2px 6px"
    rows = []
    for i, (title, score) in enumerate(results):
        rows.append(
            f"<tr><td style='{nw};text-align:center'>{i + 1}</td>"
            f"<td style='{nw};text-align:center'>{score:.3f}</td>"
            f"<td style='padding:2px 6px'>{title[:80]}</td></tr>"
        )
    return (
        f"<h3>{header}</h3>"
        "<table style='width:100%;font-size:14px;border-collapse:collapse'>"
        f"<tr style='border-bottom:2px solid #ccc'>"
        f"<th style='{nw}'>#</th><th style='{nw}'>Sim</th>"
        f"<th style='padding:2px 6px'>Title</th></tr>" + "".join(rows) + "</table>"
    )


def format_neighbors(neighbors):
    """Format similar queries as markdown."""
    if not neighbors:
        return ""
    lines = ["| Similarity | Specificity | Query |"]
    lines.append("|-----------|------------|-------|")
    for query, spec, sim in neighbors:
        lines.append(f"| {sim:.3f} | {spec:.3f} | {query} |")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Gradio demo for bag-of-documents search")
    parser.add_argument("--base-model", default="all-MiniLM-L6-v2", help="Base model")
    parser.add_argument("--port", type=int, default=7860, help="Port (default: 7860)")
    args = parser.parse_args()

    os.environ["BASE_MODEL"] = args.base_model

    resources = load_resources()

    def search_fn(query, k):
        if not query.strip():
            return "", "", "", ""

        k = int(k)
        specificity, neighbors = predict_specificity(query, resources)

        ft_results = search_products(query, resources, "retrieval_model", k=k)
        base_results = search_products(query, resources, "base_model", k=k)

        spec_text = f"**Specificity:** {specificity:.3f}" if specificity else ""
        ft_text = format_results(ft_results, "Bag of Documents Model")
        base_text = format_results(base_results, "Base MiniLM Model")
        neighbor_text = format_neighbors(neighbors)

        return ft_text, base_text, spec_text, neighbor_text

    with gr.Blocks(title="Bag-of-Documents Search") as demo:
        gr.Markdown(
            "# Bag-of-Documents Search\n"
            "6M Amazon products (1996–2023, McAuley Lab) · "
            "75K queries · Fine-tuned MiniLM vs base MiniLM"
        )

        with gr.Row():
            query = gr.Textbox(label="Search query", placeholder="Search products...", scale=4)
            k = gr.Number(label="k", value=50, minimum=1, maximum=200, scale=1)
            search_btn = gr.Button("Search", scale=1)

        spec_output = gr.Markdown(label="Specificity")

        with gr.Row():
            ft_output = gr.Markdown()
            base_output = gr.Markdown()

        neighbor_output = gr.Markdown(label="Similar queries")

        search_btn.click(
            fn=search_fn,
            inputs=[query, k],
            outputs=[ft_output, base_output, spec_output, neighbor_output],
        )
        query.submit(
            fn=search_fn,
            inputs=[query, k],
            outputs=[ft_output, base_output, spec_output, neighbor_output],
        )

    demo.launch(server_port=args.port)


if __name__ == "__main__":
    main()
