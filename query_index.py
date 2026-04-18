#!/usr/bin/env python3
"""
Query the product nearest-neighbor index.

Supports three modes:
  1. Text query — encode with the fine-tuned query model, find nearest products
  2. Product query — find products similar to a given product title
  3. Interactive — REPL for exploring the product space

Usage:
    python query_index.py combined_index/ "mechanical keyboard"
    python query_index.py combined_index/ "mechanical keyboard" --k 20
    python query_index.py combined_index/ --product "Corsair K70 RGB PRO"
    python query_index.py combined_index/ --interactive
"""

import argparse
import json
import os
import sys

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


def load_index(index_dir):
    """Load the FAISS index and title metadata."""
    index_path = os.path.join(index_dir, "index.faiss")
    meta_path = os.path.join(index_dir, "titles.json")

    index = faiss.read_index(index_path)
    with open(meta_path) as f:
        titles = json.load(f)

    return index, titles


def search(index, titles, query_vec, k=10, ef_search=64):
    """Search the index and return (title, similarity) pairs."""
    query_vec = np.array(query_vec, dtype=np.float32).reshape(1, -1)
    faiss.normalize_L2(query_vec)

    index.hnsw.efSearch = ef_search
    D, I = index.search(query_vec, k)

    results = []
    for dist, idx in zip(D[0], I[0]):
        if idx < 0:
            continue
        sim = 1 - dist / 2  # L2 on unit vectors → cosine: cos = 1 - L2²/2
        results.append((titles[idx], float(sim)))

    return results


def main():
    parser = argparse.ArgumentParser(description="Query the product nearest-neighbor index")
    parser.add_argument("index_dir", help="Directory containing index.faiss and titles.json")
    parser.add_argument("query", nargs="?", default=None, help="Text query to search for")
    parser.add_argument(
        "--product", type=str, default=None, help="Find products similar to this product title"
    )
    parser.add_argument("--k", type=int, default=10, help="Number of results (default: 10)")
    parser.add_argument(
        "--ef-search",
        type=int,
        default=128,
        help="HNSW ef_search — query-time quality (default: 128)",
    )
    parser.add_argument("--interactive", action="store_true", help="Interactive query mode")
    parser.add_argument(
        "--query-model",
        type=str,
        default=None,
        help="Path to fine-tuned query model (default: query_model/ if it exists)",
    )
    parser.add_argument(
        "--base-model",
        default="all-MiniLM-L6-v2",
        help="Base embedding model (default: all-MiniLM-L6-v2)",
    )
    args = parser.parse_args()

    # Load index
    print("Loading index...", file=sys.stderr)
    index, titles = load_index(args.index_dir)
    print(f"Index loaded: {index.ntotal} products", file=sys.stderr)

    # Load embedding model
    query_model_path = args.query_model
    if query_model_path is None:
        # Try retrieval model first, then query model, then base
        script_dir = os.path.dirname(os.path.abspath(__file__))
        for candidate in ["retrieval_model", "query_model"]:
            path = os.path.join(script_dir, candidate)
            if os.path.exists(os.path.join(path, "config.json")):
                query_model_path = path
                break
        if query_model_path is None:
            query_model_path = args.base_model

    print(f"Loading model: {query_model_path}", file=sys.stderr)
    model = SentenceTransformer(query_model_path)

    # Also load base model for comparison
    base_model = None
    if query_model_path != args.base_model:
        base_model = SentenceTransformer(args.base_model)

    def do_query(query_text, k, use_model=None):
        """Encode a text query and search."""
        m = use_model or model
        vec = m.encode(query_text, normalize_embeddings=True)
        results = search(index, titles, vec, k=k, ef_search=args.ef_search)
        return results

    def do_product_query(product_title, k):
        """Encode a product title with the base model and search."""
        m = base_model if base_model else model
        vec = m.encode(product_title, normalize_embeddings=True)
        results = search(index, titles, vec, k=k, ef_search=args.ef_search)
        return results

    # Ensure base model is always available for comparison
    if base_model is None:
        base_model = model  # no fine-tuned model, so they're the same

    def print_results(results, label):
        print(f"\n{label}")
        print(f"{'─' * 70}")
        for i, (title, sim) in enumerate(results):
            print(f"  {i + 1:>3}. [{sim:.3f}] {title[:80]}")
        print()

    # Load bag centroids for query-to-query search
    bags_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bags.jsonl")
    bag_queries = []
    bag_vectors = []
    bag_matrix = None
    if os.path.exists(bags_path):
        with open(bags_path) as f:
            for line in f:
                try:
                    bag = json.loads(line)
                    bag_queries.append(bag["query"])
                    bag_vectors.append(bag["query_vector"])
                except (json.JSONDecodeError, KeyError):
                    pass
        if bag_vectors:
            bag_matrix = np.array(bag_vectors, dtype=np.float32)
            faiss.normalize_L2(bag_matrix)
            print(
                f"Loaded {len(bag_queries)} bag centroids for query matching",
                file=sys.stderr,
            )

    def do_query_neighbors(query_text, k, use_model=None):
        """Find nearest known queries by bag centroid similarity."""
        if bag_matrix is None:
            return []
        m = use_model or model
        vec = m.encode(query_text, normalize_embeddings=True)
        vec = np.array(vec, dtype=np.float32).reshape(1, -1)
        sims = (vec @ bag_matrix.T).flatten()
        top_k = np.argsort(-sims)[:k]
        return [(bag_queries[i], float(sims[i])) for i in top_k]

    has_finetune = query_model_path != args.base_model

    if args.interactive:
        print("\nInteractive mode. Commands:")
        print("  <query>        — find nearest products (shows both models if fine-tuned)")
        print("  p:<title>      — find products similar to a product")
        print("  q:<query>      — find nearest known queries only")
        print("  quit           — exit")
        print()
        while True:
            try:
                line = input("query> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if not line or line.lower() == "quit":
                break
            if line.startswith("p:"):
                product_title = line[2:].strip()
                results = do_product_query(product_title, args.k)
                print_results(results, f"Products similar to: {product_title}")
            elif line.startswith("q:"):
                query_text = line[2:].strip()
                neighbors = do_query_neighbors(query_text, args.k)
                print_results(neighbors, f"Nearest known queries to: {query_text}")
            else:
                results = do_query(line, args.k, use_model=model)
                print_results(results, f"Products (fine-tuned model): {line}")
                if has_finetune:
                    results_base = do_query(line, args.k, use_model=base_model)
                    print_results(results_base, f"Products (base MiniLM): {line}")
                neighbors = do_query_neighbors(line, min(args.k, 5))
                if neighbors:
                    print_results(neighbors, "Nearest known queries:")

    elif args.product:
        results = do_product_query(args.product, args.k)
        print_results(results, f"Products similar to: {args.product}")

    elif args.query:
        results = do_query(args.query, args.k, use_model=model)
        print_results(results, f"Products (fine-tuned model): {args.query}")
        if has_finetune:
            results_base = do_query(args.query, args.k, use_model=base_model)
            print_results(results_base, f"Products (base MiniLM): {args.query}")
        neighbors = do_query_neighbors(args.query, min(args.k, 5))
        if neighbors:
            print_results(neighbors, "Nearest known queries:")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
