#!/usr/bin/env python3
"""
Evaluate fine-tuned query model against ESCI ground truth.

Loads the fine-tuned model, encodes ESCI queries, retrieves from FAISS,
and compares against human E/S/C/I labels.

Usage:
    python eval_model.py query_model/
    python eval_model.py query_model/ --base  # also eval base model for comparison
"""

import argparse
import json
import os
import time

import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from utils import fmt_duration

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def eval_model(model, model_name, index, titles, esci_df, k=50):
    """Evaluate a model on ESCI queries."""
    title_set = set(titles)

    # Get unique queries
    queries = esci_df["query"].unique().tolist()
    print(f"  Encoding {len(queries):,} queries...", flush=True)
    t0 = time.time()
    q_vecs = model.encode(
        queries, normalize_embeddings=True, batch_size=256, show_progress_bar=True
    )
    print(f"  Encoded in {fmt_duration(time.time() - t0)}", flush=True)

    # Retrieve top-K from FAISS for each query
    print(f"  Retrieving top-{k} from FAISS...", flush=True)
    t0 = time.time()
    q_vecs = np.array(q_vecs, dtype=np.float32)
    faiss.normalize_L2(q_vecs)
    D, I = index.search(q_vecs, k)
    print(f"  Retrieved in {fmt_duration(time.time() - t0)}", flush=True)

    # Build retrieved titles per query
    query_to_retrieved = {}
    for qi, query in enumerate(queries):
        retrieved = []
        for _dist, idx in zip(D[qi], I[qi]):
            if idx >= 0:
                retrieved.append(titles[idx])
        query_to_retrieved[query] = retrieved

    # Match against ESCI labels
    # For each ESCI query-product pair where product is in our index,
    # check if it was retrieved
    esci_in_index = esci_df[esci_df["product_title"].isin(title_set)]

    # Convert retrieved lists to sets for fast lookup
    query_to_retrieved_set = {q: set(titles) for q, titles in query_to_retrieved.items()}

    results = {
        "Exact": {"retrieved": 0, "total": 0},
        "Substitute": {"retrieved": 0, "total": 0},
        "Complement": {"retrieved": 0, "total": 0},
        "Irrelevant": {"retrieved": 0, "total": 0},
    }

    for _, row in esci_in_index.iterrows():
        label = row["esci_label"]
        query = row["query"]
        title = row["product_title"]
        results[label]["total"] += 1
        if query in query_to_retrieved_set and title in query_to_retrieved_set[query]:
            results[label]["retrieved"] += 1

    # Print results
    print(f"\n  === {model_name} ESCI Evaluation (top-{k}) ===")
    print(f"  {'Label':<12} {'Retrieved':>10} {'Total':>8} {'Rate':>8}")
    print(f"  {'-' * 42}")
    for label in ["Exact", "Substitute", "Complement", "Irrelevant"]:
        r = results[label]
        rate = r["retrieved"] / r["total"] if r["total"] > 0 else 0
        print(f"  {label:<12} {r['retrieved']:>10,} {r['total']:>8,} {rate:>8.1%}")

    # Aggregate metrics
    relevant_retrieved = results["Exact"]["retrieved"] + results["Substitute"]["retrieved"]
    relevant_total = results["Exact"]["total"] + results["Substitute"]["total"]
    irrelevant_retrieved = results["Complement"]["retrieved"] + results["Irrelevant"]["retrieved"]
    irrelevant_total = results["Complement"]["total"] + results["Irrelevant"]["total"]

    total_retrieved = relevant_retrieved + irrelevant_retrieved
    precision = relevant_retrieved / total_retrieved if total_retrieved > 0 else 0
    recall = relevant_retrieved / relevant_total if relevant_total > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0

    print(f"\n  Relevant (E+S) recall: {recall:.1%} ({relevant_retrieved:,}/{relevant_total:,})")
    irr_rate = irrelevant_retrieved / irrelevant_total if irrelevant_total > 0 else 0
    print(
        f"  Complement+Irrelevant retrieval rate: {irr_rate:.1%} ({irrelevant_retrieved:,}/{irrelevant_total:,})"
    )
    print(f"  Precision (among retrieved): {precision:.1%}")
    print(f"  F1: {f1:.1%}")

    comp = results["Complement"]
    complement_rate = comp["retrieved"] / comp["total"] if comp["total"] > 0 else 0

    return {
        "model": model_name,
        "relevant_recall": recall,
        "complement_rate": complement_rate,
        "precision": precision,
        "f1": f1,
    }


def sanity_check(model, model_name, index, titles):
    """Quick sanity check with known queries."""
    test_queries = [
        "hp laptop",
        "iphone case",
        "nvidia rtx 3080",
        "wireless keyboard",
        "acoustic guitar",
        "samsung galaxy s21",
        "yoga mat",
        "jigsaw puzzle",
    ]

    print(f"\n  === {model_name} Sanity Check ===")
    q_vecs = model.encode(test_queries, normalize_embeddings=True)
    q_vecs = np.array(q_vecs, dtype=np.float32)
    faiss.normalize_L2(q_vecs)
    D, I = index.search(q_vecs, 5)

    for qi, query in enumerate(test_queries):
        print(f"\n  {query}:")
        for rank, (dist, idx) in enumerate(zip(D[qi], I[qi])):
            if idx >= 0:
                cos = 1 - dist / 2
                print(f"    {rank + 1}. [{cos:.3f}] {titles[idx][:65]}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir", help="Fine-tuned model directory")
    parser.add_argument(
        "--base", action="store_true", help="Also evaluate base model for comparison"
    )
    parser.add_argument("--k", type=int, default=50, help="Top-K for retrieval (default: 50)")
    parser.add_argument(
        "--base-model",
        default="all-MiniLM-L6-v2",
        help="Base embedding model (default: all-MiniLM-L6-v2)",
    )
    args = parser.parse_args()

    # Load FAISS index (once)
    print("Loading FAISS index...", flush=True)
    index_path = os.path.join(SCRIPT_DIR, "combined_index")
    index = faiss.read_index(os.path.join(index_path, "index.faiss"))
    index.hnsw.efSearch = 128
    with open(os.path.join(index_path, "titles.json")) as f:
        titles = json.load(f)
    print(f"  {index.ntotal:,} products", flush=True)

    # Load ESCI
    print("Loading ESCI...", flush=True)
    esci_path = os.path.join(SCRIPT_DIR, "esci", "train.parquet")
    df = pd.read_parquet(
        esci_path, columns=["query", "product_title", "esci_label", "product_locale"]
    )
    df = df[(df["product_locale"] == "us") & df["product_title"].notna()].copy()
    print(f"  {len(df):,} US pairs, {df['query'].nunique():,} queries", flush=True)

    # Evaluate fine-tuned model
    print(f"\nLoading fine-tuned model from {args.model_dir}...", flush=True)
    ft_model = SentenceTransformer(args.model_dir)
    sanity_check(ft_model, "Fine-tuned", index, titles)
    ft_results = eval_model(ft_model, "Fine-tuned", index, titles, df, k=args.k)
    del ft_model

    # Optionally evaluate base model
    if args.base:
        print(f"\nLoading base model {args.base_model}...", flush=True)
        base_model = SentenceTransformer(args.base_model)
        sanity_check(base_model, "Base MiniLM", index, titles)
        base_results = eval_model(base_model, "Base MiniLM", index, titles, df, k=args.k)

        # Comparison
        print("\n  === COMPARISON ===")
        print(f"  {'Metric':<30} {'Base':>10} {'Fine-tuned':>10} {'Delta':>10}")
        print(f"  {'-' * 65}")
        for metric in ["relevant_recall", "complement_rate", "precision", "f1"]:
            b = base_results[metric]
            f = ft_results[metric]
            d = f - b
            print(f"  {metric:<30} {b:>10.1%} {f:>10.1%} {d:>+10.1%}")

    print("\nDone.", flush=True)


if __name__ == "__main__":
    main()
