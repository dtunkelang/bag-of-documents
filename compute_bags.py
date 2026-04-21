#!/usr/bin/env python3
"""
Recompute bags using hybrid retrieval + cross-encoder scoring.

For each query:
1. Hybrid retrieval: keyword (tantivy AND) + embedding (FAISS)
2. Cross-encoder scoring: score all candidates, threshold at --ce-threshold
3. Compute bag centroid (mean of normalized vectors) and specificity

Usage:
    python compute_bags.py queries.jsonl bags.jsonl --ce-rerank models/esci-cross-encoder
"""

import argparse
import gc
import json
import os
import random
import signal
import sys
import time

import faiss
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from utils import fmt_duration, generate_keyword_combos, tokenize_query

# --- Configuration ---
EMBED_MODEL_ID = "all-MiniLM-L6-v2"
INDEX_DIR = "combined_index"
K = 50  # max bag members
CANDIDATE_MULTIPLIER = 4  # fetch K*this from each source


# --- Signal handling ---

stop_requested = False


def handle_signal(sig, frame):
    global stop_requested
    if stop_requested:
        sys.exit(1)
    stop_requested = True
    print("\nStopping after current query...", file=sys.stderr)


signal.signal(signal.SIGTERM, handle_signal)
signal.signal(signal.SIGINT, handle_signal)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="Queries JSONL file")
    parser.add_argument("output", help="Bags JSONL output file")
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Randomize query order before applying --limit (for random sampling)",
    )
    parser.add_argument(
        "--sort-queries",
        action="store_true",
        help="Sort pending queries by embedding similarity for FAISS cache locality. "
        "Requires encoding all pending queries up front; on memory-constrained "
        "machines (16 GB) this can trigger swap thrashing. Off by default.",
    )
    parser.add_argument("--limit", type=int, default=0, help="Only process N queries (0=all)")
    parser.add_argument(
        "--model", default=EMBED_MODEL_ID, help=f"Embedding model (default: {EMBED_MODEL_ID})"
    )
    parser.add_argument(
        "--ce-rerank",
        required=True,
        help="Cross-encoder model path (e.g., models/esci-cross-encoder)",
    )
    parser.add_argument(
        "--ce-threshold",
        type=float,
        default=0.3,
        help="Minimum CE score to include a result in the bag (default: 0.3)",
    )
    args = parser.parse_args()

    # Load queries from JSONL
    queries = []
    with open(args.input) as f:
        for line in f:
            data = json.loads(line)
            queries.append(data["query"])

    if args.limit:
        if args.shuffle:
            random.shuffle(queries)
        queries = queries[: args.limit]

    # Resume: read completed queries from output file
    done_queries = set()
    if os.path.exists(args.output):
        with open(args.output) as f:
            for line in f:
                bag = json.loads(line)
                done_queries.add(bag["query"])

    pending = [q for q in queries if q not in done_queries]

    print(f"Queries: {len(queries)}")
    print(f"Already done: {len(done_queries)}")
    print(f"Pending: {len(pending)}")

    if not pending:
        print("All bags computed.")
        return

    # Optionally sort pending queries by embedding similarity for FAISS cache locality.
    # Off by default because the up-front encoding of all pending queries can consume
    # hundreds of MB and trigger swap thrashing on 16 GB machines.
    model_id = args.model
    if len(pending) > 100 and args.sort_queries:
        print("Sorting queries by embedding similarity for cache locality...")
        _sort_model = SentenceTransformer(model_id)
        _sort_vecs = _sort_model.encode(
            pending,
            normalize_embeddings=True,
            batch_size=256,
            show_progress_bar=True,
        )
        # Sort by first principal component (cheap 1D ordering)
        mean_vec = _sort_vecs.mean(axis=0)
        projections = _sort_vecs @ mean_vec
        order = np.argsort(projections)
        pending = [pending[i] for i in order]
        del _sort_model, _sort_vecs, projections, order
        print("  Queries sorted.")

    # Load models and indexes
    script_dir = os.path.dirname(os.path.abspath(__file__))
    index_path = os.path.join(script_dir, INDEX_DIR)

    print(f"Loading embedding model ({model_id})...")
    embed_model = SentenceTransformer(model_id)

    print("Loading FAISS index...")
    faiss_index = faiss.read_index(os.path.join(index_path, "index.faiss"))
    with open(os.path.join(index_path, "titles.json")) as f:
        all_titles = json.load(f)
    faiss_index.hnsw.efSearch = 64  # lower = fewer page faults, minimal recall loss
    print(f"  FAISS: {faiss_index.ntotal:,} products (efSearch=64)")

    print("Loading tantivy index...")
    import tantivy

    schema_builder = tantivy.SchemaBuilder()
    schema_builder.add_text_field("title", stored=True)
    schema = schema_builder.build()
    tantivy_index = tantivy.Index(schema, path=os.path.join(index_path, "tantivy_index"))
    tantivy_index.reload()
    tv_searcher = tantivy_index.searcher()
    print("  Tantivy index loaded")

    # Cross-encoder for scoring candidates
    from sentence_transformers import CrossEncoder

    if torch.backends.mps.is_available():
        ce_device = "mps"
    elif torch.cuda.is_available():
        ce_device = "cuda"
    else:
        ce_device = "cpu"
    print(f"Loading cross-encoder from {args.ce_rerank} (device={ce_device})...")
    ce_model = CrossEncoder(args.ce_rerank, device=ce_device)
    print("  Cross-encoder loaded")

    print("All models loaded.")

    CE_BATCH = 32  # Number of queries to batch for CE inference

    def retrieve_and_score(query):
        """Step 1: retrieve candidates via keyword + FAISS."""
        words = tokenize_query(query)

        # --- Step 1a: Keyword retrieval (tantivy AND with relaxation) ---
        seen_titles = set()
        raw_candidates = []

        for n_required, combos in generate_keyword_combos(words):
            for combo in combos:
                try:
                    parsed = tantivy_index.parse_query(" AND ".join(combo), ["title"])
                    results = tv_searcher.search(parsed, limit=K * CANDIDATE_MULTIPLIER)
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

        # --- Step 1b: FAISS embedding retrieval ---
        q_vec = embed_model.encode(query, normalize_embeddings=True)
        q_vec = np.array(q_vec, dtype=np.float32).reshape(1, -1)
        faiss.normalize_L2(q_vec)
        D, I = faiss_index.search(q_vec, K * CANDIDATE_MULTIPLIER)
        for _dist, idx in zip(D[0], I[0]):
            if idx >= 0:
                title = all_titles[idx]
                if title not in seen_titles:
                    seen_titles.add(title)
                    raw_candidates.append(title)

        n_candidates = len(raw_candidates)

        return {
            "query": query,
            "q_vec": q_vec,
            "n_candidates": n_candidates,
            "candidates": raw_candidates,
        }

    def finalize_batch(batch_items):
        """Step 2-3: Batched CE scoring + centroid computation."""
        # --- Batched CE scoring ---
        all_ce_pairs = []
        pair_offsets = []  # (start, end) for each item
        for item in batch_items:
            start = len(all_ce_pairs)
            all_ce_pairs.extend((item["query"], t) for t in item["candidates"])
            pair_offsets.append((start, len(all_ce_pairs)))

        all_ce_scores = ce_model.predict(all_ce_pairs) if all_ce_pairs else np.array([])

        for item, (start, end) in zip(batch_items, pair_offsets):
            if not item["candidates"]:
                item["relevant_titles"] = []
                item["relevant_scores"] = []
                continue
            scores = all_ce_scores[start:end]
            ce_ranked = sorted(zip(scores, item["candidates"]), key=lambda x: -x[0])
            passing = [(float(s), t) for s, t in ce_ranked if s >= args.ce_threshold][:K]
            item["relevant_titles"] = [t for _, t in passing]
            item["relevant_scores"] = [s for s, _ in passing]

        # --- Batched embedding encode for centroids ---
        all_rel_titles = []
        title_offsets = []
        for item in batch_items:
            start = len(all_rel_titles)
            all_rel_titles.extend(item["relevant_titles"])
            title_offsets.append((start, len(all_rel_titles)))

        if all_rel_titles:
            all_rel_vectors = embed_model.encode(all_rel_titles, normalize_embeddings=True)

        bags = []
        for item, (start, end) in zip(batch_items, title_offsets):
            if item["relevant_titles"]:
                rel_vectors = all_rel_vectors[start:end]
                centroid = rel_vectors.mean(axis=0)
                centroid = centroid / np.linalg.norm(centroid)
                specificity = float(np.mean([centroid @ v for v in rel_vectors]))
                bag = {
                    "query": item["query"],
                    "num_results": len(item["relevant_titles"]),
                    "query_vector": centroid.tolist(),
                    "specificity": specificity,
                    "results": [
                        {"title": t, "ce_score": s}
                        for t, s in zip(item["relevant_titles"], item["relevant_scores"])
                    ],
                }
            else:
                bag = {
                    "query": item["query"],
                    "num_results": 0,
                    "query_vector": item["q_vec"][0].tolist(),
                    "specificity": 0,
                    "results": [],
                }
            bags.append((item["n_candidates"], bag))
        return bags

    # Process queries — append bags to JSONL output
    completed = len(done_queries)
    errors = 0
    run_start = time.time()
    bags_this_run = 0
    early_bags = []
    batch_buf = []
    out_file = open(args.output, "a")

    def write_bag(bag, n_candidates):
        nonlocal completed, bags_this_run
        out_file.write(json.dumps(bag, ensure_ascii=False) + "\n")
        out_file.flush()
        completed += 1
        bags_this_run += 1
        if bags_this_run <= 10:
            early_bags.append(bag)
        run_elapsed = time.time() - run_start
        remaining = len(pending) - bags_this_run
        avg_per = run_elapsed / bags_this_run
        eta = fmt_duration(remaining * avg_per)
        print(
            f"[{completed}/{len(queries)}] {bag['query'][:50]}: "
            f"{bag['num_results']}/{n_candidates} relevant "
            f"(ETA {eta})"
        )
        # Early validation after first 10 bags
        if bags_this_run == 10:
            empty = sum(1 for b in early_bags if b["num_results"] == 0)
            mean_size = np.mean([b["num_results"] for b in early_bags])
            mean_spec = np.mean(
                [b["specificity"] for b in early_bags if b["specificity"] > 0] or [0]
            )
            print("\n  === EARLY CHECK (first 10 bags) ===")
            print(f"  Empty bags: {empty}/10")
            print(f"  Mean bag size: {mean_size:.1f}")
            print(f"  Mean specificity: {mean_spec:.3f}")
            ce_quality = []
            for b in early_bags:
                if b["num_results"] < 2:
                    continue
                titles = [r["title"] for r in b["results"][:5]]
                pairs = [(b["query"], t) for t in titles]
                scores = ce_model.predict(pairs)
                ce_quality.append(float(np.mean(scores)))
            if ce_quality:
                print(f"  Mean CE score (top 5 members): {np.mean(ce_quality):.3f}")
            if empty > 5:
                print("  WARNING: >50% empty bags — check retrieval", file=sys.stderr)
            if mean_spec > 0 and mean_spec < 0.5:
                print("  WARNING: Low specificity — bags may be noisy", file=sys.stderr)
            print()

    for query in pending:
        if stop_requested:
            break

        try:
            item = retrieve_and_score(query)
            batch_buf.append(item)
        except Exception as e:
            errors += 1
            print(f"ERROR [{query[:50]}]: {e}", file=sys.stderr)
            continue

        if len(batch_buf) < CE_BATCH:
            continue

        # Flush batch
        try:
            results = finalize_batch(batch_buf)
        except Exception as e:
            errors += len(batch_buf)
            print(f"ERROR [batch]: {e}", file=sys.stderr)
            batch_buf = []
            continue
        batch_buf = []

        # Release per-batch transient state to keep long runs from growing in memory
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

        for n_candidates, bag in results:
            write_bag(bag, n_candidates)

    # Flush remaining batch
    if batch_buf and not stop_requested:
        try:
            results = finalize_batch(batch_buf)
            for n_candidates, bag in results:
                write_bag(bag, n_candidates)
        except Exception as e:
            errors += len(batch_buf)
            print(f"ERROR [final batch]: {e}", file=sys.stderr)

    out_file.close()
    run_elapsed = time.time() - run_start
    print(
        f"\nDone. {completed}/{len(queries)} completed, "
        f"{errors} errors. Elapsed: {fmt_duration(run_elapsed)}."
    )


if __name__ == "__main__":
    main()
