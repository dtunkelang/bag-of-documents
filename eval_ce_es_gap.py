#!/usr/bin/env python3
"""Measure how well a CE separates E (Exact) from S (Substitute) judgments.

For each query in ESCI test_qrels with both E and S products, compute the CE
score for each and report the per-query gap. Aggregate stats describe how
much room the CE has for separating "exactly relevant" from "substitute".

Usage:
    python eval_ce_es_gap.py --ce-model models/esci-us-ce
"""

import argparse
import json
import os
import statistics

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
from sentence_transformers import CrossEncoder

# ESCI relevance: 3=Exact, 2=Substitute, 1=Complement, 0=Irrelevant
GRADE_TO_LABEL = {3: "E", 2: "S", 1: "C", 0: "I"}


def load_qrels(path):
    """Group qrels by query_id: {qid: {grade: [product_ids]}}."""
    by_query = {}
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            qid = r["query_id"]
            grade = r["relevance"]
            by_query.setdefault(qid, {}).setdefault(grade, []).append(r["product_id"])
    return by_query


def load_queries(path):
    out = {}
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            out[d["query_id"]] = d["query"]
    return out


def load_titles(pid_path, title_path):
    """Map product_id -> title from parallel JSON arrays."""
    with open(pid_path) as f:
        pids = json.load(f)
    with open(title_path) as f:
        titles = json.load(f)
    return dict(zip(pids, titles))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ce-model", default="models/esci-us-ce")
    parser.add_argument("--qrels", default="esci_us_data/test_qrels.jsonl")
    parser.add_argument("--queries", default="esci_us_data/test_queries.jsonl")
    parser.add_argument("--product-ids", default="esci_us_data/product_ids.json")
    parser.add_argument("--product-titles", default="esci_us_data/titles.json")
    parser.add_argument("--limit", type=int, default=500, help="Number of queries to score")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    print(f"loading qrels from {args.qrels}...", flush=True)
    qrels = load_qrels(args.qrels)
    print(f"  {len(qrels):,} queries with judgments")

    print(f"loading queries from {args.queries}...", flush=True)
    queries = load_queries(args.queries)

    print(f"loading product titles from {args.product_ids} + {args.product_titles}...", flush=True)
    titles = load_titles(args.product_ids, args.product_titles)
    print(f"  {len(titles):,} products")

    # Filter to queries that have BOTH E and S judgments
    pairs = []  # (q_text, E_title, S_title)
    for qid, grades in qrels.items():
        if 3 not in grades or 2 not in grades:
            continue
        if qid not in queries:
            continue
        q_text = queries[qid]
        # Take one E and one S per query (first available)
        e_pids = [p for p in grades[3] if p in titles]
        s_pids = [p for p in grades[2] if p in titles]
        if not e_pids or not s_pids:
            continue
        pairs.append((q_text, titles[e_pids[0]], titles[s_pids[0]]))
        if len(pairs) >= args.limit:
            break
    print(f"  collected {len(pairs)} (q, E_title, S_title) triples")

    if args.device == "auto":
        device = (
            "mps"
            if torch.backends.mps.is_available()
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )
    else:
        device = args.device
    print(f"loading CE from {args.ce_model} on {device}...", flush=True)
    ce = CrossEncoder(args.ce_model, device=device, max_length=256)

    e_pairs = [(q, e) for q, e, _ in pairs]
    s_pairs = [(q, s) for q, _, s in pairs]

    print("scoring E pairs...", flush=True)
    e_scores = ce.predict(e_pairs, batch_size=32, show_progress_bar=False)
    print("scoring S pairs...", flush=True)
    s_scores = ce.predict(s_pairs, batch_size=32, show_progress_bar=False)

    gaps = [float(e) - float(s) for e, s in zip(e_scores, s_scores)]
    e_wins = sum(1 for g in gaps if g > 0)
    print(f"\n=== E vs S separation on {len(pairs)} judged pairs ===")
    print(f"  E score (mean):  {statistics.mean(map(float, e_scores)):.4f}")
    print(f"  S score (mean):  {statistics.mean(map(float, s_scores)):.4f}")
    print(f"  Gap E-S (mean):  {statistics.mean(gaps):+.4f}")
    print(f"  Gap E-S (median):{statistics.median(gaps):+.4f}")
    print(f"  E > S frequency: {e_wins}/{len(pairs)} = {e_wins / len(pairs):.1%}")
    print(
        f"  Gap quartiles:   q1={statistics.quantiles(gaps, n=4)[0]:+.4f} "
        f"q3={statistics.quantiles(gaps, n=4)[2]:+.4f}"
    )


if __name__ == "__main__":
    main()
