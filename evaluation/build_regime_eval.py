#!/usr/bin/env python3
"""Build the per-regime eval harness for BoD diagnostics.

Samples multi-token ESCI test queries, auto-tags each with an entity token
(highest-IDF query token) and category tokens (the rest), runs base-FAISS
top-100 retrieval, and bins by AND@100 into easy/mid/hard regimes.

Output: evaluation/regime_queries.jsonl with one record per query:
    {
      "query": "...",
      "tokens": ["..."],
      "entity_token": "...",
      "category_tokens": ["..."],
      "and_at_100": <int>,
      "and_at_10": <int>,
      "regime": "easy" | "mid" | "hard"
    }

Regimes are defined by AND@100:
    easy: >= 70
    mid:  10 <= AND@100 < 70
    hard: <  10
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import json
import math
import os
import random
import time

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
from sentence_transformers import SentenceTransformer

from utils import fmt_duration, tokenize_query

REGIME_THRESHOLDS = {"easy": 70, "mid": 10}  # >=70 easy; 10–69 mid; <10 hard


def regime_for(and_at_100):
    if and_at_100 >= REGIME_THRESHOLDS["easy"]:
        return "easy"
    if and_at_100 >= REGIME_THRESHOLDS["mid"]:
        return "mid"
    return "hard"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index-path", default="combined_index_amazon")
    parser.add_argument("--queries", default="esci_us_data/test_queries.jsonl")
    parser.add_argument("--out", default="evaluation/regime_queries.jsonl")
    parser.add_argument(
        "--sample", type=int, default=1500, help="Random sample of queries to score"
    )
    parser.add_argument("--per-regime", type=int, default=15, help="Final queries per regime")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print(f"loading IDF from {args.index_path}/idf.json...", flush=True)
    with open(os.path.join(args.index_path, "idf.json")) as f:
        idf_data = json.load(f)
    df = idf_data["df"]
    n_docs = idf_data["n_docs"]

    def token_idf(w):
        return math.log((n_docs + 1) / (df.get(w, 0) + 1))

    print(f"loading queries from {args.queries}...", flush=True)
    candidates = []
    with open(args.queries) as f:
        for line in f:
            d = json.loads(line)
            q = d["query"].strip()
            tokens = tokenize_query(q)
            if len(tokens) < 2:
                continue  # skip single-token; no entity/category split possible
            # entity = max-IDF token; category = rest
            scores = [(token_idf(t), t) for t in tokens]
            scores.sort(reverse=True)
            entity_token = scores[0][1]
            category_tokens = [t for t in tokens if t != entity_token]
            candidates.append(
                {
                    "query": q,
                    "tokens": tokens,
                    "entity_token": entity_token,
                    "category_tokens": category_tokens,
                }
            )
    print(f"  multi-token queries: {len(candidates):,}")

    rng = random.Random(args.seed)
    rng.shuffle(candidates)
    sampled = candidates[: args.sample]
    print(f"  sampled {len(sampled):,} for scoring", flush=True)

    # Encode + retrieve
    print("loading base model + FAISS...", flush=True)
    base = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
    import faiss

    faiss_index = faiss.read_index(os.path.join(args.index_path, "index.faiss"))
    if hasattr(faiss_index, "nprobe"):
        faiss_index.nprobe = 32
    with open(os.path.join(args.index_path, "titles.json")) as f:
        titles = json.load(f)

    print(f"encoding {len(sampled)} queries...", flush=True)
    t0 = time.time()
    qs = [c["query"] for c in sampled]
    q_vecs = base.encode(qs, normalize_embeddings=True, batch_size=128).astype(np.float32)
    print(f"  encoded in {fmt_duration(time.time() - t0)}", flush=True)

    print("retrieving top-100...", flush=True)
    t0 = time.time()
    D, I = faiss_index.search(q_vecs, 100)
    print(f"  retrieved in {fmt_duration(time.time() - t0)}", flush=True)

    # Score AND@100 + AND@10
    for i, c in enumerate(sampled):
        e = c["entity_token"]
        cats = c["category_tokens"]
        top100 = [titles[idx].lower() for idx in I[i]]
        top10 = top100[:10]
        c["and_at_100"] = sum(1 for t in top100 if e in t and any(ct in t for ct in cats))
        c["and_at_10"] = sum(1 for t in top10 if e in t and any(ct in t for ct in cats))
        c["regime"] = regime_for(c["and_at_100"])

    # Stratified sample
    by_regime = {"easy": [], "mid": [], "hard": []}
    for c in sampled:
        by_regime[c["regime"]].append(c)
    for r in ("easy", "mid", "hard"):
        rng.shuffle(by_regime[r])
        print(f"  candidates with regime={r}: {len(by_regime[r])}")

    # Diversity: dedupe by entity_token (one query per unique entity) then take per-regime
    final = []
    for r in ("easy", "mid", "hard"):
        seen_entities = set()
        picked = []
        for c in by_regime[r]:
            if c["entity_token"] in seen_entities:
                continue
            seen_entities.add(c["entity_token"])
            picked.append(c)
            if len(picked) >= args.per_regime:
                break
        # If still under quota, fill from remaining (allowing entity duplicates)
        if len(picked) < args.per_regime:
            for c in by_regime[r]:
                if c not in picked:
                    picked.append(c)
                    if len(picked) >= args.per_regime:
                        break
        final.extend(picked)
        print(f"  selected {len(picked)} for regime={r}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        for c in final:
            f.write(json.dumps(c) + "\n")
    print(f"\nwrote {len(final)} queries to {args.out}")

    # Sanity sample
    print("\n=== Sanity sample (5 per regime) ===")
    by_regime_final = {"easy": [], "mid": [], "hard": []}
    for c in final:
        by_regime_final[c["regime"]].append(c)
    for r in ("easy", "mid", "hard"):
        print(f"\n[{r}]")
        for c in by_regime_final[r][:5]:
            print(
                f"  AND@100={c['and_at_100']:>3}  entity={c['entity_token']!r:<12} {c['query'][:60]}"
            )


if __name__ == "__main__":
    main()
