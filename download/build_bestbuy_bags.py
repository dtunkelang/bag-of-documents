#!/usr/bin/env python3
"""Build BoD training/test artifacts for BestBuy ACM clickthrough data.

Reads bestbuy_acm_data/{product_ids,titles,test_queries,test_qrels}.jsonl
(produced by download/prepare_bestbuy_acm.py) and produces:

    bestbuy_acm_data/bags.jsonl                 BoD training bags (80% of queries)
    bestbuy_acm_data/holdout_qrels.jsonl        20% holdout for retrieval eval
    bestbuy_acm_data/holdout_queries.jsonl      held-out query texts

The bag for query q is the set of distinct SKUs clicked for q in train.csv.
Each bag entry has the format expected by training/finetune_query_model.py:
    {"query": str, "query_vector": [float], "results": [{"title": str}],
     "num_results": int, "specificity": float}

`query_vector` is the L2-normalized mean of clicked-product embeddings under
all-MiniLM-L6-v2 (the BoD centroid).
"""

import argparse
import json
import os
import random

import numpy as np
import torch
from sentence_transformers import SentenceTransformer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="bestbuy_acm_data")
    ap.add_argument("--encoder", default="all-MiniLM-L6-v2")
    ap.add_argument("--test-fraction", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    print(f"loading queries + qrels from {args.data_dir}/...", flush=True)
    queries = {}
    with open(os.path.join(args.data_dir, "test_queries.jsonl")) as f:
        for line in f:
            d = json.loads(line)
            queries[d["query_id"]] = d["query"]
    qrels = {}
    with open(os.path.join(args.data_dir, "test_qrels.jsonl")) as f:
        for line in f:
            r = json.loads(line)
            qrels.setdefault(r["query_id"], []).append(r["product_id"])
    print(
        f"  {len(queries):,} queries, {sum(len(v) for v in qrels.values()):,} qrels rows",
        flush=True,
    )

    with open(os.path.join(args.data_dir, "product_ids.json")) as f:
        pids = json.load(f)
    with open(os.path.join(args.data_dir, "titles.json")) as f:
        titles = json.load(f)
    pid_to_idx = {p: i for i, p in enumerate(pids)}
    print(f"  {len(pids):,} products in catalog", flush=True)

    # 80/20 random split on QUERIES (not on (q, p) pairs).
    rng = random.Random(args.seed)
    qids = sorted(qrels.keys())
    rng.shuffle(qids)
    n_test = int(len(qids) * args.test_fraction)
    test_qids = set(qids[:n_test])
    train_qids = [q for q in qids if q not in test_qids]
    print(
        f"  split: train={len(train_qids):,}  test={len(test_qids):,}  "
        f"(seed={args.seed}, fraction={args.test_fraction})",
        flush=True,
    )

    # Encode all catalog products once (cheap on 53K).
    device = (
        "mps"
        if torch.backends.mps.is_available()
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"\nencoding {len(titles):,} titles with {args.encoder} on {device}...", flush=True)
    model = SentenceTransformer(args.encoder, device=device)
    pv = model.encode(
        titles, normalize_embeddings=True, batch_size=128, show_progress_bar=True
    ).astype(np.float32)
    print(f"  encoded shape={pv.shape}", flush=True)
    np.save(os.path.join(args.data_dir, "base_catalog.vecs.fp16.npy"), pv.astype(np.float16))
    print("  saved base_catalog.vecs.fp16.npy", flush=True)

    # Build bags.jsonl for training queries.
    bags_path = os.path.join(args.data_dir, "bags.jsonl")
    print(f"\nwriting {bags_path}...", flush=True)
    n_bags = 0
    with open(bags_path, "w") as f:
        for qid in train_qids:
            sku_list = qrels[qid]
            idxs = [pid_to_idx[s] for s in sku_list if s in pid_to_idx]
            if len(idxs) < 2:
                continue
            bag_vecs = pv[idxs]
            centroid = bag_vecs.mean(axis=0)
            centroid /= np.linalg.norm(centroid) + 1e-8
            # Specificity = mean cosine of centroid with bag members (a tightness measure).
            specificity = float(np.mean(bag_vecs @ centroid))
            results = [{"title": titles[i]} for i in idxs]
            bag = {
                "query": queries[qid],
                "query_vector": centroid.astype(np.float32).tolist(),
                "results": results,
                "num_results": len(results),
                "specificity": specificity,
            }
            f.write(json.dumps(bag) + "\n")
            n_bags += 1
    print(f"  wrote {n_bags:,} bags", flush=True)

    # Holdout files for evaluation.
    qq = os.path.join(args.data_dir, "holdout_queries.jsonl")
    qr = os.path.join(args.data_dir, "holdout_qrels.jsonl")
    n_q = 0
    n_r = 0
    with open(qq, "w") as fq, open(qr, "w") as fr:
        for qid in sorted(test_qids):
            fq.write(json.dumps({"query_id": qid, "query": queries[qid]}) + "\n")
            n_q += 1
            for sku in qrels[qid]:
                if sku in pid_to_idx:
                    fr.write(
                        json.dumps({"query_id": qid, "product_id": sku, "relevance": 1}) + "\n"
                    )
                    n_r += 1
    print(f"  wrote {n_q:,} holdout queries, {n_r:,} holdout qrels rows", flush=True)


if __name__ == "__main__":
    main()
