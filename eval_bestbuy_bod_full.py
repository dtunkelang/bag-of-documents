#!/usr/bin/env python3
"""Eval base vs BoD on the FULL 1.27M BestBuy catalog (not the 53K subset)."""

import json
import os
from collections import defaultdict

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

DATA = "bestbuy_acm_full"
K = 10


def main():
    print("loading data...", flush=True)
    with open(os.path.join(DATA, "product_ids.json")) as f:
        pids = json.load(f)
    pid_to_idx = {p: i for i, p in enumerate(pids)}
    qids, queries = [], []
    with open(os.path.join(DATA, "holdout_queries.jsonl")) as f:
        for line in f:
            d = json.loads(line)
            qids.append(d["query_id"])
            queries.append(d["query"])
    pos = defaultdict(set)
    with open(os.path.join(DATA, "holdout_qrels.jsonl")) as f:
        for line in f:
            r = json.loads(line)
            if r["product_id"] not in pid_to_idx:
                continue
            pos[r["query_id"]].add(pid_to_idx[r["product_id"]])
    gold = [pos.get(q, set()) for q in qids]
    print(f"  catalog={len(pids):,}  queries={len(queries):,}", flush=True)

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    base_pv = np.load(os.path.join(DATA, "base_catalog.vecs.fp16.npy")).astype(np.float32)
    bod_pv = np.load(os.path.join(DATA, "bod_catalog.vecs.fp16.npy")).astype(np.float32)
    print(f"  vecs loaded: base {base_pv.shape}, bod {bod_pv.shape}", flush=True)

    base = SentenceTransformer("all-MiniLM-L6-v2", device=device)
    base_qv = base.encode(
        queries, normalize_embeddings=True, batch_size=256, show_progress_bar=True
    ).astype(np.float32)
    del base
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

    bod = SentenceTransformer("query_model_bestbuy_bod", device=device)
    bod_qv = bod.encode(
        queries, normalize_embeddings=True, batch_size=256, show_progress_bar=True
    ).astype(np.float32)
    del bod
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

    n_docs = base_pv.shape[0]
    chunk = max(64, int(2.5e8 // n_docs))
    base_hit = bod_hit = 0
    base_frac = bod_frac = 0.0
    n_eval = 0
    n_chunks = (len(qids) + chunk - 1) // chunk
    for ci, start in enumerate(range(0, len(qids), chunk)):
        end = min(start + chunk, len(qids))
        bsim = base_qv[start:end] @ base_pv.T
        b_topk = np.argpartition(-bsim, K, axis=1)[:, :K]
        del bsim
        dsim = bod_qv[start:end] @ bod_pv.T
        d_topk = np.argpartition(-dsim, K, axis=1)[:, :K]
        # Sort top-k for E@1.
        for j, gi in enumerate(range(start, end)):
            g = gold[gi]
            if not g:
                continue
            n_eval += 1
            b_set = {int(x) for x in b_topk[j]}
            d_set = {int(x) for x in d_topk[j]}
            if b_set & g:
                base_hit += 1
            if d_set & g:
                bod_hit += 1
            base_frac += len(b_set & g) / len(g)
            bod_frac += len(d_set & g) / len(g)
        del dsim
        if (ci + 1) % 5 == 0 or ci + 1 == n_chunks:
            print(f"  matmul: {ci + 1}/{n_chunks} chunks", flush=True)

    print("\nFULL CATALOG eval (n_eval=" + f"{n_eval:,}" + "):")
    print(f"  base R@10 (binary hit-rate)    : {base_hit / n_eval:.4f}")
    print(f"  BoD  R@10 (binary hit-rate)    : {bod_hit / n_eval:.4f}")
    print(f"  base R@10 (fraction-recovered) : {base_frac / n_eval:.4f}")
    print(f"  BoD  R@10 (fraction-recovered) : {bod_frac / n_eval:.4f}")


if __name__ == "__main__":
    main()
