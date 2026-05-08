#!/usr/bin/env python3
"""Compute E@1 (top-1 hit-rate) on the full 1.27M catalog."""

import json
import os
from collections import defaultdict

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

DATA = "bestbuy_acm_full"


def main():
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
            if r["product_id"] in pid_to_idx:
                pos[r["query_id"]].add(pid_to_idx[r["product_id"]])
    gold = [pos.get(q, set()) for q in qids]

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    base_pv = np.load(os.path.join(DATA, "base_catalog.vecs.fp16.npy")).astype(np.float32)
    bod_pv = np.load(os.path.join(DATA, "bod_catalog.vecs.fp16.npy")).astype(np.float32)
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
    base_e1 = bod_e1 = 0
    n_eval = 0
    for start in range(0, len(qids), chunk):
        end = min(start + chunk, len(qids))
        bsim = base_qv[start:end] @ base_pv.T
        b_top1 = bsim.argmax(axis=1)
        del bsim
        dsim = bod_qv[start:end] @ bod_pv.T
        d_top1 = dsim.argmax(axis=1)
        del dsim
        for j, gi in enumerate(range(start, end)):
            g = gold[gi]
            if not g:
                continue
            n_eval += 1
            if int(b_top1[j]) in g:
                base_e1 += 1
            if int(d_top1[j]) in g:
                bod_e1 += 1
    print(f"\nE@1 over {n_eval:,} queries:")
    print(f"  base : {base_e1 / n_eval:.4f}")
    print(f"  BoD  : {bod_e1 / n_eval:.4f}")
    print(f"  Δ    : {(bod_e1 - base_e1) / n_eval:+.4f}")


if __name__ == "__main__":
    main()
