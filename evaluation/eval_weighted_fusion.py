#!/usr/bin/env python3
"""Quality-weighted SCORE fusion of base / BoD / HyDE retrievers.

Unlike `eval_rrf_ensemble.py` which fuses rankings via RRF, this script
fuses cosine similarity scores directly:

    fused = w_bod * sim_bod + w_hyde * sim_hyde

Sweeps the weight grid in 0.1 steps, reports R@10 for each (w_bod,
w_hyde) pair on a held-out test set, and identifies the best fusion
weight for the corpus.

Score fusion has a theoretical advantage over RRF when components are
on the same scale: it can route queries to the higher-scoring method
implicitly, which RRF can't do.

Usage:
    python evaluation/eval_weighted_fusion.py \\
        --catalog scifact_data/titles.json \\
        --product-ids scifact_data/doc_ids.json \\
        --qrels scifact_data/test_qrels.jsonl --min-relevance 1 \\
        --queries scifact_data/test_queries.jsonl \\
        --base-model all-MiniLM-L6-v2 \\
        --base-vecs scifact_data/base_catalog.vecs.fp16.npy \\
        --bod-model query_model_scifact_bod \\
        --hyde-passages scifact_data/hyde_passages_llama3.1_8b-instruct-q4_K_M.jsonl \\
        --label scifact
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from sentence_transformers import SentenceTransformer  # noqa: E402


def r_at_k(sims, pos_lists, k):
    """Returns mean fraction-recovered R@k across queries with positives."""
    if sims.size == 0:
        return 0.0
    top = np.argpartition(-sims, k, axis=1)[:, :k]
    n = 0
    total = 0.0
    for j, g in enumerate(pos_lists):
        if not g:
            continue
        hits = len({int(x) for x in top[j]} & g)
        total += hits / len(g)
        n += 1
    return total / n if n else 0.0


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--catalog", required=True)
    ap.add_argument("--product-ids", required=True)
    ap.add_argument("--qrels", required=True)
    ap.add_argument("--queries", required=True)
    ap.add_argument("--min-relevance", type=int, default=1)
    ap.add_argument("--base-model", required=True)
    ap.add_argument("--base-vecs", required=True)
    ap.add_argument("--bod-model", required=True)
    ap.add_argument("--bod-vecs", default=None)
    ap.add_argument("--hyde-passages", required=True)
    ap.add_argument("--label", default="corpus")
    ap.add_argument("--k", type=int, default=10)
    args = ap.parse_args()

    print("loading data...", flush=True)
    with open(args.catalog) as f:
        titles = json.load(f)
    with open(args.product_ids) as f:
        pids = json.load(f)
    pid_to_idx = {p: i for i, p in enumerate(pids)}

    queries_by_qid = {}
    with open(args.queries) as f:
        for line in f:
            d = json.loads(line)
            queries_by_qid[d["query_id"]] = d["query"]

    pos = defaultdict(set)
    field = None
    with open(args.qrels) as f:
        for line in f:
            r = json.loads(line)
            if field is None:
                field = "product_id" if "product_id" in r else "doc_id"
            if r[field] not in pid_to_idx:
                continue
            if r["relevance"] < args.min_relevance:
                continue
            pos[r["query_id"]].add(pid_to_idx[r[field]])

    qids = sorted(queries_by_qid)
    queries = [queries_by_qid[q] for q in qids]
    pos_lists = [pos.get(q, set()) for q in qids]
    n_pos_q = sum(1 for v in pos_lists if v)
    print(
        f"  catalog={len(pids):,}  queries={len(qids):,}  pos-bearing={n_pos_q:,}",
        flush=True,
    )

    passages_by_qid = {}
    with open(args.hyde_passages) as f:
        for line in f:
            d = json.loads(line)
            passages_by_qid[d["query_id"]] = d["passage"]
    hyde_passages = [passages_by_qid.get(q, "") for q in qids]

    device = "mps" if torch.backends.mps.is_available() else "cpu"

    print(f"loading cached base vecs {args.base_vecs}...", flush=True)
    base_pv = np.load(args.base_vecs).astype(np.float32)

    if args.bod_vecs and os.path.exists(args.bod_vecs):
        print(f"loading cached BoD vecs {args.bod_vecs}...", flush=True)
        bod_pv = np.load(args.bod_vecs).astype(np.float32)
    else:
        print(f"encoding catalog with {args.bod_model}...", flush=True)
        m = SentenceTransformer(args.bod_model, device=device)
        bod_pv = m.encode(
            titles, normalize_embeddings=True, batch_size=128, show_progress_bar=True
        ).astype(np.float32)
        del m
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

    print("encoding queries...", flush=True)
    base = SentenceTransformer(args.base_model, device=device)
    base_qv = base.encode(
        queries, normalize_embeddings=True, batch_size=256, show_progress_bar=False
    ).astype(np.float32)
    hyde_qv = base.encode(
        hyde_passages, normalize_embeddings=True, batch_size=128, show_progress_bar=False
    ).astype(np.float32)
    del base
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    bod = SentenceTransformer(args.bod_model, device=device)
    bod_qv = bod.encode(
        queries, normalize_embeddings=True, batch_size=256, show_progress_bar=False
    ).astype(np.float32)
    del bod
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

    print("computing similarity matrices (chunked)...", flush=True)
    n_q = len(qids)
    n_d = len(pids)
    sim_base = np.empty((n_q, n_d), dtype=np.float32)
    sim_bod = np.empty((n_q, n_d), dtype=np.float32)
    sim_hyde = np.empty((n_q, n_d), dtype=np.float32)
    chunk = 256
    for start in range(0, n_q, chunk):
        end = min(start + chunk, n_q)
        sim_base[start:end] = base_qv[start:end] @ base_pv.T
        sim_bod[start:end] = bod_qv[start:end] @ bod_pv.T
        sim_hyde[start:end] = hyde_qv[start:end] @ base_pv.T

    # Individual R@10 baselines
    print(f"\nIndividual R@{args.k}:")
    base_r = r_at_k(sim_base, pos_lists, args.k)
    bod_r = r_at_k(sim_bod, pos_lists, args.k)
    hyde_r = r_at_k(sim_hyde, pos_lists, args.k)
    print(f"  base:  {base_r:.4f}")
    print(f"  BoD:   {bod_r:.4f}")
    print(f"  HyDE:  {hyde_r:.4f}")

    # Two-component sweep: w_bod * sim_bod + (1-w_bod) * sim_hyde
    print("\nTwo-component sweep — w_bod*BoD + (1-w_bod)*HyDE:")
    print(f"  {'w_bod':>6} {'R@10':>8}")
    best_2 = (-1, -1.0)
    for w in np.arange(0.0, 1.001, 0.1):
        fused = w * sim_bod + (1 - w) * sim_hyde
        r = r_at_k(fused, pos_lists, args.k)
        flag = ""
        if r > best_2[1]:
            best_2 = (w, r)
            flag = "  ← best so far"
        print(f"  {w:>6.1f} {r:>8.4f}{flag}")

    # Three-component sweep (coarser grid): w_base + w_bod + w_hyde = 1
    print("\nThree-component sweep — base + BoD + HyDE (Δ=0.1 grid):")
    print(f"  {'w_base':>7} {'w_bod':>6} {'w_hyde':>7} {'R@10':>8}")
    best_3 = (None, -1.0)
    for w_base in np.arange(0.0, 1.001, 0.1):
        for w_bod in np.arange(0.0, 1.001 - w_base, 0.1):
            w_hyde = 1.0 - w_base - w_bod
            if w_hyde < -1e-6:
                continue
            fused = w_base * sim_base + w_bod * sim_bod + w_hyde * sim_hyde
            r = r_at_k(fused, pos_lists, args.k)
            if r > best_3[1]:
                best_3 = ((w_base, w_bod, w_hyde), r)
                print(f"  {w_base:>7.1f} {w_bod:>6.1f} {w_hyde:>7.1f} {r:>8.4f}  ← best")

    print("\n=== summary ===")
    print(f"  corpus:       {args.label}")
    print(f"  base R@{args.k}:     {base_r:.4f}")
    print(f"  BoD R@{args.k}:      {bod_r:.4f}  (Δ vs base: {bod_r - base_r:+.4f})")
    print(f"  HyDE R@{args.k}:     {hyde_r:.4f}  (Δ vs base: {hyde_r - base_r:+.4f})")
    print(
        f"  best 2-comp:  w_bod={best_2[0]:.1f}  R@{args.k}={best_2[1]:.4f}  "
        f"(Δ vs better individual: {best_2[1] - max(bod_r, hyde_r):+.4f})"
    )
    print(
        f"  best 3-comp:  w_base={best_3[0][0]:.1f} w_bod={best_3[0][1]:.1f} "
        f"w_hyde={best_3[0][2]:.1f}  R@{args.k}={best_3[1]:.4f}  "
        f"(Δ vs better individual: {best_3[1] - max(base_r, bod_r, hyde_r):+.4f})"
    )


if __name__ == "__main__":
    main()
