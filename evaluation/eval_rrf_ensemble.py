#!/usr/bin/env python3
"""Reciprocal Rank Fusion ensemble of base / BoD / HyDE retrievers.

For each test query, computes top-N candidates under each retriever,
then RRF-fuses them. Reports R@10 for each individual method plus
RRF(BoD, HyDE), RRF(base, BoD, HyDE), and the union (each gold doc
that appears in any top-K).

Reads HyDE passages from the cache produced by `eval_hyde.py`. Encodes
the BoD catalog if no cache is supplied. The base catalog vecs cache
is required.

Usage:
    python evaluation/eval_rrf_ensemble.py \\
        --catalog scifact_data/titles.json \\
        --product-ids scifact_data/doc_ids.json \\
        --qrels scifact_data/test_qrels.jsonl --min-relevance 1 \\
        --queries scifact_data/test_queries.jsonl \\
        --base-model all-MiniLM-L6-v2 \\
        --base-vecs scifact_data/base_catalog.vecs.fp16.npy \\
        --bod-model query_model_scifact_bod \\
        --hyde-passages scifact_data/hyde_passages_llama3.1_8b-instruct-q4_K_M.jsonl \\
        --label scifact \\
        --rrf-k 60 \\
        --top-n 100
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


def topn(qv, pv, n):
    """Returns (n_queries, n) array of top-n doc indices for each query."""
    sims = qv @ pv.T
    idx = np.argpartition(-sims, n, axis=1)[:, :n]
    # sort within top-n by sim descending
    rng = np.arange(idx.shape[0])[:, None]
    sorted_within = np.argsort(-sims[rng, idx], axis=1)
    return idx[rng, sorted_within]


def rrf_merge(rankings_list, top_k, rrf_k=60):
    """RRF: for each candidate doc, sum 1/(rrf_k + rank) across all rankings
    it appears in. Returns top-k indices sorted by RRF score."""
    scores = defaultdict(float)
    for rankings in rankings_list:
        for rank, doc_idx in enumerate(rankings):
            scores[int(doc_idx)] += 1.0 / (rrf_k + rank + 1)
    return sorted(scores, key=scores.get, reverse=True)[:top_k]


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
    ap.add_argument("--bod-vecs", default=None, help="optional cached BoD catalog .npy")
    ap.add_argument("--hyde-passages", required=True, help="JSONL from eval_hyde.py")
    ap.add_argument("--label", default="corpus")
    ap.add_argument("--k", type=int, default=10, help="R@k for the final metric")
    ap.add_argument("--top-n", type=int, default=100, help="top-n per method, fused into top-k")
    ap.add_argument("--rrf-k", type=int, default=60, help="RRF damping constant")
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
    n_pos_q = sum(1 for v in pos.values() if v)
    print(
        f"  catalog={len(pids):,}  queries={len(queries):,}  pos-bearing={n_pos_q:,}",
        flush=True,
    )

    # Hypothetical passages keyed by qid
    passages_by_qid = {}
    with open(args.hyde_passages) as f:
        for line in f:
            d = json.loads(line)
            passages_by_qid[d["query_id"]] = d["passage"]
    hyde_passages = [passages_by_qid.get(q, "") for q in qids]
    n_missing = sum(1 for p in hyde_passages if not p)
    if n_missing:
        print(
            f"  WARNING: {n_missing} queries have no HyDE passage (using empty string)", flush=True
        )

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

    print("encoding queries (base + HyDE)...", flush=True)
    base = SentenceTransformer(args.base_model, device=device)
    base_qv = base.encode(
        queries, normalize_embeddings=True, batch_size=256, show_progress_bar=True
    ).astype(np.float32)
    hyde_qv = base.encode(
        hyde_passages, normalize_embeddings=True, batch_size=128, show_progress_bar=True
    ).astype(np.float32)
    del base
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

    print(f"encoding queries with {args.bod_model}...", flush=True)
    bod = SentenceTransformer(args.bod_model, device=device)
    bod_qv = bod.encode(
        queries, normalize_embeddings=True, batch_size=256, show_progress_bar=True
    ).astype(np.float32)
    del bod
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

    print(f"\ngetting top-{args.top_n} per method...", flush=True)
    base_top = topn(base_qv, base_pv, args.top_n)
    bod_top = topn(bod_qv, bod_pv, args.top_n)
    hyde_top = topn(hyde_qv, base_pv, args.top_n)

    print("computing RRF ensembles...", flush=True)

    methods = {
        "base": base_top[:, : args.k],
        "BoD": bod_top[:, : args.k],
        "HyDE": hyde_top[:, : args.k],
    }
    # Ensembles via RRF on the full top_n rankings.
    rrf_bh = []
    rrf_all = []
    for j in range(len(qids)):
        rrf_bh.append(rrf_merge([bod_top[j], hyde_top[j]], args.k, args.rrf_k))
        rrf_all.append(rrf_merge([base_top[j], bod_top[j], hyde_top[j]], args.k, args.rrf_k))
    methods["RRF(BoD,HyDE)"] = np.array(rrf_bh)
    methods["RRF(base,BoD,HyDE)"] = np.array(rrf_all)

    # Union (any method's top-k counts as a hit). Best-case upper bound.
    union = []
    for j in range(len(qids)):
        u = set()
        for top in (base_top[j, : args.k], bod_top[j, : args.k], hyde_top[j, : args.k]):
            u.update(int(x) for x in top)
        union.append(list(u))
    methods["UNION (oracle upper bound)"] = union

    print(f"\nscoring R@{args.k} per method:")
    results = {}
    for name, tops in methods.items():
        n = 0
        hits_sum = 0.0
        for j, qid in enumerate(qids):
            g = pos.get(qid, set())
            if not g:
                continue
            n += 1
            top_set = set(int(x) for x in tops[j])
            hits_sum += len(top_set & g) / len(g)
        results[name] = (hits_sum / n, n)
        print(f"  {name:<30}  R@{args.k}={hits_sum / n:.3f}  (n={n:,})")

    # Compute per-query base-blind subset breakdown for RRF and union vs BoD/HyDE alone.
    base_blind_idxs = [
        j
        for j, qid in enumerate(qids)
        if pos.get(qid) and not (set(int(x) for x in base_top[j, : args.k]) & pos[qid])
    ]
    print(f"\nbase-blind subset (n={len(base_blind_idxs):,}):")
    for name, tops in methods.items():
        if name == "base":
            continue
        rescues = 0
        rescued_frac = 0.0
        for j in base_blind_idxs:
            qid = qids[j]
            g = pos[qid]
            top_set = set(int(x) for x in tops[j])
            hit = len(top_set & g)
            if hit > 0:
                rescues += 1
            rescued_frac += hit / len(g)
        if base_blind_idxs:
            print(
                f"  {name:<30}  rescues {rescues}/{len(base_blind_idxs)} "
                f"({100 * rescues / len(base_blind_idxs):.1f}%)  "
                f"avg-fraction-rescued={rescued_frac / len(base_blind_idxs):.3f}"
            )


if __name__ == "__main__":
    main()
