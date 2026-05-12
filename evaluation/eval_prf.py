#!/usr/bin/env python3
"""Embedding-based pseudo-relevance feedback (E-PRF) — "HyDE without the LLM."

For each query q:
  1. Retrieve top-K candidates with base encoder.
  2. Average their doc embeddings → q_prf (centroid of pseudo-relevant docs).
  3. Mix: q_mixed = alpha * q + (1 - alpha) * q_prf (then renormalize).
  4. Retrieve final top-N with q_mixed.

This is Rocchio relevance feedback adapted to dense retrieval — strictly
deterministic, no LLM, free at inference. The critical test of HyDE's
value: if E-PRF matches HyDE's rescue rate, the LLM isn't providing
information beyond what's already in the index.

Sweeps K in {3, 5, 10} and alpha in {0.0, 0.3, 0.5, 0.7, 1.0}.

Usage:
    python evaluation/eval_prf.py \\
        --catalog scifact_data/titles.json \\
        --product-ids scifact_data/doc_ids.json \\
        --qrels scifact_data/test_qrels.jsonl --min-relevance 1 \\
        --queries scifact_data/test_queries.jsonl \\
        --base-model all-MiniLM-L6-v2 \\
        --base-vecs scifact_data/base_catalog.vecs.fp16.npy \\
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
    """Mean fraction-recovered R@k over queries with positives."""
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
    ap.add_argument("--label", default="corpus")
    ap.add_argument("--k", type=int, default=10, help="R@k metric")
    ap.add_argument(
        "--prf-ks",
        default="3,5,10",
        help="comma-separated feedback depths to sweep",
    )
    ap.add_argument(
        "--alphas",
        default="0.0,0.3,0.5,0.7,1.0",
        help="mixing weights (alpha=1.0 is pure base, 0.0 is pure PRF)",
    )
    ap.add_argument(
        "--write-per-query",
        action="store_true",
        help="write per-query JSONL for best (K, alpha)",
    )
    args = ap.parse_args()

    print("loading data...", flush=True)
    with open(args.catalog) as f:
        titles = json.load(f)
    _ = titles  # not used after pids mapping but kept to validate loadability
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
    n_pos = sum(1 for v in pos_lists if v)
    print(
        f"  catalog={len(pids):,}  queries={len(qids):,}  pos-bearing={n_pos:,}",
        flush=True,
    )

    print(f"loading cached base vecs {args.base_vecs}...", flush=True)
    base_pv = np.load(args.base_vecs).astype(np.float32)

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"encoding queries with {args.base_model}...", flush=True)
    enc = SentenceTransformer(args.base_model, device=device)
    base_qv = enc.encode(
        queries, normalize_embeddings=True, batch_size=256, show_progress_bar=False
    ).astype(np.float32)
    del enc

    print("computing base similarity matrix...", flush=True)
    sim_base = base_qv @ base_pv.T  # (n_q, n_d)

    prf_ks = [int(x) for x in args.prf_ks.split(",")]
    alphas = [float(x) for x in args.alphas.split(",")]
    max_k = max(prf_ks)

    # Top-max_k per query, sorted by score desc; reused across all PRF depths.
    print(f"getting top-{max_k} per query for PRF feedback...", flush=True)
    top_idx = np.argpartition(-sim_base, max_k, axis=1)[:, :max_k]
    rng = np.arange(len(qids))[:, None]
    top_sorted = np.argsort(-sim_base[rng, top_idx], axis=1)
    top_idx = top_idx[rng, top_sorted]

    base_r = r_at_k(sim_base, pos_lists, args.k)
    print(f"\nBaseline R@{args.k}:")
    print(f"  base (raw query): {base_r:.4f}")

    print("\nPRF sweep — K (feedback depth) x alpha (alpha=1 is base, 0 is pure PRF):")
    print(f"  {'K':>3} {'alpha':>6} {'R@10':>8}")
    best = (None, -1.0)
    for K in prf_ks:
        prf_docs = base_pv[top_idx[:, :K]]
        q_prf = prf_docs.mean(axis=1)
        norms = np.linalg.norm(q_prf, axis=1, keepdims=True)
        norms[norms < 1e-12] = 1.0
        q_prf = q_prf / norms

        for alpha in alphas:
            q_mixed = alpha * base_qv + (1.0 - alpha) * q_prf
            norms = np.linalg.norm(q_mixed, axis=1, keepdims=True)
            norms[norms < 1e-12] = 1.0
            q_mixed = q_mixed / norms
            sim_mixed = q_mixed @ base_pv.T
            r = r_at_k(sim_mixed, pos_lists, args.k)
            flag = ""
            if r > best[1]:
                best = ((K, alpha), r)
                flag = "  <- best"
            print(f"  {K:>3} {alpha:>6.1f} {r:>8.4f}{flag}")

    print("\n=== summary ===")
    print(f"  corpus:       {args.label}")
    print(f"  base R@{args.k}:     {base_r:.4f}")
    bk, ba = best[0]
    print(
        f"  best PRF:     K={bk}  alpha={ba:.1f}  R@{args.k}={best[1]:.4f}  "
        f"(delta vs base: {best[1] - base_r:+.4f})"
    )

    if args.write_per_query:
        K, alpha = best[0]
        prf_docs = base_pv[top_idx[:, :K]]
        q_prf = prf_docs.mean(axis=1)
        norms = np.linalg.norm(q_prf, axis=1, keepdims=True)
        norms[norms < 1e-12] = 1.0
        q_prf = q_prf / norms
        q_mixed = alpha * base_qv + (1.0 - alpha) * q_prf
        norms = np.linalg.norm(q_mixed, axis=1, keepdims=True)
        norms[norms < 1e-12] = 1.0
        q_mixed = q_mixed / norms
        sim_mixed = q_mixed @ base_pv.T
        top_mixed = np.argpartition(-sim_mixed, args.k, axis=1)[:, : args.k]
        top_base = np.argpartition(-sim_base, args.k, axis=1)[:, : args.k]
        out_path = os.path.join(
            os.path.dirname(args.queries) or ".",
            f"prf_per_query_{args.label}.jsonl",
        )
        with open(out_path, "w") as f:
            for j, qid in enumerate(qids):
                g = pos_lists[j]
                if not g:
                    continue
                base_hit = len({int(x) for x in top_base[j]} & g)
                prf_hit = len({int(x) for x in top_mixed[j]} & g)
                f.write(
                    json.dumps(
                        {
                            "query_id": qid,
                            "n_gold": len(g),
                            "base_hit": base_hit,
                            "prf_hit": prf_hit,
                            "prf_k": K,
                            "prf_alpha": alpha,
                        }
                    )
                    + "\n"
                )
        print(f"  per-query results at {out_path}", flush=True)


if __name__ == "__main__":
    main()
