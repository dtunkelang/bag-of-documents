#!/usr/bin/env python3
"""Test whether base-side confidence signals predict the BoD specialization tax.

Hypothesis: BoD hurts on base-perfect queries (every corpus shows -6 to -18pp
on the 1.0 bucket). If a cheap query-time signal computed from base alone
cleanly separates base-perfect queries from base-blind ones, a router can
fire BoD only on uncertain queries — capturing the rescue while skipping
the tax.

Signals tested (all derived from base's own top-10 over the catalog):
  - top1            top-1 cosine similarity
  - top1_minus_top2 margin between top-1 and top-2
  - mean_top10      mean cosine over the top-10
  - top1_minus_top10 spread between top-1 and top-10

For each signal, report:
  - correlation with base R@10 hits
  - oracle router lift (if we set the optimal threshold on this run)

If the strongest signal yields a Pareto-better router (base on top quantile,
BoD on bottom quantile), the tax is avoidable. If no signal cleanly
separates the buckets, the tax is intrinsic.

Runs on BestBuy by default (largest n in our calibration set).
"""

import argparse
import json
import os
from collections import defaultdict

import numpy as np
import torch
from sentence_transformers import SentenceTransformer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="bestbuy_acm_data")
    ap.add_argument("--queries", default="holdout_queries.jsonl")
    ap.add_argument("--qrels", default="holdout_qrels.jsonl")
    ap.add_argument("--min-relevance", type=int, default=1)
    ap.add_argument("--base-model", default="all-MiniLM-L6-v2")
    ap.add_argument("--bod-model", default="query_model_bestbuy_bod")
    ap.add_argument("--base-vecs", default="base_catalog.vecs.fp16.npy")
    ap.add_argument("--k", type=int, default=10)
    args = ap.parse_args()

    print("loading data...", flush=True)
    with open(os.path.join(args.data_dir, "titles.json")) as f:
        titles = json.load(f)
    with open(os.path.join(args.data_dir, "product_ids.json")) as f:
        pids = json.load(f)
    pid_to_idx = {p: i for i, p in enumerate(pids)}
    qids, queries = [], []
    with open(os.path.join(args.data_dir, args.queries)) as f:
        for line in f:
            d = json.loads(line)
            qids.append(d["query_id"])
            queries.append(d["query"])
    pos = defaultdict(set)
    field = None
    with open(os.path.join(args.data_dir, args.qrels)) as f:
        for line in f:
            r = json.loads(line)
            if field is None:
                field = "product_id" if "product_id" in r else "doc_id"
            if r[field] not in pid_to_idx:
                continue
            if r["relevance"] < args.min_relevance:
                continue
            pos[r["query_id"]].add(pid_to_idx[r[field]])
    print(f"  catalog={len(pids):,}  queries={len(queries):,}", flush=True)

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    base_pv = np.load(os.path.join(args.data_dir, args.base_vecs)).astype(np.float32)
    base = SentenceTransformer(args.base_model, device=device)
    bod = SentenceTransformer(args.bod_model, device=device)

    print("encoding catalog with BoD...", flush=True)
    bod_pv = bod.encode(
        titles, normalize_embeddings=True, batch_size=128, show_progress_bar=True
    ).astype(np.float32)
    print("encoding queries...", flush=True)
    base_qv = base.encode(
        queries, normalize_embeddings=True, batch_size=256, show_progress_bar=True
    ).astype(np.float32)
    bod_qv = bod.encode(
        queries, normalize_embeddings=True, batch_size=256, show_progress_bar=True
    ).astype(np.float32)
    del base, bod
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

    print("matmul + signals...", flush=True)
    chunk = 1024
    rows = []
    k = args.k
    for start in range(0, len(qids), chunk):
        end = min(start + chunk, len(qids))
        bsim = base_qv[start:end] @ base_pv.T
        dsim = bod_qv[start:end] @ bod_pv.T
        # Top-k by base, with cosines preserved.
        btopk_idx = np.argpartition(-bsim, k, axis=1)[:, :k]
        # Sort the top-k by descending sim per row.
        for j, gi in enumerate(range(start, end)):
            g = pos.get(qids[gi], set())
            if not g:
                continue
            row_b = bsim[j]
            row_d = dsim[j]
            top_b = btopk_idx[j]
            sorted_b_idx = top_b[np.argsort(-row_b[top_b])]
            top_b_sims = row_b[sorted_b_idx]
            top_d_idx = np.argpartition(-row_d, k)[:k]
            top_d_sorted = top_d_idx[np.argsort(-row_d[top_d_idx])]
            bh = len({int(x) for x in sorted_b_idx} & g) / len(g)
            dh = len({int(x) for x in top_d_sorted} & g) / len(g)
            rows.append(
                (
                    qids[gi],
                    bh,
                    dh,
                    float(top_b_sims[0]),  # top1
                    float(top_b_sims[0] - top_b_sims[1]),  # top1_minus_top2
                    float(np.mean(top_b_sims)),  # mean_top10
                    float(top_b_sims[0] - top_b_sims[-1]),  # top1_minus_topk
                )
            )
        del bsim, dsim
    print(f"  {len(rows):,} pos-bearing queries scored", flush=True)

    arr = np.array([(r[1], r[2], r[3], r[4], r[5], r[6]) for r in rows], dtype=np.float32)
    bh = arr[:, 0]
    dh = arr[:, 1]
    delta = dh - bh

    signals = {
        "top1": arr[:, 2],
        "top1_minus_top2": arr[:, 3],
        "mean_top10": arr[:, 4],
        "top1_minus_topk": arr[:, 5],
    }

    print("\n" + "=" * 78)
    print("Tax-router probe — base-side confidence signals on BestBuy holdout")
    print("=" * 78)
    print(
        f"  base R@10 mean: {bh.mean():.3f}    BoD R@10 mean: {dh.mean():.3f}    "
        f"Δ_BoD−base: {delta.mean():+.3f}"
    )
    print()

    print("Pearson correlation of signal with (base R@10) and (Δ = BoD − base):")
    print(f"  {'signal':<22} {'r(base)':>10} {'r(Δ)':>10}")
    for name, s in signals.items():
        r_base = float(np.corrcoef(s, bh)[0, 1])
        r_delta = float(np.corrcoef(s, delta)[0, 1])
        print(f"  {name:<22} {r_base:>+10.3f} {r_delta:>+10.3f}")

    # Oracle router: for each signal, find a threshold that maximizes
    # R@10 when we route low-signal queries to BoD and high-signal to base.
    print("\nOracle router (route to base if signal >= τ, else BoD; best τ chosen on this set):")
    print(
        f"  {'signal':<22} {'best τ':>8} {'route_base%':>12} "
        f"{'router R@10':>12} {'gain vs BoD':>12} {'gain vs base':>12}"
    )
    base_only = bh.mean()
    bod_only = dh.mean()
    for name, s in signals.items():
        # Sweep τ over deciles of the signal.
        best_r = -1.0
        best_tau = None
        best_pct = None
        for tau in np.quantile(s, np.linspace(0.0, 1.0, 21)):
            mask_base = s >= tau
            r = (np.where(mask_base, bh, dh)).mean()
            if r > best_r:
                best_r = float(r)
                best_tau = float(tau)
                best_pct = 100.0 * float(mask_base.mean())
        gain_bod = best_r - bod_only
        gain_base = best_r - base_only
        print(
            f"  {name:<22} {best_tau:>+8.3f} {best_pct:>11.1f}%  "
            f"{best_r:>12.4f} {gain_bod:>+12.4f} {gain_base:>+12.4f}"
        )

    print(
        "\nA Pareto-better router needs gain > 0 vs both base AND BoD."
        " Negative or near-zero gains mean the tax is intrinsic on this corpus."
    )


if __name__ == "__main__":
    main()
