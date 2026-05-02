#!/usr/bin/env python3
"""Bootstrap 95% CIs on R@10 / nDCG@10 / E@1 / E@3 for the SOTA setups,
using cached top-100 candidate scores.

Aim: tell us whether ~0.1pp differences in the recent SOTA chain are
real or within sampling noise. With N=22,458 test queries, the std of
the mean is roughly sigma_per_query / sqrt(N). For R@10 with sigma~0.5,
that's ~0.0033 = 0.33pp. So +0.1pp differences are likely noise; +0.3pp
is a meaningful effect.

Setups bootstrapped (all use cached ce_top100 artifacts):
  H'   - bm25s top-10 alone
  CC3-100 (sumsim only)
  CC4-100 (sumsim + CE @ w=0.25)
  CC4-100 w_ce=0.50 (E@1-favoring)

For each setup, computes per-query metrics, then resamples queries with
replacement 1000 times and reports mean + 95% CI on each metric.

Usage:
    python evaluation/bootstrap_cis.py
    python evaluation/bootstrap_cis.py --n-bootstrap 2000
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import os  # noqa: E402
from collections import defaultdict  # noqa: E402

import numpy as np  # noqa: E402

SCRIPT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INDEX_DIR = os.path.join(SCRIPT_DIR, "combined_index_us_minilm")
K_EVAL = 10


def per_query_metrics(retrieved_pids, qrels_q):
    if not retrieved_pids:
        return None
    pos_e = {pid for pid, g in qrels_q.items() if g >= 3}
    pos_es = {pid for pid, g in qrels_q.items() if g >= 2}
    if not pos_es:
        return None
    top_k = retrieved_pids[:K_EVAL]
    recall = sum(1 for p in top_k if p in pos_es) / len(pos_es)
    gains = [1.0 if p in pos_e else (0.1 if p in pos_es else 0.0) for p in top_k]
    dcg = sum(g / math.log2(i + 2) for i, g in enumerate(gains))
    ideal = sorted((1.0 if p in pos_e else 0.1 for p in pos_es), reverse=True)[:K_EVAL]
    idcg = sum(g / math.log2(i + 2) for i, g in enumerate(ideal))
    ndcg = dcg / idcg if idcg > 0 else 0.0
    if pos_e:
        e1 = sum(1 for p in top_k[:1] if p in pos_e) / min(1, len(pos_e))
        e3 = sum(1 for p in top_k[:3] if p in pos_e) / min(3, len(pos_e))
    else:
        e1 = e3 = float("nan")
    return recall, ndcg, e1, e3


def normalize_per_query(scores, valid_mask):
    out = scores.copy()
    for qi in range(out.shape[0]):
        v = out[qi, valid_mask[qi]]
        if v.size == 0:
            continue
        lo, hi = float(v.min()), float(v.max())
        out[qi, valid_mask[qi]] = (v - lo) / max(hi - lo, 1e-8)
    return out


def per_query_setup(scores, candidate_pos, valid, qids, qrels, faiss_pos_to_pid):
    """Return per-query (R@10, nDCG, E@1, E@3) under top-10 by `scores`.
    NaN where the query has no E qrel (E@1/E@3) or no E+S (skipped)."""
    n = len(qids)
    rs = np.full(n, np.nan, dtype=np.float32)
    ns = np.full(n, np.nan, dtype=np.float32)
    e1s = np.full(n, np.nan, dtype=np.float32)
    e3s = np.full(n, np.nan, dtype=np.float32)
    for qi, qid in enumerate(qids):
        if not valid[qi].any():
            continue
        s = scores[qi].copy()
        s[~valid[qi]] = -np.inf
        order = np.argsort(-s)[:K_EVAL]
        ordering = [faiss_pos_to_pid[int(candidate_pos[qi, int(j)])] for j in order]
        m = per_query_metrics(ordering, qrels[qid])
        if m is None:
            continue
        rs[qi] = m[0]
        ns[qi] = m[1]
        e1s[qi] = m[2]
        e3s[qi] = m[3]
    return rs, ns, e1s, e3s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-bootstrap", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)

    qrels = defaultdict(dict)
    with open(os.path.join(SCRIPT_DIR, "esci_us_data/test_qrels.jsonl")) as f:
        for line in f:
            r = json.loads(line)
            qrels[r["query_id"]][r["product_id"]] = r["relevance"]
    queries_all = {}
    with open(os.path.join(SCRIPT_DIR, "esci_us_data/test_queries.jsonl")) as f:
        for line in f:
            d = json.loads(line)
            queries_all[d["query_id"]] = d["query"]
    with open(os.path.join(SCRIPT_DIR, "esci_us_data/product_ids.json")) as f:
        esci_pids = json.load(f)
    with open(os.path.join(SCRIPT_DIR, "esci_us_data/titles.json")) as f:
        esci_titles_arr = json.load(f)
    with open(os.path.join(INDEX_DIR, "titles.json")) as f:
        index_titles = json.load(f)
    title_to_pid = {t: p for p, t in zip(esci_pids, esci_titles_arr)}
    faiss_pos_to_pid = [title_to_pid.get(t) for t in index_titles]

    qids = [qid for qid in queries_all if qid in qrels and any(g >= 2 for g in qrels[qid].values())]
    print(f"  {len(qids):,} eval queries", flush=True)

    candidate_pos = np.load(os.path.join(INDEX_DIR, "ce_top100_candidates.npy"))
    sumsim = np.load(os.path.join(INDEX_DIR, "ce_top100_sumsim.npy"))
    ce_scores = np.load(os.path.join(INDEX_DIR, "ce_top100_scores.npy"))
    valid = candidate_pos >= 0

    nm_sum = normalize_per_query(sumsim, valid)
    nm_ce = normalize_per_query(ce_scores, valid)

    # Build per-query metric arrays for each setup.
    print("computing per-query metrics for each setup...", flush=True)
    setups = {}
    # H' (BM25 alone): top-10 = first 10 of bm25s top-100 = identity ranking.
    bm25_alone_scores = -np.tile(
        np.arange(candidate_pos.shape[1], dtype=np.float32), (len(qids), 1)
    )
    setups["H' (bm25s alone)"] = per_query_setup(
        bm25_alone_scores, candidate_pos, valid, qids, qrels, faiss_pos_to_pid
    )
    setups["CC3-100 (sumsim)"] = per_query_setup(
        sumsim, candidate_pos, valid, qids, qrels, faiss_pos_to_pid
    )
    cc4_25 = 0.75 * nm_sum + 0.25 * nm_ce
    setups["CC4-100 (w_ce=0.25)"] = per_query_setup(
        cc4_25, candidate_pos, valid, qids, qrels, faiss_pos_to_pid
    )
    cc4_50 = 0.5 * nm_sum + 0.5 * nm_ce
    setups["CC4-100 (w_ce=0.50)"] = per_query_setup(
        cc4_50, candidate_pos, valid, qids, qrels, faiss_pos_to_pid
    )
    cc3_50_scores = sumsim.copy()
    cc3_50_scores[:, 50:] = -np.inf
    setups["CC3-50 (sumsim, top-50)"] = per_query_setup(
        cc3_50_scores, candidate_pos, valid, qids, qrels, faiss_pos_to_pid
    )

    # Bootstrap
    n = len(qids)
    print(f"\nbootstrapping {args.n_bootstrap} resamples of {n:,} queries...", flush=True)

    print(
        f"\n{'setup':<32} | {'R@10 (95% CI)':<22} {'nDCG (95% CI)':<22} {'E@1 (95% CI)':<22} {'E@3 (95% CI)':<22}"
    )
    print("-" * 130)
    means_per_setup = {}
    for label, (rs, ns, e1s, e3s) in setups.items():
        # Bootstrap each metric independently — sample query indices with replacement.
        bs_r, bs_n, bs_e1, bs_e3 = [], [], [], []
        for _ in range(args.n_bootstrap):
            idx = rng.integers(0, n, size=n)
            bs_r.append(np.nanmean(rs[idx]))
            bs_n.append(np.nanmean(ns[idx]))
            bs_e1.append(np.nanmean(e1s[idx]))
            bs_e3.append(np.nanmean(e3s[idx]))
        bs_r = np.asarray(bs_r)
        bs_n = np.asarray(bs_n)
        bs_e1 = np.asarray(bs_e1)
        bs_e3 = np.asarray(bs_e3)

        def fmt(arr):
            mu = arr.mean()
            lo = np.percentile(arr, 2.5)
            hi = np.percentile(arr, 97.5)
            return f"{mu:.2%} [{lo:.2%},{hi:.2%}]"

        print(f"{label:<32} | {fmt(bs_r):<22} {fmt(bs_n):<22} {fmt(bs_e1):<22} {fmt(bs_e3):<22}")
        means_per_setup[label] = (rs, ns, e1s, e3s)

    # Pairwise paired-bootstrap deltas.
    print(f"\n{'pair':<60} | {'ΔR@10 (95% CI)':<22} {'ΔE@1 (95% CI)':<22}")
    print("-" * 110)
    pairs_to_compare = [
        ("CC4-100 (w_ce=0.25)", "CC3-100 (sumsim)"),
        ("CC4-100 (w_ce=0.50)", "CC4-100 (w_ce=0.25)"),
        ("CC3-100 (sumsim)", "CC3-50 (sumsim, top-50)"),
        ("CC3-100 (sumsim)", "H' (bm25s alone)"),
        ("CC4-100 (w_ce=0.25)", "CC3-50 (sumsim, top-50)"),
    ]
    for a, b in pairs_to_compare:
        rs_a, _, e1s_a, _ = means_per_setup[a]
        rs_b, _, e1s_b, _ = means_per_setup[b]
        diffs_r = rs_a - rs_b  # per-query delta
        diffs_e1 = e1s_a - e1s_b
        bs_r, bs_e1 = [], []
        for _ in range(args.n_bootstrap):
            idx = rng.integers(0, n, size=n)
            bs_r.append(np.nanmean(diffs_r[idx]))
            bs_e1.append(np.nanmean(diffs_e1[idx]))
        bs_r = np.asarray(bs_r)
        bs_e1 = np.asarray(bs_e1)

        def fmt(arr, mu_only=False):
            mu = arr.mean()
            lo = np.percentile(arr, 2.5)
            hi = np.percentile(arr, 97.5)
            sig = "**" if (lo > 0 and hi > 0) or (lo < 0 and hi < 0) else "  "
            return f"{sig}{mu:+.2%} [{lo:+.2%},{hi:+.2%}]"

        label = f"{a} vs {b}"
        print(f"{label:<60} | {fmt(bs_r):<22} {fmt(bs_e1):<22}")

    print("\n** = 95% CI excludes 0 (statistically significant)")


if __name__ == "__main__":
    main()
