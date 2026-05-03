#!/usr/bin/env python3
"""Bootstrap CIs on BGE-reranker fusion variants vs CC4-LiYuan and CC3.

Runs after evaluation/eval_bge_reranker.py with --max-queries 0 has saved
bge_rerank/bge_scores_top100_all.npy.

Compares (per-query bootstrap, 1000 resamples):
  CC3-100 (sumsim only)
  CC4-100 LiYuan w=0.25 (current quality SOTA)
  BGE alone
  sumsim + BGE w=0.25, 0.50, 0.75
  3-way (sumsim + LiYuan + BGE) equal mean

Tells us which BGE-fusion variant is actually significantly better than
CC4-LiYuan, and by how much.

Usage:
    python evaluation/bootstrap_bge.py
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
    n = len(qids)
    rs = np.full(n, np.nan, dtype=np.float32)
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
        e1s[qi] = m[2] if m[2] is not None else float("nan")
        e3s[qi] = m[3] if m[3] is not None else float("nan")
    return rs, e1s, e3s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-bootstrap", type=int, default=1000)
    ap.add_argument(
        "--bge-scores",
        default=os.path.join(INDEX_DIR, "bge_rerank", "bge_scores_top100_all.npy"),
    )
    args = ap.parse_args()

    rng = np.random.default_rng(42)

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

    cand = np.load(os.path.join(INDEX_DIR, "ce_top100_candidates.npy"))
    sumsim = np.load(os.path.join(INDEX_DIR, "ce_top100_sumsim.npy"))
    liyuan = np.load(os.path.join(INDEX_DIR, "ce_top100_scores.npy"))
    bge = np.load(args.bge_scores)
    valid = cand >= 0
    valid_bge = valid & ~np.isnan(bge)

    nm_sum = normalize_per_query(sumsim, valid)
    nm_li = normalize_per_query(liyuan, valid)
    nm_bge = normalize_per_query(np.nan_to_num(bge, nan=0.0), valid_bge)

    setups = {
        "CC3-100 (sumsim)": sumsim,
        "CC4-100 LiYuan w=0.25 (current SOTA)": 0.75 * nm_sum + 0.25 * nm_li,
        "BGE alone": bge,
        "sumsim + BGE w=0.25": 0.75 * nm_sum + 0.25 * nm_bge,
        "sumsim + BGE w=0.50": 0.50 * nm_sum + 0.50 * nm_bge,
        "sumsim + BGE w=0.75": 0.25 * nm_sum + 0.75 * nm_bge,
        "3-way (sumsim+LiYuan+BGE) mean": (nm_sum + nm_li + nm_bge) / 3,
    }

    print("computing per-query metrics for each setup...", flush=True)
    metrics = {}
    for label, scores in setups.items():
        metrics[label] = per_query_setup(scores, cand, valid, qids, qrels, faiss_pos_to_pid)

    # Aggregate + bootstrap CIs.
    n = len(qids)
    print(f"\nbootstrapping {args.n_bootstrap} resamples...\n", flush=True)
    print(f"{'setup':<42} | {'R@10 (95% CI)':<22} {'E@1 (95% CI)':<22} {'E@3 (95% CI)':<22}")
    print("-" * 116)

    for label, (rs, e1s, e3s) in metrics.items():
        bs_r, bs_e1, bs_e3 = [], [], []
        for _ in range(args.n_bootstrap):
            idx = rng.integers(0, n, size=n)
            bs_r.append(np.nanmean(rs[idx]))
            bs_e1.append(np.nanmean(e1s[idx]))
            bs_e3.append(np.nanmean(e3s[idx]))
        bs_r, bs_e1, bs_e3 = map(np.asarray, (bs_r, bs_e1, bs_e3))

        def fmt(arr):
            return (
                f"{arr.mean():.2%} [{np.percentile(arr, 2.5):.2%},{np.percentile(arr, 97.5):.2%}]"
            )

        print(f"{label:<42} | {fmt(bs_r):<22} {fmt(bs_e1):<22} {fmt(bs_e3):<22}")

    # Pairwise deltas vs current SOTA.
    print(f"\n{'pair':<60} | {'ΔR@10 (95% CI)':<22} {'ΔE@1 (95% CI)':<22}")
    print("-" * 110)
    base_label = "CC4-100 LiYuan w=0.25 (current SOTA)"
    base_r, base_e1, _ = metrics[base_label]
    pairs = [
        "BGE alone",
        "sumsim + BGE w=0.25",
        "sumsim + BGE w=0.50",
        "sumsim + BGE w=0.75",
        "3-way (sumsim+LiYuan+BGE) mean",
    ]
    for label in pairs:
        rs, e1s, _ = metrics[label]
        diffs_r = rs - base_r
        diffs_e1 = e1s - base_e1
        bs_r, bs_e1 = [], []
        for _ in range(args.n_bootstrap):
            idx = rng.integers(0, n, size=n)
            bs_r.append(np.nanmean(diffs_r[idx]))
            bs_e1.append(np.nanmean(diffs_e1[idx]))
        bs_r, bs_e1 = map(np.asarray, (bs_r, bs_e1))

        def fmt2(arr):
            mu, lo, hi = arr.mean(), np.percentile(arr, 2.5), np.percentile(arr, 97.5)
            sig = "**" if (lo > 0 and hi > 0) or (lo < 0 and hi < 0) else "  "
            return f"{sig}{mu:+.2%} [{lo:+.2%},{hi:+.2%}]"

        line = f"{label} vs SOTA"
        print(f"{line:<60} | {fmt2(bs_r):<22} {fmt2(bs_e1):<22}")
    print("\n** = 95% CI excludes 0 (statistically significant)")


if __name__ == "__main__":
    main()
