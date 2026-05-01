#!/usr/bin/env python3
"""CE + sumsim fusion sweep using cached CE scores from eval_ce_rerank.py.

Reuses combined_index_us_minilm/{ce_candidates,ce_sumsim,ce_scores}.npy.
Sweeps fusion weight w_ce in {0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0} with
two normalizations (min-max per query, z-score per query) and finds the
best mixer.

Usage:
    python evaluation/eval_ce_fusion.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

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
    e1 = (1.0 if any(p in pos_e for p in top_k[:1]) else 0.0) if pos_e else None
    e3 = (1.0 if any(p in pos_e for p in top_k[:3]) else 0.0) if pos_e else None
    return recall, ndcg, e1, e3


def normalize_per_query(scores, valid_mask, mode):
    out = scores.copy()
    if mode == "minmax":
        for qi in range(out.shape[0]):
            v = out[qi, valid_mask[qi]]
            if v.size == 0:
                continue
            lo, hi = v.min(), v.max()
            if hi > lo:
                out[qi, valid_mask[qi]] = (v - lo) / (hi - lo)
            else:
                out[qi, valid_mask[qi]] = 0.0
    elif mode == "zscore":
        for qi in range(out.shape[0]):
            v = out[qi, valid_mask[qi]]
            if v.size <= 1:
                continue
            mu, sd = v.mean(), v.std()
            if sd > 0:
                out[qi, valid_mask[qi]] = (v - mu) / sd
            else:
                out[qi, valid_mask[qi]] = 0.0
    return out


def main():
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

    candidate_pos = np.load(os.path.join(INDEX_DIR, "ce_candidates.npy"))
    sumsim_scores = np.load(os.path.join(INDEX_DIR, "ce_sumsim.npy"))
    ce_scores = np.load(os.path.join(INDEX_DIR, "ce_scores.npy"))
    K_RET = candidate_pos.shape[1]
    valid = candidate_pos >= 0
    print(f"  loaded ce_scores shape={ce_scores.shape}, K_RET={K_RET}", flush=True)

    print(
        f"\n{'setup':<48} | {'R@10':>7} {'nDCG@10':>9} {'E@1':>7} {'E@3':>7}",
        flush=True,
    )
    print("-" * 92)

    weights = [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]
    norms = ["minmax", "zscore"]

    for norm in norms:
        nm_sum = normalize_per_query(sumsim_scores, valid, norm)
        nm_ce = normalize_per_query(ce_scores, valid, norm)
        for w in weights:
            fused = (1 - w) * nm_sum + w * nm_ce
            fused[~valid] = -np.inf
            rs, ns, e1s, e3s = [], [], [], []
            for qi, qid in enumerate(qids):
                if not valid[qi].any():
                    continue
                order = np.argsort(-fused[qi])[:K_EVAL]
                ordering = [faiss_pos_to_pid[int(candidate_pos[qi, int(j)])] for j in order]
                m = per_query_metrics(ordering, qrels[qid])
                if m is None:
                    continue
                r, n, e1, e3 = m
                rs.append(r)
                ns.append(n)
                if e1 is not None:
                    e1s.append(e1)
                    e3s.append(e3)
            label = f"{norm}: w_ce={w:.2f} w_sum={1 - w:.2f}"
            print(
                f"{label:<48} | {np.mean(rs):>6.2%} {np.mean(ns):>9.4f} "
                f"{np.mean(e1s):>6.2%} {np.mean(e3s):>6.2%}",
                flush=True,
            )


if __name__ == "__main__":
    main()
