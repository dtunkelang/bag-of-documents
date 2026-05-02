#!/usr/bin/env python3
"""Per-bin breakdown of CC4-100 - CC3-100 (CE-fusion lift).

Bins the 22,458 test queries by CC3-100's per-query R@10 to surface where
the CE fusion lift is concentrated. If hard-regime: CE rescues queries
the bi-encoder ensemble missed (structural signal). If middle-regime: CE
is just polishing — same shape as the rerank_G effect we measured earlier.
If easy-regime: CE smooths the top.

Pure CPU; reuses cached top-100 artifacts.

Usage:
    python evaluation/per_bin_cc4_vs_cc3.py
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
        e1s[qi] = m[2]
    return rs, e1s


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

    candidate_pos = np.load(os.path.join(INDEX_DIR, "ce_top100_candidates.npy"))
    sumsim = np.load(os.path.join(INDEX_DIR, "ce_top100_sumsim.npy"))
    ce_scores = np.load(os.path.join(INDEX_DIR, "ce_top100_scores.npy"))
    valid = candidate_pos >= 0

    nm_sum = normalize_per_query(sumsim, valid)
    nm_ce = normalize_per_query(ce_scores, valid)
    cc4_25 = 0.75 * nm_sum + 0.25 * nm_ce
    cc4_50 = 0.5 * nm_sum + 0.5 * nm_ce

    rs_cc3, e1_cc3 = per_query_setup(sumsim, candidate_pos, valid, qids, qrels, faiss_pos_to_pid)
    rs_cc4_25, e1_cc4_25 = per_query_setup(
        cc4_25, candidate_pos, valid, qids, qrels, faiss_pos_to_pid
    )
    rs_cc4_50, e1_cc4_50 = per_query_setup(
        cc4_50, candidate_pos, valid, qids, qrels, faiss_pos_to_pid
    )

    # Bin queries by CC3-100 R@10.
    bins = {
        "0 (CC3 zero recall)": (lambda r: r == 0.0),
        "(0, 0.25]": (lambda r: 0 < r <= 0.25),
        "(0.25, 0.50]": (lambda r: 0.25 < r <= 0.50),
        "(0.50, 1.00] (easy)": (lambda r: r > 0.50),
    }

    print(
        f"\n{'bin':<28} {'n':>7} | {'CC3 R@10':>8} {'CC4 R@10':>8} {'Δ R@10':>7} | "
        f"{'CC3 E@1':>8} {'CC4 E@1':>8} {'Δ E@1':>7}"
    )
    print("-" * 105)
    for bin_label, predicate in bins.items():
        mask = np.array([(not np.isnan(r)) and predicate(r) for r in rs_cc3])
        if not mask.any():
            continue
        c3_r = float(np.nanmean(rs_cc3[mask]))
        c4_r = float(np.nanmean(rs_cc4_25[mask]))
        c3_e = float(np.nanmean(e1_cc3[mask]))
        c4_e = float(np.nanmean(e1_cc4_25[mask]))
        n = int(mask.sum())
        print(
            f"{bin_label:<28} {n:>7,} | "
            f"{c3_r:>7.2%} {c4_r:>7.2%} {(c4_r - c3_r) * 100:>+6.2f} | "
            f"{c3_e:>7.2%} {c4_e:>7.2%} {(c4_e - c3_e) * 100:>+6.2f}"
        )
    print("\nAll setups use BM25 top-100 (no bi-encoder filter); CC4 fusion is w_ce=0.25 minmax.")


if __name__ == "__main__":
    main()
