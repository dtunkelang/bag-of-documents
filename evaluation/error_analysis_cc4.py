#!/usr/bin/env python3
"""Where does CC4-100 fail? For queries with R@10 < 0.1 under CC4-100,
diagnose whether the failure is candidate-pool (BM25 missed it) or
ranking (BM25 had it but rerank couldn't surface it).

Per-failure breakdown:
  - n_relevant: how many E+S products in qrels
  - n_in_pool: how many of those are in the BM25 top-100
  - top_rank_in_ce: best rank among those in pool when sorted by CE
  - top_rank_in_sumsim: best rank when sorted by sumsim

If n_in_pool == 0: the candidate pool is the bottleneck.
If n_in_pool > 0 but top_rank > 50: the rerank is the bottleneck.

Pure CPU; reuses cached top-100 artifacts.

Usage:
    python evaluation/error_analysis_cc4.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json  # noqa: E402
import os  # noqa: E402
from collections import defaultdict  # noqa: E402

import numpy as np  # noqa: E402

SCRIPT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INDEX_DIR = os.path.join(SCRIPT_DIR, "combined_index_us_minilm")
K_EVAL = 10


def per_query_recall(retrieved_pids, qrels_q):
    if not retrieved_pids:
        return None
    pos_es = {pid for pid, g in qrels_q.items() if g >= 2}
    if not pos_es:
        return None
    top_k = retrieved_pids[:K_EVAL]
    return sum(1 for p in top_k if p in pos_es) / len(pos_es), pos_es


def normalize_per_query(scores, valid_mask):
    out = scores.copy()
    for qi in range(out.shape[0]):
        v = out[qi, valid_mask[qi]]
        if v.size == 0:
            continue
        lo, hi = float(v.min()), float(v.max())
        out[qi, valid_mask[qi]] = (v - lo) / max(hi - lo, 1e-8)
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
    queries = [queries_all[qid] for qid in qids]
    print(f"  {len(qids):,} eval queries", flush=True)

    candidate_pos = np.load(os.path.join(INDEX_DIR, "ce_top100_candidates.npy"))
    sumsim = np.load(os.path.join(INDEX_DIR, "ce_top100_sumsim.npy"))
    ce_scores = np.load(os.path.join(INDEX_DIR, "ce_top100_scores.npy"))
    valid = candidate_pos >= 0
    nm_sum = normalize_per_query(sumsim, valid)
    nm_ce = normalize_per_query(ce_scores, valid)
    cc4 = 0.75 * nm_sum + 0.25 * nm_ce

    # CC4 top-10 per query.
    failure_records = []
    for qi, qid in enumerate(qids):
        if not valid[qi].any():
            continue
        s = cc4[qi].copy()
        s[~valid[qi]] = -np.inf
        order = np.argsort(-s)[:K_EVAL]
        ordering = [faiss_pos_to_pid[int(candidate_pos[qi, int(j)])] for j in order]
        m = per_query_recall(ordering, qrels[qid])
        if m is None:
            continue
        recall, pos_es = m
        if recall >= 0.10:
            continue

        # Failure: how many relevant products are in BM25 top-100?
        candidate_pids_in_pool = {faiss_pos_to_pid[int(p)] for p in candidate_pos[qi] if p >= 0}
        relevant_in_pool = pos_es & candidate_pids_in_pool
        n_relevant = len(pos_es)
        n_in_pool = len(relevant_in_pool)

        # Best rank for any relevant product under CE / sumsim.
        ce_order = np.argsort(-ce_scores[qi].copy())
        sum_order = np.argsort(-sumsim[qi].copy())
        cc4_order = np.argsort(-cc4[qi].copy())
        ce_best_rank = -1
        sum_best_rank = -1
        cc4_best_rank = -1
        for j_ce, j_sum, j_cc4 in zip(ce_order, sum_order, cc4_order):
            cand_ce = (
                faiss_pos_to_pid[int(candidate_pos[qi, int(j_ce)])]
                if candidate_pos[qi, int(j_ce)] >= 0
                else None
            )
            cand_sum = (
                faiss_pos_to_pid[int(candidate_pos[qi, int(j_sum)])]
                if candidate_pos[qi, int(j_sum)] >= 0
                else None
            )
            cand_cc4 = (
                faiss_pos_to_pid[int(candidate_pos[qi, int(j_cc4)])]
                if candidate_pos[qi, int(j_cc4)] >= 0
                else None
            )
            if ce_best_rank < 0 and cand_ce in relevant_in_pool:
                ce_best_rank = int(j_ce) + 1
            if sum_best_rank < 0 and cand_sum in relevant_in_pool:
                sum_best_rank = int(j_sum) + 1
            if cc4_best_rank < 0 and cand_cc4 in relevant_in_pool:
                cc4_best_rank = int(j_cc4) + 1
            if ce_best_rank > 0 and sum_best_rank > 0 and cc4_best_rank > 0:
                break

        failure_records.append(
            {
                "qid": qid,
                "query": queries[qi],
                "n_relevant": n_relevant,
                "n_in_pool": n_in_pool,
                "ce_best": ce_best_rank,
                "sum_best": sum_best_rank,
                "cc4_best": cc4_best_rank,
            }
        )

    print(f"\nCC4-100 failures (R@10 < 0.10): {len(failure_records):,}", flush=True)

    # Bucket by candidate-pool coverage.
    none_in_pool = [r for r in failure_records if r["n_in_pool"] == 0]
    some_in_pool = [r for r in failure_records if 0 < r["n_in_pool"] < r["n_relevant"]]
    all_in_pool = [r for r in failure_records if r["n_in_pool"] == r["n_relevant"]]

    print(
        f"  none in BM25 top-100:   {len(none_in_pool):,} ({len(none_in_pool) / len(failure_records):.1%})  <-- candidate-pool floor"
    )
    print(
        f"  partial in pool:        {len(some_in_pool):,} ({len(some_in_pool) / len(failure_records):.1%})"
    )
    print(
        f"  all in pool, but R<0.1: {len(all_in_pool):,} ({len(all_in_pool) / len(failure_records):.1%})  <-- pure rerank failure"
    )

    # For "all in pool" failures, where do the relevant products rank under CE / sumsim / CC4?
    if all_in_pool:
        cc4_ranks = [r["cc4_best"] for r in all_in_pool if r["cc4_best"] > 0]
        sum_ranks = [r["sum_best"] for r in all_in_pool if r["sum_best"] > 0]
        ce_ranks = [r["ce_best"] for r in all_in_pool if r["ce_best"] > 0]
        print("\n'all in pool' failure: best-rank distribution of any relevant product")
        for label, ranks in [("CC4", cc4_ranks), ("sumsim", sum_ranks), ("CE", ce_ranks)]:
            if not ranks:
                continue
            r = np.asarray(ranks)
            print(
                f"  {label:<6}: median={int(np.median(r))} p25={int(np.percentile(r, 25))} "
                f"p75={int(np.percentile(r, 75))} max={int(r.max())} (out of 100)"
            )

    # Sample some examples.
    print("\nsample 'none in pool' failures (BM25 missed all relevant):")
    for r in none_in_pool[:8]:
        print(f"  qid={r['qid']} '{r['query']}' n_rel={r['n_relevant']}")
    print("\nsample 'all in pool' failures (rerank couldn't surface):")
    for r in all_in_pool[:8]:
        print(
            f"  qid={r['qid']} '{r['query']}' n_rel={r['n_relevant']} "
            f"sum_best={r['sum_best']} ce_best={r['ce_best']} cc4_best={r['cc4_best']}"
        )


if __name__ == "__main__":
    main()
