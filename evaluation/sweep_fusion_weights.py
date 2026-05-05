#!/usr/bin/env python3
"""Sweep non-uniform fusion weights for the 3-way (sumsim, liyuan, bge) ensemble.

CC5 currently does (sumsim_norm + liyuan_norm + bge_norm) / 3. Try other
weight tuples summing to 1 to see if there's a better Pareto point on R@10
or E@1 over the same input streams.
"""

import json
import os
import statistics
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INDEX_DIR = os.path.join(SCRIPT_DIR, "combined_index_us_minilm")


def per_query_minmax(scores):
    lo = scores.min(axis=1, keepdims=True)
    hi = scores.max(axis=1, keepdims=True)
    rng = np.maximum(hi - lo, 1e-9)
    return (scores - lo) / rng


def main():
    qrels = defaultdict(dict)
    with open(os.path.join(SCRIPT_DIR, "esci_us_data/test_qrels.jsonl")) as f:
        for line in f:
            r = json.loads(line)
            qrels[r["query_id"]][r["product_id"]] = r["relevance"]
    qrels = dict(qrels)
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

    cands = np.load(os.path.join(INDEX_DIR, "ce_top100_candidates.npy"))
    sumsim = np.load(os.path.join(INDEX_DIR, "ce_top100_sumsim.npy"))
    liyuan = np.load(os.path.join(INDEX_DIR, "ce_top100_scores.npy"))
    bge = np.load(os.path.join(INDEX_DIR, "bge_rerank/bge_scores_top100_all.npy"))

    test_queries_order = list(queries_all.keys())
    qid_to_row = {qid: i for i, qid in enumerate(test_queries_order)}
    rows = np.array([qid_to_row[qid] for qid in qids if qid in qid_to_row])
    cands = cands[rows]
    sumsim_n = per_query_minmax(sumsim[rows])
    liyuan_n = per_query_minmax(liyuan[rows])
    bge_n = per_query_minmax(bge[rows])

    n_q = cands.shape[0]
    K_EVAL = 10

    def eval_weights(w_s, w_l, w_b, K=100):
        sc = w_s * sumsim_n[:, :K] + w_l * liyuan_n[:, :K] + w_b * bge_n[:, :K]
        sub_cands = cands[:, :K]
        top = np.argpartition(-sc, K_EVAL, axis=1)[:, :K_EVAL]
        recalls, e1s = [], []
        for i in range(n_q):
            qid = qids[i]
            qr = qrels[qid]
            ord_i = top[i][np.argsort(-sc[i, top[i]])]
            top_pos = sub_cands[i, ord_i]
            top_pids = [
                faiss_pos_to_pid[p] if 0 <= p < len(faiss_pos_to_pid) else None for p in top_pos
            ]
            es = {p for p, g in qr.items() if g >= 2}
            ee = {p for p, g in qr.items() if g == 3}
            if es:
                recalls.append(sum(1 for p in top_pids if p in es) / len(es))
            if ee:
                e1s.append(1.0 if top_pids[0] in ee else 0.0)
        return statistics.mean(recalls), statistics.mean(e1s)

    # Generate weight combinations summing to 1, in 0.1 steps
    weight_combos = []
    for ws_int in range(0, 11):
        for wl_int in range(0, 11 - ws_int):
            wb_int = 10 - ws_int - wl_int
            if wb_int < 0:
                continue
            weight_combos.append((ws_int / 10, wl_int / 10, wb_int / 10))

    print(f"sweeping {len(weight_combos)} weight combinations at K=100...", flush=True)
    print(f"{'w_sumsim':<9} {'w_liyuan':<9} {'w_bge':<9} {'R@10':<8} {'E@1':<8}", flush=True)
    results = []
    for ws, wl, wb in weight_combos:
        r10, e1 = eval_weights(ws, wl, wb, K=100)
        results.append({"w_sumsim": ws, "w_liyuan": wl, "w_bge": wb, "R@10": r10, "E@1": e1})
        if r10 * 100 >= 23.0 or (ws, wl, wb) in [(1 / 3, 1 / 3, 1 / 3), (0.5, 0, 0.5)]:
            print(
                f"{ws:<9.2f} {wl:<9.2f} {wb:<9.2f} {r10 * 100:<8.2f} {e1 * 100:<8.2f}",
                flush=True,
            )

    print("\n=== TOP 10 by R@10 ===", flush=True)
    results.sort(key=lambda r: -r["R@10"])
    print(f"{'w_sumsim':<9} {'w_liyuan':<9} {'w_bge':<9} {'R@10':<8} {'E@1':<8}")
    for r in results[:10]:
        print(
            f"{r['w_sumsim']:<9.2f} {r['w_liyuan']:<9.2f} {r['w_bge']:<9.2f} "
            f"{r['R@10'] * 100:<8.2f} {r['E@1'] * 100:<8.2f}"
        )

    print("\n=== TOP 10 by E@1 ===", flush=True)
    results.sort(key=lambda r: -r["E@1"])
    print(f"{'w_sumsim':<9} {'w_liyuan':<9} {'w_bge':<9} {'R@10':<8} {'E@1':<8}")
    for r in results[:10]:
        print(
            f"{r['w_sumsim']:<9.2f} {r['w_liyuan']:<9.2f} {r['w_bge']:<9.2f} "
            f"{r['R@10'] * 100:<8.2f} {r['E@1'] * 100:<8.2f}"
        )

    with open("/tmp/fusion_weight_sweep.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nsaved to /tmp/fusion_weight_sweep.json", flush=True)


if __name__ == "__main__":
    main()
