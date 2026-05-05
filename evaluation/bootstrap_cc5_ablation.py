#!/usr/bin/env python3
"""Paired bootstrap CIs for the CC5 ablation deltas (especially LiYuan-out).

Computes per-query R@10/E@1 for each ablation variant against full CC5,
then runs 1000-resample paired bootstrap on the deltas.
"""

import json
import os
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
    print(f"  {len(qids):,} eval queries", flush=True)

    cands = np.load(os.path.join(INDEX_DIR, "ce_top100_candidates.npy"))
    sumsim = np.load(os.path.join(INDEX_DIR, "ce_top100_sumsim.npy"))
    liyuan = np.load(os.path.join(INDEX_DIR, "ce_top100_scores.npy"))
    bge = np.load(os.path.join(INDEX_DIR, "bge_rerank/bge_scores_top100_all.npy"))

    test_queries_order = list(queries_all.keys())
    qid_to_row = {qid: i for i, qid in enumerate(test_queries_order)}
    rows = np.array([qid_to_row[qid] for qid in qids if qid in qid_to_row])
    cands = cands[rows]
    sumsim = sumsim[rows]
    liyuan = liyuan[rows]
    bge = bge[rows]
    n_q = cands.shape[0]
    K_EVAL = 10

    def per_query_eval(streams, K_RERANK=100):
        if cands.shape[1] > K_RERANK:
            sub_cands = cands[:, :K_RERANK]
            sub_streams = [s[:, :K_RERANK] for s in streams]
        else:
            sub_cands = cands
            sub_streams = streams
        normed = [per_query_minmax(s) for s in sub_streams]
        fused = np.mean(np.stack(normed, axis=0), axis=0)
        top = np.argpartition(-fused, K_EVAL, axis=1)[:, :K_EVAL]
        recalls = np.full(n_q, -1.0)
        e1s = np.full(n_q, -1.0)
        for i in range(n_q):
            qid = qids[i]
            qr = qrels[qid]
            ord_i = top[i][np.argsort(-fused[i, top[i]])]
            top_pos = sub_cands[i, ord_i]
            top_pids = [
                faiss_pos_to_pid[p] if 0 <= p < len(faiss_pos_to_pid) else None for p in top_pos
            ]
            es_pids = {p for p, g in qr.items() if g >= 2}
            e_pids = {p for p, g in qr.items() if g == 3}
            if es_pids:
                hits = sum(1 for p in top_pids if p in es_pids)
                recalls[i] = hits / len(es_pids)
            if e_pids:
                e1s[i] = 1.0 if top_pids[0] in e_pids else 0.0
        return recalls, e1s

    print("computing per-query metrics...", flush=True)
    base_r, base_e = per_query_eval([sumsim, liyuan, bge], 100)
    no_li_r, no_li_e = per_query_eval([sumsim, bge], 100)
    no_li_50_r, no_li_50_e = per_query_eval([sumsim, bge], 50)
    no_bge_r, no_bge_e = per_query_eval([sumsim, liyuan], 100)
    K50_r, K50_e = per_query_eval([sumsim, liyuan, bge], 50)

    def boot(a, b, n_iter=1000, seed=42):
        a = np.asarray(a)
        b = np.asarray(b)
        mask = (a >= 0) & (b >= 0)
        a = a[mask]
        b = b[mask]
        rng = np.random.default_rng(seed)
        n = len(a)
        deltas = np.zeros(n_iter)
        for t in range(n_iter):
            idx = rng.integers(0, n, n)
            deltas[t] = b[idx].mean() - a[idx].mean()
        return {
            "n": int(n),
            "delta_mean": float(deltas.mean()),
            "ci_low": float(np.percentile(deltas, 2.5)),
            "ci_high": float(np.percentile(deltas, 97.5)),
            "p_b_better": float((deltas > 0).mean()),
        }

    pairs = [
        ("no_liyuan_K100 R@10", base_r, no_li_r),
        ("no_liyuan_K100 E@1 ", base_e, no_li_e),
        ("no_liyuan_K50  R@10", base_r, no_li_50_r),
        ("no_liyuan_K50  E@1 ", base_e, no_li_50_e),
        ("no_bge_K100    R@10", base_r, no_bge_r),
        ("no_bge_K100    E@1 ", base_e, no_bge_e),
        ("full_K50       R@10", base_r, K50_r),
        ("full_K50       E@1 ", base_e, K50_e),
    ]
    print("\n=== bootstrap deltas vs CC5_full_K100 ===")
    print(f"{'setup':<22}  delta(b - base) [95% CI]            n     p(b>base)")
    print("-" * 90)
    for label, a, b in pairs:
        res = boot(a, b)
        print(
            f"{label:<22}  {res['delta_mean'] * 100:+6.2f}pp [{res['ci_low'] * 100:+6.2f}, "
            f"{res['ci_high'] * 100:+6.2f}]  {res['n']:>5}  {res['p_b_better']:.3f}"
        )


if __name__ == "__main__":
    main()
