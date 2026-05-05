#!/usr/bin/env python3
"""Free CC5 ablations against existing precomputed scores.

CC5 fusion: per-query min-max normalize each of (sumsim, liyuan, bge) over
the BM25 top-100 candidates, then average. Top-10 of the fused score.

Variants tested (all use cached scores; no new compute):
  CC5_full      : sumsim + liyuan + bge        (current production)
  CC5_no_liyuan : sumsim + bge                 (drops 1s/query CPU)
  CC5_no_sumsim : liyuan + bge                 (drops 3 bi-enc passes)
  CC5_no_bge    : sumsim + liyuan = CC4-100    (drops 5s/query CPU)
  CC5_K50       : full fusion but only top-50 candidates (halves CE cost)
  CC5_K25       : full fusion top-25
  CC5_only_bge  : just bge                     (lower bound for BGE alone)
  CC5_only_liyuan: just liyuan                 (lower bound for LiYuan alone)
  CC5_only_sumsim: just sumsim                 (= CC3-50 essentially)

Reports R@10 / E@1 deltas vs CC5_full to identify the cheapest drop.
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
    """Per-row min-max normalization to [0, 1]. Constant rows -> 0."""
    lo = scores.min(axis=1, keepdims=True)
    hi = scores.max(axis=1, keepdims=True)
    rng = np.maximum(hi - lo, 1e-9)
    return (scores - lo) / rng


def main():
    print("loading qrels + queries + product map...", flush=True)
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
    print(f"  {len(qids):,} eval queries (full test set)", flush=True)

    print("\nloading cached scores...", flush=True)
    cands = np.load(os.path.join(INDEX_DIR, "ce_top100_candidates.npy"))
    sumsim = np.load(os.path.join(INDEX_DIR, "ce_top100_sumsim.npy"))
    liyuan = np.load(os.path.join(INDEX_DIR, "ce_top100_scores.npy"))
    bge = np.load(os.path.join(INDEX_DIR, "bge_rerank/bge_scores_top100_all.npy"))
    print(
        f"  cands {cands.shape}, sumsim {sumsim.shape}, liyuan {liyuan.shape}, bge {bge.shape}",
        flush=True,
    )

    # The CE/BGE caches are aligned to all 22458 test queries (in test_queries.jsonl order)
    # We need to align to the eligible-qids subset.
    test_queries_order = list(queries_all.keys())
    qid_to_row = {qid: i for i, qid in enumerate(test_queries_order)}
    rows = np.array([qid_to_row[qid] for qid in qids if qid in qid_to_row])

    cands = cands[rows]
    sumsim = sumsim[rows]
    liyuan = liyuan[rows]
    bge = bge[rows]
    print(f"  after subsetting to eligible queries: {cands.shape}", flush=True)

    n_q = cands.shape[0]
    K_EVAL = 10

    def fuse_and_eval(streams, K_RERANK=100, label=""):
        """Per-query min-max norm each stream, average, top-K_EVAL within K_RERANK candidates."""
        if cands.shape[1] > K_RERANK:
            sub_cands = cands[:, :K_RERANK]
            sub_streams = [s[:, :K_RERANK] for s in streams]
        else:
            sub_cands = cands
            sub_streams = streams
        normed = [per_query_minmax(s) for s in sub_streams]
        fused = np.mean(np.stack(normed, axis=0), axis=0)  # (n_q, K_RERANK)
        # top-K_EVAL by fused score
        top = np.argpartition(-fused, K_EVAL, axis=1)[:, :K_EVAL]
        recalls, e1s = [], []
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
                recalls.append(hits / len(es_pids))
            if e_pids:
                e1s.append(1.0 if top_pids[0] in e_pids else 0.0)
        return {
            "label": label,
            "R@10": statistics.mean(recalls) if recalls else 0,
            "E@1": statistics.mean(e1s) if e1s else 0,
            "n_es": len(recalls),
            "n_e": len(e1s),
        }

    setups = [
        # full ensemble at varying K
        ("CC5_full_K100", [sumsim, liyuan, bge], 100),
        ("CC5_full_K50", [sumsim, liyuan, bge], 50),
        ("CC5_full_K25", [sumsim, liyuan, bge], 25),
        # drop one stream
        ("CC5_no_liyuan_K100", [sumsim, bge], 100),
        ("CC5_no_bge_K100", [sumsim, liyuan], 100),  # ≈ CC4-100
        ("CC5_no_sumsim_K100", [liyuan, bge], 100),
        # singles
        ("only_bge_K100", [bge], 100),
        ("only_liyuan_K100", [liyuan], 100),
        ("only_sumsim_K100", [sumsim], 100),  # ≈ CC3 over CE candidates
        # Drop+K combos
        ("CC5_no_liyuan_K50", [sumsim, bge], 50),
        ("CC5_no_bge_K50", [sumsim, liyuan], 50),
    ]

    print("\ncomputing fusions...", flush=True)
    results = []
    for label, streams, k_rerank in setups:
        r = fuse_and_eval(streams, K_RERANK=k_rerank, label=label)
        results.append(r)
        print(f"  {label:<24}  R@10={r['R@10'] * 100:5.2f}  E@1={r['E@1'] * 100:5.2f}", flush=True)

    print("\n=== DELTAS vs CC5_full_K100 ===", flush=True)
    base = next(r for r in results if r["label"] == "CC5_full_K100")
    print(f"{'setup':<24}  R@10    delta   E@1    delta")
    for r in results:
        dr = (r["R@10"] - base["R@10"]) * 100
        de = (r["E@1"] - base["E@1"]) * 100
        print(
            f"{r['label']:<24}  {r['R@10'] * 100:5.2f}   {dr:+5.2f}  "
            f"{r['E@1'] * 100:5.2f}   {de:+5.2f}"
        )

    out = {"results": results, "base": base}
    with open("/tmp/cc5_ablation.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\nsaved to /tmp/cc5_ablation.json", flush=True)


if __name__ == "__main__":
    main()
