#!/usr/bin/env python3
"""Compare BGE-reranker-base 5K result to BGE-reranker-v2-m3 5K (cached backup).

Fires after the BGE-base 5K eval finishes. Loads the freshly-saved BGE-base
scores and the backed-up v2-m3 scores, computes CC5_no_liyuan_K100 R@10/E@1
under each, reports deltas. Writes POSITIVE_5K or NEGATIVE_5K verdict.

Tolerance: BGE-base within -0.5pp R@10 of v2-m3 = green-light full run.
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
    cands_all = np.load(os.path.join(INDEX_DIR, "ce_top100_candidates.npy"))
    sumsim_all = np.load(os.path.join(INDEX_DIR, "ce_top100_sumsim.npy"))

    test_queries_order = list(queries_all.keys())
    qid_to_row = {qid: i for i, qid in enumerate(test_queries_order)}
    rows = np.array([qid_to_row[qid] for qid in qids if qid in qid_to_row])
    cands_all = cands_all[rows]
    sumsim_all = sumsim_all[rows]

    # Subsample to the same 5K queries the eval used (seed=42)
    rng = np.random.default_rng(42)
    sample_idx = rng.choice(len(qids), size=5000, replace=False)
    sample_idx.sort()

    cands = cands_all[sample_idx]
    sumsim = sumsim_all[sample_idx]
    sub_qids = [qids[int(i)] for i in sample_idx]

    # Load BGE-base 5K (just-saved) and v2-m3 5K (backed up)
    bge_base = np.load(os.path.join(INDEX_DIR, "bge_rerank/bge_scores_top100_n5000.npy")).astype(
        np.float32
    )
    bge_v2m3 = np.load("/tmp/bge_v2m3_5k_backup.npy").astype(np.float32)
    print(f"BGE-base shape: {bge_base.shape}, v2-m3 shape: {bge_v2m3.shape}", flush=True)

    K = 100
    K_EVAL = 10

    def fuse_eval(bge_scores, name):
        # Note: BGE eval saves scores in ALL-query order, only with 5K queries having non-zero scores.
        # If BGE-base shape is (n_full, K), we already have it for the full 22458 rows but only
        # 5K of them have scores filled in.
        if bge_scores.shape[0] == len(qids):
            bge_sub = bge_scores[sample_idx]
        elif bge_scores.shape[0] == 5000:
            bge_sub = bge_scores
        else:
            print(f"WARN: unexpected shape {bge_scores.shape}; assuming first 5000 rows align")
            bge_sub = bge_scores[: len(sample_idx)]

        # Mask invalid (zero/no-score) entries
        valid = bge_sub != 0
        if not valid.all():
            n_valid_rows = (valid.sum(axis=1) > 0).sum()
            print(f"  {name}: {n_valid_rows} rows with at least one valid score", flush=True)

        sumsim_norm = per_query_minmax(sumsim[:, :K])
        bge_norm = per_query_minmax(bge_sub[:, :K])

        fused = 0.5 * sumsim_norm + 0.5 * bge_norm  # CC5_no_liyuan
        top = np.argpartition(-fused, K_EVAL, axis=1)[:, :K_EVAL]

        recalls, e1s = [], []
        for i in range(len(sub_qids)):
            qid = sub_qids[i]
            qr = qrels[qid]
            ord_i = top[i][np.argsort(-fused[i, top[i]])]
            top_pos = cands[i, ord_i]
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

    print("\nCC5_no_liyuan_K100 on 5K subsample...", flush=True)
    r_v, e_v = fuse_eval(bge_v2m3, "v2-m3")
    print(f"  v2-m3 (568M):    R@10={r_v * 100:5.2f}  E@1={e_v * 100:5.2f}", flush=True)
    r_b, e_b = fuse_eval(bge_base, "base")
    print(f"  base (110M):     R@10={r_b * 100:5.2f}  E@1={e_b * 100:5.2f}", flush=True)
    dr = (r_b - r_v) * 100
    de = (e_b - e_v) * 100
    print(f"\n  delta (base - v2-m3):  R@10={dr:+5.2f}  E@1={de:+5.2f}", flush=True)

    # Verdict
    TOL = 0.5
    if dr >= -TOL:
        print(f"\nPOSITIVE_5K  (base within {TOL}pp R@10 of v2-m3; recommend full run)", flush=True)
    else:
        print(
            f"\nNEGATIVE_5K  (base losing > {TOL}pp R@10 vs v2-m3; not worth full run)", flush=True
        )


if __name__ == "__main__":
    main()
