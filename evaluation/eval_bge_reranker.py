#!/usr/bin/env python3
"""Probe: BAAI/bge-reranker-v2-m3 as a CE replacement for the CC4 quality SOTA.

The current quality SOTA uses LiYuan/Amazon-Cup-Cross-Encoder-Regression
(RoBERTa-base, ~125M params). bge-reranker-v2-m3 is a stronger XLM-RoBERTa-
large reranker (~568M params, multilingual, BEIR-tested). Already cached
locally from prior work.

Approach: re-score the same BM25 top-K candidate pool with bge-reranker,
fuse the same way as CC4 (per-query min-max + 0.75*sumsim + 0.25*ce).
Compare to LiYuan-CC4 at top-50 and top-100.

Latency expectation: bge-reranker is ~2-4x slower than LiYuan per pair.
We probe on a 5K-query subsample first (~30 min) to screen for lift; if
positive, queue the full 22,458-query run for overnight.

Usage:
    python evaluation/eval_bge_reranker.py --top-k 100 --max-queries 5000  # screen
    python evaluation/eval_bge_reranker.py --top-k 100  # full run
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import os  # noqa: E402
import time  # noqa: E402
from collections import defaultdict  # noqa: E402

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np  # noqa: E402
import torch  # noqa: E402
from sentence_transformers import CrossEncoder  # noqa: E402

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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="BAAI/bge-reranker-v2-m3")
    ap.add_argument("--top-k", type=int, default=100, help="candidate pool size")
    ap.add_argument("--max-queries", type=int, default=0, help="0 = all queries")
    ap.add_argument("--batch-size", type=int, default=16)
    args = ap.parse_args()

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

    # Subsample if requested.
    if args.max_queries > 0 and args.max_queries < len(qids):
        rng = np.random.default_rng(42)
        sample_idx = rng.choice(len(qids), size=args.max_queries, replace=False)
        sample_idx.sort()
        print(f"  sampling {args.max_queries:,} queries (seed=42)", flush=True)
    else:
        sample_idx = np.arange(len(qids))

    # Load cached top-100 artifacts.
    cand = np.load(os.path.join(INDEX_DIR, "ce_top100_candidates.npy"))
    sumsim = np.load(os.path.join(INDEX_DIR, "ce_top100_sumsim.npy"))
    liyuan_ce = np.load(os.path.join(INDEX_DIR, "ce_top100_scores.npy"))
    valid = cand >= 0
    K_POOL = min(args.top_k, cand.shape[1])

    # CE-score with BGE-reranker.
    device = (
        "mps"
        if torch.backends.mps.is_available()
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"\nloading {args.model} on {device}...", flush=True)
    t0 = time.time()
    bge = CrossEncoder(args.model, device=device)
    print(f"  loaded in {time.time() - t0:.0f}s", flush=True)

    n_pairs_total = int(args.max_queries if args.max_queries else len(qids)) * K_POOL
    bge_scores = np.full((len(qids), K_POOL), np.nan, dtype=np.float32)

    print(f"BGE-rescoring {len(sample_idx):,} queries x {K_POOL} candidates...", flush=True)
    pairs_buf = []
    locs_buf = []
    n_done = 0
    t0 = time.time()
    for qi in sample_idx:
        q = queries[qi]
        for j in range(K_POOL):
            pos = int(cand[qi, j])
            if pos < 0:
                continue
            pairs_buf.append((q, index_titles[pos]))
            locs_buf.append((qi, j))
        if len(pairs_buf) >= 2048:
            scores = bge.predict(pairs_buf, batch_size=args.batch_size, show_progress_bar=False)
            for (qi2, j2), sc in zip(locs_buf, scores):
                bge_scores[qi2, j2] = float(sc)
            n_done += len(pairs_buf)
            elapsed = time.time() - t0
            rate = n_done / max(elapsed, 1e-3)
            eta = (n_pairs_total - n_done) / max(rate, 1e-3)
            print(
                f"  {n_done:,}/{n_pairs_total:,} ({n_done / n_pairs_total:.1%}) "
                f"@ {rate:.0f}/s eta {eta / 60:.1f}m",
                flush=True,
            )
            pairs_buf.clear()
            locs_buf.clear()
    if pairs_buf:
        scores = bge.predict(pairs_buf, batch_size=args.batch_size, show_progress_bar=False)
        for (qi2, j2), sc in zip(locs_buf, scores):
            bge_scores[qi2, j2] = float(sc)

    # Save scores for reuse.
    out_dir = os.path.join(INDEX_DIR, "bge_rerank")
    os.makedirs(out_dir, exist_ok=True)
    suffix = f"_n{args.max_queries}" if args.max_queries else "_all"
    np.save(os.path.join(out_dir, f"bge_scores_top{K_POOL}{suffix}.npy"), bge_scores)
    print("\nsaved BGE scores", flush=True)

    # Eval setups (only over the sampled subset).

    def eval_setup(score_matrix, label, K=K_POOL):
        rs, ns, e1s, e3s = [], [], [], []
        for qi in sample_idx:
            qid = qids[int(qi)]
            s = score_matrix[qi].copy()
            s[~valid[qi]] = -np.inf
            if score_matrix.shape[1] > K:
                s[K:] = -np.inf
            order = np.argsort(-s)[:K_EVAL]
            ordering = [faiss_pos_to_pid[int(cand[qi, int(j)])] for j in order]
            m = per_query_metrics(ordering, qrels[qid])
            if m is None:
                continue
            r, n, e1, e3 = m
            rs.append(r)
            ns.append(n)
            if e1 is not None and not math.isnan(e1):
                e1s.append(e1)
                e3s.append(e3)
        print(
            f"  {label:<48} | R@10 {np.mean(rs):.2%}  nDCG {np.mean(ns):.4f}  "
            f"E@1 {np.mean(e1s):.2%}  E@3 {np.mean(e3s):.2%}",
            flush=True,
        )

    print(f"\neval over {len(sample_idx):,} sampled queries:", flush=True)
    nm_sum = normalize_per_query(sumsim, valid)
    nm_li = normalize_per_query(liyuan_ce, valid)
    nm_bge = normalize_per_query(np.nan_to_num(bge_scores, nan=0.0), valid & ~np.isnan(bge_scores))

    eval_setup(sumsim, f"CC3-{K_POOL} (3-way sumsim only)")
    eval_setup(0.75 * nm_sum + 0.25 * nm_li, f"CC4-{K_POOL} (sumsim + LiYuan, w=0.25)")
    eval_setup(0.5 * nm_sum + 0.5 * nm_li, f"CC4-{K_POOL} (sumsim + LiYuan, w=0.50)")
    eval_setup(nm_bge, f"BGE alone (over BM25 top-{K_POOL})")
    eval_setup(0.75 * nm_sum + 0.25 * nm_bge, "sumsim + BGE w=0.25")
    eval_setup(0.5 * nm_sum + 0.5 * nm_bge, "sumsim + BGE w=0.50")
    eval_setup(0.25 * nm_sum + 0.75 * nm_bge, "sumsim + BGE w=0.75")
    # 4-way fusion: sumsim + LiYuan + BGE, equal-weight subset variant.
    nm4 = (nm_sum + nm_li + nm_bge) / 3
    eval_setup(nm4, "3-way (sumsim + LiYuan + BGE) equal mean")


if __name__ == "__main__":
    main()
