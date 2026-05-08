#!/usr/bin/env python3
"""4-architecture comparison: base / BM25 / BoD-retriever / BoD-reranker.

Generalizes evaluation/eval_bestbuy_bod_reranker.py to any corpus with
a trained BoD model. Tests the hypothesis from CHS_RESULTS.md Pattern
10: rerank > retrieve when BM25 first-stage recall is high enough,
which approximates "BM25 R@10 ≥ base R@10."

Usage:
    python evaluation/eval_rerank_vs_retrieve.py \\
        --catalog NEW/titles.json \\
        --product-ids NEW/product_ids.json \\
        --qrels NEW/test_qrels.jsonl --min-relevance 1 \\
        --queries NEW/test_queries.jsonl \\
        --base-model all-MiniLM-L6-v2 \\
        --bod-model query_model_NEW_bod \\
        --base-vecs NEW/base_catalog.vecs.fp16.npy \\
        --label NEW
"""

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import bm25s  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from sentence_transformers import SentenceTransformer  # noqa: E402

K = 10
BM25_TOP = 50


def fraction_recovered(per_q):
    if not per_q:
        return float("nan")
    return sum(h / n if n else 0 for h, n in per_q) / len(per_q)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--catalog", required=True)
    ap.add_argument("--product-ids", required=True)
    ap.add_argument("--qrels", required=True)
    ap.add_argument("--queries", required=True)
    ap.add_argument("--min-relevance", type=int, default=1)
    ap.add_argument("--base-model", default="all-MiniLM-L6-v2")
    ap.add_argument("--bod-model", required=True)
    ap.add_argument("--base-vecs", default=None)
    ap.add_argument("--bod-vecs", default=None)
    ap.add_argument("--label", default="corpus")
    args = ap.parse_args()

    print(f"=== {args.label} ===", flush=True)
    with open(args.catalog) as f:
        titles = json.load(f)
    with open(args.product_ids) as f:
        pids = json.load(f)
    pid_to_idx = {p: i for i, p in enumerate(pids)}
    qids, queries = [], []
    with open(args.queries) as f:
        for line in f:
            d = json.loads(line)
            qids.append(d["query_id"])
            queries.append(d["query"])
    pos = defaultdict(set)
    field = None
    with open(args.qrels) as f:
        for line in f:
            r = json.loads(line)
            if field is None:
                field = "product_id" if "product_id" in r else "doc_id"
            if r[field] not in pid_to_idx:
                continue
            if r["relevance"] < args.min_relevance:
                continue
            pos[r["query_id"]].add(pid_to_idx[r[field]])
    gold = [pos.get(q, set()) for q in qids]
    print(f"  catalog={len(pids):,}  queries={len(queries):,}", flush=True)

    device = "mps" if torch.backends.mps.is_available() else "cpu"

    if args.base_vecs and os.path.exists(args.base_vecs):
        print(f"  loading base vecs from {args.base_vecs}", flush=True)
        base_pv = np.load(args.base_vecs).astype(np.float32)
    else:
        print(f"  encoding catalog with {args.base_model}", flush=True)
        m = SentenceTransformer(args.base_model, device=device)
        base_pv = m.encode(
            titles, normalize_embeddings=True, batch_size=128, show_progress_bar=True
        ).astype(np.float32)
        del m
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

    if args.bod_vecs and os.path.exists(args.bod_vecs):
        print(f"  loading BoD vecs from {args.bod_vecs}", flush=True)
        bod_pv = np.load(args.bod_vecs).astype(np.float32)
    else:
        print(f"  encoding catalog with {args.bod_model}", flush=True)
        m = SentenceTransformer(args.bod_model, device=device)
        bod_pv = m.encode(
            titles, normalize_embeddings=True, batch_size=128, show_progress_bar=True
        ).astype(np.float32)
        del m
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

    print("  encoding queries (base + BoD)", flush=True)
    bm = SentenceTransformer(args.base_model, device=device)
    base_qv = bm.encode(
        queries, normalize_embeddings=True, batch_size=256, show_progress_bar=True
    ).astype(np.float32)
    del bm
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    dm = SentenceTransformer(args.bod_model, device=device)
    bod_qv = dm.encode(
        queries, normalize_embeddings=True, batch_size=256, show_progress_bar=True
    ).astype(np.float32)
    del dm
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

    print("  building BM25 index", flush=True)
    t0 = time.time()
    retriever = bm25s.BM25()
    tokenized_corpus = bm25s.tokenize(titles, stopwords="en", show_progress=False)
    retriever.index(tokenized_corpus, show_progress=False)
    tokenized_queries = bm25s.tokenize(queries, stopwords="en", show_progress=False)
    bm25_top, _ = retriever.retrieve(tokenized_queries, k=BM25_TOP, show_progress=False)
    print(f"  BM25 index+retrieve in {time.time() - t0:.0f}s", flush=True)

    print("  scoring", flush=True)
    base_rows, bm25_rows, bod_retr_rows, bod_rerk_rows = [], [], [], []
    # Chunk size scales inversely with catalog size to keep peak memory low.
    # Each `sim` chunk is (chunk × N) float32; keep < ~1GB per chunk.
    n_docs = base_pv.shape[0]
    chunk = max(64, int(2.5e8 // n_docs))
    for start in range(0, len(qids), chunk):
        end = min(start + chunk, len(qids))
        bsim = base_qv[start:end] @ base_pv.T
        b_topk = np.argpartition(-bsim, K, axis=1)[:, :K]
        del bsim
        dsim = bod_qv[start:end] @ bod_pv.T
        d_topk = np.argpartition(-dsim, K, axis=1)[:, :K]
        for j, gi in enumerate(range(start, end)):
            g = gold[gi]
            if not g:
                continue
            n = len(g)
            base_rows.append((len({int(x) for x in b_topk[j]} & g), n))
            bod_retr_rows.append((len({int(x) for x in d_topk[j]} & g), n))
            cand = bm25_top[gi]
            cand = cand[cand >= 0][:BM25_TOP].astype(np.int64)
            if len(cand) == 0:
                bm25_rows.append((0, n))
                bod_rerk_rows.append((0, n))
                continue
            top_bm25 = cand[:K]
            bm25_rows.append((len({int(x) for x in top_bm25} & g), n))
            cand_sims = dsim[j, cand]
            order = np.argsort(-cand_sims)[:K]
            rerank_top = cand[order]
            bod_rerk_rows.append((len({int(x) for x in rerank_top} & g), n))
        del dsim

    base_r = fraction_recovered(base_rows)
    bm25_r = fraction_recovered(bm25_rows)
    bod_retr_r = fraction_recovered(bod_retr_rows)
    bod_rerk_r = fraction_recovered(bod_rerk_rows)
    n_eval = len(base_rows)

    rer_wins = sum(1 for a, b in zip(bod_rerk_rows, bod_retr_rows) if a[0] > b[0])
    rer_losses = sum(1 for a, b in zip(bod_rerk_rows, bod_retr_rows) if a[0] < b[0])
    rer_ties = n_eval - rer_wins - rer_losses

    print(f"\n--- {args.label} R@10 (fraction-recovered, n={n_eval:,}) ---")
    print(f"  base                : {base_r:.4f}")
    print(f"  BM25                : {bm25_r:.4f}")
    print(f"  BoD-as-retriever    : {bod_retr_r:.4f}")
    print(f"  BoD-as-reranker     : {bod_rerk_r:.4f}")
    print(f"  Δ rerk−retr         : {bod_rerk_r - bod_retr_r:+.4f}")
    print(
        f"  BM25 vs base        : {bm25_r - base_r:+.4f}  "
        f"({'BM25 stronger' if bm25_r >= base_r else 'BM25 weaker'} → "
        f"{'rerank should win' if bm25_r >= base_r else 'retrieve should win'})"
    )
    print(
        f"  rerank H2H (W/L/T)  : "
        f"{100 * rer_wins / n_eval:>5.1f}% / "
        f"{100 * rer_losses / n_eval:>5.1f}% / "
        f"{100 * rer_ties / n_eval:>5.1f}%"
    )
    # Emit summary line for chain script aggregation.
    print(
        f"\nSUMMARY {args.label}\t{base_r:.4f}\t{bm25_r:.4f}\t"
        f"{bod_retr_r:.4f}\t{bod_rerk_r:.4f}\t"
        f"{rer_wins}\t{rer_losses}\t{rer_ties}"
    )


if __name__ == "__main__":
    main()
