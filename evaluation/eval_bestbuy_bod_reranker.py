#!/usr/bin/env python3
"""BoD-as-reranker vs BoD-as-retriever on BestBuy holdout.

We've shown on ESCI that BoD-as-reranker (BM25 top-50 reranked by BoD
cosine) beats BoD-as-retriever (BoD cosine over the full catalog) on
~24% of queries, ties on ~55%, loses on ~16%. This script checks
whether that pattern generalizes to clicks-trained BoD on BestBuy.

Compares 4 architectures on the BestBuy 12,128-query holdout:

  base MiniLM           dense retrieval over the full 53K catalog
  BoD-as-retriever      query_model_bestbuy_bod over the full 53K catalog
  BM25 alone            bm25s top-K, no rerank
  BoD-as-reranker       BM25 top-50, reranked by BoD cosine

Metric: fraction-recovered R@10 (mean over queries of
positives_in_top_10 / total_positives) — same metric used in the
4-corpus framework table.
"""

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

DATA = "bestbuy_acm_data"
K = 10
BM25_TOP = 50


def fraction_recovered(per_q):
    """Mean over queries of (positives_in_top_K / total_positives)."""
    if not per_q:
        return float("nan")
    return sum(h / n if n else 0 for h, n in per_q) / len(per_q)


def main():
    print("loading data...", flush=True)
    with open(os.path.join(DATA, "titles.json")) as f:
        titles = json.load(f)
    with open(os.path.join(DATA, "product_ids.json")) as f:
        pids = json.load(f)
    pid_to_idx = {p: i for i, p in enumerate(pids)}
    qids, queries = [], []
    with open(os.path.join(DATA, "holdout_queries.jsonl")) as f:
        for line in f:
            d = json.loads(line)
            qids.append(d["query_id"])
            queries.append(d["query"])
    pos = defaultdict(set)
    with open(os.path.join(DATA, "holdout_qrels.jsonl")) as f:
        for line in f:
            r = json.loads(line)
            if r["product_id"] not in pid_to_idx:
                continue
            pos[r["query_id"]].add(pid_to_idx[r["product_id"]])
    gold = [pos.get(q, set()) for q in qids]
    n_with_gold = sum(1 for g in gold if g)
    print(
        f"  catalog={len(pids):,}  queries={len(queries):,}  with-gold={n_with_gold:,}",
        flush=True,
    )

    # --- Vectors ---------------------------------------------------------
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    base_pv = np.load(os.path.join(DATA, "base_catalog.vecs.fp16.npy")).astype(np.float32)

    print("loading base model + encoding base queries...", flush=True)
    base = SentenceTransformer("all-MiniLM-L6-v2", device=device)
    base_qv = base.encode(
        queries, normalize_embeddings=True, batch_size=256, show_progress_bar=True
    ).astype(np.float32)
    del base
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

    print("loading BoD model + encoding BoD catalog...", flush=True)
    bod = SentenceTransformer("query_model_bestbuy_bod", device=device)
    bod_pv = bod.encode(
        titles, normalize_embeddings=True, batch_size=128, show_progress_bar=True
    ).astype(np.float32)
    print("encoding BoD queries...", flush=True)
    bod_qv = bod.encode(
        queries, normalize_embeddings=True, batch_size=256, show_progress_bar=True
    ).astype(np.float32)
    del bod
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

    # --- BM25 index ------------------------------------------------------
    print("building BM25 index over titles...", flush=True)
    t0 = time.time()
    retriever = bm25s.BM25()
    tokenized_corpus = bm25s.tokenize(titles, stopwords="en", show_progress=False)
    retriever.index(tokenized_corpus, show_progress=False)
    print(f"  indexed {len(titles):,} titles in {time.time() - t0:.0f}s", flush=True)

    print(f"running BM25 top-{BM25_TOP} for {len(queries):,} queries...", flush=True)
    t0 = time.time()
    tokenized_queries = bm25s.tokenize(queries, stopwords="en", show_progress=False)
    bm25_top, _scores = retriever.retrieve(tokenized_queries, k=BM25_TOP, show_progress=False)
    print(f"  retrieved in {time.time() - t0:.0f}s", flush=True)

    # --- Eval each architecture ------------------------------------------
    print("\nscoring per query...", flush=True)
    base_rows = []
    bod_retr_rows = []
    bm25_rows = []
    bod_rerk_rows = []
    chunk = 1024
    for start in range(0, len(qids), chunk):
        end = min(start + chunk, len(qids))
        bsim = base_qv[start:end] @ base_pv.T
        dsim = bod_qv[start:end] @ bod_pv.T
        b_topk = np.argpartition(-bsim, K, axis=1)[:, :K]
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
        del bsim, dsim

    base_r = fraction_recovered(base_rows)
    bod_retr_r = fraction_recovered(bod_retr_rows)
    bm25_r = fraction_recovered(bm25_rows)
    bod_rerk_r = fraction_recovered(bod_rerk_rows)

    # Per-query head-to-head: BoD-as-reranker vs BoD-as-retriever.
    rer_wins = sum(1 for a, b in zip(bod_rerk_rows, bod_retr_rows) if a[0] > b[0])
    rer_losses = sum(1 for a, b in zip(bod_rerk_rows, bod_retr_rows) if a[0] < b[0])
    rer_ties = sum(1 for a, b in zip(bod_rerk_rows, bod_retr_rows) if a[0] == b[0])

    n_eval = len(base_rows)
    print("\n" + "=" * 72)
    print(f"BestBuy holdout R@10 (fraction-recovered, n={n_eval:,})")
    print("=" * 72)
    print(f"  base MiniLM retrieval                : {base_r:.4f}")
    print(f"  BM25 alone (bm25s, default tokenizer): {bm25_r:.4f}")
    print(f"  BoD-as-retriever (full catalog)      : {bod_retr_r:.4f}")
    print(f"  BoD-as-reranker (BM25 top-50)        : {bod_rerk_r:.4f}")
    print()
    print(f"  Δ BoD-reranker − BoD-retriever       : {bod_rerk_r - bod_retr_r:+.4f}")
    print(f"  Δ BoD-reranker − BM25                : {bod_rerk_r - bm25_r:+.4f}")
    print(f"  Δ BoD-reranker − base                : {bod_rerk_r - base_r:+.4f}")
    print()
    print("  Per-query head-to-head: BoD-reranker vs BoD-retriever")
    pct = lambda x: 100.0 * x / n_eval  # noqa: E731
    print(f"    reranker wins : {rer_wins:>6,}  ({pct(rer_wins):>5.1f}%)")
    print(f"    reranker loses: {rer_losses:>6,}  ({pct(rer_losses):>5.1f}%)")
    print(f"    ties          : {rer_ties:>6,}  ({pct(rer_ties):>5.1f}%)")


if __name__ == "__main__":
    main()
