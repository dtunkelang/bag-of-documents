#!/usr/bin/env python3
"""Find ESCI test queries where BoD-as-reranker beats BoD-as-retriever.

Compares per-query top-10 R@10 (E-grade only) under two modes that use the
SAME BoD model (6M-MNRL), so the only difference is the architecture:

  bod_retriever   cosine top-10 over the full 1.2M-product catalog
  bod_reranker    BM25 top-50 candidates, reranked by the same BoD cosine

Reports the top-N queries where the reranker beats the retriever — useful
demo examples for showing why rerank-over-BM25 wins on lexically-specific
queries even when the BoD model is identical.
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

INDEX = "combined_index_us_minilm"
DATA = "esci_us_data"
K = 10
BM25_TOP = 50


def main():
    print("loading data...", flush=True)
    with open(f"{DATA}/product_ids.json") as f:
        pids = json.load(f)
    pid_to_idx = {p: i for i, p in enumerate(pids)}
    queries = {}
    with open(f"{DATA}/test_queries.jsonl") as f:
        for line in f:
            d = json.loads(line)
            queries[d["query_id"]] = d["query"]
    e_pos = defaultdict(set)
    with open(f"{DATA}/test_qrels.jsonl") as f:
        for line in f:
            r = json.loads(line)
            if r["product_id"] not in pid_to_idx:
                continue
            if r["relevance"] == 3:
                e_pos[r["query_id"]].add(pid_to_idx[r["product_id"]])

    qids = [q for q in sorted(queries) if e_pos.get(q)]
    qtexts = [queries[q] for q in qids]
    print(f"  {len(qids):,} pos-bearing test queries", flush=True)

    bm25_top = np.load(f"{INDEX}/bm25_top200.npy")[:, :BM25_TOP]
    with open(f"{INDEX}/bm25s_qids.json") as f:
        bm25_qid_order = json.load(f)
    bm25_qid_to_row = {q: i for i, q in enumerate(bm25_qid_order)}

    print("loading rerank_A vecs (BoD catalog, fp16 -> fp32)...", flush=True)
    pv = np.load(f"{INDEX}/rerank_A.vecs.fp16.npy").astype(np.float32)

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print("encoding queries with rerank_A (6M-MNRL)...", flush=True)
    a = SentenceTransformer("query_model_us_full_6m_mnrl", device=device)
    qv = a.encode(qtexts, normalize_embeddings=True, batch_size=256, show_progress_bar=True).astype(
        np.float32
    )
    del a
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

    print("scoring per query (chunked)...", flush=True)
    rows = []
    chunk = 256
    for start in range(0, len(qids), chunk):
        end = min(start + chunk, len(qids))
        sims = qv[start:end] @ pv.T  # (chunk, N) full-catalog cosine for retriever
        topk_retr = np.argpartition(-sims, K, axis=1)[:, :K]
        for j, gi in enumerate(range(start, end)):
            qid = qids[gi]
            qtext = qtexts[gi]
            gold = e_pos.get(qid, set())
            if not gold:
                continue
            r_hits = len({int(x) for x in topk_retr[j]} & gold)
            row_idx = bm25_qid_to_row.get(qid)
            if row_idx is None:
                continue
            cand = bm25_top[row_idx]
            cand = cand[cand >= 0][:BM25_TOP].astype(np.int64)
            if len(cand) == 0:
                continue
            cand_sims = sims[j, cand]
            order = np.argsort(-cand_sims)[:K]
            rerank_top = cand[order]
            d_hits = len({int(x) for x in rerank_top} & gold)
            rows.append((qid, qtext, len(gold), r_hits, d_hits, d_hits - r_hits))
        del sims, topk_retr
        print(f"  {end:,}/{len(qids):,} queries scored", flush=True)

    print(f"\nscored {len(rows):,} queries\n", flush=True)

    # Filter: at least 2 gold, reranker strictly wins.
    wins = [r for r in rows if r[2] >= 2 and r[5] > 0]
    wins.sort(key=lambda r: (-r[5], -r[4]))

    print("=" * 78)
    print("TOP queries where BoD-reranker beats BoD-retriever (R@10 on E-grade)")
    print("Same BoD model on both sides; only architecture differs.")
    print("=" * 78)
    print(f"{'Δ':>2} {'gold':>4} {'retr':>4} {'rerk':>4}  query")
    for r in wins[:50]:
        print(f"{r[5]:>2} {r[2]:>4} {r[3]:>4} {r[4]:>4}  {r[1]!r}")

    # Aggregate stats.
    print(f"\nTotal queries where reranker beats retriever (Δ>0, ≥2 gold): {len(wins):,}")
    print(
        f"Total queries where retriever beats reranker (Δ<0, ≥2 gold):  "
        f"{sum(1 for r in rows if r[2] >= 2 and r[5] < 0):,}"
    )
    print(
        f"Ties:                                                         "
        f"{sum(1 for r in rows if r[2] >= 2 and r[5] == 0):,}"
    )


if __name__ == "__main__":
    main()
