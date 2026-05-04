#!/usr/bin/env python3
"""Paired bootstrap CIs: FAISS-hn variant vs rerank_A baseline (retriever and rerank)."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

with open("/tmp/faisshn_skip30_probe_per_query.json") as _f:
    DATA = json.load(_f)
PER_Q = DATA["per_query"]


def boot(arr_a, arr_k, n_iter=1000, seed=42):
    arr_a = np.array(arr_a, dtype=float)
    arr_k = np.array(arr_k, dtype=float)
    mask = (arr_a >= 0) & (arr_k >= 0)
    a, k = arr_a[mask], arr_k[mask]
    rng = np.random.default_rng(seed)
    n = len(a)
    deltas = []
    for _ in range(n_iter):
        idx = rng.integers(0, n, n)
        deltas.append(k[idx].mean() - a[idx].mean())
    deltas = np.array(deltas)
    return {
        "n": int(n),
        "delta_mean": float(deltas.mean()),
        "ci_low": float(np.percentile(deltas, 2.5)),
        "ci_high": float(np.percentile(deltas, 97.5)),
        "p_k_better": float((deltas > 0).mean()),
    }


pairs = [
    (
        "retriever R@10: faisshn vs rerank_A",
        "K1_rerank_K_faisshn_retriever",
        "A1_rerank_A_retriever",
        "recall",
    ),
    (
        "retriever E@1:  faisshn vs rerank_A",
        "K1_rerank_K_faisshn_retriever",
        "A1_rerank_A_retriever",
        "e_at_1",
    ),
    (
        "rerank R@10:    faisshn vs rerank_A",
        "K2_bm25top50_rerank_K_faisshn",
        "A2_bm25top50_rerank_A",
        "recall",
    ),
    (
        "rerank E@1:     faisshn vs rerank_A",
        "K2_bm25top50_rerank_K_faisshn",
        "A2_bm25top50_rerank_A",
        "e_at_1",
    ),
]

print("setup                                  delta(K-A) [95% CI]              n     p(K>A)")
print("-" * 90)
for label, k_setup, a_setup, metric in pairs:
    res = boot(PER_Q[a_setup][metric], PER_Q[k_setup][metric])
    print(
        f"{label}  {res['delta_mean'] * 100:+6.2f}pp [{res['ci_low'] * 100:+6.2f}, "
        f"{res['ci_high'] * 100:+6.2f}]  {res['n']:>5}  {res['p_k_better']:.3f}"
    )
