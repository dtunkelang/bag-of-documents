#!/usr/bin/env python3
"""Per-query Spearman rank correlations between A / B / G / CE scoring
streams over BM25 top-100 candidates.

Tells us how orthogonal the four signals are. If CE is highly correlated
with mean(A,B,G), distillation is theoretically doomed (CE just amplifies
existing bi-encoder signal). If correlation is low (~0.5 or below), CE
has substantial orthogonal signal and scaled distillation has real
headroom.

Reuses cached top-100 artifacts. Pure CPU.

Usage:
    python evaluation/encoder_correlations.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json  # noqa: E402
import os  # noqa: E402
from collections import defaultdict  # noqa: E402

import numpy as np  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

SCRIPT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INDEX_DIR = os.path.join(SCRIPT_DIR, "combined_index_us_minilm")


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
    qids = [qid for qid in queries_all if qid in qrels and any(g >= 2 for g in qrels[qid].values())]
    print(f"  {len(qids):,} eval queries", flush=True)

    candidate_pos = np.load(os.path.join(INDEX_DIR, "ce_top100_candidates.npy"))
    sims_a = np.load(os.path.join(INDEX_DIR, "ce_top100_sims_a.npy"))
    sims_b = np.load(os.path.join(INDEX_DIR, "ce_top100_sims_b.npy"))
    sims_g = np.load(os.path.join(INDEX_DIR, "ce_top100_sims_g.npy"))
    ce_scores = np.load(os.path.join(INDEX_DIR, "ce_top100_scores.npy"))
    valid = candidate_pos >= 0

    # Per query, compute Spearman corr between each pair of streams.
    print("computing per-query Spearman correlations...", flush=True)
    pairs = [
        ("A", "B", sims_a, sims_b),
        ("A", "G", sims_a, sims_g),
        ("B", "G", sims_b, sims_g),
        ("A", "CE", sims_a, ce_scores),
        ("B", "CE", sims_b, ce_scores),
        ("G", "CE", sims_g, ce_scores),
        ("mean(ABG)", "CE", (sims_a + sims_b + sims_g) / 3, ce_scores),
    ]

    n_q = len(qids)
    corrs_per_pair = {f"{a} vs {b}": np.full(n_q, np.nan, dtype=np.float64) for a, b, _, _ in pairs}

    for qi in range(n_q):
        m = valid[qi]
        if m.sum() < 3:
            continue
        for a_label, b_label, sa, sb in pairs:
            xa = sa[qi, m]
            xb = sb[qi, m]
            if np.std(xa) < 1e-10 or np.std(xb) < 1e-10:
                continue
            rho, _ = spearmanr(xa, xb)
            corrs_per_pair[f"{a_label} vs {b_label}"][qi] = rho

    print(f"\n{'pair':<24} | {'mean':>7} {'median':>7} {'p10':>7} {'p90':>7}")
    print("-" * 64)
    for pair_label, vals in corrs_per_pair.items():
        v = vals[~np.isnan(vals)]
        if v.size == 0:
            continue
        print(
            f"{pair_label:<24} | "
            f"{v.mean():>6.3f} {np.median(v):>7.3f} "
            f"{np.percentile(v, 10):>7.3f} {np.percentile(v, 90):>7.3f}"
        )

    # Headline reading
    mean_abg_ce = corrs_per_pair["mean(ABG) vs CE"]
    v = mean_abg_ce[~np.isnan(mean_abg_ce)]
    median_corr = float(np.median(v))
    print(f"\nmedian Spearman(mean(A,B,G), CE) = {median_corr:.3f}")
    if median_corr > 0.85:
        verdict = "HIGH redundancy: CE is mostly the same ranking as bi-encoder mean. Distillation has limited theoretical headroom."
    elif median_corr > 0.7:
        verdict = "MODERATE redundancy. Distillation could capture some CE signal but ceiling is near current ensemble."
    else:
        verdict = "LOW redundancy: CE has substantial orthogonal signal. Scaled distillation has real headroom."
    print(f"  → {verdict}")


if __name__ == "__main__":
    main()
