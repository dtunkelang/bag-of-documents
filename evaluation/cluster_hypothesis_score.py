#!/usr/bin/env python3
"""Cluster-Hypothesis Score (CHS) for a corpus + encoder.

Operationalizes the van Rijsbergen cluster hypothesis: "closely related
documents tend to be relevant to the same requests." For BoD specifically:
documents that share a query (i.e., would land in the same bag) should be
close to each other in embedding space, and closer to each other than to
documents that don't share the query.

Given a corpus and an encoder, the score answers: how well does the
embedding geometry under this encoder line up with the relevance structure
of the corpus? Higher CHS predicts BoD will generalize.

Inputs (qrels mode):
  - qrels.jsonl: {query_id, product_id, relevance}
  - product_id -> title mapping
  - encoder name (sentence-transformers compatible)

Output: a structured score report.

Two metrics layered:
  1. mean_intra: average cosine between two positives for the same query
  2. mean_inter_neg: average cosine between a positive and a negative for
     the same query
  3. mean_random: average cosine between two random products (baseline)
  4. CHS_raw      = mean_intra - mean_inter_neg
  5. CHS_normed   = (mean_intra - mean_inter_neg) / max(mean_intra - mean_random, eps)
                    Range typically [0, 1]: 1 means the in-bag/out-of-bag
                    gap matches the in-bag/random gap (cluster hypothesis
                    saturates the available signal). 0 means in-bag pairs
                    are no closer to each other than to negatives.
  6. strong_inv_rate: % of eligible queries where some pos-neg cosine
                     exceeds the best pos-pos cosine. The cluster
                     hypothesis fails for that query.

Calibration (single confirmed positive: ESCI-US under all-MiniLM-L6-v2):
  ESCI-US strict (E vs I):
    mean_intra=0.577  mean_inter_neg=0.412  mean_random=0.131
    CHS_normed=0.37   strong_inv_rate=16.2%
  ESCI-US is BoD-positive (CC5 R@10 23.57 vs base 15.60), so 0.37 is the
  empirical floor for "BoD works." Anything materially below 0.20 is in
  prior-negative territory (NFCorpus / MS MARCO weren't measured here but
  failed to lift in prior probes; rough expectation: chs_normed < 0.2).

Suggested thresholds (with the caveat that we have ONE calibration point):
  - chs_normed >= 0.30: GREEN — at-or-above ESCI level; BoD plausibly works
  - 0.15 <= chs_normed < 0.30: YELLOW — pilot before full investment
  - chs_normed < 0.15: RED — BoD unlikely to lift over base
These should be validated on more corpora before being load-bearing.

Usage:
  python evaluation/cluster_hypothesis_score.py \
      --qrels esci_us_data/test_qrels.jsonl \
      --pids esci_us_data/product_ids.json \
      --titles esci_us_data/titles.json \
      --encoder all-MiniLM-L6-v2 \
      --partition strict
  python evaluation/cluster_hypothesis_score.py --partition relaxed
"""

import argparse
import json
import os
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

SCRIPT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qrels", default=os.path.join(SCRIPT_DIR, "esci_us_data/test_qrels.jsonl"))
    ap.add_argument("--pids", default=os.path.join(SCRIPT_DIR, "esci_us_data/product_ids.json"))
    ap.add_argument("--titles", default=os.path.join(SCRIPT_DIR, "esci_us_data/titles.json"))
    ap.add_argument(
        "--encoder",
        default="all-MiniLM-L6-v2",
        help="sentence-transformers encoder for the embedding space",
    )
    ap.add_argument(
        "--partition",
        choices=["strict", "relaxed"],
        default="strict",
        help="strict = E vs I; relaxed = E+S vs I+C",
    )
    ap.add_argument(
        "--max-queries",
        type=int,
        default=0,
        help="0 = use all eligible queries; otherwise sample first N",
    )
    ap.add_argument(
        "--n-random-pairs",
        type=int,
        default=200_000,
        help="number of random product pairs for baseline cosine",
    )
    ap.add_argument(
        "--cache-vecs",
        default=None,
        help=(
            "optional path to a .npy of precomputed product vectors aligned "
            "with --pids order. Skips encoding if present."
        ),
    )
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    print(f"loading qrels from {args.qrels}...", flush=True)
    qrels = defaultdict(dict)
    with open(args.qrels) as f:
        for line in f:
            r = json.loads(line)
            qrels[r["query_id"]][r["product_id"]] = r["relevance"]
    qrels = dict(qrels)

    print(f"loading product map from {args.pids} + {args.titles}...", flush=True)
    with open(args.pids) as f:
        pids = json.load(f)
    with open(args.titles) as f:
        titles = json.load(f)
    pid_to_pos = {p: i for i, p in enumerate(pids)}
    print(f"  {len(pids):,} products", flush=True)

    if args.partition == "strict":
        is_pos = lambda g: g == 3  # noqa: E731
        is_neg = lambda g: g == 0  # noqa: E731
        partition_label = "E vs I"
    else:
        is_pos = lambda g: g >= 2  # noqa: E731
        is_neg = lambda g: g <= 1  # noqa: E731
        partition_label = "E+S vs I+C"

    # Eligible queries: >=2 positives and >=1 negative in the catalog.
    eligible = []
    for qid, qr in qrels.items():
        pos = [p for p, g in qr.items() if is_pos(g) and p in pid_to_pos]
        neg = [p for p, g in qr.items() if is_neg(g) and p in pid_to_pos]
        if len(pos) >= 2 and len(neg) >= 1:
            eligible.append((qid, pos, neg))
    print(
        f"\npartition: {partition_label}; {len(eligible):,} eligible queries "
        f"(>=2 pos, >=1 neg in catalog)",
        flush=True,
    )

    if args.max_queries > 0 and len(eligible) > args.max_queries:
        eligible = eligible[: args.max_queries]
        print(f"  limited to first {len(eligible):,}", flush=True)

    # Resolve product positions touched by these queries (saves encoding cost)
    touched = set()
    for _, pos, neg in eligible:
        for p in pos:
            touched.add(pid_to_pos[p])
        for p in neg:
            touched.add(pid_to_pos[p])
    touched = sorted(touched)
    print(f"  {len(touched):,} unique products touched by eligible queries", flush=True)

    # Encode just the touched subset (or load cached)
    if args.cache_vecs and os.path.exists(args.cache_vecs):
        print(f"loading cached product vectors from {args.cache_vecs}...", flush=True)
        pv_full = np.load(args.cache_vecs).astype(np.float32)
        if pv_full.shape[0] != len(pids):
            raise ValueError(
                f"cached vecs have {pv_full.shape[0]} rows, expected {len(pids)} "
                f"(must align with --pids order)"
            )
        pv = pv_full[touched]
    else:
        device = (
            "mps"
            if torch.backends.mps.is_available()
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        print(f"encoding {len(touched):,} products with {args.encoder} on {device}...", flush=True)
        t0 = time.time()
        model = SentenceTransformer(args.encoder, device=device)
        texts = [titles[i] for i in touched]
        pv = model.encode(
            texts,
            normalize_embeddings=True,
            batch_size=128,
            show_progress_bar=True,
        ).astype(np.float32)
        print(f"  done in {time.time() - t0:.0f}s, shape={pv.shape}", flush=True)

    pos_remap = {orig_idx: subset_idx for subset_idx, orig_idx in enumerate(touched)}

    # Per-query metrics
    pp_means, pp_mins, pp_maxes = [], [], []
    pn_means, pn_maxes = [], []
    n_strong_inv = 0
    n_total = 0
    for _qid, pos, neg in eligible:
        pos_idx = [pos_remap[pid_to_pos[p]] for p in pos]
        neg_idx = [pos_remap[pid_to_pos[p]] for p in neg]
        pp = pv[pos_idx]
        nn = pv[neg_idx]

        pp_sims = pp @ pp.T
        iu = np.triu_indices(len(pp_sims), k=1)
        pp_pairs = pp_sims[iu]
        pp_means.append(float(pp_pairs.mean()))
        pp_mins.append(float(pp_pairs.min()))
        pp_max = float(pp_pairs.max())
        pp_maxes.append(pp_max)

        pn_sims = pp @ nn.T
        pn_means.append(float(pn_sims.mean()))
        pn_max = float(pn_sims.max())
        pn_maxes.append(pn_max)

        n_total += 1
        if pn_max > pp_max:
            n_strong_inv += 1

    # Random-pair baseline (over the touched subset, fast)
    rng = np.random.default_rng(args.seed)
    n_subset = pv.shape[0]
    # Sample without replacement of pairs
    pairs_a = rng.integers(0, n_subset, args.n_random_pairs)
    pairs_b = rng.integers(0, n_subset, args.n_random_pairs)
    same = pairs_a == pairs_b
    pairs_a = pairs_a[~same]
    pairs_b = pairs_b[~same]
    random_sims = (pv[pairs_a] * pv[pairs_b]).sum(axis=1)
    mean_random = float(random_sims.mean())

    mean_intra = float(np.mean(pp_means))
    mean_inter_neg = float(np.mean(pn_means))
    chs_raw = mean_intra - mean_inter_neg
    chs_normed = chs_raw / max(mean_intra - mean_random, 1e-6)
    strong_inv_rate = n_strong_inv / max(n_total, 1)

    print("\n" + "=" * 72)
    print(f"CLUSTER HYPOTHESIS SCORE — {os.path.basename(args.qrels)} under {args.encoder}")
    print(f"partition: {partition_label}; n_eligible_queries: {n_total:,}")
    print("=" * 72)
    print()
    print(f"  mean intra-bag cosine     (pos-pos):     {mean_intra:.4f}")
    print(f"  mean inter-bag cosine     (pos-neg):     {mean_inter_neg:.4f}")
    print(f"  mean random-pair cosine   (baseline):    {mean_random:.4f}")
    print()
    print(f"  CHS_raw    (intra - inter_neg):          {chs_raw:+.4f}")
    print(f"  CHS_normed (raw / (intra - random)):     {chs_normed:.4f}")
    print(f"  strong_inv_rate (pn_max > pp_max):       {strong_inv_rate:.1%}")
    print()
    print("  Distribution (per-query):")
    print(
        f"    pp_mean  median={statistics.median(pp_means):.3f}  "
        f"p25={np.percentile(pp_means, 25):.3f}  p75={np.percentile(pp_means, 75):.3f}"
    )
    print(
        f"    pp_min   median={statistics.median(pp_mins):.3f}  "
        f"p25={np.percentile(pp_mins, 25):.3f}  p75={np.percentile(pp_mins, 75):.3f}"
    )
    print(
        f"    pn_mean  median={statistics.median(pn_means):.3f}  "
        f"p25={np.percentile(pn_means, 25):.3f}  p75={np.percentile(pn_means, 75):.3f}"
    )
    print()

    if chs_normed >= 0.30:
        verdict = "GREEN — at-or-above ESCI-US calibration point; BoD plausibly works"
    elif chs_normed >= 0.15:
        verdict = "YELLOW — below ESCI baseline; pilot before full investment"
    else:
        verdict = "RED — well below ESCI; BoD unlikely to lift over base"
    print(f"  Verdict (ESCI-US-anchored): {verdict}")
    print("    Calibration: ESCI-US (BoD-positive) gives chs_normed=0.37 under all-MiniLM-L6-v2.")
    print()

    out = {
        "qrels": args.qrels,
        "encoder": args.encoder,
        "partition": partition_label,
        "n_eligible": n_total,
        "mean_intra_bag_cosine": mean_intra,
        "mean_inter_neg_cosine": mean_inter_neg,
        "mean_random_pair_cosine": mean_random,
        "chs_raw": chs_raw,
        "chs_normed": chs_normed,
        "strong_inv_rate": strong_inv_rate,
        "pp_means_median": statistics.median(pp_means),
        "pp_mins_median": statistics.median(pp_mins),
        "pn_means_median": statistics.median(pn_means),
        "verdict": verdict,
    }
    out_path = "/tmp/cluster_hypothesis_score.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"  saved structured score to {out_path}")


if __name__ == "__main__":
    main()
