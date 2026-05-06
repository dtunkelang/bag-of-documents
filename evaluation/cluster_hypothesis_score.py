#!/usr/bin/env python3
"""CLI: Cluster Hypothesis Score (CHS) for one (corpus, encoder) pair.

Loads qrels + items from local files, runs `bagofdocs.cluster_hypothesis`,
prints a formatted report. For multi-corpus comparison, see
`evaluation/chs_corpus_compare.py`.

Quick start (ESCI-US strict, default encoder):

    python evaluation/cluster_hypothesis_score.py

Custom corpus (e.g., NFCorpus or any ESCI-shaped data):

    python evaluation/cluster_hypothesis_score.py \\
        --qrels nfcorpus_data/test_qrels.jsonl \\
        --pids nfcorpus_data/doc_ids.json \\
        --titles nfcorpus_data/titles.json \\
        --id-field doc_id \\
        --partition strict

Input format expected:
    --qrels:  JSONL, each line {"query_id": ..., "<id_field>": ..., "relevance": INT}
    --pids:   JSON list of all item ids in the catalog
    --titles: JSON list of title/text strings, parallel to --pids
    --id-field: name of the item-id field in qrels (default "product_id";
                NFCorpus uses "doc_id")

See `bagofdocs/cluster_hypothesis.py` for the metric definitions and
`evaluation/CHS_RESULTS.md` for calibration data across corpora.
"""

import argparse
import json
import os
import statistics
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from bagofdocs.cluster_hypothesis import compute_chs, schs_verdict  # noqa: E402

SCRIPT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
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
        help="strict: top-grade vs bottom-grade; relaxed: include adjacent grades",
    )
    ap.add_argument(
        "--id-field",
        default="product_id",
        help="qrels item-id field name (e.g., 'product_id' for ESCI, 'doc_id' for NFCorpus)",
    )
    ap.add_argument(
        "--pos-grade",
        type=int,
        default=None,
        help="positive relevance grade (default: max grade in qrels)",
    )
    ap.add_argument("--neg-grade", type=int, default=0, help="negative relevance grade (default 0)")
    ap.add_argument(
        "--n-random-pairs",
        type=int,
        default=200_000,
        help="number of random product pairs for baseline cosine",
    )
    ap.add_argument(
        "--cache-vecs",
        default=None,
        help="optional .npy of precomputed item vectors aligned with --pids order",
    )
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    print(f"loading qrels from {args.qrels}...", flush=True)
    qrels = defaultdict(dict)
    with open(args.qrels) as f:
        for line in f:
            r = json.loads(line)
            qrels[r["query_id"]][r[args.id_field]] = r["relevance"]
    qrels = dict(qrels)

    print(f"loading items from {args.pids} + {args.titles}...", flush=True)
    with open(args.pids) as f:
        pids = json.load(f)
    with open(args.titles) as f:
        titles = json.load(f)
    print(f"  {len(pids):,} items", flush=True)

    cache_vecs = None
    if args.cache_vecs and os.path.exists(args.cache_vecs):
        print(f"loading cached vectors from {args.cache_vecs}...", flush=True)
        cache_vecs = np.load(args.cache_vecs).astype(np.float32, copy=False)

    res = compute_chs(
        qrels=qrels,
        pids=pids,
        titles=titles,
        encoder_name=args.encoder,
        partition=args.partition,
        pos_grade=args.pos_grade,
        neg_grade=args.neg_grade,
        seed=args.seed,
        n_random_pairs=args.n_random_pairs,
        cache_vecs=cache_vecs,
    )

    print("\n" + "=" * 72)
    print(f"CLUSTER HYPOTHESIS SCORE — {os.path.basename(args.qrels)} under {args.encoder}")
    print(f"partition: {args.partition}")
    print(
        f"pos-bearing queries (>=2 pos): {res.n_pos_bearing:,}; "
        f"explicit-neg queries: {res.n_explicit_neg:,}; "
        f"items touched: {res.n_products_touched:,}"
    )
    print("=" * 72)
    print()
    print(f"  mean intra-bag cosine     (pos-pos):     {res.mean_intra:.4f}")
    if res.has_explicit_negs:
        print(f"  mean inter-bag cosine     (pos-neg):     {res.mean_inter_neg:.4f}")
    else:
        print("  mean inter-bag cosine     (pos-neg):     n/a (no explicit negatives)")
    print(f"  mean random-pair cosine   (baseline):    {res.mean_random:.4f}")
    print()
    print(f"  SCHS  (intra vs random):                 {res.schs:.4f}")
    print("        (intra - random) / (1 - random); range [0, 1]")
    if res.has_explicit_negs:
        print(f"  HCHS  (intra vs in-query neg):           {res.hchs:.4f}")
        print("        (intra - inter_neg) / (intra - random); range [0, 1]")
        print(f"  strong_inv_rate (pn_max > pp_max):       {res.strong_inv_rate:.1%}")
    else:
        print("  HCHS:                                    n/a (no explicit negatives)")
        print("  strong_inv_rate:                         n/a")
    print()

    if res.pp_means:
        print("  Per-query distribution:")
        print(
            f"    pp_mean  median={statistics.median(res.pp_means):.3f}  "
            f"p25={np.percentile(res.pp_means, 25):.3f}  "
            f"p75={np.percentile(res.pp_means, 75):.3f}"
        )
        print(
            f"    pp_min   median={statistics.median(res.pp_mins):.3f}  "
            f"p25={np.percentile(res.pp_mins, 25):.3f}  "
            f"p75={np.percentile(res.pp_mins, 75):.3f}"
        )
        if res.pn_means:
            print(
                f"    pn_mean  median={statistics.median(res.pn_means):.3f}  "
                f"p25={np.percentile(res.pn_means, 25):.3f}  "
                f"p75={np.percentile(res.pn_means, 75):.3f}"
            )
        else:
            print("    pn_mean  n/a (no explicit negatives)")

    print()
    print(f"  Verdict: {schs_verdict(res.schs, res.n_pos_bearing)}")
    print()

    out_path = "/tmp/cluster_hypothesis_score.json"
    with open(out_path, "w") as f:
        json.dump({"args": vars(args), **res.to_dict()}, f, indent=2)
    print(f"  saved structured score to {out_path}")


if __name__ == "__main__":
    main()
