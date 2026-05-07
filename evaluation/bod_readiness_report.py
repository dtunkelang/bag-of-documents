#!/usr/bin/env python3
"""One-shot BoD readiness report — predict lift on a new corpus before training.

Combines two pre-training measurements into a verdict:

  1. SCHS (cluster hypothesis score, `bagofdocs.cluster_hypothesis`):
     does the corpus's relevance structure cluster under the encoder?
     Necessary floor — if SCHS < 0.40, BoD is unlikely to lift regardless.

  2. Base-difficulty distribution (`measure_base_difficulty.py`-style):
     how much headroom does the base encoder leave? Specifically the
     base-blind subset (queries where base R@10 = 0) and the
     base-perfect subset (queries where base R@10 = 1).

The framework's predicted lift comes from the 5-corpus calibration table
in `evaluation/CHS_RESULTS.md`:

    lift ≈ (base_blind_size × rescue_rate) − (base_perfect_size × spec_tax)

Rescue rate and spec tax depend on bag signal quality, which can't be
measured without training. We give a band over plausible signal levels:

    pessimistic   rescue ~5pp,  tax ~15pp  (NFCorpus-like, noisy qrels)
    realistic     rescue ~12pp, tax ~10pp  (ESCI-like, graded qrels)
    optimistic    rescue ~25pp, tax ~6pp   (BestBuy-like, click data)

Then a verdict:
    GO            optimistic >= 5pp AND realistic > 1pp AND SCHS >= 0.40
    CONDITIONAL   realistic > 1pp but optimistic < 5pp; depends on bag signal
    SKIP          realistic <= 1pp OR SCHS < 0.40

Usage:
    python evaluation/bod_readiness_report.py \\
        --catalog corpus/titles.json \\
        --product-ids corpus/product_ids.json \\
        --qrels corpus/test_qrels.jsonl --min-relevance 1 \\
        --queries corpus/test_queries.jsonl \\
        --encoder all-MiniLM-L6-v2 \\
        --vecs-cache corpus/base_catalog.vecs.fp16.npy
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from sentence_transformers import SentenceTransformer  # noqa: E402

from bagofdocs.cluster_hypothesis import compute_chs, schs_verdict  # noqa: E402

# Calibration constants from CHS_RESULTS.md (5-corpus rescue/tax table).
RESCUE_BANDS = {"pessimistic": 0.05, "realistic": 0.12, "optimistic": 0.25}
TAX_BANDS = {"pessimistic": 0.15, "realistic": 0.10, "optimistic": 0.06}
SCHS_FLOOR = 0.40


def predict_lift(base_blind, base_perfect):
    """Return dict of {scenario: predicted_lift_pp} for the 3 signal bands."""
    out = {}
    for scenario in RESCUE_BANDS:
        out[scenario] = base_blind * RESCUE_BANDS[scenario] - base_perfect * TAX_BANDS[scenario]
    return out


def verdict(schs, base_blind, base_perfect, predicted):
    if schs < SCHS_FLOOR:
        return (
            "SKIP",
            f"SCHS={schs:.3f} below the {SCHS_FLOOR:.2f} floor; clustering structure is too weak.",
        )
    realistic = predicted["realistic"]
    optimistic = predicted["optimistic"]
    if realistic > 0.01 and optimistic >= 0.05:
        return (
            "GO",
            f"realistic predicted lift {realistic * 100:+.1f}pp; "
            f"optimistic up to {optimistic * 100:+.1f}pp.",
        )
    if realistic > 0.01:
        return (
            "CONDITIONAL",
            f"realistic predicted lift {realistic * 100:+.1f}pp but optimistic only "
            f"{optimistic * 100:+.1f}pp — depends on whether your bag signal is sharper "
            f"than graded qrels (clicks/engagement, or tightly-curated CE filtering).",
        )
    return (
        "SKIP",
        f"even optimistic predicted lift is only {optimistic * 100:+.1f}pp; "
        "too small to justify the BoD pipeline cost.",
    )


def load_corpus(args):
    print("loading data...", flush=True)
    with open(args.catalog) as f:
        titles = json.load(f)
    with open(args.product_ids) as f:
        pids = json.load(f)
    pid_to_idx = {p: i for i, p in enumerate(pids)}
    queries_by_qid = {}
    with open(args.queries) as f:
        for line in f:
            d = json.loads(line)
            queries_by_qid[d["query_id"]] = d["query"]
    qrels_full = defaultdict(dict)  # for SCHS (preserves grades)
    pos = defaultdict(set)  # for base-difficulty (binary)
    field = None
    with open(args.qrels) as f:
        for line in f:
            r = json.loads(line)
            if field is None:
                field = "product_id" if "product_id" in r else "doc_id"
            if r[field] not in pid_to_idx:
                continue
            qrels_full[r["query_id"]][r[field]] = r["relevance"]
            if r["relevance"] >= args.min_relevance:
                pos[r["query_id"]].add(pid_to_idx[r[field]])
    return titles, pids, pid_to_idx, queries_by_qid, qrels_full, pos


def base_difficulty(args, titles, pids, queries_by_qid, pos):
    qids = sorted(queries_by_qid)
    queries = [queries_by_qid[q] for q in qids]
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    if args.vecs_cache and os.path.exists(args.vecs_cache):
        print(f"loading cached catalog vecs {args.vecs_cache}...", flush=True)
        pv = np.load(args.vecs_cache).astype(np.float32)
    else:
        print(f"encoding catalog with {args.encoder} on {device}...", flush=True)
        m = SentenceTransformer(args.encoder, device=device)
        pv = m.encode(
            titles, normalize_embeddings=True, batch_size=128, show_progress_bar=True
        ).astype(np.float32)
        if args.vecs_cache:
            np.save(args.vecs_cache, pv.astype(np.float16))
            print(f"  cached at {args.vecs_cache}", flush=True)
        del m
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

    print("encoding queries...", flush=True)
    m = SentenceTransformer(args.encoder, device=device)
    qv = m.encode(
        queries, normalize_embeddings=True, batch_size=256, show_progress_bar=True
    ).astype(np.float32)
    del m

    print("matmul + bucketing...", flush=True)
    per_q = []
    chunk = args.chunk
    n_chunks = (len(qids) + chunk - 1) // chunk
    for ci, start in enumerate(range(0, len(qids), chunk)):
        end = min(start + chunk, len(qids))
        sims = qv[start:end] @ pv.T
        topk = np.argpartition(-sims, args.k, axis=1)[:, : args.k]
        del sims
        for j, gi in enumerate(range(start, end)):
            qid = qids[gi]
            g = pos.get(qid, set())
            if not g:
                continue
            h = len({int(x) for x in topk[j]} & g)
            per_q.append((h, len(g)))
        if (ci + 1) % 10 == 0 or ci + 1 == n_chunks:
            print(f"  {ci + 1}/{n_chunks} chunks", flush=True)

    if not per_q:
        return None

    base_blind = sum(1 for h, n in per_q if h == 0) / len(per_q)
    base_perfect = sum(1 for h, n in per_q if h == n) / len(per_q)
    overall_r = sum(h / n for h, n in per_q) / len(per_q)
    return {
        "n_queries": len(per_q),
        "base_blind": base_blind,
        "base_perfect": base_perfect,
        "overall_R10": overall_r,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--catalog", required=True, help="titles.json")
    ap.add_argument("--product-ids", required=True, help="product_ids.json (or doc_ids.json)")
    ap.add_argument(
        "--qrels", required=True, help="qrels.jsonl with query_id, product_id, relevance"
    )
    ap.add_argument("--queries", required=True, help="queries.jsonl with query_id, query")
    ap.add_argument("--min-relevance", type=int, default=1)
    ap.add_argument("--encoder", default="all-MiniLM-L6-v2")
    ap.add_argument("--vecs-cache", default=None, help="optional .npy cache for catalog vecs")
    ap.add_argument("--label", default=None)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--chunk", type=int, default=512)
    args = ap.parse_args()

    label = args.label or os.path.basename(os.path.dirname(args.catalog) or ".")

    titles, pids, _pid_to_idx, queries_by_qid, qrels_full, pos = load_corpus(args)
    print(
        f"  catalog={len(pids):,}  queries={len(queries_by_qid):,}  "
        f"qrels rows={sum(len(v) for v in qrels_full.values()):,}",
        flush=True,
    )

    print("\n--- 1/2: SCHS (cluster hypothesis score) ---", flush=True)
    chs = compute_chs(dict(qrels_full), pids, titles, args.encoder, partition="strict")

    print("\n--- 2/2: base-difficulty distribution ---", flush=True)
    bd = base_difficulty(args, titles, pids, queries_by_qid, pos)
    if bd is None:
        print("ERROR: no positive-bearing queries — corpus has no usable signal.", flush=True)
        sys.exit(1)

    predicted = predict_lift(bd["base_blind"], bd["base_perfect"])
    v_label, v_msg = verdict(chs.schs, bd["base_blind"], bd["base_perfect"], predicted)

    print("\n" + "=" * 78)
    print(f"BoD READINESS REPORT — {label}")
    print("=" * 78)
    print(f"  encoder:           {args.encoder}")
    print(f"  catalog size:      {len(pids):,} docs")
    print(f"  pos-bearing queries: {bd['n_queries']:,}")
    print()
    print("  Cluster hypothesis (SCHS):")
    print(f"    SCHS:            {chs.schs:.3f}")
    print(f"    n_pos_bearing:   {chs.n_pos_bearing:,}")
    print(f"    verdict:         {schs_verdict(chs.schs, chs.n_pos_bearing)}")
    print()
    print("  Base difficulty (base R@10 distribution):")
    print(f"    overall R@10:    {bd['overall_R10']:.3f}")
    print(f"    base-blind:      {bd['base_blind']:.1%}  (queries where base finds 0 positives)")
    print(f"    base-perfect:    {bd['base_perfect']:.1%}  (queries where base gets all positives)")
    print()
    print("  Predicted lift bands (per CHS_RESULTS.md calibration):")
    for scenario, lift in predicted.items():
        rescue = RESCUE_BANDS[scenario]
        tax = TAX_BANDS[scenario]
        print(
            f"    {scenario:<13}  rescue~{rescue * 100:.0f}pp tax~{tax * 100:.0f}pp  "
            f"=>  predicted Δ R@10 = {lift * 100:+.1f}pp"
        )
    print()
    print(f"  VERDICT: {v_label}")
    print(f"  reason:  {v_msg}")
    print()
    print("  Next steps:")
    if v_label == "GO":
        print("    1. Build bags from your relevance signal (training/bags_from_qrels.py).")
        print("    2. Add hardnegs (download/add_random_hardnegs_bestbuy.py is generic).")
        print("    3. Train BoD via training/finetune_with_hardnegs.py.")
        print("    4. Run evaluation/diagnose_lift.py to confirm the predicted band.")
    elif v_label == "CONDITIONAL":
        print("    1. Audit your relevance signal — clicks beat graded qrels beat noisy qrels.")
        print("    2. If clicks/engagement are available, optimistic band applies; train BoD.")
        print(
            "    3. Otherwise, run a small-scale ablation first (5-10K bags, 1 epoch) "
            "before committing to the full pipeline."
        )
    else:
        print("    BoD is unlikely to pay off on this corpus. Possible levers:")
        print("    - Use a weaker base encoder (more headroom; see Pattern 7 in CHS_RESULTS.md).")
        print("    - Augment qrels with click data or CE-filtered hybrid retrieval.")
        print("    - For SCHS<0.40: consider whether the corpus has any cluster structure at all.")


if __name__ == "__main__":
    main()
