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
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from sentence_transformers import SentenceTransformer  # noqa: E402

from bagofdocs.cluster_hypothesis import compute_chs, schs_verdict  # noqa: E402

# Calibration constants from CHS_RESULTS.md (15-corpus rescue/tax table).
#
# Tax magnitude tracks (1 - base R@10) closely across the calibration set —
# empirically `tax / (1 - base R@10)` clusters around 0.07-0.13. Scaling the
# per-bucket tax by (1 - base R@10) correctly classifies CQADupStack/programmers
# and CQADupStack/gaming (which the fixed-tax v1 formula misclassified as
# false-SKIPs).
RESCUE_BANDS = {"pessimistic": 0.05, "realistic": 0.12, "optimistic": 0.25}
TAX_K = {"pessimistic": 0.15, "realistic": 0.10, "optimistic": 0.06}
SCHS_FLOOR = 0.40
# Deployment-architecture rule of thumb (Pattern 10 in CHS_RESULTS.md):
# rerank wins when BM25 ≥ base by ~2pp; retrieve wins when BM25 ≤ base by ~2pp;
# in between, expect ties or small wins in either direction.
RERANK_VS_RETRIEVE_THRESHOLD = 0.02

# Rescue-rate predictor (Pattern 8a in CHS_RESULTS.md). Linear regression
# fit on the 14 calibration corpora with base R@10 < 0.85. Quora (base R@10
# = 0.95) is excluded — its extreme leverage drops LOO R² from 0.78 to 0.54
# and inflates LOO RMSE from 2.64pp to 3.74pp. Within the validated regime:
# in-sample R²=0.869 / RMSE=2.04pp; LOO R²=0.780 / RMSE=2.64pp.
#
# Above the threshold the linear model has no support, so we return None
# and the readiness tool falls back to the wide v1 5/12/25pp bands.
#
# rescue_pp = log_n_bags*W_LOG_N + median_size*W_SIZE + median_spec*W_SPEC + INTERCEPT
RESCUE_W_LOG_N = 5.270
RESCUE_W_SIZE = -0.008
RESCUE_W_SPEC = 54.345
RESCUE_INTERCEPT = -48.228
RESCUE_RMSE_PP = 2.64  # LOO RMSE (in-sample 2.04pp); use as ±band
RESCUE_BASE_R10_MAX = 0.85  # gate: above this, the predictor is unreliable


def compute_bag_stats(qrels_full, pid_to_idx, base_pv, min_relevance, k_cap=50):
    """Compute (n_bags, median_size, median_spec) using already-encoded catalog.

    No extra encoding step — pulls per-bag positive vectors from `base_pv`
    (the catalog encoded with the base model) and computes spherical-mean
    centroid + intra-bag mean cosine. Mirrors `training/bags_from_qrels.py`
    but without writing to disk.
    """
    sizes, specs = [], []
    for _qid, doc_grades in qrels_full.items():
        pos = sorted(
            (
                (pid_to_idx[pid], rel)
                for pid, rel in doc_grades.items()
                if pid in pid_to_idx and rel >= min_relevance
            ),
            key=lambda x: -x[1],
        )
        idxs = [d for d, _ in pos[:k_cap]]
        if len(idxs) < 2:
            continue
        bag_vecs = base_pv[idxs]
        centroid = bag_vecs.mean(axis=0)
        norm = float(np.linalg.norm(centroid))
        if norm < 1e-12:
            continue
        centroid /= norm
        spec = float(np.mean(bag_vecs @ centroid))
        sizes.append(len(idxs))
        specs.append(spec)
    if not sizes:
        return None
    return {
        "n_bags": len(sizes),
        "median_size": float(np.median(sizes)),
        "median_spec": float(np.median(specs)),
    }


def predict_rescue_rate(bag_stats, base_r10=None):
    """Linear-regression rescue-rate point estimate (Pattern 8a in
    CHS_RESULTS.md). Returns rescue rate as a fraction (e.g., 0.12 = 12pp),
    or None when the predictor is out of its validated regime:

      - `bag_stats` missing or `n_bags` < 10 (too small a sample to trust).
      - `base_r10` >= RESCUE_BASE_R10_MAX (no calibration support; Quora
        was the only training point above this threshold and its LOO
        residual was −7.7pp).
    """
    if bag_stats is None or bag_stats.get("n_bags", 0) < 10:
        return None
    if base_r10 is not None and base_r10 >= RESCUE_BASE_R10_MAX:
        return None
    pp = (
        RESCUE_W_LOG_N * np.log10(bag_stats["n_bags"])
        + RESCUE_W_SIZE * bag_stats["median_size"]
        + RESCUE_W_SPEC * bag_stats["median_spec"]
        + RESCUE_INTERCEPT
    )
    # Clamp to plausible range (0 to 40pp); convert pp -> fraction.
    return max(0.0, min(0.40, float(pp) / 100.0))


def predict_lift(base_blind, base_perfect, base_overall_r10=None, predicted_rescue=None):
    """Return dict of {scenario: predicted_lift_pp} for the 3 signal bands.

    Tax magnitude scales with (1 - base R@10) when `base_overall_r10` is
    provided — this matches the 15-corpus calibration data (CHS_RESULTS.md
    Pattern 9).

    When `predicted_rescue` is provided (from `predict_rescue_rate()`), the
    rescue band collapses around that point estimate ± RESCUE_RMSE_PP/100,
    sharpening every prediction. When omitted, falls back to the wide
    fixed-band defaults from RESCUE_BANDS (v1-compatible).
    """
    out = {}
    headroom = (1.0 - base_overall_r10) if base_overall_r10 is not None else 1.0
    if predicted_rescue is not None:
        # Point estimate ± RMSE — much tighter than the wide v1 bands.
        rmse = RESCUE_RMSE_PP / 100.0
        rescue_bands = {
            "pessimistic": max(0.0, predicted_rescue - rmse),
            "realistic": predicted_rescue,
            "optimistic": predicted_rescue + rmse,
        }
    else:
        rescue_bands = RESCUE_BANDS
    for scenario in rescue_bands:
        rescue = rescue_bands[scenario]
        tax = TAX_K[scenario] * headroom
        out[scenario] = base_blind * rescue - base_perfect * tax
    return out


def verdict(schs, base_blind, base_perfect, predicted, n_bags=None):
    # No multi-positive queries → no bags → BoD has nothing to train on.
    # This trumps every other signal; the chain is a non-starter.
    if n_bags is not None and n_bags == 0:
        return (
            "SKIP",
            "no multi-positive queries (n_bags=0); BoD requires ≥2 positives per "
            "query to form a bag — this corpus has none.",
        )
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
    }, pv


def bm25_r10(titles, queries, qids, pos, k=10):
    """Compute fraction-recovered R@k for BM25 alone (no rerank).

    Returns None if bm25s isn't installed; the readiness verdict still works,
    just without the architecture recommendation.
    """
    try:
        import bm25s
    except ImportError:
        return None
    print("computing BM25 R@10 for the architecture recommendation...", flush=True)
    t0 = time.time()
    retriever = bm25s.BM25()
    tokenized_corpus = bm25s.tokenize(titles, stopwords="en", show_progress=False)
    retriever.index(tokenized_corpus, show_progress=False)
    tokenized_queries = bm25s.tokenize(queries, stopwords="en", show_progress=False)
    bm25_top, _ = retriever.retrieve(tokenized_queries, k=k, show_progress=False)
    n = 0
    total = 0.0
    for i, qid in enumerate(qids):
        g = pos.get(qid, set())
        if not g:
            continue
        cand = bm25_top[i]
        cand = cand[cand >= 0]
        hits = len({int(x) for x in cand} & g)
        total += hits / len(g)
        n += 1
    print(f"  BM25 indexed + scored in {time.time() - t0:.0f}s", flush=True)
    return total / n if n else None


def architecture_recommendation(bm25_r, base_r):
    if bm25_r is None or base_r is None:
        return None, "BM25 not measured (install `bm25s` to enable)."
    delta = bm25_r - base_r
    if delta >= RERANK_VS_RETRIEVE_THRESHOLD:
        return "rerank", (
            f"BM25 R@10 {bm25_r:.3f} > base {base_r:.3f} by {delta * 100:+.1f}pp; "
            "use BoD as a reranker over BM25 top-50 (per Pattern 10)."
        )
    if delta <= -RERANK_VS_RETRIEVE_THRESHOLD:
        return "retrieve", (
            f"BM25 R@10 {bm25_r:.3f} < base {base_r:.3f} by {delta * 100:+.1f}pp; "
            "use BoD as a retriever over the full catalog (BM25 misses too much)."
        )
    return "either", (
        f"BM25 ({bm25_r:.3f}) ≈ base ({base_r:.3f}) within ±2pp; "
        "rerank and retrieve are roughly tied — pick whichever is cheaper to "
        "operate, expect 5-25% per-query difference but small overall lift gap."
    )


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

    print("\n--- 2/3: base-difficulty distribution ---", flush=True)
    bd_result = base_difficulty(args, titles, pids, queries_by_qid, pos)
    if bd_result is None:
        print("ERROR: no positive-bearing queries — corpus has no usable signal.", flush=True)
        sys.exit(1)
    bd, base_pv = bd_result

    # Bag stats + rescue-rate point estimate (Pattern 8a).
    pid_to_idx = {p: i for i, p in enumerate(pids)}
    bag_stats = compute_bag_stats(qrels_full, pid_to_idx, base_pv, args.min_relevance)
    predicted_rescue = predict_rescue_rate(bag_stats, base_r10=bd["overall_R10"])

    predicted = predict_lift(
        bd["base_blind"], bd["base_perfect"], bd["overall_R10"], predicted_rescue
    )
    v_label, v_msg = verdict(
        chs.schs,
        bd["base_blind"],
        bd["base_perfect"],
        predicted,
        n_bags=bag_stats["n_bags"] if bag_stats else 0,
    )

    print("\n--- 3/3: BM25 R@10 (architecture recommendation) ---", flush=True)
    qids_eval = sorted(queries_by_qid)
    queries_eval = [queries_by_qid[q] for q in qids_eval]
    bm25_r = bm25_r10(titles, queries_eval, qids_eval, pos, k=args.k)
    arch, arch_msg = architecture_recommendation(bm25_r, bd["overall_R10"])

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
    headroom = 1.0 - bd["overall_R10"]
    if predicted_rescue is not None:
        rmse_pp = RESCUE_RMSE_PP
        rescue_label = (
            f"rescue~{predicted_rescue * 100:.1f}pp ±{rmse_pp:.1f}pp "
            f"(predicted from bag stats; n_bags={bag_stats['n_bags']:,}, "
            f"median_size={bag_stats['median_size']:.0f}, "
            f"median_spec={bag_stats['median_spec']:.3f})"
        )
        print(f"    {rescue_label}")
        for scenario, lift in predicted.items():
            tax_eff = TAX_K[scenario] * headroom
            print(
                f"    {scenario:<13}  tax~{tax_eff * 100:.1f}pp  "
                f"=>  predicted Δ R@10 = {lift * 100:+.1f}pp"
            )
    else:
        if bd["overall_R10"] >= RESCUE_BASE_R10_MAX:
            print(
                f"    (base R@10 {bd['overall_R10']:.3f} >= {RESCUE_BASE_R10_MAX} — "
                f"predictor out of validated regime; using fixed v1 bands)"
            )
        else:
            print("    (rescue rate point-estimate unavailable; using fixed v1 bands)")
        for scenario, lift in predicted.items():
            rescue = RESCUE_BANDS[scenario]
            tax_eff = TAX_K[scenario] * headroom
            print(
                f"    {scenario:<13}  rescue~{rescue * 100:.0f}pp tax~{tax_eff * 100:.1f}pp  "
                f"=>  predicted Δ R@10 = {lift * 100:+.1f}pp"
            )
    print()
    if bm25_r is not None:
        print("  Deployment architecture (when GO):")
        print(f"    BM25 R@10:        {bm25_r:.3f}  (vs base {bd['overall_R10']:.3f})")
        print(f"    recommendation:   {arch or 'n/a'}")
        print(f"    reason:           {arch_msg}")
        print()
    else:
        print("  Deployment architecture: BM25 not measured (install `bm25s`).")
        print()
    print(f"  VERDICT: {v_label}")
    print(f"  reason:  {v_msg}")
    print()
    print("  Next steps:")
    if v_label == "GO":
        print("    1. Build bags from your relevance signal (training/bags_from_qrels.py).")
        print("    2. Add hardnegs (download/add_random_hardnegs.py).")
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
        # Known false-SKIP zone: SCHS just below the floor + low base-perfect (low tax
        # exposure) can still produce a real lift even with mediocre clustering. SCIDOCS
        # is the canonical example (SCHS 0.367, BP 0.8% -> actual +4.1pp despite SKIP).
        # NaN check via x == x.
        if (
            chs.schs is not None
            and chs.schs == chs.schs  # noqa: PLR0124  not nan
            and 0.30 <= chs.schs < SCHS_FLOOR
            and bd["base_perfect"] < 0.05
        ):
            print()
            print("    Note: this corpus sits in the 'false-SKIP zone' (Pattern 9 in")
            print("    CHS_RESULTS.md): SCHS just below the 0.40 floor AND base-perfect")
            print("    fraction below 5% (low tax exposure). The realistic-band lift")
            print(f"    prediction was {predicted['realistic'] * 100:+.1f}pp; SCIDOCS in this")
            print("    zone delivered +4.1pp actual. Consider piloting BoD anyway with a")
            print("    small-scale ablation before fully committing.")


if __name__ == "__main__":
    main()
