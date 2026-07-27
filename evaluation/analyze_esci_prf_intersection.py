#!/usr/bin/env python3
"""Does pseudo-relevance feedback restricted to the BM25-and-dense INTERSECTION
only help on the queries where lexical and dense already agree?

Motivation
----------
A SIGIR paper reportedly runs PRF over the INTERSECTION of a lexical (BM25) and
a dense retriever's top ranks -- i.e. feedback documents that two independent
retrievers both like -- and reports good aggregate results. The worry this
script tests: the intersection is only large when the two retrievers already
agree, and agreement is a proxy for "easy query". If so, an aggregate gain can
be entirely a "the easy queries got easier" effect, while the queries that most
need help (small or EMPTY intersection -- where PRF is a literal no-op) get
nothing. An unstratified average would hide that completely.

THIS IS A SIMPLIFIED PROXY, NOT A REPRODUCTION. We do not have the paper's
implementation details (their feedback-term selection, weighting, whether the
PRF acts on the lexical side, dense side, or both, their depths, their corpus).
What is tested here is the STRUCTURAL claim -- "benefit of PRF-on-intersection
concentrates in queries with a large intersection" -- with the most tractable
local instantiation of the idea: a dense-side Rocchio update, since this repo
already has cached catalog embeddings. No claim is made about the paper's
actual numbers.

Chain so far (this repo's ESCI lexical-bias / hybrid-retrieval line)
-------------------------------------------------------------------
1. `eval_esci_llm_judge_lexical_bias.py` -- paraphrasing the query does not
   debias an LLM judge's literal-overlap bias.
2. `analyze_esci_paraphrase_fidelity.py` -- the harm concentrates in drifted
   paraphrases.
3. `analyze_esci_recall_union.py` -- the recall-union gain DIED under a
   size-matched control: a pool-size artifact, not a real signal.
4. `analyze_esci_bm25_paraphrase_rrf.py` -- real BM25+dense RRF retrieval. The
   "dense also likes this doc" split turned out to be a ~20x precision filter
   on ANY candidate list (it showed up identically in the zero-paraphrase
   controls), i.e. a property of dense-top-D, not of the technique riding on
   top of it.
5. THIS SCRIPT -- same failure family, different technique. (3) was a pool-size
   artifact and (4) was a "dense-top-D is just a better pool" artifact. The
   candidate artifact here is a QUERY-SUBSET artifact: the treatment is only
   active on an unrepresentative, already-easy slice of queries.

Conditions (per query, depth D=100 per retrieval leg)
-----------------------------------------------------
    I_N     = BM25-orig-top-N  n  Dense-orig-top-N        (doc-index sets)
    A       = RRF(BM25-orig-top-D, Dense-orig-top-D)               [baseline]
    prf_qv  = normalize((1 - alpha) * orig_qv
                        + alpha * mean(catalog_vecs[d] for d in I_N))
              ... and prf_qv = orig_qv exactly when I_N is empty
    P       = RRF(BM25-orig-top-D, Dense-PRF-top-D)          [PRF treatment]

A and P differ ONLY in the query vector used for the dense leg. By construction
delta(P, A) is EXACTLY 0 for every empty-intersection query -- that is not a
bug, it is the cleanest possible statement of the concern: on the queries with
no lexical/dense agreement, this method cannot do anything at all. The script
asserts that identity as a self-check.

Strata
------
By |I_N| at the primary depth N=20: an explicit EMPTY bucket plus tercile-like
integer cuts over the NON-empty queries, cut points derived from the observed
distribution (closest-to-target thirds), printed and stored in the output. Cuts
are re-derived independently at each sensitivity depth.

The two-part read this script exists to produce
-----------------------------------------------
(a) Is BASELINE (no-PRF) quality already higher in the large-|I| stratum than in
    the small/empty ones? That is the "these are the easy queries" test, and it
    does not involve PRF at all.
(b) Is delta(P, A) positive in the large-|I| stratum but flat/negative in the
    small/empty ones? Together with (a) that confirms the concern: the method's
    measured benefit lives on the slice that needed it least.
The pooled/unstratified delta over all 250 queries is reported alongside, to
show how misleading the aggregate would be if that pattern holds.

Conventions (unchanged from the rest of the chain)
--------------------------------------------------
True positive: ESCI Exact (relevance 3) only, primary; E+S (relevance >= 2) as
a sensitivity check. Substitute is the adversarial lexical near-miss class of
the parent experiment, so counting S as a positive would score the failure mode
as a success. nDCG uses linear gains = ESCI grade (E=3,S=2,C=1,I=0) via the
parent script's `ndcg_at_k`, ideal = all judged docs for the query. Recall is
|fused top-K & TP| / |TP| over queries with >= 1 TP. Same 250-query ESCI-US
sample; gold from esci_us_data/test_qrels.jsonl.

No paraphrases anywhere in this experiment -- the ORIGINAL query only.

Reused artifacts (nothing expensive is recomputed)
--------------------------------------------------
* catalog embeddings: /tmp/esci_lexbias/esci_us_minilm_catalog.fp16.npy
  (all-MiniLM-L6-v2 over all 360,873 ESCI-US titles), built by
  analyze_esci_bm25_paraphrase_rrf.py. Never re-encoded.
* BM25 lists: `bm25_orig` from
  /tmp/esci_lexbias/esci_bm25para_rrf_retrieval.npz (top-200, of which we use
  the top-100). The BM25 index is NEVER rebuilt; if that cache is missing this
  script refuses to run rather than silently doing an expensive rebuild.
* The dense QUERY side is recomputed (cheap: 250 queries) so that conditions A
  and P come from bit-identical machinery and differ only in the query vector.
  Agreement with the cached `dense_orig` list is measured and reported as
  `dense_orig_vs_cache_agreement` -- if it is ~1.0 the choice is immaterial,
  and if it is not, recomputing was necessary for a valid A/P comparison.

Zero API calls: pure local compute (numpy + a local sentence-transformers
encode of 250 short queries).

Alpha sweep mode (--alpha-sweep)
-------------------------------
The single-alpha run at alpha=0.5 found the large-|I| stratum got WORSE while
the small-|I| stratum got a small lift. `--alpha-sweep 0.1,0.2,0.3,0.4,0.5`
answers whether that harm is a ceiling effect of too aggressive a perturbation
(it should shrink toward 0 as alpha -> 0) or structural. It runs at the PRIMARY
intersection depth and primary TP only, and the alpha-INDEPENDENT work (BM25
lists, dense-orig retrieval, catalog vecs, intersections, strata) is computed
ONCE -- only the Rocchio update -> dense-PRF retrieval -> RRF -> metrics leg
reruns per alpha. Output goes to --sweep-out, never to --out.

Usage (this repo's .venv has none of numpy/sentence-transformers; use uv):
    uv run --no-project --with numpy --with sentence-transformers --with torch \\
        python evaluation/analyze_esci_prf_intersection.py

    uv run --no-project --with numpy --with sentence-transformers --with torch \\
        python evaluation/analyze_esci_prf_intersection.py \\
        --alpha-sweep 0.1,0.2,0.3,0.4,0.5
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
from eval_esci_llm_judge_lexical_bias import (  # noqa: E402
    GRADE_LETTER,
    ndcg_at_k,
    work_paths,
)

KS = (5, 10)
TP_DEFS = {
    "E": lambda g: g == 3,
    "E+S": lambda g: g >= 2,
}
PRIMARY_TP = "E"
ARMS = ("A_baseline_hybrid", "P_prf_intersection")
STRATA_ORDER = ("empty", "small", "mid", "large", "pooled_all_queries")


# --------------------------------------------------------------------------
# stats (same constructions as analyze_esci_bm25_paraphrase_rrf.py)
# --------------------------------------------------------------------------
def _paired_boot(dif, n_boot, seed):
    """Paired bootstrap over queries on the mean of a per-query delta."""
    v = np.asarray([x for x in dif if np.isfinite(x)], dtype=np.float64)
    if v.size < 2:
        return None
    rng = np.random.default_rng(seed)
    draws = v[rng.integers(0, v.size, size=(n_boot, v.size))].mean(axis=1)
    return {
        "mean": float(v.mean()),
        "ci95": [float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))],
        "p_le_0": float(np.mean(draws <= 0)),
        "wins": int((v > 0).sum()),
        "losses": int((v < 0).sum()),
        "ties": int((v == 0).sum()),
        "n": int(v.size),
    }


def _mean_ci(values, n_boot, seed):
    v = np.asarray([x for x in values if x is not None and np.isfinite(x)], dtype=np.float64)
    if v.size < 2:
        return {"mean": float(v.mean()) if v.size else None, "ci95": None, "n": int(v.size)}
    rng = np.random.default_rng(seed)
    draws = v[rng.integers(0, v.size, size=(n_boot, v.size))].mean(axis=1)
    return {
        "mean": float(v.mean()),
        "ci95": [float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))],
        "n": int(v.size),
    }


def _unpaired_diff_boot(a, b, n_boot, seed):
    """mean(a) - mean(b), independent bootstrap per side (disjoint query groups)."""
    a = np.asarray([x for x in a if np.isfinite(x)], dtype=np.float64)
    b = np.asarray([x for x in b if np.isfinite(x)], dtype=np.float64)
    if a.size < 2 or b.size < 2:
        return None
    rng = np.random.default_rng(seed)
    da = a[rng.integers(0, a.size, size=(n_boot, a.size))].mean(axis=1)
    db = b[rng.integers(0, b.size, size=(n_boot, b.size))].mean(axis=1)
    d = da - db
    return {
        "mean_a": float(a.mean()),
        "mean_b": float(b.mean()),
        "diff": float(a.mean() - b.mean()),
        "ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
        "p_a_lt_b": float(np.mean(d < 0)),
        "n_a": int(a.size),
        "n_b": int(b.size),
    }


# --------------------------------------------------------------------------
# retrieval / fusion
# --------------------------------------------------------------------------
def rrf_merge(rankings_list, top_k, rrf_k=60):
    """RRF: for each candidate doc, sum 1/(rrf_k + rank + 1) across every
    ranking it appears in. Returns top-k doc indices by RRF score.

    Verbatim construction from evaluation/eval_rrf_ensemble.py, same as
    analyze_esci_bm25_paraphrase_rrf.py.
    """
    scores = defaultdict(float)
    for rankings in rankings_list:
        for rank, doc_idx in enumerate(rankings):
            scores[int(doc_idx)] += 1.0 / (rrf_k + rank + 1)
    return sorted(scores, key=scores.get, reverse=True)[:top_k]


def encode_queries(model_name, queries):
    import torch
    from sentence_transformers import SentenceTransformer

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"  encoding {len(queries)} queries with {model_name} on {device}", flush=True)
    m = SentenceTransformer(model_name, device=device)
    qv = m.encode(
        queries, normalize_embeddings=True, batch_size=256, show_progress_bar=False
    ).astype(np.float32)
    del m
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    return qv


def dense_topk_from_qv(qv, pv, k, chunk=32):
    """Exact top-k by cosine (both sides L2-normalized) via chunked matmul."""
    out = np.zeros((qv.shape[0], k), dtype=np.int64)
    for s in range(0, qv.shape[0], chunk):
        e = min(s + chunk, qv.shape[0])
        sims = qv[s:e] @ pv.T
        part = np.argpartition(-sims, k, axis=1)[:, :k]
        rows = np.arange(part.shape[0])[:, None]
        order = np.argsort(-sims[rows, part], axis=1)
        out[s:e] = part[rows, order]
        del sims
    return out


def prf_query_vecs(qv, pv, intersections, alpha):
    """Dense-side Rocchio on the BM25/dense intersection.

    prf_qv = normalize((1 - alpha) * orig_qv + alpha * centroid(I)), and
    prf_qv = orig_qv exactly when I is empty (a recorded no-op).
    """
    out = qv.copy()
    for i, docs in enumerate(intersections):
        if not docs:
            continue
        centroid = pv[sorted(docs)].mean(axis=0)
        v = (1.0 - alpha) * qv[i] + alpha * centroid
        n = np.linalg.norm(v)
        out[i] = v / n if n > 0 else qv[i]
    return out


# --------------------------------------------------------------------------
# strata
# --------------------------------------------------------------------------
def tercile_cuts(sizes):
    """Integer cut points splitting the NON-EMPTY |I| values into three
    near-equal groups. Greedy closest-to-target over the cumulative histogram,
    so the cuts are a deterministic function of the observed distribution
    rather than an assumed shape.

    Returns (t1, t2): small = 1..t1, mid = t1+1..t2, large = > t2.
    """
    nz = sorted(s for s in sizes if s > 0)
    if len(nz) < 3:
        return None
    vals = sorted(set(nz))
    if len(vals) < 3:
        return None

    def _pick(candidates, pool, target):
        best, best_dev = None, None
        for t in candidates:
            dev = abs(sum(1 for s in pool if s <= t) - target)
            if best_dev is None or dev < best_dev:
                best, best_dev = t, dev
        return best

    t1 = _pick(vals[:-2], nz, len(nz) / 3.0)
    rest = [s for s in nz if s > t1]
    t2 = _pick([v for v in vals if t1 < v < vals[-1]], rest, len(rest) / 2.0)
    if t2 is None or t2 <= t1:
        return None
    return int(t1), int(t2)


def assign_strata(sizes, cuts):
    """qi lists per stratum, plus a pooled bucket of every query."""
    n = len(sizes)
    strata = {"empty": [i for i in range(n) if sizes[i] == 0]}
    if cuts is None:
        strata["nonempty"] = [i for i in range(n) if sizes[i] > 0]
    else:
        t1, t2 = cuts
        strata["small"] = [i for i in range(n) if 0 < sizes[i] <= t1]
        strata["mid"] = [i for i in range(n) if t1 < sizes[i] <= t2]
        strata["large"] = [i for i in range(n) if sizes[i] > t2]
    strata["pooled_all_queries"] = list(range(n))
    return strata


def size_distribution(sizes):
    a = np.asarray(sizes, dtype=np.int64)
    return {
        "n": int(a.size),
        "mean": float(a.mean()),
        "median": float(np.median(a)),
        "min": int(a.min()),
        "max": int(a.max()),
        "quartiles": {p: int(np.percentile(a, p)) for p in (25, 50, 75)},
        "deciles": {p: int(np.percentile(a, p)) for p in (10, 90)},
        "n_empty": int((a == 0).sum()),
        "frac_empty": float((a == 0).mean()),
        "histogram": np.bincount(a).tolist(),
    }


# --------------------------------------------------------------------------
# analysis for one (stratum, K, tp_def)
# --------------------------------------------------------------------------
def analyse(per_q, qis, k, tp_pred, n_boot, seed):
    rec = {a: [] for a in ARMS}
    ndcg = {a: [] for a in ARMS}
    isizes, n_no_tp, n_prf_noop = [], 0, 0

    for qi in qis:
        q = per_q[qi]
        grades = q["grades"]
        tp = {pid for pid, g in grades.items() if tp_pred(g)}
        isizes.append(q["isize"])
        n_prf_noop += int(q["isize"] == 0)
        all_grades = list(grades.values())
        for arm in ARMS:
            ranked = q["fused"][arm][:k]
            ndcg[arm].append(ndcg_at_k([grades.get(d, 0) for d in ranked], all_grades, k))
        if not tp:
            n_no_tp += 1
            continue
        for arm in ARMS:
            top = set(q["fused"][arm][:k])
            rec[arm].append(len(top & tp) / len(tp))

    out = {
        "k": k,
        "n_queries": len(qis),
        "n_queries_with_tp": len(rec[ARMS[0]]),
        "n_queries_no_tp": n_no_tp,
        "n_prf_noop_empty_intersection": n_prf_noop,
        "intersection_size": {
            "mean": float(np.mean(isizes)) if isizes else None,
            "median": float(np.median(isizes)) if isizes else None,
            "min": int(np.min(isizes)) if isizes else None,
            "max": int(np.max(isizes)) if isizes else None,
        },
        "recall": {a: _mean_ci(rec[a], n_boot, seed) for a in ARMS},
        "ndcg": {a: _mean_ci(ndcg[a], n_boot, seed) for a in ARMS},
    }
    out["delta_recall_P_minus_A"] = _paired_boot(
        [p - a for p, a in zip(rec["P_prf_intersection"], rec["A_baseline_hybrid"])],
        n_boot,
        seed,
    )
    out["delta_ndcg_P_minus_A"] = _paired_boot(
        [
            p - a
            for p, a in zip(ndcg["P_prf_intersection"], ndcg["A_baseline_hybrid"])
            if np.isfinite(p) and np.isfinite(a)
        ],
        n_boot,
        seed,
    )
    # raw per-query vectors are kept only long enough for the cross-stratum
    # contrasts assembled by the caller
    out["_recall"] = rec
    out["_ndcg"] = ndcg
    return out


def strip_private(blk):
    return {kk: vv for kk, vv in blk.items() if not kk.startswith("_")}


# --------------------------------------------------------------------------
# reporting
# --------------------------------------------------------------------------
def _fmt_ci(d, key="ci95"):
    if not d or not d.get(key):
        return "n/a"
    return f"[{d[key][0]:+.4f}, {d[key][1]:+.4f}]"


def _print_depth(depth_blk, tp_name):
    print(
        f"\n================ N={depth_blk['intersection_depth']} (TP = {tp_name}) ================",
        flush=True,
    )
    cuts = depth_blk["strata_cuts"]
    if cuts:
        print(
            f"  strata cuts on |I| : empty = 0 | small = 1..{cuts['t1']} | "
            f"mid = {cuts['t1'] + 1}..{cuts['t2']} | large = >{cuts['t2']}",
            flush=True,
        )
    else:
        print(
            "  strata cuts on |I| : DEGENERATE (too few distinct sizes) -> empty / nonempty only",
            flush=True,
        )
    for name in STRATA_ORDER:
        if name in depth_blk["n_by_stratum"]:
            print(f"    n[{name}] = {depth_blk['n_by_stratum'][name]}", flush=True)

    for k in KS:
        print(f"\n  ---- K={k} ----", flush=True)
        hdr = (
            f"  {'stratum':<20s} {'n':>4s} {'|I|':>5s} "
            f"{'R@K A':>8s} {'R@K P':>8s} {'dR':>8s} {'dR CI':>20s} "
            f"{'nD A':>7s} {'nD P':>7s} {'dnD':>8s} {'dnD CI':>20s}"
        )
        print(hdr, flush=True)
        for name in STRATA_ORDER:
            blk = depth_blk["strata"].get(name, {}).get(f"k{k}")
            if not blk:
                continue
            ra = blk["recall"]["A_baseline_hybrid"]["mean"]
            rp = blk["recall"]["P_prf_intersection"]["mean"]
            na = blk["ndcg"]["A_baseline_hybrid"]["mean"]
            npv = blk["ndcg"]["P_prf_intersection"]["mean"]
            dr = blk["delta_recall_P_minus_A"]
            dn = blk["delta_ndcg_P_minus_A"]
            print(
                f"  {name:<20s} {blk['n_queries']:>4d} "
                f"{blk['intersection_size']['mean']:>5.1f} "
                f"{ra:>8.4f} {rp:>8.4f} "
                f"{(dr['mean'] if dr else float('nan')):>+8.4f} {_fmt_ci(dr):>20s} "
                f"{na:>7.4f} {npv:>7.4f} "
                f"{(dn['mean'] if dn else float('nan')):>+8.4f} {_fmt_ci(dn):>20s}",
                flush=True,
            )
        base = depth_blk["baseline_difficulty_contrasts"].get(f"k{k}", {})
        print("  baseline-only (no PRF) difficulty contrasts, large vs small/empty:", flush=True)
        for lbl, key in (
            ("A recall  large - small", "recall_large_minus_small"),
            ("A recall  large - empty", "recall_large_minus_empty"),
            ("A nDCG    large - small", "ndcg_large_minus_small"),
            ("A nDCG    large - empty", "ndcg_large_minus_empty"),
        ):
            d = base.get(key)
            if d:
                print(
                    f"    {lbl}: {d['diff']:+.4f}  CI {_fmt_ci(d)}  "
                    f"P(large<other)={d['p_a_lt_b']:.3f}",
                    flush=True,
                )
        gate = depth_blk["delta_concentration_contrasts"].get(f"k{k}", {})
        print("  delta(P,A) concentration, large vs small:", flush=True)
        for lbl, key in (
            ("dRecall  large - small", "delta_recall_large_minus_small"),
            ("dnDCG    large - small", "delta_ndcg_large_minus_small"),
        ):
            d = gate.get(key)
            if d:
                print(f"    {lbl}: {d['diff']:+.4f}  CI {_fmt_ci(d)}", flush=True)


# --------------------------------------------------------------------------
def prepare_depth(depth_n, bm25_100, dense_orig_100, n_q):
    """ALPHA-INDEPENDENT half of the experiment at intersection depth N: the
    intersections themselves, their size distribution, the tercile cut points
    and the strata assignment. None of this depends on the Rocchio alpha, so a
    sweep computes it exactly once and reuses it for every alpha.
    """
    print(f"\nintersection depth N={depth_n}", flush=True)
    intersections = [
        set(bm25_100[i, :depth_n].tolist()) & set(dense_orig_100[i, :depth_n].tolist())
        for i in range(n_q)
    ]
    sizes = [len(s) for s in intersections]
    dist = size_distribution(sizes)
    print(
        f"  |I| mean {dist['mean']:.2f} median {dist['median']:.0f} "
        f"range [{dist['min']}, {dist['max']}] quartiles "
        f"{dist['quartiles']}  empty {dist['n_empty']} ({dist['frac_empty']:.1%})",
        flush=True,
    )
    cuts = tercile_cuts(sizes)
    strata = assign_strata(sizes, cuts)
    print("  strata: " + "  ".join(f"{k}={len(v)}" for k, v in strata.items()), flush=True)
    return {
        "depth_n": depth_n,
        "intersections": intersections,
        "sizes": sizes,
        "dist": dist,
        "cuts": cuts,
        "strata": strata,
    }


def cuts_block(cuts):
    """Serializable description of the strata cut points."""
    if not cuts:
        return None
    return {
        "t1": cuts[0],
        "t2": cuts[1],
        "rule": (
            f"empty: |I| == 0 | small: 1 <= |I| <= {cuts[0]} | "
            f"mid: {cuts[0] + 1} <= |I| <= {cuts[1]} | large: |I| > {cuts[1]}"
        ),
        "derivation": (
            "closest-to-target tercile split of the NON-EMPTY |I| values "
            "over the observed cumulative histogram"
        ),
    }


def prf_fuse(alpha, prep, qids, queries, bm25_100, dense_orig_100, qv, pv, grade_of, args):
    """ALPHA-DEPENDENT half: Rocchio query update -> dense-PRF retrieval ->
    RRF fusion -> per-query records. This is the only part a sweep reruns.
    """
    intersections = prep["intersections"]
    sizes = prep["sizes"]
    prf_qv = prf_query_vecs(qv, pv, intersections, alpha)
    n_changed = int((np.abs(prf_qv - qv) > 0).any(axis=1).sum())
    print(
        f"  PRF (dense Rocchio, alpha={alpha}): {n_changed} of {len(qids)} "
        f"query vectors changed; {len(qids) - n_changed} are exact no-ops",
        flush=True,
    )
    dense_prf_100 = dense_topk_from_qv(prf_qv, pv, args.depth)

    per_q = []
    for i, qid in enumerate(qids):
        bo = [int(x) for x in bm25_100[i]]
        de = [int(x) for x in dense_orig_100[i]]
        dp = [int(x) for x in dense_prf_100[i]]
        # NOTE: fused_p is always computed from the PRF dense list, never
        # short-circuited to fused_a for empty intersections. That keeps the
        # empty-intersection self-check below a real test (prf_qv == orig_qv
        # exactly => identical dense list => identical fusion) instead of a
        # tautology.
        fused_a = rrf_merge([bo, de], max(KS), args.rrf_k)
        fused_p = rrf_merge([bo, dp], max(KS), args.rrf_k)
        per_q.append(
            {
                "qi": i,
                "qid": qid,
                "query": queries[i],
                "isize": sizes[i],
                "fused": {"A_baseline_hybrid": fused_a, "P_prf_intersection": fused_p},
                "grades": grade_of.get(qid, {}),
            }
        )

    # self-check: empty intersection => PRF is a literal no-op => identical rankings
    bad = [
        q["qid"]
        for q in per_q
        if q["isize"] == 0 and q["fused"]["A_baseline_hybrid"] != q["fused"]["P_prf_intersection"]
    ]
    if bad:
        raise SystemExit(
            f"self-check failed: {len(bad)} empty-intersection queries have "
            f"A != P rankings (e.g. {bad[:3]}); A and P must differ only via prf_qv"
        )

    return per_q, n_changed


def analyse_tps(per_q, strata, tp_defs, args):
    """Stratified metrics + cross-stratum contrasts, per TP definition."""
    by_tp = {}
    for tp_name, tp_pred in tp_defs.items():
        blocks = {}
        for sname, qis in strata.items():
            for k in KS:
                blocks.setdefault(sname, {})[f"k{k}"] = analyse(
                    per_q, qis, k, tp_pred, args.n_boot, args.seed
                )
        # cross-stratum contrasts: unpaired (disjoint query groups)
        base_c, conc_c = {}, {}
        for k in KS:
            kk = f"k{k}"
            for tgt, other in (("large", "small"), ("large", "empty")):
                if tgt not in blocks or other not in blocks:
                    continue
                for metric in ("recall", "ndcg"):
                    key = f"_{metric}"
                    base_c.setdefault(kk, {})[f"{metric}_{tgt}_minus_{other}"] = (
                        _unpaired_diff_boot(
                            blocks[tgt][kk][key]["A_baseline_hybrid"],
                            blocks[other][kk][key]["A_baseline_hybrid"],
                            args.n_boot,
                            args.seed,
                        )
                    )
                    da = [
                        p - a
                        for p, a in zip(
                            blocks[tgt][kk][key]["P_prf_intersection"],
                            blocks[tgt][kk][key]["A_baseline_hybrid"],
                        )
                        if np.isfinite(p) and np.isfinite(a)
                    ]
                    db = [
                        p - a
                        for p, a in zip(
                            blocks[other][kk][key]["P_prf_intersection"],
                            blocks[other][kk][key]["A_baseline_hybrid"],
                        )
                        if np.isfinite(p) and np.isfinite(a)
                    ]
                    conc_c.setdefault(kk, {})[f"delta_{metric}_{tgt}_minus_{other}"] = (
                        _unpaired_diff_boot(da, db, args.n_boot, args.seed)
                    )
        by_tp[tp_name] = {
            "strata": {s: {k: strip_private(b) for k, b in d.items()} for s, d in blocks.items()},
            "baseline_difficulty_contrasts": base_c,
            "delta_concentration_contrasts": conc_c,
        }
    return by_tp


def run_depth(depth_n, qids, queries, bm25_100, dense_orig_100, qv, pv, grade_of, args):
    """One full single-alpha experiment at a given intersection depth N."""
    prep = prepare_depth(depth_n, bm25_100, dense_orig_100, len(qids))
    per_q, n_changed = prf_fuse(
        args.alpha, prep, qids, queries, bm25_100, dense_orig_100, qv, pv, grade_of, args
    )
    return {
        "intersection_depth": depth_n,
        "alpha": args.alpha,
        "intersection_size_distribution": prep["dist"],
        "strata_cuts": cuts_block(prep["cuts"]),
        "n_by_stratum": {k: len(v) for k, v in prep["strata"].items()},
        "n_query_vecs_changed_by_prf": n_changed,
        "by_tp": analyse_tps(per_q, prep["strata"], TP_DEFS, args),
    }


def run_alpha_sweep(depth_n, qids, queries, bm25_100, dense_orig_100, qv, pv, grade_of, args):
    """Sweep the Rocchio alpha at ONE intersection depth.

    The alpha-independent work (intersections at depth N, size distribution,
    tercile cuts, strata) is done once by prepare_depth(); only the Rocchio
    update -> dense-PRF retrieval -> RRF -> metrics leg reruns per alpha.
    Primary TP only (Exact), since this sweep asks a single question: how do the
    per-stratum delta(P, A) values move as alpha shrinks?
    """
    prep = prepare_depth(depth_n, bm25_100, dense_orig_100, len(qids))
    tp_defs = {PRIMARY_TP: TP_DEFS[PRIMARY_TP]}
    by_alpha = {}
    for alpha in args.alphas:
        print(f"\n---- alpha = {alpha} ----", flush=True)
        per_q, n_changed = prf_fuse(
            alpha, prep, qids, queries, bm25_100, dense_orig_100, qv, pv, grade_of, args
        )
        by_alpha[f"alpha{alpha:g}"] = {
            "alpha": alpha,
            "n_query_vecs_changed_by_prf": n_changed,
            "by_tp": analyse_tps(per_q, prep["strata"], tp_defs, args),
        }

    # flattened alpha-major -> stratum-major view: the actual trend table
    trend = {}
    for k in KS:
        kk = f"k{k}"
        for sname in STRATA_ORDER:
            for metric in ("recall", "ndcg"):
                series = []
                for alpha in args.alphas:
                    blk = (
                        by_alpha[f"alpha{alpha:g}"]["by_tp"][PRIMARY_TP]["strata"]
                        .get(sname, {})
                        .get(kk)
                    )
                    if not blk:
                        continue
                    d = blk[f"delta_{metric}_P_minus_A"]
                    series.append(
                        {
                            "alpha": alpha,
                            "n": blk["n_queries"],
                            "mean_A": blk[metric]["A_baseline_hybrid"]["mean"],
                            "mean_P": blk[metric]["P_prf_intersection"]["mean"],
                            "delta": d["mean"] if d else None,
                            "ci95": d["ci95"] if d else None,
                            "p_le_0": d["p_le_0"] if d else None,
                            "wins": d["wins"] if d else None,
                            "losses": d["losses"] if d else None,
                            "ties": d["ties"] if d else None,
                        }
                    )
                if series:
                    trend.setdefault(kk, {}).setdefault(sname, {})[metric] = series
    return {
        "intersection_depth": depth_n,
        "alphas": list(args.alphas),
        "intersection_size_distribution": prep["dist"],
        "strata_cuts": cuts_block(prep["cuts"]),
        "n_by_stratum": {k: len(v) for k, v in prep["strata"].items()},
        "by_alpha": by_alpha,
        "trend_by_k_stratum_metric": trend,
    }


def _print_alpha_sweep(blk):
    print(
        f"\n================ ALPHA SWEEP, N={blk['intersection_depth']} "
        f"(TP = {PRIMARY_TP}) ================",
        flush=True,
    )
    cuts = blk["strata_cuts"]
    if cuts:
        print(
            f"  strata cuts on |I| : empty = 0 | small = 1..{cuts['t1']} | "
            f"mid = {cuts['t1'] + 1}..{cuts['t2']} | large = >{cuts['t2']}",
            flush=True,
        )
    for name in STRATA_ORDER:
        if name in blk["n_by_stratum"]:
            print(f"    n[{name}] = {blk['n_by_stratum'][name]}", flush=True)

    for k in KS:
        kk = f"k{k}"
        for metric, label in (("recall", f"R@{k}"), ("ndcg", f"nDCG@{k}")):
            print(f"\n  ---- delta(P,A) {label} vs alpha ----", flush=True)
            print(
                f"  {'stratum':<20s} {'n':>4s} {'alpha':>6s} "
                f"{'A':>8s} {'P':>8s} {'delta':>9s} {'CI95':>22s} "
                f"{'p<=0':>6s} {'W/L/T':>14s}",
                flush=True,
            )
            for name in STRATA_ORDER:
                series = blk["trend_by_k_stratum_metric"].get(kk, {}).get(name, {}).get(metric)
                if not series:
                    continue
                for row in series:
                    ci = f"[{row['ci95'][0]:+.4f}, {row['ci95'][1]:+.4f}]" if row["ci95"] else "n/a"
                    wlt = (
                        f"{row['wins']}/{row['losses']}/{row['ties']}"
                        if row["wins"] is not None
                        else "n/a"
                    )
                    dv = row["delta"] if row["delta"] is not None else float("nan")
                    p0 = row["p_le_0"] if row["p_le_0"] is not None else float("nan")
                    print(
                        f"  {name:<20s} {row['n']:>4d} {row['alpha']:>6.2f} "
                        f"{row['mean_A']:>8.4f} {row['mean_P']:>8.4f} "
                        f"{dv:>+9.4f} {ci:>22s} {p0:>6.3f} {wlt:>14s}",
                        flush=True,
                    )
                print("", flush=True)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--data-dir", default="esci_us_data")
    ap.add_argument("--work-dir", default="/tmp/esci_lexbias")
    ap.add_argument("--tag", default="esci_us")
    ap.add_argument("--depth", type=int, default=100, help="top-D per retrieval leg (fusion)")
    ap.add_argument(
        "--intersection-depths",
        default="20,10,50",
        help="comma-separated N for BM25-topN n Dense-topN; the FIRST is primary",
    )
    ap.add_argument("--alpha", type=float, default=0.5, help="Rocchio weight on the PRF centroid")
    ap.add_argument(
        "--alpha-sweep",
        default=None,
        help=(
            "comma-separated Rocchio alphas, e.g. '0.1,0.2,0.3,0.4,0.5'. When given, "
            "runs an ALPHA SWEEP at the PRIMARY intersection depth only (the "
            "sensitivity depths are skipped) with the primary TP definition only, "
            "reusing one set of intersections/strata across all alphas. Writes to "
            "--sweep-out instead of --out. Without this flag the script behaves "
            "exactly as before: single --alpha, all --intersection-depths, both TPs."
        ),
    )
    ap.add_argument("--rrf-k", type=int, default=60)
    ap.add_argument("--model", default="all-MiniLM-L6-v2")
    ap.add_argument("--vecs-cache", default="/tmp/esci_lexbias/esci_us_minilm_catalog.fp16.npy")
    ap.add_argument(
        "--retrieval-cache", default="/tmp/esci_lexbias/esci_bm25para_rrf_retrieval.npz"
    )
    ap.add_argument("--out", default="evaluation/results/esci_prf_intersection.json")
    ap.add_argument(
        "--sweep-out",
        default="evaluation/results/esci_prf_intersection_alpha_sweep.json",
        help="output path used when --alpha-sweep is given (never overwrites --out)",
    )
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    depths = [int(x) for x in args.intersection_depths.split(",") if x.strip()]
    if not depths:
        raise SystemExit("--intersection-depths produced no depths")
    if max(depths) > args.depth:
        raise SystemExit(f"intersection depth {max(depths)} exceeds --depth {args.depth}")

    args.alphas = None
    if args.alpha_sweep:
        args.alphas = [float(x) for x in args.alpha_sweep.split(",") if x.strip()]
        if not args.alphas:
            raise SystemExit("--alpha-sweep produced no alpha values")
        if any(not 0.0 <= a <= 1.0 for a in args.alphas):
            raise SystemExit(f"--alpha-sweep values must be in [0, 1]: {args.alphas}")
        depths = depths[:1]  # sweep runs at the PRIMARY intersection depth only
    alpha_desc = f"alpha swept over {args.alphas}" if args.alphas else f"alpha={args.alpha}"

    paths = work_paths(args.work_dir, args.tag)
    data = Path(args.data_dir)
    ret_cache = Path(args.retrieval_cache)
    vecs_cache = Path(args.vecs_cache)
    for p, why in (
        (ret_cache, "BM25 lists (run analyze_esci_bm25_paraphrase_rrf.py first)"),
        (vecs_cache, "catalog embeddings (run analyze_esci_bm25_paraphrase_rrf.py first)"),
    ):
        if not p.exists():
            raise SystemExit(f"missing required cache {p}: {why}")

    with open(paths["sample"]) as f:
        rows = json.load(f)["rows"]
    qids = [r["query_id"] for r in rows]
    queries = [r["query"] for r in rows]
    qid_set = set(qids)
    print(f"{len(rows)} sampled queries from {paths['sample']}", flush=True)

    with open(data / "product_ids.json") as f:
        pids = json.load(f)
    pid_to_idx = {p: i for i, p in enumerate(pids)}
    print(f"  catalog: {len(pids):,} products", flush=True)

    grade_of = defaultdict(dict)
    n_qrel, n_unmapped = 0, 0
    with open(data / "test_qrels.jsonl") as f:
        for line in f:
            r = json.loads(line)
            if r["query_id"] not in qid_set:
                continue
            i = pid_to_idx.get(r["product_id"])
            if i is None:
                n_unmapped += 1
                continue
            grade_of[r["query_id"]][i] = int(r["relevance"])
            n_qrel += 1
    lab = defaultdict(int)
    for d in grade_of.values():
        for g in d.values():
            lab[GRADE_LETTER[g]] += 1
    print(
        f"  gold: {n_qrel:,} judged (query, product) pairs over {len(grade_of)} of "
        f"{len(qids)} queries ({n_unmapped} qrels rows not in the catalog); "
        f"labels {dict(lab)}",
        flush=True,
    )

    print(f"\nloading cached BM25 lists {ret_cache}", flush=True)
    z = np.load(ret_cache)
    bm25_orig = z["bm25_orig"]
    dense_orig_cached = z["dense_orig"]
    if bm25_orig.shape[0] != len(qids) or bm25_orig.shape[1] < args.depth:
        raise SystemExit(
            f"cached bm25_orig {bm25_orig.shape} does not cover "
            f"{len(qids)} queries x depth {args.depth}"
        )
    bm25_100 = bm25_orig[:, : args.depth]

    print(f"loading cached catalog vecs {vecs_cache}", flush=True)
    pv = np.load(vecs_cache).astype(np.float32)
    if pv.shape[0] != len(pids):
        raise SystemExit(f"catalog vecs {pv.shape} vs {len(pids)} product ids")
    print(f"  {pv.shape[0]:,} x {pv.shape[1]} float32", flush=True)

    qv = encode_queries(args.model, queries)
    dense_orig_100 = dense_topk_from_qv(qv, pv, args.depth)

    # validity check on reusing the cached dense list vs recomputing it
    agree = [
        len(set(dense_orig_100[i].tolist()) & set(dense_orig_cached[i].tolist())) / args.depth
        for i in range(len(qids))
        if dense_orig_cached.shape[1] >= args.depth
    ]
    exact = [
        bool(np.array_equal(dense_orig_100[i], dense_orig_cached[i])) for i in range(len(qids))
    ]
    agreement = {
        "mean_overlap_at_depth": float(np.mean(agree)) if agree else None,
        "min_overlap_at_depth": float(np.min(agree)) if agree else None,
        "frac_rankings_bit_identical": float(np.mean(exact)),
        "note": (
            "Conditions A and P both use the RECOMPUTED dense-orig list so they "
            "differ only in the query vector. This is the agreement with the "
            "cached dense_orig from analyze_esci_bm25_paraphrase_rrf.py; ~1.0 "
            "means the choice is immaterial."
        ),
    }
    print(
        f"  dense-orig vs cached: mean overlap@{args.depth} "
        f"{agreement['mean_overlap_at_depth']:.4f}, "
        f"{agreement['frac_rankings_bit_identical']:.1%} bit-identical",
        flush=True,
    )

    results = {
        "question": (
            "Does the benefit of pseudo-relevance feedback restricted to the "
            "BM25/dense INTERSECTION concentrate in queries with a large "
            "intersection (where the two retrievers already agree, i.e. the "
            "already-easy queries) and vanish or reverse for queries with a "
            "small or empty intersection (the ones that most need help)?"
        ),
        "proxy_disclaimer": (
            "SIMPLIFIED PROXY, NOT A REPRODUCTION. The SIGIR paper's exact "
            "method is not available to us (feedback-term selection, weighting, "
            "which retriever the expanded query is issued against, depths, "
            "corpus). This tests the STRUCTURAL claim only, using a dense-side "
            "Rocchio update over the intersection because this repo already has "
            "cached catalog embeddings. No claim is made about that paper's "
            "numbers or about its method as implemented by its authors."
        ),
        "chain": [
            "eval_esci_llm_judge_lexical_bias.py (paraphrase does not debias an LLM judge)",
            "analyze_esci_paraphrase_fidelity.py (harm concentrates in drifted paraphrases)",
            "analyze_esci_recall_union.py (judge-score union gain = pool-size artifact)",
            "analyze_esci_bm25_paraphrase_rrf.py ('dense also likes this doc' = a ~20x "
            "precision filter on ANY candidate list, not a property of the technique)",
            "analyze_esci_prf_intersection.py (THIS: is PRF-on-intersection a "
            "QUERY-SUBSET artifact -- active only on the already-easy slice?)",
        ],
        "systems": {
            "bm25": (
                "bm25s, k1=1.5, b=0.75, english stemmer + stopwords, full catalog of "
                f"{len(pids):,} titles; lists REUSED from {ret_cache} (index not rebuilt)"
            ),
            "dense": (
                f"{args.model} via sentence-transformers, normalized embeddings, cosine = "
                f"dot product; catalog vecs REUSED from {vecs_cache} (not re-encoded), "
                "query side recomputed so A and P differ only in the query vector"
            ),
            "fusion": (
                "RRF, score += 1/(rrf_k + rank + 1) summed over lists, "
                f"rrf_k={args.rrf_k} (as evaluation/eval_rrf_ensemble.py)"
            ),
            "depth_per_leg": args.depth,
        },
        "conditions": {
            "A_baseline_hybrid": "RRF(BM25-orig-top-D, Dense-orig-top-D)",
            "P_prf_intersection": (
                "RRF(BM25-orig-top-D, Dense-PRF-top-D) where Dense-PRF uses "
                "prf_qv = normalize((1-alpha) * orig_qv + alpha * "
                "mean(catalog_vecs[d] for d in BM25-topN n Dense-topN)), "
                f"{alpha_desc}. Empty intersection => prf_qv = orig_qv "
                "exactly, so P == A by construction for those queries (asserted "
                "as a self-check, and reported as its own stratum)."
            ),
            "why_dense_side_prf": (
                "The dense leg is where cached vectors make a Rocchio update "
                "tractable with zero extra indexing. It is also the leg the "
                "intersection's centroid is defined in."
            ),
        },
        "conventions": {
            "true_positive_primary": "ESCI Exact (relevance 3) only",
            "true_positive_secondary": "Exact + Substitute (relevance >= 2)",
            "why": (
                "Substitute is the adversarial lexical near-miss class of the parent "
                "experiment, so counting S as a positive would score the bias failure "
                "mode as a success."
            ),
            "recall": "|fused top-K & TP| / |TP|, over queries with >=1 TP",
            "ndcg": (
                "ndcg_at_k() from eval_esci_llm_judge_lexical_bias.py: linear gains = "
                "ESCI grade (E=3,S=2,C=1,I=0); unjudged retrieved docs get gain 0; "
                "ideal = all judged docs for that query"
            ),
            "strata": (
                "by |BM25-topN n Dense-topN| at each depth N: an explicit empty bucket "
                "plus closest-to-target tercile cuts over the NON-empty values, derived "
                "from the observed histogram (cuts stored per depth)"
            ),
            "bootstrap": (
                f"n_boot={args.n_boot}, seed={args.seed}. delta(P,A) is PAIRED over "
                "queries within a stratum. Cross-stratum contrasts (large vs small, "
                "large vs empty) are UNPAIRED -- disjoint query groups of unequal size."
            ),
            "baseline_difficulty_contrasts": (
                "condition A's OWN recall/nDCG compared across strata, with no PRF "
                "involved. This is the direct test of 'large-intersection queries are "
                "already easier' and is independent of whether PRF helps."
            ),
        },
        "source_files": {
            "sample": str(paths["sample"]),
            "catalog_ids": str(data / "product_ids.json"),
            "qrels": str(data / "test_qrels.jsonl"),
            "bm25_lists_cache": str(ret_cache),
            "catalog_vecs_cache": str(vecs_cache),
        },
        "dense_orig_vs_cache_agreement": agreement,
        "primary_intersection_depth": depths[0],
        "primary_tp": PRIMARY_TP,
        "n_queries": len(qids),
    }

    if args.alphas:
        results["mode"] = "alpha_sweep"
        results["alpha_sweep"] = {
            "alphas": list(args.alphas),
            "question": (
                "At alpha=0.5 the large-|I| (already-easy) stratum got WORSE while "
                "the small-|I| stratum got a small lift. Is that harm a ceiling "
                "effect of too aggressive a query-vector perturbation on an "
                "already-good query (it should shrink toward 0 as alpha -> 0), or "
                "is it structural (any dense-side Rocchio nudge hurts them)?"
            ),
            "design": (
                "The alpha-INDEPENDENT work (BM25 lists, dense-orig retrieval, "
                "catalog vecs, intersections at the primary depth, tercile cuts, "
                "strata) is computed ONCE. Only the alpha-DEPENDENT leg (Rocchio "
                "query update -> dense-PRF top-D -> RRF fusion -> metrics) reruns "
                "per alpha, so every alpha is compared on identical strata and "
                "identical baseline condition A."
            ),
            "scope_note": (
                "Primary TP (Exact) only and the primary intersection depth only; "
                "the E+S and N=10/N=50 sensitivity checks live in the single-alpha "
                "run (esci_prf_intersection.json) and are skipped here."
            ),
            "empty_stratum_expectation": (
                "delta(P, A) must be EXACTLY 0.0 in the empty stratum at every "
                "alpha, since prf_qv = orig_qv there by construction regardless of "
                "alpha. Asserted by the same self-check as the single-alpha run."
            ),
        }
        results["sweep"] = run_alpha_sweep(
            depths[0], qids, queries, bm25_100, dense_orig_100, qv, pv, grade_of, args
        )
        _print_alpha_sweep(results["sweep"])
        out = Path(args.sweep_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nsaved -> {out}", flush=True)
        return

    results["by_depth"] = {}
    for n in depths:
        results["by_depth"][f"N{n}"] = run_depth(
            n, qids, queries, bm25_100, dense_orig_100, qv, pv, grade_of, args
        )

    for n in depths:
        blk = results["by_depth"][f"N{n}"]
        for tp_name in TP_DEFS:
            merged = {
                "intersection_depth": n,
                "strata_cuts": blk["strata_cuts"],
                "n_by_stratum": blk["n_by_stratum"],
                **blk["by_tp"][tp_name],
            }
            if tp_name == PRIMARY_TP or n == depths[0]:
                _print_depth(merged, tp_name)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nsaved -> {out}", flush=True)


if __name__ == "__main__":
    main()
