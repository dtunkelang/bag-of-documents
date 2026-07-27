#!/usr/bin/env python3
"""Does adding a BM25-on-paraphrase leg to a BM25+dense hybrid help, and is the
help concentrated in docs that dense-on-the-ORIGINAL-query independently likes?

Chain so far
------------
1. `eval_esci_llm_judge_lexical_bias.py` -- a gpt-4o-mini pointwise judge over
   ESCI. Judging a PARAPHRASE instead of the literal query does NOT debias the
   judge's literal-overlap bias, and as a reranking replacement it hurt nDCG@10.
2. `analyze_esci_paraphrase_fidelity.py` -- the harm concentrates in LOW-fidelity
   (semantically drifted) paraphrases. Faithful ones (fidelity_argmax >= 4,
   n=123) are nDCG-NEUTRAL as a replacement.
3. `analyze_esci_recall_union.py` -- reframed as recall expansion: union the
   literal top-K with the paraphrase top-K (both by judge score). The raw gain
   was real (+0.065 R@5 faithful) but DIED under a size-matched control
   (literal-alone expanded to the union's pool size: +0.0065, CI crossing 0),
   and the sharp attribution test failed too. Verdict: pool-size artifact --
   the paraphrase was adding generic diversity, not selectively surfacing
   bias-suppressed relevant items.
4. THIS SCRIPT -- two changes from (3).

   (a) REAL RETRIEVAL, not judge-score reordering of a 20-doc judged pool.
       BM25 (bm25s, k1=1.5, b=0.75, en stem+stopwords) and dense
       (all-MiniLM-L6-v2) over the FULL 360,873-doc ESCI-US catalog, fused with
       RRF (rrf_k=60). Gold comes from esci_us_data/test_qrels.jsonl.

   (b) A NEW DISCRIMINATING TEST that (3) did not have: condition on whether
       BM25-on-paraphrase AGREES with dense-on-the-original-query. BM25 is
       lexically brittle (blind to synonym-only matches); dense is comparatively
       phrasing-robust. So:
         - a doc that BM25-paraphrase surfaces AND dense-orig also independently
           ranks in its top-100 has two independent signals corroborating it;
         - a doc that BM25-paraphrase surfaces and dense-orig does NOT like is a
           single brittle signal, and looks much more like the generic-diversity
           noise (3) found.
       Prediction: TP rate of `dense_corroborated` B-added docs >>
       `dense_uncorroborated`.

Only the BM25 leg gets a paraphrase; the dense leg is untouched. That is the
point of the hypothesis -- we are trying to patch BM25's lexical brittleness
specifically, and we need dense-orig to stay an INDEPENDENT corroborator.

Conditions (per query, depth D=100 per list)
--------------------------------------------
    S_A     = BM25-orig-top-D  u  Dense-orig-top-D
    A       = RRF(BM25-orig-top-D, Dense-orig-top-D)                [baseline_hybrid]
    S_B     = S_A  u  BM25-para-top-D
    B       = RRF(BM25-orig-top-D, Dense-orig-top-D, BM25-para-top-D)  [treatment]
    extra_n = |S_B| - |S_A|     candidates the paraphrase actually ADDED
    C       = RRF(BM25-orig-top-(D+extra_n), Dense-orig-top-D)  [size_matched_control]
    D       = RRF(BM25-orig-top-D, Dense-orig-top-D,
                  BM25-orig-ranks-(D+1..2D) as a fresh rank-1..D list)
                                                            [rank_matched_control]

C is the discipline that killed (3): give the baseline an equally large
candidate pool with ZERO paraphrase signal (just go deeper on the same original
query). ASSUME any raw B-vs-A gain is a pool-size artifact until B beats C.

D was added after seeing C come out nearly inert in this RRF setting: lengthening
one existing list's tail only injects RRF mass of 1/(60+101) and below, so C
almost never displaces anything from A's top-K, which makes it a weak control
even though it is honestly size-matched. D instead supplies a THIRD leg with the
same rank-weight profile as B's paraphrase leg, built from the second page of the
same BM25 original query -- generic diversity, zero paraphrase. B must beat D for
the paraphrase signal itself to matter.

Decision rule:

    delta(B,A) significant but delta(B,C) / delta(B,D) crossing 0  ->  replicates
    the pool-size artifact, hypothesis dead. Only a delta over the controls with
    a CI excluding 0 counts as a real result.

Conventions
-----------
True positive: ESCI Exact (relevance 3) only, primary -- same as (3), because
Substitute is the adversarial lexical near-miss class the bias mechanism
favours, so counting S as a positive would score the failure mode as a success.
E+S reported as a sensitivity check. nDCG uses linear gains = ESCI grade
(E=3,S=2,C=1,I=0) via the parent script's `ndcg_at_k`, over ALL judged docs for
the query as the ideal.

Subsets mirror (3): faithful = fidelity_argmax >= 4 (n=123), drifted = the rest
(n=127), unrestricted = all 250. Queries whose paraphrase status is not "ok"
(5 of 250 came back "identical") get NO paraphrase leg, so B == A for them; they
are kept, not filtered -- dropping them would inflate the treatment arm.

No API calls. Paraphrases and fidelity labels are read from the cached files
under --work-dir. The catalog embedding and the retrieval lists are cached to
--work-dir so re-running the analysis is cheap.

Usage (this repo's .venv has none of numpy/bm25s/sentence-transformers; use uv):
    uv run --no-project --with numpy --with bm25s --with PyStemmer \\
        --with sentence-transformers --with torch --with openai \\
        --with python-dotenv python \\
        evaluation/analyze_esci_bm25_paraphrase_rrf.py
"""

import argparse
import json
import sys
import time
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
# TP conventions. Primary is Exact-only; see module docstring.
TP_DEFS = {
    "E": lambda g: g == 3,
    "E+S": lambda g: g >= 2,
}
PRIMARY_TP = "E"
ARMS = (
    "A_baseline_hybrid",
    "B_treatment",
    "C_size_matched_control",
    "D_rank_matched_control",
)
ADDED_TAGS = ("B_added", "C_added", "D_added")


# --------------------------------------------------------------------------
# stats (same constructions as analyze_esci_recall_union.py)
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
    """mean(a) - mean(b), independent bootstrap on each side.

    For the attribution check: the two groups are disjoint ITEM sets of
    different sizes, so no pairing is available.
    """
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


def _did_boot(a1, a0, b1, b0, n_boot, seed):
    """(mean(a1)-mean(a0)) - (mean(b1)-mean(b0)), 4 independent bootstraps.

    a* = B-added corroborated/uncorroborated, b* = control-added corroborated/
    uncorroborated. Isolates "corroboration matters MORE when the extra signal
    came from a paraphrase" from "dense-top-100 docs are just better docs".
    """
    arrs = []
    for x in (a1, a0, b1, b0):
        v = np.asarray([t for t in x if np.isfinite(t)], dtype=np.float64)
        if v.size < 2:
            return None
        arrs.append(v)
    rng = np.random.default_rng(seed)
    draws = [v[rng.integers(0, v.size, size=(n_boot, v.size))].mean(axis=1) for v in arrs]
    d = (draws[0] - draws[1]) - (draws[2] - draws[3])
    obs = (arrs[0].mean() - arrs[1].mean()) - (arrs[2].mean() - arrs[3].mean())
    return {
        "did": float(obs),
        "ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
        "p_le_0": float(np.mean(d <= 0)),
        "n": [int(v.size) for v in arrs],
    }


# --------------------------------------------------------------------------
# retrieval
# --------------------------------------------------------------------------
def rrf_merge(rankings_list, top_k, rrf_k=60):
    """RRF: for each candidate doc, sum 1/(rrf_k + rank + 1) across every
    ranking it appears in. Returns top-k doc indices by RRF score.

    Verbatim construction from evaluation/eval_rrf_ensemble.py.
    """
    scores = defaultdict(float)
    for rankings in rankings_list:
        for rank, doc_idx in enumerate(rankings):
            scores[int(doc_idx)] += 1.0 / (rrf_k + rank + 1)
    return sorted(scores, key=scores.get, reverse=True)[:top_k]


def build_bm25(titles, k1, b):
    import bm25s
    from Stemmer import Stemmer

    stemmer = Stemmer("english")
    print(f"  tokenizing {len(titles):,} docs (stem=en, stopwords=en)...", flush=True)
    t0 = time.time()
    tok = bm25s.tokenize(titles, stopwords="en", stemmer=stemmer, show_progress=False)
    print(f"    {time.time() - t0:.1f}s", flush=True)
    print(f"  indexing BM25 k1={k1} b={b}...", flush=True)
    t0 = time.time()
    idx = bm25s.BM25(k1=k1, b=b)
    idx.index(tok, show_progress=False)
    print(f"    {time.time() - t0:.1f}s", flush=True)
    return idx, stemmer


def bm25_topk(idx, stemmer, queries, k):
    import bm25s

    t0 = time.time()
    qtok = bm25s.tokenize(queries, stopwords="en", stemmer=stemmer, show_progress=False)
    res_idx, _ = idx.retrieve(qtok, k=k, show_progress=False)
    print(
        f"  BM25 top-{k} for {len(queries):,} queries in {time.time() - t0:.1f}s",
        flush=True,
    )
    return np.asarray(res_idx, dtype=np.int64)


def load_or_encode_catalog(titles, model_name, cache_path):
    import torch
    from sentence_transformers import SentenceTransformer

    cache_path = Path(cache_path)
    if cache_path.exists():
        print(f"  loading cached catalog vecs {cache_path}", flush=True)
        return np.load(cache_path).astype(np.float32)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(
        f"  encoding {len(titles):,} titles with {model_name} on {device} "
        f"(this takes a few minutes)...",
        flush=True,
    )
    m = SentenceTransformer(model_name, device=device)
    v = m.encode(titles, normalize_embeddings=True, batch_size=128, show_progress_bar=True).astype(
        np.float32
    )
    del m
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache_path, v.astype(np.float16))
    print(f"    cached -> {cache_path}", flush=True)
    return v


def dense_topk(model_name, queries, pv, k, chunk=64):
    import torch
    from sentence_transformers import SentenceTransformer

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    m = SentenceTransformer(model_name, device=device)
    qv = m.encode(
        queries, normalize_embeddings=True, batch_size=256, show_progress_bar=False
    ).astype(np.float32)
    del m
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    out = np.zeros((len(queries), k), dtype=np.int64)
    for s in range(0, len(queries), chunk):
        e = min(s + chunk, len(queries))
        sims = qv[s:e] @ pv.T
        part = np.argpartition(-sims, k, axis=1)[:, :k]
        rows = np.arange(part.shape[0])[:, None]
        order = np.argsort(-sims[rows, part], axis=1)
        out[s:e] = part[rows, order]
        del sims
    print(f"  dense top-{k} for {len(queries):,} queries done", flush=True)
    return out


def get_retrieval(args, titles, orig_queries, para_queries, cache):
    """Returns (bm25_orig[n, deep], dense_orig[n, D], bm25_para[n, D] with -1
    rows for queries lacking an ok paraphrase)."""
    cache = Path(cache)
    if cache.exists() and not args.force_retrieve:
        print(f"loading cached retrieval lists {cache}", flush=True)
        z = np.load(cache)
        return z["bm25_orig"], z["dense_orig"], z["bm25_para"]

    deep = 2 * args.depth  # extra_n <= depth, so depth*2 always suffices for C
    print("\nbuilding BM25 index over the full catalog...", flush=True)
    idx, stemmer = build_bm25(titles, args.k1, args.b)
    print(f"\nretrieving BM25 (orig, depth {deep})...", flush=True)
    bm25_orig = bm25_topk(idx, stemmer, orig_queries, deep)

    ok = [i for i, q in enumerate(para_queries) if q is not None]
    bm25_para = np.full((len(orig_queries), args.depth), -1, dtype=np.int64)
    print(f"retrieving BM25 (paraphrase, depth {args.depth}, n={len(ok)})...", flush=True)
    sub = bm25_topk(idx, stemmer, [para_queries[i] for i in ok], args.depth)
    for j, i in enumerate(ok):
        bm25_para[i] = sub[j]
    del idx

    print("\ndense leg...", flush=True)
    pv = load_or_encode_catalog(titles, args.model, args.vecs_cache)
    dense_orig = dense_topk(args.model, orig_queries, pv, args.depth)
    del pv

    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, bm25_orig=bm25_orig, dense_orig=dense_orig, bm25_para=bm25_para)
    print(f"cached retrieval -> {cache}", flush=True)
    return bm25_orig, dense_orig, bm25_para


# --------------------------------------------------------------------------
# per-query fusion table
# --------------------------------------------------------------------------
def build_table(qids, queries, bm25_orig, dense_orig, bm25_para, grade_of, depth, rrf_k, max_k):
    """One record per query with the three fused rankings and the raw dense list."""
    per_q = []
    for i, qid in enumerate(qids):
        bo_full = [int(x) for x in bm25_orig[i]]
        bo = bo_full[:depth]
        de = [int(x) for x in dense_orig[i]]
        bp = [int(x) for x in bm25_para[i] if int(x) >= 0]

        bo_next = bo_full[depth : 2 * depth]  # ranks D+1..2D, re-based to rank 1..D

        s_a = set(bo) | set(de)
        s_b = s_a | set(bp)
        extra_n = len(s_b) - len(s_a)
        extra_n_d = len(s_a | set(bo_next)) - len(s_a)

        fused_a = rrf_merge([bo, de], max_k, rrf_k)
        fused_b = rrf_merge([bo, de, bp], max_k, rrf_k) if bp else list(fused_a)
        bo_ext = bo_full[: depth + extra_n]
        fused_c = rrf_merge([bo_ext, de], max_k, rrf_k) if extra_n else list(fused_a)
        fused_d = rrf_merge([bo, de, bo_next], max_k, rrf_k)

        per_q.append(
            {
                "qi": i,
                "qid": qid,
                "query": queries[i],
                "has_para": bool(bp),
                "extra_n": extra_n,
                "extra_n_rank_matched": extra_n_d,
                "size_S_A": len(s_a),
                "size_S_B": len(s_b),
                "dense_raw": set(de),
                "bm25_orig_raw": bo,
                "bm25_para_raw": bp,
                "bm25_orig_next_raw": bo_next,
                "fused": {
                    "A_baseline_hybrid": fused_a,
                    "B_treatment": fused_b,
                    "C_size_matched_control": fused_c,
                    "D_rank_matched_control": fused_d,
                },
                "grades": grade_of.get(qid, {}),
            }
        )
    return per_q


# --------------------------------------------------------------------------
# analysis for one (subset, K, tp_def)
# --------------------------------------------------------------------------
def analyse(per_q, qis, k, tp_pred, n_boot, seed):
    rec = {a: [] for a in ARMS}
    ndcg = {a: [] for a in ARMS}
    extra_ns, extra_ns_d, n_no_tp = [], [], 0

    # item-level TP indicators for the disagreement-conditioned attribution test
    att = {f"{tag}_dense_{c}": [] for tag in ADDED_TAGS for c in ("corroborated", "uncorroborated")}
    # judged-only variant (drop docs absent from the ESCI qrels for that query)
    att_judged = {name: [] for name in att}
    # pool-level TP rates: the "these are just better docs" null explanation
    pool = {
        "bm25_orig_top_D": [],
        "dense_orig_top_D": [],
        "bm25_para_top_D": [],
        "bm25_para_in_dense_top_D": [],
        "bm25_para_not_in_dense_top_D": [],
        "bm25_orig_ranks_D_to_2D": [],
    }

    for qi in qis:
        q = per_q[qi]
        grades = q["grades"]
        tp = {pid for pid, g in grades.items() if tp_pred(g)}
        extra_ns.append(q["extra_n"])
        extra_ns_d.append(q["extra_n_rank_matched"])
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

    # attribution is defined on the PRIMARY TP convention only, and is computed
    # over every query in the subset (a query with no TP still contributes
    # negative items, which is exactly the denominator we want).
    for qi in qis:
        q = per_q[qi]
        grades = q["grades"]
        a_top = set(q["fused"]["A_baseline_hybrid"][:k])
        dense_raw = q["dense_raw"]
        for name, docs in (
            ("bm25_orig_top_D", q["bm25_orig_raw"]),
            ("dense_orig_top_D", sorted(dense_raw)),
            ("bm25_para_top_D", q["bm25_para_raw"]),
            ("bm25_para_in_dense_top_D", [d for d in q["bm25_para_raw"] if d in dense_raw]),
            (
                "bm25_para_not_in_dense_top_D",
                [d for d in q["bm25_para_raw"] if d not in dense_raw],
            ),
            ("bm25_orig_ranks_D_to_2D", q["bm25_orig_next_raw"]),
        ):
            pool[name].extend(1.0 if grades.get(d, 0) == 3 else 0.0 for d in docs)
        for arm, tag in (
            ("B_treatment", "B_added"),
            ("C_size_matched_control", "C_added"),
            ("D_rank_matched_control", "D_added"),
        ):
            for d in set(q["fused"][arm][:k]) - a_top:
                bucket = (
                    f"{tag}_dense_corroborated"
                    if d in q["dense_raw"]
                    else f"{tag}_dense_uncorroborated"
                )
                is_tp = 1.0 if grades.get(d, 0) == 3 else 0.0
                att[bucket].append(is_tp)
                if d in grades:
                    att_judged[bucket].append(is_tp)

    out = {
        "k": k,
        "n_queries": len(qis),
        "n_queries_with_tp": len(rec[ARMS[0]]),
        "n_queries_no_tp": n_no_tp,
        "extra_n": {
            "mean": float(np.mean(extra_ns)) if extra_ns else None,
            "median": float(np.median(extra_ns)) if extra_ns else None,
            "max": int(np.max(extra_ns)) if extra_ns else None,
            "frac_zero": float(np.mean([e == 0 for e in extra_ns])) if extra_ns else None,
        },
        "extra_n_rank_matched_control": {
            "mean": float(np.mean(extra_ns_d)) if extra_ns_d else None,
            "median": float(np.median(extra_ns_d)) if extra_ns_d else None,
            "max": int(np.max(extra_ns_d)) if extra_ns_d else None,
        },
        "recall": {a: _mean_ci(rec[a], n_boot, seed) for a in ARMS},
        "ndcg": {a: _mean_ci(ndcg[a], n_boot, seed) for a in ARMS},
        "delta_recall_B_minus_A": _paired_boot(
            [b - a for b, a in zip(rec["B_treatment"], rec["A_baseline_hybrid"])],
            n_boot,
            seed,
        ),
        "delta_recall_B_minus_C": _paired_boot(
            [b - c for b, c in zip(rec["B_treatment"], rec["C_size_matched_control"])],
            n_boot,
            seed,
        ),
        "delta_recall_C_minus_A": _paired_boot(
            [c - a for c, a in zip(rec["C_size_matched_control"], rec["A_baseline_hybrid"])],
            n_boot,
            seed,
        ),
        "delta_recall_B_minus_D": _paired_boot(
            [b - d for b, d in zip(rec["B_treatment"], rec["D_rank_matched_control"])],
            n_boot,
            seed,
        ),
        "delta_recall_D_minus_A": _paired_boot(
            [d - a for d, a in zip(rec["D_rank_matched_control"], rec["A_baseline_hybrid"])],
            n_boot,
            seed,
        ),
        "delta_ndcg_B_minus_A": _paired_boot(
            [
                b - a
                for b, a in zip(ndcg["B_treatment"], ndcg["A_baseline_hybrid"])
                if np.isfinite(b) and np.isfinite(a)
            ],
            n_boot,
            seed,
        ),
        "delta_ndcg_B_minus_C": _paired_boot(
            [
                b - c
                for b, c in zip(ndcg["B_treatment"], ndcg["C_size_matched_control"])
                if np.isfinite(b) and np.isfinite(c)
            ],
            n_boot,
            seed,
        ),
        "delta_ndcg_C_minus_A": _paired_boot(
            [
                c - a
                for c, a in zip(ndcg["C_size_matched_control"], ndcg["A_baseline_hybrid"])
                if np.isfinite(c) and np.isfinite(a)
            ],
            n_boot,
            seed,
        ),
        "delta_ndcg_B_minus_D": _paired_boot(
            [
                b - d
                for b, d in zip(ndcg["B_treatment"], ndcg["D_rank_matched_control"])
                if np.isfinite(b) and np.isfinite(d)
            ],
            n_boot,
            seed,
        ),
        "delta_ndcg_D_minus_A": _paired_boot(
            [
                d - a
                for d, a in zip(ndcg["D_rank_matched_control"], ndcg["A_baseline_hybrid"])
                if np.isfinite(d) and np.isfinite(a)
            ],
            n_boot,
            seed,
        ),
        "pool_level_tp_rate_primary": {
            name: {
                "n_items": len(v),
                **{kk: vv for kk, vv in _mean_ci(v, n_boot, seed).items() if kk != "n"},
            }
            for name, v in pool.items()
        },
        "pool_level_contrast_para_in_vs_not_in_dense": _unpaired_diff_boot(
            pool["bm25_para_in_dense_top_D"],
            pool["bm25_para_not_in_dense_top_D"],
            n_boot,
            seed,
        ),
    }

    def _att_block(store):
        blk = {
            "n_items": {name: len(v) for name, v in store.items()},
            "tp_rate": {name: (float(np.mean(v)) if v else None) for name, v in store.items()},
            "tp_rate_ci": {name: _mean_ci(v, n_boot, seed) for name, v in store.items()},
            "B_corroborated_minus_uncorroborated": _unpaired_diff_boot(
                store["B_added_dense_corroborated"],
                store["B_added_dense_uncorroborated"],
                n_boot,
                seed,
            ),
            # control contrasts: the same corroboration split among docs each
            # zero-paraphrase control adds. If the corroboration gap shows up
            # there too, it is a property of dense-top-D, not of the paraphrase.
            "C_corroborated_minus_uncorroborated": _unpaired_diff_boot(
                store["C_added_dense_corroborated"],
                store["C_added_dense_uncorroborated"],
                n_boot,
                seed,
            ),
            "D_corroborated_minus_uncorroborated": _unpaired_diff_boot(
                store["D_added_dense_corroborated"],
                store["D_added_dense_uncorroborated"],
                n_boot,
                seed,
            ),
            # does the paraphrase's corroborated bucket beat the controls'
            # added docs at all?
            "B_corroborated_minus_C_added_all": _unpaired_diff_boot(
                store["B_added_dense_corroborated"],
                store["C_added_dense_corroborated"] + store["C_added_dense_uncorroborated"],
                n_boot,
                seed,
            ),
            "B_corroborated_minus_D_added_all": _unpaired_diff_boot(
                store["B_added_dense_corroborated"],
                store["D_added_dense_corroborated"] + store["D_added_dense_uncorroborated"],
                n_boot,
                seed,
            ),
            "did_B_minus_C_corroboration_effect": _did_boot(
                store["B_added_dense_corroborated"],
                store["B_added_dense_uncorroborated"],
                store["C_added_dense_corroborated"],
                store["C_added_dense_uncorroborated"],
                n_boot,
                seed,
            ),
            "did_B_minus_D_corroboration_effect": _did_boot(
                store["B_added_dense_corroborated"],
                store["B_added_dense_uncorroborated"],
                store["D_added_dense_corroborated"],
                store["D_added_dense_uncorroborated"],
                n_boot,
                seed,
            ),
        }
        blk["small_bucket_warning"] = sorted(name for name, v in store.items() if len(v) < 20)
        return blk

    out["attribution_all_items"] = _att_block(att)
    out["attribution_judged_items_only"] = _att_block(att_judged)
    return out


# --------------------------------------------------------------------------
def _fmt_ci(d, key="ci95"):
    if not d or not d.get(key):
        return "n/a"
    return f"[{d[key][0]:+.4f}, {d[key][1]:+.4f}]"


def _print_block(title, blk):
    print(
        f"\n-- {title}  (K={blk['k']}, n={blk['n_queries']} queries, "
        f"{blk['n_queries_with_tp']} with >=1 TP) --",
        flush=True,
    )
    e = blk["extra_n"]
    ed = blk["extra_n_rank_matched_control"]
    print(
        f"   extra_n (paraphrase-added candidates): mean {e['mean']:.2f} "
        f"median {e['median']:.0f} max {e['max']}  "
        f"({e['frac_zero']:.1%} of queries add nothing)   "
        f"| rank-matched control adds mean {ed['mean']:.2f}",
        flush=True,
    )
    for metric in ("recall", "ndcg"):
        for arm in ARMS:
            v = blk[metric][arm]
            ci = v["ci95"]
            ci_s = f"  [{ci[0]:.4f}, {ci[1]:.4f}]" if ci else ""
            print(f"   {metric}@{blk['k']} {arm:<26s} {v['mean']:.4f}{ci_s}", flush=True)
        for lbl, key in (
            ("B - A         ", f"delta_{metric}_B_minus_A"),
            ("B - C (SIZE)  ", f"delta_{metric}_B_minus_C"),
            ("B - D (RANK)  ", f"delta_{metric}_B_minus_D"),
            ("C - A         ", f"delta_{metric}_C_minus_A"),
            ("D - A         ", f"delta_{metric}_D_minus_A"),
        ):
            d = blk[key]
            if d:
                print(
                    f"     d {metric} {lbl} {d['mean']:+.4f}  CI {_fmt_ci(d)}  "
                    f"W/L/T {d['wins']}/{d['losses']}/{d['ties']}",
                    flush=True,
                )
    a = blk["attribution_all_items"]
    print("   attribution (B-added docs, primary TP = Exact):", flush=True)
    for name in (
        f"{tag}_dense_{c}" for tag in ADDED_TAGS for c in ("corroborated", "uncorroborated")
    ):
        n = a["n_items"][name]
        r = a["tp_rate"][name]
        ci = a["tp_rate_ci"][name]["ci95"]
        ci_s = f"  [{ci[0]:.3f}, {ci[1]:.3f}]" if ci else ""
        flag = "  <-- SMALL (n<20), CI unreliable" if n < 20 else ""
        print(
            f"     {name:<32s} n={n:<5d} TP rate "
            + (f"{r:.3f}{ci_s}" if r is not None else "n/a")
            + flag,
            flush=True,
        )
    for lbl, key in (
        ("B corrob - B uncorrob", "B_corroborated_minus_uncorroborated"),
        ("C corrob - C uncorrob", "C_corroborated_minus_uncorroborated"),
        ("D corrob - D uncorrob", "D_corroborated_minus_uncorroborated"),
        ("B corrob - C added   ", "B_corroborated_minus_C_added_all"),
        ("B corrob - D added   ", "B_corroborated_minus_D_added_all"),
    ):
        d = a[key]
        if d:
            print(
                f"     d TP {lbl} {d['diff']:+.4f}  CI {_fmt_ci(d)}  P(a<b)={d['p_a_lt_b']:.3f}",
                flush=True,
            )
    for lbl, key in (
        ("vs C", "did_B_minus_C_corroboration_effect"),
        ("vs D", "did_B_minus_D_corroboration_effect"),
    ):
        did = a[key]
        if did:
            print(
                f"     DiD {lbl} (B corrob effect - control corrob effect) "
                f"{did['did']:+.4f}  CI {_fmt_ci(did)}",
                flush=True,
            )
    print(
        "   pool-level TP rate (primary, whole raw lists -- the "
        "'these are just better docs' null):",
        flush=True,
    )
    for name, v in blk["pool_level_tp_rate_primary"].items():
        ci = v["ci95"]
        ci_s = f"  [{ci[0]:.3f}, {ci[1]:.3f}]" if ci else ""
        if v["mean"] is not None:
            print(
                f"     {name:<32s} n={v['n_items']:<6d} TP rate {v['mean']:.3f}{ci_s}",
                flush=True,
            )
    d = blk["pool_level_contrast_para_in_vs_not_in_dense"]
    if d:
        print(
            f"     d TP para-in-dense - para-not-in-dense {d['diff']:+.4f}  CI {_fmt_ci(d)}",
            flush=True,
        )


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--data-dir", default="esci_us_data")
    ap.add_argument("--work-dir", default="/tmp/esci_lexbias")
    ap.add_argument("--tag", default="esci_us")
    ap.add_argument("--fidelity", default=None)
    ap.add_argument(
        "--fidelity-threshold",
        type=int,
        default=4,
        help="fidelity_argmax >= this counts as faithful (matches "
        "analyze_esci_recall_union.py / primary_argmax_ge_4)",
    )
    ap.add_argument("--depth", type=int, default=100, help="top-D per retrieval leg")
    ap.add_argument("--rrf-k", type=int, default=60)
    ap.add_argument("--k1", type=float, default=1.5)
    ap.add_argument("--b", type=float, default=0.75)
    ap.add_argument("--model", default="all-MiniLM-L6-v2")
    ap.add_argument("--vecs-cache", default="/tmp/esci_lexbias/esci_us_minilm_catalog.fp16.npy")
    ap.add_argument(
        "--retrieval-cache", default="/tmp/esci_lexbias/esci_bm25para_rrf_retrieval.npz"
    )
    ap.add_argument("--force-retrieve", action="store_true")
    ap.add_argument("--out", default="evaluation/results/esci_bm25_paraphrase_rrf.json")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    paths = work_paths(args.work_dir, args.tag)
    fid_path = Path(
        args.fidelity or (Path(args.work_dir) / f"esci_lexbias_fidelity_{args.tag}.json")
    )
    data = Path(args.data_dir)

    with open(paths["sample"]) as f:
        rows = json.load(f)["rows"]
    with open(paths["para"]) as f:
        para_of = {p["query_id"]: p for p in json.load(f)["paraphrases"]}
    with open(fid_path) as f:
        fid_of = {e["query_id"]: e for e in json.load(f)["fidelity"]}
    missing = [r["query_id"] for r in rows if r["query_id"] not in fid_of]
    if missing:
        raise SystemExit(f"{len(missing)} sampled queries lack a fidelity label")

    print(f"loading catalog from {data}...", flush=True)
    with open(data / "titles.json") as f:
        titles = json.load(f)
    with open(data / "product_ids.json") as f:
        pids = json.load(f)
    if len(titles) != len(pids):
        raise SystemExit(f"titles ({len(titles)}) / ids ({len(pids)}) length mismatch")
    pid_to_idx = {p: i for i, p in enumerate(pids)}
    print(f"  {len(pids):,} products", flush=True)

    qids = [r["query_id"] for r in rows]
    queries = [r["query"] for r in rows]
    qid_set = set(qids)

    # gold: doc_index -> ESCI grade, restricted to the sampled queries
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
        f"  gold: {n_qrel:,} judged (query, product) pairs over "
        f"{len(grade_of)} of {len(qids)} sampled queries "
        f"({n_unmapped} qrels rows not in the catalog); labels {dict(lab)}",
        flush=True,
    )

    para_queries = []
    n_ok = 0
    for r in rows:
        p = para_of.get(r["query_id"])
        if p and p.get("status") == "ok" and p.get("paraphrase"):
            para_queries.append(p["paraphrase"])
            n_ok += 1
        else:
            para_queries.append(None)
    print(
        f"  paraphrases: {n_ok}/{len(rows)} usable (status == 'ok'); the rest get "
        f"no paraphrase leg, so B == A for them",
        flush=True,
    )

    bm25_orig, dense_orig, bm25_para = get_retrieval(
        args, titles, queries, para_queries, args.retrieval_cache
    )
    del titles

    max_k = max(KS)
    per_q = build_table(
        qids,
        queries,
        bm25_orig,
        dense_orig,
        bm25_para,
        grade_of,
        args.depth,
        args.rrf_k,
        max_k,
    )

    thr = args.fidelity_threshold
    subsets = {
        "faithful": [q["qi"] for q in per_q if fid_of[q["qid"]]["fidelity_argmax"] >= thr],
        "drifted": [q["qi"] for q in per_q if fid_of[q["qid"]]["fidelity_argmax"] < thr],
        "unrestricted": [q["qi"] for q in per_q],
    }
    print(
        "subsets: "
        + "  ".join(f"{k}={len(v)}" for k, v in subsets.items())
        + f"   (rule: fidelity_argmax >= {thr})",
        flush=True,
    )

    results = {
        "question": (
            "In a real BM25+dense RRF hybrid over the full ESCI-US catalog, does "
            "adding a BM25-on-paraphrase leg improve recall/nDCG beyond a "
            "size-matched deeper-BM25-orig control, and is any gain concentrated "
            "in docs that dense-on-the-ORIGINAL-query independently corroborates?"
        ),
        "chain": [
            "eval_esci_llm_judge_lexical_bias.py (paraphrase does not debias an LLM judge)",
            "analyze_esci_paraphrase_fidelity.py (harm concentrates in drifted paraphrases)",
            "analyze_esci_recall_union.py (judge-score union gain = pool-size artifact)",
            "analyze_esci_bm25_paraphrase_rrf.py (THIS: real hybrid retrieval + "
            "dense-corroboration-conditioned attribution)",
        ],
        "systems": {
            "bm25": f"bm25s, k1={args.k1}, b={args.b}, english stemmer + stopwords, "
            f"full catalog of {len(pids):,} titles (as evaluation/bm25_retrieve.py)",
            "dense": f"{args.model} via sentence-transformers, normalized embeddings, "
            f"cosine = dot product, catalog vecs cached at {args.vecs_cache}",
            "fusion": f"RRF, score += 1/(rrf_k + rank + 1) summed over lists, "
            f"rrf_k={args.rrf_k} (as evaluation/eval_rrf_ensemble.py)",
            "depth_per_leg": args.depth,
        },
        "conditions": {
            "A_baseline_hybrid": "RRF(BM25-orig-top-D, Dense-orig-top-D)",
            "B_treatment": "RRF(BM25-orig-top-D, Dense-orig-top-D, BM25-para-top-D)",
            "C_size_matched_control": (
                "RRF(BM25-orig-top-(D+extra_n), Dense-orig-top-D), where "
                "extra_n = |S_A u BM25-para-top-D| - |S_A|. Same candidate-pool "
                "size as B with ZERO paraphrase signal. CAVEAT found in this run: "
                "extending ONE list's tail injects only low-rank RRF mass "
                "(1/(60+101) at rank 101), whereas B's third leg injects mass from "
                "rank 1, so C is nearly inert -- it displaces almost nothing from "
                "A's top-K. It is size-matched but not rank-matched."
            ),
            "D_rank_matched_control": (
                "RRF(BM25-orig-top-D, Dense-orig-top-D, BM25-orig-ranks-(D+1..2D) "
                "re-based to rank 1..D). A THIRD LEG with the same rank-weight "
                "profile as B's paraphrase leg and zero paraphrase signal, added "
                "because C turned out nearly inert. Its added pool is >= B's, so it "
                "is a generous control: if B cannot beat D, the paraphrase leg is "
                "not doing anything a second page of the same BM25 query cannot."
            ),
            "decision_rule": (
                "delta(B,A) positive but delta(B,C) crossing 0 replicates the "
                "pool-size artifact of analyze_esci_recall_union.py and kills the "
                "hypothesis. Only delta(B,C) with a CI excluding 0 counts."
            ),
        },
        "conventions": {
            "true_positive_primary": "ESCI Exact (relevance 3) only",
            "true_positive_secondary": "Exact + Substitute (relevance >= 2)",
            "why": (
                "Substitute is the adversarial lexical near-miss class of the "
                "parent experiment, so counting S as a positive would score the "
                "bias failure mode as a success."
            ),
            "recall": "|fused top-K & TP| / |TP|, over queries with >=1 TP",
            "ndcg": "ndcg_at_k() from eval_esci_llm_judge_lexical_bias.py: linear "
            "gains = ESCI grade (E=3,S=2,C=1,I=0); unjudged retrieved docs "
            "get gain 0; ideal = all judged docs for that query",
            "attribution": (
                "B-added = docs in B's fused top-K but not A's fused top-K. "
                "dense_corroborated = also present in Dense-orig's RAW top-D list; "
                "dense_uncorroborated = absent from it (BM25-paraphrase is the sole "
                "signal). TP rate = fraction Exact. Reported over ALL added items "
                "(unjudged counted as non-TP, the honest denominator for a "
                "retrieval experiment) and over judged items only as a sensitivity. "
                "C-added and D-added get the same split as no-paraphrase controls, "
                "plus a difference-in-differences on the corroboration effect. "
                "pool_level_tp_rate_primary is the null this test has to beat: if "
                "BM25-para docs that fall inside dense-top-D are already more "
                "precise than those outside it at the RAW LIST level, the "
                "corroborated-vs-uncorroborated gap is just restating that dense's "
                "top-D is a better pool, with no fusion or paraphrase synergy."
            ),
            "paraphrase_gating": (
                "only status == 'ok' paraphrases are used; queries without one keep "
                "B == A rather than being dropped"
            ),
            "fidelity_split": f"fidelity_argmax >= {thr} -> faithful (mirrors "
            f"primary_argmax_ge_4 in analyze_esci_paraphrase_fidelity.py)",
            "bootstrap": f"n_boot={args.n_boot}, seed={args.seed}; recall/nDCG deltas "
            f"PAIRED over queries, attribution contrasts unpaired over "
            f"items (disjoint groups of unequal size)",
        },
        "source_files": {
            "sample": str(paths["sample"]),
            "paraphrases": str(paths["para"]),
            "fidelity": str(fid_path),
            "catalog_titles": str(data / "titles.json"),
            "catalog_ids": str(data / "product_ids.json"),
            "qrels": str(data / "test_qrels.jsonl"),
        },
        "n_queries_by_subset": {k: len(v) for k, v in subsets.items()},
        "n_paraphrases_usable": n_ok,
        "subsets": {},
    }

    for tp_name, tp_pred in TP_DEFS.items():
        for sub_name, qis in subsets.items():
            for k in KS:
                blk = analyse(per_q, qis, k, tp_pred, args.n_boot, args.seed)
                results["subsets"].setdefault(tp_name, {}).setdefault(sub_name, {})[f"k{k}"] = blk

    print(f"\n===== TP = {PRIMARY_TP} (primary) =====", flush=True)
    for sub_name in ("faithful", "drifted", "unrestricted"):
        for k in KS:
            _print_block(sub_name, results["subsets"][PRIMARY_TP][sub_name][f"k{k}"])

    print("\n===== TP = E+S (sensitivity), recall deltas only =====", flush=True)
    for sub_name in ("faithful", "drifted", "unrestricted"):
        for k in KS:
            blk = results["subsets"]["E+S"][sub_name][f"k{k}"]
            ba = blk["delta_recall_B_minus_A"]
            bc = blk["delta_recall_B_minus_C"]
            print(
                f"  {sub_name:<13s} K={k:<3d} A={blk['recall']['A_baseline_hybrid']['mean']:.4f} "
                f"B={blk['recall']['B_treatment']['mean']:.4f} "
                f"C={blk['recall']['C_size_matched_control']['mean']:.4f}   "
                f"B-A {ba['mean']:+.4f} {_fmt_ci(ba)}   "
                f"B-C {bc['mean']:+.4f} {_fmt_ci(bc)}",
                flush=True,
            )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nsaved -> {out}", flush=True)


if __name__ == "__main__":
    main()
