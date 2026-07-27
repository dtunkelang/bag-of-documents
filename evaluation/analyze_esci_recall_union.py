#!/usr/bin/env python3
"""Does UNION-ing top-K-by-literal-score with top-K-by-faithful-paraphrase-score
recover relevant candidates that the literal query's lexical bias buried?

Chain so far
------------
1. `eval_esci_llm_judge_lexical_bias.py` -- a gpt-4o-mini pointwise judge over
   ESCI (query, candidate) pairs, scored twice: once with the literal query and
   once with a paraphrase. Paraphrasing does NOT debias the judge, and used as a
   *reranking replacement* it looked like it hurt nDCG@10.
2. `analyze_esci_paraphrase_fidelity.py` -- adds a 1-5 fidelity judge over the
   paraphrases and shows the nDCG harm concentrates in LOW-fidelity paraphrases.
   HIGH-fidelity ones (argmax >= 4, n=123) are nDCG-NEUTRAL as a replacement.
3. THIS SCRIPT -- a different use case. If faithful-paraphrase scoring is neutral
   rather than harmful, and it keys on partly different tokens than the literal
   query, then the union of the two top-K lists is a *candidate-pool expansion*:
   it may surface true positives the literal query's lexical bias specifically
   suppressed. Recall-expansion, not reranking-replacement.

Three things have to hold for that story, and this script tests all three:

  (1) RECALL. union recall@K > literal recall@K, paired-bootstrap CI clear of 0.
      The union pool is larger than K by construction, so a raw union-vs-literal
      recall win is partly free. The honest control is `literal_size_matched`:
      literal-alone extended to the same pool size as that query's union. If the
      union does not beat THAT, the "gain" is just a bigger pool.

  (2) ATTRIBUTION. The true positives the union adds should be LOW literal-query
      lexical overlap -- exactly the ones the Alaofi-style bias mechanism buries.
      Two references matter, not one:
        - vs. TPs literal already catches (weak test: almost anything outside
          the literal top-K is lower overlap than what's inside it), and
        - vs. ALL TPs literal misses, in or out of the union (sharp test: if
          union-recovered TPs look like a random draw from literal's misses,
          the paraphrase is adding diversity, not undoing a bias).

  (3) PRECISION COST. A recall gain bought with a precision collapse just moves
      work downstream. Reported as TP-precision and as not-Irrelevant precision.

"True positive" = ESCI Exact (grade 3) as the primary convention: the parent
script's whole premise is that Substitute is the adversarial *near-miss* class
("iphone 13 case" -> an iphone 12 case), so folding S into the positives would
score the bias mechanism's favourite failure as a success. E+S is reported
alongside as a sensitivity check. Graded nDCG conventions (linear gains, E=3,
S=2, C=1, I=0) are untouched -- this analysis is set-based, not ranking-based.

Subsets mirror `analyze_esci_paraphrase_fidelity.py`'s primary split:
faithful = fidelity_argmax >= 4 (n=123), drifted = the rest (n=127), plus the
unrestricted population (n=250) as the "what if you don't gate on fidelity"
contrast.

Read-only: consumes the cached sample / score matrices / fidelity / paraphrase
files under --work-dir. No API calls.

Usage (this repo's .venv lacks numpy; use uv):
    uv run --no-project --with numpy --with openai --with python-dotenv python \
        evaluation/analyze_esci_recall_union.py
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
    overlap_metrics,
    work_paths,
)

KS = (5, 10)
# TP conventions. Primary is Exact-only; see module docstring.
TP_DEFS = {
    "E": lambda g: g == 3,
    "E+S": lambda g: g >= 2,
}
PRIMARY_TP = "E"


# --------------------------------------------------------------------------
# stats
# --------------------------------------------------------------------------
def _paired_boot(dif, n_boot, seed):
    """Paired bootstrap over queries on the mean of a per-query delta.

    Same construction as `analyze_esci_paraphrase_fidelity._paired_boot`: the
    two arms are computed on the SAME query, so resample query indices once.
    """
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
    """mean(a) - mean(b) with an independent bootstrap on each side.

    Used for the attribution check, where the two groups are disjoint *item*
    sets of different sizes (no pairing available).
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


# --------------------------------------------------------------------------
# per-query table
# --------------------------------------------------------------------------
def build_table(rows, para_of, lit, par):
    """Per-query candidate records, dropping NaN-scored pairs exactly as
    `_analyze_scorer` does (a pair needs BOTH conditions finite to be usable)."""
    per_q = []
    n_nan = 0
    for qi, r in enumerate(rows):
        pinfo = para_of[r["query_id"]]
        cands = []
        for ci, (pid, g, title) in enumerate(
            zip(r["product_ids"], r["grades"], r["titles"])
        ):
            s_lit, s_par = float(lit[qi, ci]), float(par[qi, ci])
            if not (np.isfinite(s_lit) and np.isfinite(s_par)):
                n_nan += 1
                continue
            cov_lit, jac_lit = overlap_metrics(r["query"], title)
            cov_par, jac_par = overlap_metrics(pinfo["paraphrase"], title)
            cands.append(
                {
                    "pid": pid,
                    "grade": g,
                    "label": GRADE_LETTER[g],
                    "cov_lit": cov_lit,
                    "jac_lit": jac_lit,
                    "cov_par": cov_par,
                    "jac_par": jac_par,
                    "s_lit": s_lit,
                    "s_par": s_par,
                }
            )
        per_q.append({"qi": qi, "qid": r["query_id"], "query": r["query"], "cands": cands})
    return per_q, n_nan


def _topk_idx(cands, field, k):
    """Indices of the top-k by `field`, descending. Ties broken by the original
    candidate order (mergesort = stable), matching the parent script's ranking."""
    sc = np.asarray([c[field] for c in cands], dtype=np.float64)
    return list(np.argsort(-sc, kind="mergesort")[:k])


# --------------------------------------------------------------------------
# core analysis for one (subset, K, tp_def)
# --------------------------------------------------------------------------
def analyse(per_q, qis, k, tp_pred, n_boot, seed):
    rec = {"literal": [], "paraphrase": [], "union": [], "literal_size_matched": []}
    prec_tp = {"literal": [], "union": []}
    prec_notirrel = {"literal": [], "union": []}
    union_sizes, union_extra = [], []
    n_skipped = 0

    # item-level overlap pools for the attribution check
    ov = {
        "literal_caught_tp": {"cov": [], "jac": []},
        "union_added_tp": {"cov": [], "jac": []},
        "all_literal_missed_tp": {"cov": [], "jac": []},
        "union_missed_tp": {"cov": [], "jac": []},
    }
    # within-query paired form of the same contrast
    paired_added_minus_caught = []
    paired_added_minus_missed = []

    for qi in qis:
        cands = per_q[qi]["cands"]
        tp = {i for i, c in enumerate(cands) if tp_pred(c["grade"])}
        if not tp or len(cands) <= k:
            # No positives, or the pool is not bigger than K so top-K is the
            # whole pool and every arm is trivially identical.
            n_skipped += 1
            continue
        L = set(_topk_idx(cands, "s_lit", k))
        P = set(_topk_idx(cands, "s_par", k))
        U = L | P
        M = len(U)
        Lm = set(_topk_idx(cands, "s_lit", M))  # size-matched literal control

        rec["literal"].append(len(L & tp) / len(tp))
        rec["paraphrase"].append(len(P & tp) / len(tp))
        rec["union"].append(len(U & tp) / len(tp))
        rec["literal_size_matched"].append(len(Lm & tp) / len(tp))
        union_sizes.append(M)
        union_extra.append(M - k)

        prec_tp["literal"].append(len(L & tp) / len(L))
        prec_tp["union"].append(len(U & tp) / len(U))
        ni = {i for i, c in enumerate(cands) if c["grade"] >= 1}
        prec_notirrel["literal"].append(len(L & ni) / len(L))
        prec_notirrel["union"].append(len(U & ni) / len(U))

        caught = tp & L
        added = (tp & U) - L
        missed_by_lit = tp - L
        missed_by_union = tp - U
        for name, idxs in (
            ("literal_caught_tp", caught),
            ("union_added_tp", added),
            ("all_literal_missed_tp", missed_by_lit),
            ("union_missed_tp", missed_by_union),
        ):
            for i in idxs:
                ov[name]["cov"].append(cands[i]["cov_lit"])
                ov[name]["jac"].append(cands[i]["jac_lit"])
        if added and caught:
            paired_added_minus_caught.append(
                float(np.mean([cands[i]["cov_lit"] for i in added]))
                - float(np.mean([cands[i]["cov_lit"] for i in caught]))
            )
        if added and (missed_by_lit - added):
            rest = missed_by_lit - added
            paired_added_minus_missed.append(
                float(np.mean([cands[i]["cov_lit"] for i in added]))
                - float(np.mean([cands[i]["cov_lit"] for i in rest]))
            )

    d_union_lit = [u - l for u, l in zip(rec["union"], rec["literal"])]
    d_union_sizematched = [
        u - l for u, l in zip(rec["union"], rec["literal_size_matched"])
    ]
    d_para_lit = [p - l for p, l in zip(rec["paraphrase"], rec["literal"])]

    out = {
        "k": k,
        "n_queries_used": len(rec["literal"]),
        "n_queries_skipped": n_skipped,
        "recall": {name: _mean_ci(v, n_boot, seed) for name, v in rec.items()},
        "union_pool_size": {
            "mean": float(np.mean(union_sizes)) if union_sizes else None,
            "median": float(np.median(union_sizes)) if union_sizes else None,
            "max": int(np.max(union_sizes)) if union_sizes else None,
            "mean_extra_over_k": float(np.mean(union_extra)) if union_extra else None,
            "frac_queries_union_equals_k": (
                float(np.mean([s == k for s in union_sizes])) if union_sizes else None
            ),
        },
        "delta_union_minus_literal": _paired_boot(d_union_lit, n_boot, seed),
        "delta_union_minus_literal_size_matched": _paired_boot(
            d_union_sizematched, n_boot, seed
        ),
        "delta_paraphrase_minus_literal": _paired_boot(d_para_lit, n_boot, seed),
        "precision_tp": {
            name: _mean_ci(v, n_boot, seed) for name, v in prec_tp.items()
        },
        "precision_tp_delta_union_minus_literal": _paired_boot(
            [u - l for u, l in zip(prec_tp["union"], prec_tp["literal"])], n_boot, seed
        ),
        "precision_not_irrelevant": {
            name: _mean_ci(v, n_boot, seed) for name, v in prec_notirrel.items()
        },
        "precision_not_irrelevant_delta_union_minus_literal": _paired_boot(
            [u - l for u, l in zip(prec_notirrel["union"], prec_notirrel["literal"])],
            n_boot,
            seed,
        ),
        "attribution": {
            "n_items": {name: len(v["cov"]) for name, v in ov.items()},
            "mean_cov_lit": {
                name: (float(np.mean(v["cov"])) if v["cov"] else None)
                for name, v in ov.items()
            },
            "mean_jac_lit": {
                name: (float(np.mean(v["jac"])) if v["jac"] else None)
                for name, v in ov.items()
            },
            # weak test: recovered TPs vs the TPs literal already had
            "cov_union_added_minus_literal_caught": _unpaired_diff_boot(
                ov["union_added_tp"]["cov"], ov["literal_caught_tp"]["cov"], n_boot, seed
            ),
            # sharp test: recovered TPs vs EVERY TP literal missed. If the
            # paraphrase were undoing lexical bias specifically, it should pick
            # the low-overlap end of literal's misses, not a random draw.
            "cov_union_added_minus_all_literal_missed": _unpaired_diff_boot(
                ov["union_added_tp"]["cov"],
                ov["all_literal_missed_tp"]["cov"],
                n_boot,
                seed,
            ),
            "cov_union_added_minus_union_missed": _unpaired_diff_boot(
                ov["union_added_tp"]["cov"], ov["union_missed_tp"]["cov"], n_boot, seed
            ),
            "paired_within_query_cov_added_minus_caught": _paired_boot(
                paired_added_minus_caught, n_boot, seed
            ),
            "paired_within_query_cov_added_minus_other_missed": _paired_boot(
                paired_added_minus_missed, n_boot, seed
            ),
        },
    }
    return out


# --------------------------------------------------------------------------
def _print_block(title, blk):
    r = blk["recall"]
    u = blk["union_pool_size"]
    print(f"\n-- {title}  (K={blk['k']}, n={blk['n_queries_used']} queries) --", flush=True)
    for name in ("literal", "paraphrase", "union", "literal_size_matched"):
        ci = r[name]["ci95"]
        ci_s = f"  [{ci[0]:.4f}, {ci[1]:.4f}]" if ci else ""
        print(f"   recall {name:<22s} {r[name]['mean']:.4f}{ci_s}", flush=True)
    print(
        f"   union pool size mean {u['mean']:.2f} (median {u['median']:.0f}, "
        f"+{u['mean_extra_over_k']:.2f} over K; {u['frac_queries_union_equals_k']:.1%} "
        f"of queries have union == K)",
        flush=True,
    )
    for key, lbl in (
        ("delta_union_minus_literal", "union - literal        "),
        ("delta_union_minus_literal_size_matched", "union - literal@sizematch"),
        ("delta_paraphrase_minus_literal", "paraphrase - literal   "),
    ):
        d = blk[key]
        if d:
            print(
                f"   d {lbl} {d['mean']:+.4f}  CI [{d['ci95'][0]:+.4f}, {d['ci95'][1]:+.4f}]"
                f"  W/L/T {d['wins']}/{d['losses']}/{d['ties']}",
                flush=True,
            )
    pt, pn = blk["precision_tp"], blk["precision_not_irrelevant"]
    print(
        f"   precision(TP)  literal {pt['literal']['mean']:.4f} -> union "
        f"{pt['union']['mean']:.4f}  "
        f"(d {blk['precision_tp_delta_union_minus_literal']['mean']:+.4f})",
        flush=True,
    )
    print(
        f"   precision(!I)  literal {pn['literal']['mean']:.4f} -> union "
        f"{pn['union']['mean']:.4f}  "
        f"(d {blk['precision_not_irrelevant_delta_union_minus_literal']['mean']:+.4f})",
        flush=True,
    )
    a = blk["attribution"]
    m, n = a["mean_cov_lit"], a["n_items"]
    print(
        f"   cov_lit  literal-caught TP {m['literal_caught_tp']:.4f} (n={n['literal_caught_tp']})"
        f" | union-added TP "
        + (
            f"{m['union_added_tp']:.4f} (n={n['union_added_tp']})"
            if m["union_added_tp"] is not None
            else "n/a"
        )
        + f" | all literal-missed TP {m['all_literal_missed_tp']:.4f} (n={n['all_literal_missed_tp']})",
        flush=True,
    )
    for key, lbl in (
        ("cov_union_added_minus_literal_caught", "added - caught      "),
        ("cov_union_added_minus_all_literal_missed", "added - all lit-miss"),
    ):
        d = a[key]
        if d:
            print(
                f"     d cov {lbl} {d['diff']:+.4f}  CI [{d['ci95'][0]:+.4f}, "
                f"{d['ci95'][1]:+.4f}]  P(added<ref)={d['p_a_lt_b']:.3f}",
                flush=True,
            )


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--work-dir", default="/tmp/esci_lexbias")
    ap.add_argument("--tag", default="esci_us")
    ap.add_argument("--scorer", default="graded", choices=["graded", "yesno"])
    ap.add_argument(
        "--fidelity",
        default=None,
        help="fidelity JSON (default: <work-dir>/esci_lexbias_fidelity_<tag>.json)",
    )
    ap.add_argument("--fidelity-threshold", type=int, default=4,
                    help="fidelity_argmax >= this counts as faithful (matches the "
                         "fidelity script's primary_argmax_ge_4 split)")
    ap.add_argument("--out", default="evaluation/results/esci_recall_union.json")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    paths = work_paths(args.work_dir, args.tag)
    fid_path = Path(args.fidelity or (Path(args.work_dir) / f"esci_lexbias_fidelity_{args.tag}.json"))

    with open(paths["sample"]) as f:
        sample = json.load(f)
    rows = sample["rows"]
    with open(paths["para"]) as f:
        para_of = {p["query_id"]: p for p in json.load(f)["paraphrases"]}
    with open(fid_path) as f:
        fid_of = {e["query_id"]: e for e in json.load(f)["fidelity"]}
    lit = np.load(paths[f"score_{args.scorer}_literal"])
    par = np.load(paths[f"score_{args.scorer}_paraphrased"])

    missing = [r["query_id"] for r in rows if r["query_id"] not in fid_of]
    if missing:
        raise SystemExit(f"{len(missing)} sampled queries lack a fidelity label")

    per_q, n_nan = build_table(rows, para_of, lit, par)
    print(
        f"{sum(len(q['cands']) for q in per_q):,} usable pairs over {len(per_q)} queries "
        f"({n_nan} dropped for NaN)",
        flush=True,
    )

    thr = args.fidelity_threshold
    subsets = {
        "faithful": [
            q["qi"] for q in per_q if fid_of[q["qid"]]["fidelity_argmax"] >= thr
        ],
        "drifted": [
            q["qi"] for q in per_q if fid_of[q["qid"]]["fidelity_argmax"] < thr
        ],
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
            "Does UNION-ing top-K-by-literal-judge-score with "
            "top-K-by-faithful-paraphrase-judge-score expand recall by recovering "
            "true positives that the literal query's lexical bias buried?"
        ),
        "scorer": args.scorer,
        "judge_model": "gpt-4o-mini (cached scores from eval_esci_llm_judge_lexical_bias.py)",
        "source_files": {
            "sample": str(paths["sample"]),
            "scores_literal": str(paths[f"score_{args.scorer}_literal"]),
            "scores_paraphrased": str(paths[f"score_{args.scorer}_paraphrased"]),
            "fidelity": str(fid_path),
            "paraphrases": str(paths["para"]),
        },
        "conventions": {
            "true_positive_primary": "ESCI Exact (grade 3) only",
            "true_positive_secondary": "Exact + Substitute (grade >= 2)",
            "why": (
                "Substitute is the parent script's adversarial near-miss class "
                "(lexically close, wrong product), so counting S as a positive "
                "would score the lexical-bias failure mode as a success."
            ),
            "not_irrelevant_precision": "grade >= 1 (E, S or C)",
            "lexical_overlap": (
                "overlap_metrics() from the parent script: cov = |q&d|/|q| "
                "content-token coverage of the LITERAL query in the title "
                "(the parent's primary bias metric); jac = |q&d|/|q|d|"
            ),
            "ties": "stable mergesort on -score, i.e. original candidate order",
            "skipped_queries": (
                "queries with no true positive, or with pool size <= K (top-K "
                "would be the whole pool and all arms coincide)"
            ),
            "size_matched_control": (
                "literal_size_matched = literal-alone taken down to that query's "
                "own union pool size, so the union is compared against an "
                "equally large literal pool rather than against top-K"
            ),
            "fidelity_split": f"fidelity_argmax >= {thr} -> faithful "
                              f"(mirrors primary_argmax_ge_4 in analyze_esci_paraphrase_fidelity.py)",
            "bootstrap": f"n_boot={args.n_boot}, seed={args.seed}; recall/precision "
                         f"deltas are PAIRED over queries, attribution contrasts are "
                         f"unpaired over items (disjoint groups of unequal size)",
        },
        "n_queries_by_subset": {k: len(v) for k, v in subsets.items()},
        "subsets": {},
    }

    for tp_name, tp_pred in TP_DEFS.items():
        for sub_name, qis in subsets.items():
            for k in KS:
                blk = analyse(per_q, qis, k, tp_pred, args.n_boot, args.seed)
                results["subsets"].setdefault(tp_name, {}).setdefault(sub_name, {})[
                    f"k{k}"
                ] = blk

    # console summary: primary TP definition only
    print(f"\n===== TP = {PRIMARY_TP} (primary) =====", flush=True)
    for sub_name in ("faithful", "drifted", "unrestricted"):
        for k in KS:
            _print_block(sub_name, results["subsets"][PRIMARY_TP][sub_name][f"k{k}"])

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nsaved -> {out}", flush=True)


if __name__ == "__main__":
    main()
