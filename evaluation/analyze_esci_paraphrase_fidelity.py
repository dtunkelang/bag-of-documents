#!/usr/bin/env python3
"""Does the paraphrase-hurts-nDCG result survive controlling for paraphrase quality?

Follow-up to `evaluation/eval_esci_llm_judge_lexical_bias.py`. That experiment
found that paraphrasing a query before LLM-judging it does NOT debias the judge
(the score-vs-overlap correlation just re-anchors to the paraphrase's own
tokens) and costs -0.0143 nDCG@10.

Confound: manual inspection of the 250 generated paraphrases found a minority
with real *semantic drift*, not just lexical rewording --

    sick puppies              -> ill puppies        (band name destroyed)
    banshee                   -> wraith             (different entity)
    burton step on bindings   -> burton step in ...  (different binding tech)
    sandvik saw               -> cutting tool       (category broadened)
    hanuka candles            -> hanukkah lights    (different product type)

So the nDCG harm may be two mechanisms stacked: (1) bias re-anchoring, the
thing under test, and (2) plain paraphrase-quality noise. This script separates
them by adding one cheap LLM fidelity label per (query, paraphrase) pair and
re-running the ORIGINAL analysis, unchanged, on the faithful and drifted
subsets separately.

No judge re-scoring: the (query x candidate) score matrices from the parent
experiment are reused verbatim from --work-dir. The only new API spend is 250
short fidelity calls (~$0.01).

Metric code is imported from the parent module (`_analyze_scorer`) rather than
reimplemented, so a subset number is computed by exactly the code that produced
the corresponding full-sample number. One consequence to note when reading the
output: `_analyze_scorer` derives BM25 IDF from whatever row set it is handed,
so the "BM25 (pure lexical)" baseline is subset-internal. That affects only
that reference row, not the judge comparisons.

Usage (this repo's .venv lacks numpy; use uv):
    RUN="uv run --no-project --with numpy --with openai --with python-dotenv python"
    $RUN evaluation/analyze_esci_paraphrase_fidelity.py --phase estimate
    $RUN evaluation/analyze_esci_paraphrase_fidelity.py --phase fidelity
    $RUN evaluation/analyze_esci_paraphrase_fidelity.py --phase split
"""

import argparse
import asyncio
import json
import math
import sys
import time
import types
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from eval_esci_llm_judge_lexical_bias import (  # noqa: E402
    SCORERS,
    Usage,
    _analyze_scorer,
    _chat,
    estimate_cost,
    make_client,
    ndcg_at_k,
    record_spend,
    work_paths,
)

# Rubric is deliberately about *shopping intent*, not string similarity: the
# failure mode we are separating out is "different referent / different product
# type", which a lexical similarity score cannot see (`banshee` -> `wraith` is
# a total rewrite; `hanuka candles` -> `hanukkah lights` is a near-copy).
FIDELITY_PROMPT = """A shopping search query was rewritten. Judge whether the \
rewrite preserves the shopper's intent EXACTLY.

Original query: {query}
Rewritten query: {paraphrase}

Would the rewritten query send a shopper after exactly the same products?
Check all of: same product category/type; same brand, model, or named entity \
if one is named; no broadening or narrowing of scope; no substitution of a \
different referent.

5 = identical shopping intent; pure synonym substitution
4 = essentially identical; any difference is trivial
3 = mostly the same, but a requirement is slightly loosened or blurred
2 = noticeably different intent: wrong referent, wrong product type, or \
scope changed
1 = completely different shopping intent

Answer with a single digit (1, 2, 3, 4, or 5)."""

FIDELITY_LEVELS = (1, 2, 3, 4, 5)


def _fidelity_from_logprobs(choice):
    """(expected value, argmax digit) over the first-token distribution.

    Same mechanism as the parent script's `graded` scorer: continuous, tie-free,
    and it exposes how confident the fidelity call was, which a text-parsed
    single digit would throw away.
    """
    if choice is None or not choice.logprobs or not choice.logprobs.content:
        return float("nan"), None
    top = choice.logprobs.content[0].top_logprobs
    if not top:
        return float("nan"), None
    ascii_digits = {str(d): d for d in FIDELITY_LEVELS}
    p = {d: 0.0 for d in FIDELITY_LEVELS}
    for x in top:
        t = x.token.strip()
        if t in ascii_digits:
            p[ascii_digits[t]] += math.exp(x.logprob)
    tot = sum(p.values())
    if tot <= 0:
        return float("nan"), None
    ev = sum(d * v for d, v in p.items()) / tot
    argmax = max(p, key=p.get)
    return ev, argmax


# --------------------------------------------------------------------------
# phase: estimate
# --------------------------------------------------------------------------
def _load_paras(paths):
    with open(paths["para"]) as f:
        payload = json.load(f)
    return payload["paraphrases"], payload


def phase_estimate(args, paths, quiet=False):
    paras, _ = _load_paras(paths)
    envelope = 8
    tin = sum(
        len(FIDELITY_PROMPT.format(query=p["query"], paraphrase=p["paraphrase"])) / 4.0 + envelope
        for p in paras
    )
    tout = len(paras) * 1  # max_tokens=1
    cost = estimate_cost(args.model, tin, tout)
    out = {
        "model": args.model,
        "n_calls": len(paras),
        "est_tokens_in": int(tin),
        "est_tokens_out": int(tout),
        "est_cost_usd": cost,
        "ceiling_usd": args.cost_ceiling,
    }
    if not quiet:
        print(json.dumps(out, indent=2), flush=True)
        print(
            f"\nPROJECTED COST: ${cost:.4f} ({len(paras)} calls) "
            f"vs ceiling ${args.cost_ceiling:.2f}",
            flush=True,
        )
    return out


# --------------------------------------------------------------------------
# phase: fidelity
# --------------------------------------------------------------------------
async def _run_fidelity(args, paras, usage):
    client = make_client()
    sem = asyncio.Semaphore(args.concurrency)

    async def one(p):
        ch = await _chat(
            client, sem, usage, args.model,
            FIDELITY_PROMPT.format(query=p["query"], paraphrase=p["paraphrase"]),
            1, logprobs=True,
        )
        try:
            return _fidelity_from_logprobs(ch)
        except Exception:
            usage.errors += 1
            return float("nan"), None

    return await asyncio.gather(*[one(p) for p in paras])


def phase_fidelity(args, paths, fid_path):
    paras, _ = _load_paras(paths)
    est = phase_estimate(args, paths, quiet=True)
    print(
        f"[cost guard] projected ${est['est_cost_usd']:.4f} "
        f"({est['n_calls']} calls, ceiling ${args.cost_ceiling:.2f})",
        flush=True,
    )
    if est["est_cost_usd"] > args.cost_ceiling:
        raise SystemExit(
            f"Refusing to run: projected ${est['est_cost_usd']:.4f} exceeds "
            f"ceiling ${args.cost_ceiling:.2f}."
        )

    usage = Usage()
    t0 = time.time()
    res = asyncio.run(_run_fidelity(args, paras, usage))
    elapsed = time.time() - t0

    out = []
    for p, (ev, argmax) in zip(paras, res):
        out.append(
            {
                "query_id": p["query_id"],
                "query": p["query"],
                "paraphrase": p["paraphrase"],
                "para_status": p["status"],
                "jaccard_with_original": p["jaccard_with_original"],
                "fidelity_ev": float(ev) if np.isfinite(ev) else None,
                "fidelity_argmax": argmax,
            }
        )

    cost = estimate_cost(args.model, usage.tin, usage.tout)
    record_spend(args.model, usage.tin, usage.tout, cost, "esci lexical-bias: paraphrase fidelity")
    evs = [o["fidelity_ev"] for o in out if o["fidelity_ev"] is not None]
    hist = defaultdict(int)
    for o in out:
        hist[str(o["fidelity_argmax"])] += 1
    payload = {
        "model": args.model,
        "prompt": FIDELITY_PROMPT,
        "scoring": "logprob expected value over digits 1-5, max_tokens=1, top_logprobs=20",
        "n": len(out),
        "wall_clock_s": elapsed,
        "api_calls": usage.calls,
        "api_errors": usage.errors,
        "tokens_in": usage.tin,
        "tokens_out": usage.tout,
        "cost_usd": cost,
        "mean_fidelity_ev": float(np.mean(evs)) if evs else None,
        "median_fidelity_ev": float(np.median(evs)) if evs else None,
        "argmax_histogram": dict(hist),
        "fidelity": out,
    }
    with open(fid_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(
        f"fidelity done in {elapsed:.0f}s  mean EV={payload['mean_fidelity_ev']:.3f}  "
        f"argmax hist={dict(hist)}  errors={usage.errors}  cost=${cost:.4f}",
        flush=True,
    )
    print(f"saved -> {fid_path}", flush=True)


# --------------------------------------------------------------------------
# phase: split
# --------------------------------------------------------------------------
def _per_query_ndcg_deltas(recs):
    """Per-query nDCG@10 for each ranker, from the same `recs` table the parent
    analysis ranks on. Mirrors `_analyze_scorer`'s ranking block for the two
    judge conditions (the pair the paired-bootstrap CI is taken over)."""
    by_q = defaultdict(list)
    for x in recs:
        by_q[x["qi"]].append(x)
    out = {"literal": [], "paraphrased": [], "qi": []}
    for qi in sorted(by_q):
        g = by_q[qi]
        grades_all = [x["grade"] for x in g]
        row = {}
        for cond, fld in (("literal", "s_lit"), ("paraphrased", "s_par")):
            sc = np.asarray([x[fld] for x in g], dtype=np.float64)
            order = np.argsort(-sc, kind="mergesort")
            row[cond] = ndcg_at_k([g[i]["grade"] for i in order], grades_all)
        if not (np.isfinite(row["literal"]) and np.isfinite(row["paraphrased"])):
            continue
        out["literal"].append(row["literal"])
        out["paraphrased"].append(row["paraphrased"])
        out["qi"].append(qi)
    return out


def _paired_boot(dif, n_boot, seed):
    v = np.asarray(dif, dtype=np.float64)
    if v.size < 2:
        return None
    rng = np.random.default_rng(seed)
    draws = v[rng.integers(0, v.size, size=(n_boot, v.size))].mean(axis=1)
    return [float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))]


def _diff_of_diffs_boot(dif_a, dif_b, n_boot, seed):
    """Independent (unpaired) bootstrap on mean(dif_a) - mean(dif_b).

    dif_a / dif_b are the per-query (paraphrased - literal) nDCG deltas of two
    disjoint query subsets, so resampling them independently is the right
    null: 'is the harm bigger in one subset than the other?'
    """
    a = np.asarray(dif_a, dtype=np.float64)
    b = np.asarray(dif_b, dtype=np.float64)
    if a.size < 2 or b.size < 2:
        return None
    rng = np.random.default_rng(seed)
    da = a[rng.integers(0, a.size, size=(n_boot, a.size))].mean(axis=1)
    db = b[rng.integers(0, b.size, size=(n_boot, b.size))].mean(axis=1)
    d = da - db
    return {
        "mean": float(a.mean() - b.mean()),
        "ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
        "p_a_worse_than_b": float(np.mean(d < 0)),
    }


def _label_subset(entry, mode, threshold, restrict=None):
    """faithful / drifted / excluded.

    `restrict` is an optional (field, min_value) gate applied first: queries
    failing it are excluded from BOTH subsets, which is how the conditional
    "fidelity split holding lexical distance roughly fixed" scheme is built.
    """
    if restrict is not None:
        rf, rmin = restrict
        if entry.get(rf) is None or entry[rf] < rmin:
            return "excluded"
    if mode == "argmax":
        v = entry["fidelity_argmax"]
    elif mode == "jaccard":
        v = entry["jaccard_with_original"]
    else:
        v = entry["fidelity_ev"]
    if v is None:
        return "excluded"
    return "faithful" if v >= threshold else "drifted"


def phase_split(args, paths, fid_path, out_path):
    with open(paths["sample"]) as f:
        sample = json.load(f)
    rows = sample["rows"]
    with open(paths["para"]) as f:
        para_payload = json.load(f)
    para_of = {p["query_id"]: p for p in para_payload["paraphrases"]}
    with open(fid_path) as f:
        fid_payload = json.load(f)
    fid_of = {e["query_id"]: e for e in fid_payload["fidelity"]}

    missing = [r["query_id"] for r in rows if r["query_id"] not in fid_of]
    if missing:
        raise SystemExit(f"{len(missing)} sampled queries have no fidelity label; run --phase fidelity")

    # primary split, plus the alternatives, so the headline is not a threshold
    # artefact. Declared here in one place rather than picked after seeing nDCG.
    # The ladder matters because the fidelity judge is stricter than a human
    # skim: it books material/colour-word swaps ("silk"->"satin",
    # "nylon"->"polyester") as drift, which a reader calls a synonym. So the
    # question is asked at several strictness levels, and the answer should be
    # read as a trend across them, not off any single row.
    schemes = {
        "primary_argmax_ge_4": ("argmax", 4),
        "strict_ev_ge_4.0": ("ev", 4.0),
        "purest_argmax_ge_5": ("argmax", 5),
        "loose_argmax_ge_3": ("argmax", 3),
    }
    # Placebo. Splits on lexical distance from the original alone -- no
    # semantic judgement at all. If this reproduces the fidelity split's
    # pattern, the fidelity labels carried no information beyond "how many
    # words changed", and the semantic reading of the result is unsupported.
    jaccs = [fid_of[r["query_id"]]["jaccard_with_original"] for r in rows]
    schemes["placebo_jaccard_median"] = ("jaccard", float(np.median(jaccs)))
    # Conditional: fidelity split applied only to queries in the high-jaccard
    # half, i.e. holding lexical distance from the original roughly fixed. If
    # the harm still concentrates in the drifted side here, the fidelity label
    # carries semantic information the placebo cannot see.
    schemes["conditional_hi_jaccard_argmax_ge_4"] = (
        "argmax", 4, ("jaccard_with_original", float(np.median(jaccs)))
    )
    fid_vs_jac = float(
        np.corrcoef(
            [fid_of[r["query_id"]]["fidelity_ev"] or 0.0 for r in rows], jaccs
        )[0, 1]
    )
    print(f"[placebo] pearson(fidelity_ev, jaccard_with_original) = {fid_vs_jac:+.4f}", flush=True)

    mats = {}
    for scorer in SCORERS:
        lp, pp = paths[f"score_{scorer}_literal"], paths[f"score_{scorer}_paraphrased"]
        if Path(lp).exists() and Path(pp).exists():
            mats[scorer] = (np.load(lp), np.load(pp))
        else:
            print(f"  [{scorer}] score matrices missing; skipping", flush=True)

    sub_args = types.SimpleNamespace(seed=args.seed, n_boot=args.n_boot)

    out = {
        "experiment": "esci_llm_judge_lexical_bias_fidelity_split",
        "question": (
            "Does the nDCG harm from judging a paraphrased query concentrate in "
            "semantically drifted paraphrases (i.e. it is paraphrase-quality noise), "
            "or does it persist among faithful paraphrases (i.e. it is the bias "
            "re-anchoring itself)?"
        ),
        "parent_experiment": "evaluation/results/esci_llm_judge_lexical_bias.json",
        "parent_script": "evaluation/eval_esci_llm_judge_lexical_bias.py",
        "judge_scores_reused": (
            "cached (query x candidate) matrices from the parent run; no judge "
            "re-scoring. Only the 250 fidelity calls are new spend."
        ),
        "metric_code": (
            "subset metrics computed by importing the parent module's "
            "_analyze_scorer unchanged; BM25 IDF is therefore subset-internal "
            "(affects only the BM25 reference row)"
        ),
        "fidelity_judge": {k: v for k, v in fid_payload.items() if k != "fidelity"},
        "n_queries_total": len(rows),
        "n_pairs_total": sample["n_pairs"],
        "n_boot": args.n_boot,
        "seed": args.seed,
        "split_schemes": {},
    }
    out["pearson_fidelity_ev_vs_jaccard_with_original"] = fid_vs_jac
    out["fidelity_judge"]["known_miss"] = (
        "'sick puppies' -> 'ill puppies' scores 5.00: the fidelity judge does not "
        "recognise the band-name referent, so world-knowledge drift of that kind "
        "leaks into the faithful subset and biases the split against the "
        "'harm is drift-only' reading (i.e. conservatively)."
    )

    for scheme_name, spec in schemes.items():
        mode, thr = spec[0], spec[1]
        restrict = spec[2] if len(spec) > 2 else None
        labels = {}
        for r in rows:
            labels[r["query_id"]] = _label_subset(fid_of[r["query_id"]], mode, thr, restrict)
        idx = {
            "faithful": [i for i, r in enumerate(rows) if labels[r["query_id"]] == "faithful"],
            "drifted": [i for i, r in enumerate(rows) if labels[r["query_id"]] == "drifted"],
        }
        scheme_out = {
            "rule": f"{mode} >= {thr} -> faithful",
            "n_queries": {k: len(v) for k, v in idx.items()},
            "n_pairs": {
                k: int(sum(len(rows[i]["titles"]) for i in v)) for k, v in idx.items()
            },
            "n_identical_paraphrases": {
                k: int(sum(1 for i in v if para_of[rows[i]["query_id"]]["status"] != "ok"))
                for k, v in idx.items()
            },
            "mean_jaccard_with_original": {
                k: (
                    float(np.mean([para_of[rows[i]["query_id"]]["jaccard_with_original"] for i in v]))
                    if v else None
                )
                for k, v in idx.items()
            },
            "subsets": {},
        }
        if scheme_name == "primary_argmax_ge_4":
            scheme_out["drifted_examples"] = [
                {
                    "query": rows[i]["query"],
                    "paraphrase": para_of[rows[i]["query_id"]]["paraphrase"],
                    "fidelity_ev": fid_of[rows[i]["query_id"]]["fidelity_ev"],
                    "fidelity_argmax": fid_of[rows[i]["query_id"]]["fidelity_argmax"],
                }
                for i in sorted(
                    idx["drifted"],
                    key=lambda i: fid_of[rows[i]["query_id"]]["fidelity_ev"] or 0.0,
                )[:15]
            ]
            # borderline: faithful-labelled but with the lowest EV
            scheme_out["borderline_faithful_examples"] = [
                {
                    "query": rows[i]["query"],
                    "paraphrase": para_of[rows[i]["query_id"]]["paraphrase"],
                    "fidelity_ev": fid_of[rows[i]["query_id"]]["fidelity_ev"],
                    "fidelity_argmax": fid_of[rows[i]["query_id"]]["fidelity_argmax"],
                }
                for i in sorted(
                    idx["faithful"],
                    key=lambda i: fid_of[rows[i]["query_id"]]["fidelity_ev"] or 0.0,
                )[:10]
            ]

        for scorer, (lit, par) in mats.items():
            per_subset_dif = {}
            for sub_name, ii in idx.items():
                if len(ii) < 5:
                    scheme_out["subsets"].setdefault(sub_name, {})[scorer] = {
                        "skipped": f"only {len(ii)} queries"
                    }
                    continue
                sub_rows = [rows[i] for i in ii]
                sub_lit = lit[ii, :]
                sub_par = par[ii, :]
                print(
                    f"\n########## scheme={scheme_name} scorer={scorer} "
                    f"subset={sub_name} (n_q={len(ii)}) ##########",
                    flush=True,
                )
                res, recs, n_nan, _corr, _disc, _fooled = _analyze_scorer(
                    sub_args, sub_rows, len(sub_rows), para_of, sub_lit, sub_par
                )
                res["n_queries"] = len(sub_rows)
                res["n_pairs_scored"] = len(recs)
                res["n_pairs_dropped_nan"] = n_nan
                pq = _per_query_ndcg_deltas(recs)
                dif = [p - l for p, l in zip(pq["paraphrased"], pq["literal"])]
                per_subset_dif[sub_name] = dif
                res["ndcg10_paired_delta_para_minus_literal"] = {
                    "mean": float(np.mean(dif)),
                    "ci95": _paired_boot(dif, args.n_boot, args.seed),
                    "wins": int(sum(1 for d in dif if d > 0)),
                    "losses": int(sum(1 for d in dif if d < 0)),
                    "ties": int(sum(1 for d in dif if d == 0)),
                    "n_queries": len(dif),
                }
                scheme_out["subsets"].setdefault(sub_name, {})[scorer] = res

            if "faithful" in per_subset_dif and "drifted" in per_subset_dif:
                scheme_out.setdefault("faithful_minus_drifted_ndcg_delta", {})[scorer] = (
                    _diff_of_diffs_boot(
                        per_subset_dif["faithful"], per_subset_dif["drifted"],
                        args.n_boot, args.seed,
                    )
                )
        out["split_schemes"][scheme_name] = scheme_out

    outp = Path(out_path)
    outp.parent.mkdir(parents=True, exist_ok=True)
    with open(outp, "w") as f:
        json.dump(out, f, indent=2)
    _print_summary(out)
    print(f"\nsaved -> {outp}", flush=True)


def _print_summary(out):
    for scheme_name, sc in out["split_schemes"].items():
        print(f"\n================ SPLIT: {scheme_name} ({sc['rule']}) ================")
        print(
            f"  queries  faithful={sc['n_queries']['faithful']} "
            f"drifted={sc['n_queries']['drifted']}   "
            f"pairs faithful={sc['n_pairs']['faithful']} drifted={sc['n_pairs']['drifted']}"
        )
        for scorer in ("graded", "yesno"):
            have = [
                s for s in ("faithful", "drifted")
                if scorer in sc["subsets"].get(s, {})
                and "skipped" not in sc["subsets"][s][scorer]
            ]
            if not have:
                continue
            print(f"\n  --- scorer: {scorer} ---")
            print(
                f"  {'subset':<10s} {'E rho lit':>10s} {'E rho par':>10s} "
                f"{'S rho lit':>10s} {'S rho par':>10s} {'z-gap lit':>10s} "
                f"{'z-gap par':>10s} {'inv lit':>8s} {'inv par':>8s} "
                f"{'dNDCG':>9s} {'CI':>22s}"
            )
            for s in have:
                r = sc["subsets"][s][scorer]
                c = r["overlap_score_correlation_by_label"]
                d = r["e_vs_s_discrimination"]
                f_ = r["es_inversion_rates"]
                nd = r["ndcg10_paired_delta_para_minus_literal"]
                ci = nd["ci95"]
                print(
                    f"  {s:<10s} "
                    f"{c['E']['spearman_literalcov_vs_score_literal']:>10.4f} "
                    f"{c['E']['spearman_owncov_vs_score_paraphrased']:>10.4f} "
                    f"{c['S']['spearman_literalcov_vs_score_literal']:>10.4f} "
                    f"{c['S']['spearman_owncov_vs_score_paraphrased']:>10.4f} "
                    f"{d['literal']['z_gap_E_minus_S']:>10.4f} "
                    f"{d['paraphrased']['z_gap_E_minus_S']:>10.4f} "
                    f"{f_['literal']['inversion_rate']:>8.4f} "
                    f"{f_['paraphrased']['inversion_rate']:>8.4f} "
                    f"{nd['mean']:>+9.4f} "
                    f"[{ci[0]:+.4f},{ci[1]:+.4f}]".rjust(0)
                )
            dd = sc.get("faithful_minus_drifted_ndcg_delta", {}).get(scorer)
            if dd:
                print(
                    f"  faithful-minus-drifted nDCG delta: {dd['mean']:+.4f} "
                    f"CI [{dd['ci95'][0]:+.4f}, {dd['ci95'][1]:+.4f}]  "
                    f"P(faithful worse)={dd['p_a_worse_than_b']:.3f}"
                )


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--phase", required=True, choices=["estimate", "fidelity", "split"])
    ap.add_argument("--work-dir", default="/tmp/esci_lexbias")
    ap.add_argument("--tag", default="esci_us")
    ap.add_argument(
        "--out", default="evaluation/results/esci_llm_judge_lexical_bias_fidelity_split.json"
    )
    ap.add_argument("--model", default="gpt-4o-mini")
    ap.add_argument("--concurrency", type=int, default=32)
    ap.add_argument("--cost-ceiling", type=float, default=2.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-boot", type=int, default=2000)
    args = ap.parse_args()

    paths = work_paths(args.work_dir, args.tag)
    fid_path = Path(args.work_dir) / f"esci_lexbias_fidelity_{args.tag}.json"
    {
        "estimate": lambda: phase_estimate(args, paths),
        "fidelity": lambda: phase_fidelity(args, paths, fid_path),
        "split": lambda: phase_split(args, paths, fid_path, args.out),
    }[args.phase]()


if __name__ == "__main__":
    main()
