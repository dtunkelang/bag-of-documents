#!/usr/bin/env python3
"""Does a ONE-SIDED "BoD doesn't trust itself" fallback beat always-BoD?

Follow-up to `eval_bestbuy_confidence_routing.py`, which found that result-set
coherence (mean pairwise cosine among the top-10 result vectors) genuinely
correlates with correctness -- BoD's own coherence is 0.723 on its 70
categorical wins vs 0.628 on its 25 losses (delta +0.095, CI [+0.035,+0.151],
AUC 0.70) -- and yet every routing rule built on it was a wash or worse. The
rules that lost badly were all SYMMETRIC: "serve whichever model has the higher
coherence" reroutes 138/250 queries, and rerouting >50% of traffic swamps a
75-80% BoD win rate no matter how good the signal is.

This script tests the obvious asymmetric alternative the symmetric sweep never
isolated:

    default to BoD; fall back to base ONLY when BoD's OWN ABSOLUTE coherence
    is below tau.

Two differences from what was already tested. (1) The trigger is BoD's raw
coherence value, not a difference or z-difference against base -- base's
confidence never enters the decision. (2) Because it only fires when BoD itself
looks lost, rather than every time the two models happen to disagree about
which is more confident, the intervention rate is a free parameter we can dial
down to a few percent instead of 46-55%.

The whole point is the intervention-rate/payoff curve, so the primary output is
the full tau sweep -- one row per percentile of BoD's coherence distribution,
each with how many of the 250 queries get switched and the paired-bootstrap
delta vs always-BoD. On top of that:

  * 5-fold CV tau selection (pick tau on 4 folds, apply to the held-out fold),
    for the honest out-of-sample number, under all three of the selection
    objectives the parent script used (combined / junk / hit);
  * the in-sample optimum, labelled as the optimistic upper bound it is;
  * switch precision -- of the queries a given tau switches, what share were
    genuinely BoD categorical losses (i.e. worth switching) vs BoD wins
    (i.e. actively damaged);
  * the same sweep restricted to the 166 divergent queries, so the numbers are
    directly comparable to the zdiff_coherence rule's 13/166.

$0. Pure numpy re-analysis: BoD coherence and both arms' per-query junk@10,
hit@10, R@10, nDCG@10 and E@1 all come out of the already-written
evaluation/results/bestbuy_confidence_routing.json. No retrieval, no judge.

Usage:
  python evaluation/eval_bestbuy_asymmetric_coherence_fallback.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

DEFAULT_ROUTING = "evaluation/results/bestbuy_confidence_routing.json"
DEFAULT_OUT = "evaluation/results/bestbuy_asymmetric_coherence_fallback.json"

TRIGGER = "bod_coherence"

METRICS = (
    ("junk_rate", -1),  # lower is better
    ("hit", +1),
    ("recall", +1),
    ("ndcg", +1),
    ("e1", +1),
)

# Reference points from the parent run, quoted so the comparison lives in the
# same artifact rather than in a chat message.
REFERENCE_POINTS = {
    "always_bod": {
        "rule": "reference -- serve BoD on every query",
        "n_switched": 0,
        "junk_rate": 0.1456,
        "hit": 0.5,
    },
    "symmetric_pick_higher_coherence": {
        "rule": "serve whichever model has the higher coherence (zdiff_coherence > 0)",
        "n_switched": 138,
        "junk_delta": 0.0252,
        "junk_ci95": [0.004, 0.0476],
        "hit_delta": -0.068,
        "hit_ci95": [-0.112, -0.024],
        "verdict": "significantly WORSE on both metrics",
    },
    "zdiff_coherence_threshold_in_sample": {
        "rule": "serve base when zdiff_coherence < -1.488 (in-sample optimum, junk objective)",
        "n_switched": 13,
        "junk_delta": -0.0036,
        "junk_ci95": [-0.014, 0.0052],
        "hit_delta": -0.008,
        "hit_ci95": [-0.02, 0.0],
        "verdict": "non-significant wash, and this is the in-sample optimum",
    },
    "oracle_categorical": {
        "rule": "ORACLE -- serve the per-query categorical winner",
        "n_switched": 34,
        "junk_delta": -0.0296,
        "junk_ci95": [-0.0428, -0.0184],
        "hit_delta": -0.008,
    },
    "oracle_click": {
        "rule": "ORACLE -- serve the per-query click winner",
        "n_switched": 14,
        "junk_delta": 0.0008,
        "hit_delta": 0.056,
        "hit_ci95": [0.032, 0.084],
    },
}


# --------------------------------------------------------------------------
# stats helpers (same convention as eval_bestbuy_confidence_routing.py)
# --------------------------------------------------------------------------
def _paired_delta_ci(a, b, n_boot=2000, seed=0):
    """Paired bootstrap over queries on mean(a) - mean(b)."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    if a.size == 0:
        return float("nan"), (None, None)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, a.size, size=(n_boot, a.size))
    d = a[idx].mean(axis=1) - b[idx].mean(axis=1)
    return float(a.mean() - b.mean()), (
        float(np.percentile(d, 2.5)),
        float(np.percentile(d, 97.5)),
    )


# --------------------------------------------------------------------------
# policy
# --------------------------------------------------------------------------
def _choice(rows, tau):
    """Asymmetric fallback: base iff BoD's own coherence is below tau."""
    return ["base" if r[TRIGGER] < tau else "bod" for r in rows]


def _served(rows, choice):
    out = {}
    for m, _ in METRICS:
        out[m] = np.asarray(
            [r[f"{c}_{m}"] for r, c in zip(rows, choice, strict=True)], dtype=np.float64
        )
    return out


def _metrics_block(rows, choice, n_boot, seed):
    served = _served(rows, choice)
    ref = _served(rows, ["bod"] * len(rows))
    n_sw = int(sum(1 for c in choice if c == "base"))
    out = {
        "n_switched_to_base": n_sw,
        "intervention_rate": n_sw / len(rows) if rows else float("nan"),
    }
    for m, sign in METRICS:
        delta, (lo, hi) = _paired_delta_ci(served[m], ref[m], n_boot, seed)
        out[m] = {
            "value": float(np.nanmean(served[m])),
            "reference": float(np.nanmean(ref[m])),
            "delta_vs_always_bod": delta,
            "ci95": [lo, hi],
            "significant": bool(lo is not None and (lo > 0 or hi < 0)),
            "better_is": "lower" if sign < 0 else "higher",
        }
    return out


def _switch_precision(rows, choice):
    """Of the switched queries, how many were BoD losses (right call) vs wins?"""
    sw = [r for r, c in zip(rows, choice, strict=True) if c == "base"]
    won_by_base = sum(1 for r in sw if r["categorical_winner"] == "base")
    won_by_bod = sum(1 for r in sw if r["categorical_winner"] == "bod")
    ties = len(sw) - won_by_base - won_by_bod
    return {
        "n_switched": len(sw),
        "switched_that_were_bod_losses": won_by_base,
        "switched_that_were_bod_wins": won_by_bod,
        "switched_that_were_ties": ties,
        "precision_vs_bod_losses": (won_by_base / len(sw)) if sw else float("nan"),
        "recall_of_all_bod_losses": (
            won_by_base / max(1, sum(1 for r in rows if r["categorical_winner"] == "base"))
        ),
    }


def _objective(rows, choice, objective):
    s = _served(rows, choice)
    if objective == "junk":
        return float(-np.nanmean(s["junk_rate"]))
    if objective == "hit":
        return float(np.nanmean(s["hit"]))
    return float(np.nanmean(s["hit"]) - np.nanmean(s["junk_rate"]))


OBJECTIVES = ("combined", "junk", "hit")


# --------------------------------------------------------------------------
# sweep / CV / in-sample
# --------------------------------------------------------------------------
def _tau_grid(rows, pct_step=1):
    v = np.asarray([r[TRIGGER] for r in rows], dtype=np.float64)
    pcts = np.arange(0, 100 + pct_step, pct_step)
    return pcts, np.percentile(v, pcts)


def sweep(rows, n_boot, seed, pct_step=1):
    pcts, taus = _tau_grid(rows, pct_step)
    table = []
    for p, tau in zip(pcts, taus, strict=True):
        ch = _choice(rows, float(tau))
        row = {"percentile": int(p), "tau": float(tau)}
        row.update(_metrics_block(rows, ch, n_boot, seed))
        row["switch_quality"] = _switch_precision(rows, ch)
        table.append(row)
    return table


def cv_tau(rows, n_folds, n_boot, seed, objective="combined", n_thr=39):
    """Pick tau on 4 folds, apply to the held-out fold. Fold split matches the
    parent script (default_rng(seed).permutation, strided folds)."""
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(rows))
    folds = [order[i::n_folds] for i in range(n_folds)]
    choice = ["bod"] * len(rows)
    picked = []
    for f in range(n_folds):
        test_idx = {int(i) for i in folds[f]}
        train = [r for i, r in enumerate(rows) if i not in test_idx]
        tv = np.asarray([r[TRIGGER] for r in train], dtype=np.float64)
        cands = np.unique(np.percentile(tv, np.linspace(0.0, 95.0, n_thr)))
        # always-BoD (tau = -inf) is the incumbent; only beat it strictly.
        best_tau, best_obj = None, _objective(train, ["bod"] * len(train), objective)
        for tau in cands:
            obj = _objective(train, _choice(train, float(tau)), objective)
            if obj > best_obj + 1e-12:
                best_tau, best_obj = float(tau), obj
        n_train_sw = 0 if best_tau is None else int(sum(1 for r in train if r[TRIGGER] < best_tau))
        picked.append(
            {
                "fold": f,
                "n_test": len(test_idx),
                "tau": best_tau,
                "train_intervention_rate": n_train_sw / len(train),
            }
        )
        if best_tau is None:
            continue
        for i in sorted(test_idx):
            if rows[i][TRIGGER] < best_tau:
                choice[i] = "base"
    return {
        "objective": objective,
        "n_folds": n_folds,
        "selected_per_fold": picked,
        "n_folds_that_chose_to_intervene": sum(1 for p in picked if p["tau"] is not None),
        "metrics": _metrics_block(rows, choice, n_boot, seed),
        "switch_quality": _switch_precision(rows, choice),
    }


def best_in_sample(rows, n_boot, seed, objective="combined", n_thr=39):
    v = np.asarray([r[TRIGGER] for r in rows], dtype=np.float64)
    cands = np.unique(np.percentile(v, np.linspace(0.0, 95.0, n_thr)))
    best_tau, best_obj = None, _objective(rows, ["bod"] * len(rows), objective)
    for tau in cands:
        obj = _objective(rows, _choice(rows, float(tau)), objective)
        if obj > best_obj + 1e-12:
            best_tau, best_obj = float(tau), obj
    if best_tau is None:
        return {
            "objective": objective,
            "tau": None,
            "note": (
                "OPTIMISTIC UPPER BOUND -- but no tau beat always-BoD even with the "
                "threshold fitted on the same queries it is scored on"
            ),
        }
    ch = _choice(rows, best_tau)
    return {
        "objective": objective,
        "tau": best_tau,
        "note": "OPTIMISTIC UPPER BOUND -- tau fitted and scored on the same queries",
        "metrics": _metrics_block(rows, ch, n_boot, seed),
        "switch_quality": _switch_precision(rows, ch),
    }


# --------------------------------------------------------------------------
def _fmt(d, ci):
    lo, hi = ci
    star = "*" if (lo > 0 or hi < 0) else " "
    return f"{d:+.4f} [{lo:+.4f},{hi:+.4f}]{star}"


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--routing", default=DEFAULT_ROUTING)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--n-folds", type=int, default=5)
    ap.add_argument("--n-thresholds", type=int, default=39)
    ap.add_argument("--pct-step", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    with open(args.routing) as f:
        src = json.load(f)
    rows = src["per_query"]
    divergent = [r for r in rows if r["is_divergent"]]
    print(f"  {len(rows)} queries loaded, {len(divergent)} divergent", flush=True)

    coh = np.asarray([r[TRIGGER] for r in rows], dtype=np.float64)
    dist = {
        "mean": float(coh.mean()),
        "std": float(coh.std(ddof=1)),
        "percentiles": {
            str(p): float(np.percentile(coh, p)) for p in (1, 5, 10, 25, 50, 75, 90, 95, 99)
        },
    }

    out = {
        "experiment": "bestbuy_asymmetric_coherence_fallback",
        "question": (
            "Does a one-sided fallback -- default to BoD, serve base only when BoD's OWN "
            "absolute coherence is below tau -- beat always-BoD at a low intervention rate, "
            "where the symmetric 'pick the more coherent model' rules failed?"
        ),
        "config": {
            "source": args.routing,
            "policy": "serve base iff bod_coherence < tau, else serve BoD",
            "trigger_feature": TRIGGER,
            "trigger_definition": src["config"]["coherence_definition"],
            "population": "all 250 holdout queries (the real serving population)",
            "reference": "always serve BoD",
            "n_queries": len(rows),
            "n_divergent": len(divergent),
            "n_boot": args.n_boot,
            "n_folds": args.n_folds,
            "n_thresholds_cv": args.n_thresholds,
            "seed": args.seed,
            "api_spend_usd": 0.0,
            "api_note": "$0 -- pure numpy re-analysis of cached features and labels",
        },
        "bod_coherence_distribution": dist,
        "reference_points": REFERENCE_POINTS,
        "sweep_all_250": sweep(rows, args.n_boot, args.seed, args.pct_step),
        "sweep_divergent_166": sweep(divergent, args.n_boot, args.seed, args.pct_step),
        "cv_selected": {
            o: cv_tau(rows, args.n_folds, args.n_boot, args.seed, o, args.n_thresholds)
            for o in OBJECTIVES
        },
        "best_in_sample": {
            o: best_in_sample(rows, args.n_boot, args.seed, o, args.n_thresholds)
            for o in OBJECTIVES
        },
    }

    # ---- headline verdict
    sw = out["sweep_all_250"]
    any_sig_better_junk = [
        r for r in sw if r["junk_rate"]["ci95"][1] < 0 and r["n_switched_to_base"] > 0
    ]
    any_sig_better_hit = [r for r in sw if r["hit"]["ci95"][0] > 0 and r["n_switched_to_base"] > 0]
    any_sig_worse = [
        r
        for r in sw
        if r["n_switched_to_base"] > 0
        and (r["junk_rate"]["ci95"][0] > 0 or r["hit"]["ci95"][1] < 0)
    ]
    # best point estimate on junk among low-intervention taus
    low = [r for r in sw if 0 < r["intervention_rate"] <= 0.20]
    best_low = min(low, key=lambda r: r["junk_rate"]["delta_vs_always_bod"]) if low else None
    out["verdict"] = {
        "any_tau_significantly_better_on_junk": len(any_sig_better_junk) > 0,
        "any_tau_significantly_better_on_hit": len(any_sig_better_hit) > 0,
        "n_taus_significantly_worse_on_some_metric": len(any_sig_worse),
        "smallest_intervention_rate_that_is_significantly_worse": (
            min(r["intervention_rate"] for r in any_sig_worse) if any_sig_worse else None
        ),
        "best_low_intervention_point": None
        if best_low is None
        else {
            "percentile": best_low["percentile"],
            "tau": best_low["tau"],
            "n_switched": best_low["n_switched_to_base"],
            "intervention_rate": best_low["intervention_rate"],
            "junk_delta": best_low["junk_rate"]["delta_vs_always_bod"],
            "junk_ci95": best_low["junk_rate"]["ci95"],
            "hit_delta": best_low["hit"]["delta_vs_always_bod"],
            "hit_ci95": best_low["hit"]["ci95"],
            "switch_precision_vs_bod_losses": best_low["switch_quality"]["precision_vs_bod_losses"],
        },
        "cv_beats_always_bod": {
            o: bool(
                out["cv_selected"][o]["metrics"]["junk_rate"]["ci95"][1] < 0
                or out["cv_selected"][o]["metrics"]["hit"]["ci95"][0] > 0
            )
            for o in OBJECTIVES
        },
    }

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)

    # ---- console report
    print()
    print("=" * 104)
    print("ASYMMETRIC FALLBACK: serve base iff bod_coherence < tau   (all 250, ref = always BoD)")
    print("=" * 104)
    print(
        f"{'pct':>4} {'tau':>7} {'n_sw':>5} {'rate':>6}  "
        f"{'junk@10 delta [95% CI]':<28} {'hit@10 delta [95% CI]':<28} {'sw prec':>8}"
    )
    for r in sw:
        if r["percentile"] % 2 and r["percentile"] > 20:
            continue
        p = r["switch_quality"]["precision_vs_bod_losses"]
        print(
            f"{r['percentile']:>4} {r['tau']:>7.4f} {r['n_switched_to_base']:>5} "
            f"{r['intervention_rate']:>6.1%}  "
            f"{_fmt(r['junk_rate']['delta_vs_always_bod'], r['junk_rate']['ci95']):<28} "
            f"{_fmt(r['hit']['delta_vs_always_bod'], r['hit']['ci95']):<28} "
            f"{(f'{p:.2f}' if np.isfinite(p) else '   -'):>8}"
        )
    print("  (* = 95% CI excludes zero; sw prec = share of switched queries BoD actually lost)")

    print()
    print("5-FOLD CV-SELECTED tau (out of sample)")
    for o in OBJECTIVES:
        c = out["cv_selected"][o]
        m = c["metrics"]
        taus = [p["tau"] for p in c["selected_per_fold"]]
        print(
            f"  objective={o:<9} folds intervening={c['n_folds_that_chose_to_intervene']}/5 "
            f"taus={['-' if t is None else round(t, 4) for t in taus]}"
        )
        print(
            f"      n_switched={m['n_switched_to_base']:>3} ({m['intervention_rate']:.1%})  "
            f"junk {_fmt(m['junk_rate']['delta_vs_always_bod'], m['junk_rate']['ci95'])}  "
            f"hit {_fmt(m['hit']['delta_vs_always_bod'], m['hit']['ci95'])}"
        )

    print()
    print("IN-SAMPLE OPTIMUM (optimistic upper bound -- tau fitted on the scored queries)")
    for o in OBJECTIVES:
        b = out["best_in_sample"][o]
        if b.get("tau") is None:
            print(f"  objective={o:<9} no tau beat always-BoD even in sample")
            continue
        m = b["metrics"]
        print(
            f"  objective={o:<9} tau={b['tau']:.4f} n_switched={m['n_switched_to_base']:>3} "
            f"({m['intervention_rate']:.1%})  "
            f"junk {_fmt(m['junk_rate']['delta_vs_always_bod'], m['junk_rate']['ci95'])}  "
            f"hit {_fmt(m['hit']['delta_vs_always_bod'], m['hit']['ci95'])}"
        )

    print()
    print("REFERENCE POINTS (from bestbuy_confidence_routing.json)")
    for k, v in REFERENCE_POINTS.items():
        if k == "always_bod":
            continue
        print(
            f"  {k:<38} n_sw={v['n_switched']:>3}  junk {v.get('junk_delta'):+.4f}  "
            f"hit {v.get('hit_delta'):+.4f}"
        )

    print()
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
