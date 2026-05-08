#!/usr/bin/env python3
"""Hunt for a pre-training proxy of rescue rate.

CHS_RESULTS.md Pattern 9 establishes that lift = (rescue × base-blind) −
(tax × base-perfect), and that tax/(1−base) clusters around 0.07–0.13.
But rescue rate varies wildly (4–25pp across our calibration corpora)
with no current pre-training proxy. This script scans the bags.jsonl
files for every trainable corpus and correlates several candidate
metrics against the measured rescue rates.

Candidate proxies (all computable without training):
  - mean bag specificity (already precomputed in bags.jsonl)
  - median bag specificity
  - mean bag size, median bag size
  - log(bag count)
  - SCHS (already in CHS_RESULTS.md)
  - base R@10 (already measured)
  - base-blind subset size

Reports Pearson r against rescue rate for each candidate; flags any with
|r| > 0.5 as candidate proxies.
"""

import json
import os

import numpy as np

# Measured rescue rates from CHS_RESULTS.md, plus base R@10 / base-blind /
# base-perfect for context. Pulled by hand from the validation table.
CALIBRATION = [
    # (label, bags_path, rescue_pp, base_r10, base_blind_pct, base_perfect_pct, schs)
    ("BestBuy ACM", "bestbuy_acm_data/bags.jsonl", 24.9, 0.306, 44.0, 13.0, 0.525),
    ("ESCI-Spanish", "esci_es_data/bags.jsonl", 15.1, 0.074, 67.0, 1.1, 0.45),
    ("FiQA-2018", "fiqa_data/bags.jsonl", 13.0, 0.441, 34.0, 26.5, 0.44),
    ("SciFact", "scifact_data/bags.jsonl", 12.1, 0.783, 20.7, 77.3, float("nan")),
    ("NFCorpus", "nfcorpus_data/bags.jsonl", 4.2, 0.159, 31.0, 4.3, 0.38),
    ("TREC-COVID", "trec_covid_data/bags.jsonl", 0.8, 0.013, 8.0, 0.0, 0.28),  # noisy
    ("Quora", "quora_data/bags.jsonl", 14.0, 0.950, 2.4, 92.0, 0.85),
    ("SCIDOCS", "scidocs_data/bags.jsonl", 6.5, 0.231, 36.1, 0.8, 0.367),
    (
        "CQADup/programmers",
        "cqadupstack_programmers_data/bags.jsonl",
        10.2,
        0.529,
        39.7,
        47.1,
        float("nan"),
    ),
    ("CQADup/gaming", "cqadupstack_gaming_data/bags.jsonl", 16.1, 0.712, 24.5, 66.9, float("nan")),
    ("CQADup/tex", "cqadupstack_tex_data/bags.jsonl", 11.7, 0.416, 52.3, 36.8, float("nan")),
    ("CQADup/gis", "cqadupstack_gis_data/bags.jsonl", 9.7, 0.557, 40.9, 52.4, float("nan")),
    (
        "CQADup/mathematica",
        "cqadupstack_mathematica_data/bags.jsonl",
        13.5,
        0.425,
        50.7,
        37.3,
        float("nan"),
    ),
    ("CQADup/physics", "cqadupstack_physics_data/bags.jsonl", 9.4, 0.597, 32.0, 52.6, float("nan")),
    ("CQADup/stats", "cqadupstack_stats_data/bags.jsonl", 6.4, 0.463, 49.4, 42.9, float("nan")),
]


def measure(path):
    """Returns dict of pre-training metrics for a single corpus's bags.jsonl."""
    if not os.path.exists(path):
        return None
    specs, sizes = [], []
    n = 0
    with open(path) as f:
        for line in f:
            bag = json.loads(line)
            n += 1
            sp = bag.get("specificity")
            if sp is not None and sp == sp:  # not nan
                specs.append(sp)
            sz = bag.get("num_results", len(bag.get("results", [])))
            sizes.append(sz)
    if not n:
        return None
    return {
        "n_bags": n,
        "mean_spec": float(np.mean(specs)) if specs else float("nan"),
        "median_spec": float(np.median(specs)) if specs else float("nan"),
        "std_spec": float(np.std(specs)) if specs else float("nan"),
        "mean_size": float(np.mean(sizes)),
        "median_size": float(np.median(sizes)),
        "log_n_bags": float(np.log10(max(1, n))),
    }


def pearson(xs, ys):
    """Returns Pearson r, dropping any (nan, _) or (_, nan) pairs."""
    pairs = [(x, y) for x, y in zip(xs, ys) if x == x and y == y]
    if len(pairs) < 3:
        return float("nan"), len(pairs)
    xx, yy = zip(*pairs)
    return float(np.corrcoef(xx, yy)[0, 1]), len(pairs)


def main():
    print(
        f"{'corpus':<22} {'n_bags':>7} {'mean_spec':>10} {'med_size':>9} {'rescue':>7} {'base_R10':>9} {'BB%':>5} {'BP%':>5} {'SCHS':>6}"
    )
    rows = []
    for label, path, rescue, base_r, bb, bp, schs in CALIBRATION:
        m = measure(path)
        if m is None:
            print(f"{label:<22}  (no bags.jsonl found at {path})")
            continue
        rows.append(
            {
                "label": label,
                "rescue": rescue,
                "base_r10": base_r,
                "bb_pct": bb,
                "bp_pct": bp,
                "schs": schs,
                **m,
            }
        )
        print(
            f"{label:<22} {m['n_bags']:>7,} {m['mean_spec']:>10.3f} "
            f"{m['median_size']:>9.0f} {rescue:>7.1f} {base_r:>9.3f} "
            f"{bb:>5.1f} {bp:>5.1f} {schs:>6.3f}"
        )

    rescues = [r["rescue"] for r in rows]
    print("\n" + "=" * 72)
    print("Pearson correlation of each candidate proxy vs measured rescue rate:")
    print("=" * 72)
    print(f"  {'proxy':<25} {'r(rescue)':>12} {'n':>4}")
    for key in [
        "mean_spec",
        "median_spec",
        "std_spec",
        "mean_size",
        "median_size",
        "log_n_bags",
        "base_r10",
        "bb_pct",
        "bp_pct",
        "schs",
    ]:
        xs = [r.get(key, float("nan")) for r in rows]
        r, n = pearson(xs, rescues)
        flag = "  ★" if abs(r) > 0.5 else ""
        print(f"  {key:<25} {r:>+12.3f} {n:>4}{flag}")

    # Combined predictor: linear regression over the 3 proxies that passed.
    print("\n" + "=" * 72)
    print("Multi-feature linear regression (rescue ~ log_n_bags + median_size + median_spec):")
    print("=" * 72)
    feature_keys = ["log_n_bags", "median_size", "median_spec"]
    X = np.array([[r[k] for k in feature_keys] for r in rows], dtype=np.float64)
    y = np.array(rescues, dtype=np.float64)
    # Add intercept column.
    Xb = np.hstack([X, np.ones((len(X), 1))])
    # Least squares.
    coef, *_ = np.linalg.lstsq(Xb, y, rcond=None)
    y_pred = Xb @ coef
    residuals = y - y_pred
    ss_res = float(np.sum(residuals**2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1 - ss_res / ss_tot
    print("  fitted weights:")
    for k, w in zip(feature_keys + ["(intercept)"], coef):
        print(f"    {k:<25} {w:>+10.3f}")
    print(f"  R²              = {r2:.3f}")
    print(f"  RMSE            = {np.sqrt(ss_res / len(y)):.2f}pp")

    # Per-corpus residuals — are any clearly mispredicted?
    print("\n  per-corpus predicted vs actual:")
    print(f"    {'corpus':<22} {'actual':>8} {'pred':>8} {'resid':>8}")
    for r, p in zip(rows, y_pred):
        print(f"    {r['label']:<22} {r['rescue']:>8.1f} {p:>8.1f} {r['rescue'] - p:>+8.1f}")

    # Leave-one-out cross-validation: fit on 14, predict the held-out one.
    # Tells us whether the in-sample RMSE is honest or overfit (4 params,
    # 15 points → real risk of overfit).
    print("\n" + "=" * 72)
    print("Leave-one-out cross-validation (3-feature model):")
    print("=" * 72)
    print(f"    {'held-out corpus':<22} {'actual':>8} {'LOO_pred':>9} {'LOO_resid':>10}")
    loo_residuals = []
    for i in range(len(rows)):
        mask = np.ones(len(rows), dtype=bool)
        mask[i] = False
        Xb_train = Xb[mask]
        y_train = y[mask]
        coef_loo, *_ = np.linalg.lstsq(Xb_train, y_train, rcond=None)
        pred_i = float(Xb[i] @ coef_loo)
        resid_i = float(y[i] - pred_i)
        loo_residuals.append(resid_i)
        print(f"    {rows[i]['label']:<22} {y[i]:>8.1f} {pred_i:>9.1f} {resid_i:>+10.1f}")
    loo_residuals = np.array(loo_residuals)
    loo_rmse = float(np.sqrt(np.mean(loo_residuals**2)))
    loo_mae = float(np.mean(np.abs(loo_residuals)))
    loo_ss_res = float(np.sum(loo_residuals**2))
    loo_r2 = 1 - loo_ss_res / ss_tot
    print()
    print(f"  LOO RMSE        = {loo_rmse:.2f}pp  (in-sample {np.sqrt(ss_res / len(y)):.2f}pp)")
    print(f"  LOO MAE         = {loo_mae:.2f}pp")
    print(f"  LOO R²          = {loo_r2:.3f}    (in-sample {r2:.3f})")
    print(
        f"  worst LOO miss  = {float(np.max(np.abs(loo_residuals))):.1f}pp "
        f"({rows[int(np.argmax(np.abs(loo_residuals)))]['label']})"
    )

    # Does a qrels-only model (no encoding needed) also work?
    print("\n" + "=" * 72)
    print("Qrels-only fit (rescue ~ log_n_bags + median_size; no bag encoding):")
    print("=" * 72)
    qrels_keys = ["log_n_bags", "median_size"]
    Xq = np.array([[r[k] for k in qrels_keys] for r in rows], dtype=np.float64)
    Xqb = np.hstack([Xq, np.ones((len(Xq), 1))])
    coef_q, *_ = np.linalg.lstsq(Xqb, y, rcond=None)
    y_pred_q = Xqb @ coef_q
    ss_res_q = float(np.sum((y - y_pred_q) ** 2))
    r2_q = 1 - ss_res_q / ss_tot
    print(
        f"  weights: log_n_bags={coef_q[0]:+.3f}  "
        f"median_size={coef_q[1]:+.3f}  intercept={coef_q[2]:+.3f}"
    )
    print(f"  R²    = {r2_q:.3f}  (vs {r2:.3f} with bag-encoding)")
    print(f"  RMSE  = {np.sqrt(ss_res_q / len(y)):.2f}pp")
    # LOO for qrels-only.
    loo_q = []
    for i in range(len(rows)):
        mask = np.ones(len(rows), dtype=bool)
        mask[i] = False
        c, *_ = np.linalg.lstsq(Xqb[mask], y[mask], rcond=None)
        loo_q.append(float(y[i] - Xqb[i] @ c))
    loo_q = np.array(loo_q)
    loo_rmse_q = float(np.sqrt(np.mean(loo_q**2)))
    loo_r2_q = 1 - float(np.sum(loo_q**2)) / ss_tot
    print(f"  LOO RMSE = {loo_rmse_q:.2f}pp")
    print(f"  LOO R²   = {loo_r2_q:.3f}")

    # Add base R@10 (free — already measured in the readiness tool).
    print("\n" + "=" * 72)
    print("With base R@10 added (rescue ~ log_n_bags + median_size + base_r10):")
    print("=" * 72)
    keys3 = ["log_n_bags", "median_size", "base_r10"]
    X3 = np.array([[r[k] for k in keys3] for r in rows], dtype=np.float64)
    X3b = np.hstack([X3, np.ones((len(X3), 1))])
    coef3, *_ = np.linalg.lstsq(X3b, y, rcond=None)
    y_pred3 = X3b @ coef3
    ss_res3 = float(np.sum((y - y_pred3) ** 2))
    r2_3 = 1 - ss_res3 / ss_tot
    print(
        f"  weights: log_n_bags={coef3[0]:+.3f}  "
        f"median_size={coef3[1]:+.3f}  base_r10={coef3[2]:+.3f}  "
        f"intercept={coef3[3]:+.3f}"
    )
    print(f"  R²    = {r2_3:.3f}")
    print(f"  RMSE  = {np.sqrt(ss_res3 / len(y)):.2f}pp")
    # LOO for the +base_r10 variant.
    loo_3 = []
    for i in range(len(rows)):
        mask = np.ones(len(rows), dtype=bool)
        mask[i] = False
        c, *_ = np.linalg.lstsq(X3b[mask], y[mask], rcond=None)
        loo_3.append(float(y[i] - X3b[i] @ c))
    loo_3 = np.array(loo_3)
    loo_rmse_3 = float(np.sqrt(np.mean(loo_3**2)))
    loo_r2_3 = 1 - float(np.sum(loo_3**2)) / ss_tot
    print(f"  LOO RMSE = {loo_rmse_3:.2f}pp")
    print(f"  LOO R²   = {loo_r2_3:.3f}")

    # base_r10 threshold sweep — does gating out high-base-R@10 corpora
    # tighten the LOO band? The 15-corpus LOO is dragged down by Quora
    # (base R@10 = 0.95, LOO residual −7.7pp). Refit on subsets below
    # each threshold to find the regime where the predictor is reliable.
    print("\n" + "=" * 72)
    print("base_r10 threshold sweep (3-feature model, in-sample + LOOCV):")
    print("=" * 72)
    print(
        f"  {'threshold':>10} {'n':>3} {'in_RMSE':>8} {'in_R²':>7} "
        f"{'LOO_RMSE':>9} {'LOO_R²':>8}  {'worst miss':<22} {'resid':>6}"
    )

    def _refit_loo(subset_rows):
        Xs = np.array([[r[k] for k in feature_keys] for r in subset_rows], dtype=np.float64)
        ys = np.array([r["rescue"] for r in subset_rows], dtype=np.float64)
        Xsb = np.hstack([Xs, np.ones((len(Xs), 1))])
        coef_full, *_ = np.linalg.lstsq(Xsb, ys, rcond=None)
        in_pred = Xsb @ coef_full
        ss_tot_s = float(np.sum((ys - ys.mean()) ** 2))
        in_rmse_s = float(np.sqrt(np.mean((ys - in_pred) ** 2)))
        in_r2_s = 1 - float(np.sum((ys - in_pred) ** 2)) / ss_tot_s
        loo = []
        for j in range(len(subset_rows)):
            mask = np.ones(len(subset_rows), dtype=bool)
            mask[j] = False
            c, *_ = np.linalg.lstsq(Xsb[mask], ys[mask], rcond=None)
            loo.append(float(ys[j] - Xsb[j] @ c))
        loo = np.array(loo)
        loo_rmse_s = float(np.sqrt(np.mean(loo**2)))
        loo_r2_s = 1 - float(np.sum(loo**2)) / ss_tot_s
        worst_j = int(np.argmax(np.abs(loo)))
        return {
            "n": len(subset_rows),
            "in_rmse": in_rmse_s,
            "in_r2": in_r2_s,
            "loo_rmse": loo_rmse_s,
            "loo_r2": loo_r2_s,
            "worst": subset_rows[worst_j]["label"],
            "worst_resid": loo[worst_j],
            "coef": coef_full,
        }

    for thr in [1.01, 0.90, 0.85, 0.80, 0.75, 0.70]:
        sub = [r for r in rows if r["base_r10"] < thr]
        if len(sub) < 5:
            continue
        f = _refit_loo(sub)
        label = f"base<{thr:.2f}" if thr < 1.0 else "all-15"
        print(
            f"  {label:>10} {f['n']:>3} {f['in_rmse']:>8.2f} {f['in_r2']:>7.3f} "
            f"{f['loo_rmse']:>9.2f} {f['loo_r2']:>8.3f}  "
            f"{f['worst']:<22} {f['worst_resid']:>+6.1f}"
        )

    # Production gate: base_r10 < 0.85 drops Quora only, doubles LOO R².
    print("\n  Production gate (base_r10 < 0.85), fitted weights:")
    sub = [r for r in rows if r["base_r10"] < 0.85]
    f_prod = _refit_loo(sub)
    for k, w in zip(feature_keys + ["(intercept)"], f_prod["coef"]):
        print(f"    {k:<25} {w:>+10.3f}")
    print(
        f"  Use these in bod_readiness_report.py with RESCUE_RMSE_PP="
        f"{f_prod['loo_rmse']:.2f} and RESCUE_BASE_R10_MAX=0.85."
    )


if __name__ == "__main__":
    main()
