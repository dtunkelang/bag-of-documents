#!/usr/bin/env python3
"""Bulk readiness sweep across every locally-available corpus.

Calls the bod_readiness_report.py functions directly (no subprocess
parsing) and emits a single TSV with: corpus, n_queries, base_R@10,
base-blind%, base-perfect%, n_bags, median_size, median_spec,
predicted rescue (or N/A if out of regime), predicted realistic lift,
SCHS, verdict.

Useful as a retrospective validation of the full readiness pipeline
(SCHS gate + base-difficulty + Pattern 8a predictor + v2 tax bands)
against measured outcomes from CHS_RESULTS.md, and as a screening tool
for new corpora before committing to training.
"""

import os
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "evaluation"))

import bod_readiness_report as rr  # noqa: E402

from bagofdocs.cluster_hypothesis import compute_chs  # noqa: E402

ENCODER = "all-MiniLM-L6-v2"
# (label, data_dir, pid_file, optional measured rescue from CHS_RESULTS for
# retrospective validation; None when unmeasured.)
CORPORA = [
    ("BestBuy ACM", "bestbuy_acm_data", "product_ids.json", 24.9),
    ("ESCI-Spanish", "esci_es_data", "product_ids.json", 15.1),
    ("FiQA-2018", "fiqa_data", "product_ids.json", 13.0),
    ("SciFact", "scifact_data", "product_ids.json", 12.1),
    ("NFCorpus", "nfcorpus_data", "product_ids.json", 4.2),
    ("TREC-COVID", "trec_covid_data", "product_ids.json", 0.8),
    ("Quora", "quora_data", "product_ids.json", 14.0),
    ("SCIDOCS", "scidocs_data", "product_ids.json", 6.5),
    ("CQADup/programmers", "cqadupstack_programmers_data", "product_ids.json", 10.2),
    ("CQADup/gaming", "cqadupstack_gaming_data", "product_ids.json", 16.1),
    ("CQADup/tex", "cqadupstack_tex_data", "product_ids.json", 11.7),
    ("CQADup/gis", "cqadupstack_gis_data", "product_ids.json", 9.7),
    ("CQADup/mathematica", "cqadupstack_mathematica_data", "product_ids.json", 13.5),
    ("CQADup/physics", "cqadupstack_physics_data", "product_ids.json", 9.4),
    ("CQADup/stats", "cqadupstack_stats_data", "product_ids.json", 6.4),
    ("ArguAna", "arguana_data", "product_ids.json", None),  # never measured
]


def run_one(label, data_dir, pid_file):
    """Returns a dict of summary metrics for one corpus, or None on failure."""
    catalog = os.path.join(data_dir, "titles.json")
    pids = os.path.join(data_dir, pid_file)
    qrels = os.path.join(data_dir, "test_qrels.jsonl")
    queries = os.path.join(data_dir, "test_queries.jsonl")
    vecs = os.path.join(data_dir, "base_catalog.vecs.fp16.npy")
    if not all(os.path.exists(p) for p in [catalog, pids, qrels, queries]):
        return None

    args = SimpleNamespace(
        catalog=catalog,
        product_ids=pids,
        qrels=qrels,
        queries=queries,
        min_relevance=1,
        encoder=ENCODER,
        vecs_cache=vecs if os.path.exists(vecs) else None,
        label=label,
        k=10,
        chunk=512,
    )
    titles, pid_list, _pid_to_idx, queries_by_qid, qrels_full, pos = rr.load_corpus(args)
    chs = compute_chs(dict(qrels_full), pid_list, titles, ENCODER, partition="strict")
    bd_result = rr.base_difficulty(args, titles, pid_list, queries_by_qid, pos)
    if bd_result is None:
        return None
    bd, base_pv = bd_result
    pid_to_idx = {p: i for i, p in enumerate(pid_list)}
    bag_stats = rr.compute_bag_stats(qrels_full, pid_to_idx, base_pv, args.min_relevance)
    predicted_rescue = rr.predict_rescue_rate(bag_stats, base_r10=bd["overall_R10"])
    predicted = rr.predict_lift(
        bd["base_blind"], bd["base_perfect"], bd["overall_R10"], predicted_rescue
    )
    v_label, _ = rr.verdict(chs.schs, bd["base_blind"], bd["base_perfect"], predicted)
    return {
        "n_queries": bd["n_queries"],
        "base_R10": bd["overall_R10"],
        "base_blind_pct": bd["base_blind"] * 100,
        "base_perfect_pct": bd["base_perfect"] * 100,
        "n_bags": bag_stats["n_bags"] if bag_stats else 0,
        "median_size": bag_stats["median_size"] if bag_stats else float("nan"),
        "median_spec": bag_stats["median_spec"] if bag_stats else float("nan"),
        "predicted_rescue_pp": predicted_rescue * 100 if predicted_rescue is not None else None,
        "realistic_lift_pp": predicted["realistic"] * 100,
        "optimistic_lift_pp": predicted["optimistic"] * 100,
        "schs": chs.schs,
        "verdict": v_label,
    }


def main():
    out_path = ROOT / "logs" / "readiness_sweep.tsv"
    out_path.parent.mkdir(exist_ok=True)
    rows = []
    for label, data_dir, pid_file, measured_rescue in CORPORA:
        print(f"[{label}] starting...", flush=True)
        try:
            r = run_one(label, data_dir, pid_file)
        except Exception as e:
            print(f"[{label}] ERROR: {e}", flush=True)
            continue
        if r is None:
            print(f"[{label}] missing data — skipped", flush=True)
            continue
        r["label"] = label
        r["measured_rescue_pp"] = measured_rescue
        rows.append(r)
        pred_str = (
            f"{r['predicted_rescue_pp']:5.1f}" if r["predicted_rescue_pp"] is not None else "  N/A"
        )
        meas_str = f"{measured_rescue:5.1f}" if measured_rescue is not None else "  N/A"
        print(
            f"[{label}] R@10={r['base_R10']:.3f} BB={r['base_blind_pct']:.1f}% "
            f"BP={r['base_perfect_pct']:.1f}% rescue_pred={pred_str} "
            f"rescue_meas={meas_str} verdict={r['verdict']}",
            flush=True,
        )

    headers = [
        "corpus",
        "n_queries",
        "base_R10",
        "base_blind_pct",
        "base_perfect_pct",
        "schs",
        "n_bags",
        "median_size",
        "median_spec",
        "predicted_rescue_pp",
        "measured_rescue_pp",
        "realistic_lift_pp",
        "optimistic_lift_pp",
        "verdict",
    ]

    def cell(r, h):
        if h == "corpus":
            return r["label"]
        v = r.get(h)
        return "" if v is None else str(v)

    with open(out_path, "w") as f:
        f.write("\t".join(headers) + "\n")
        for r in rows:
            f.write("\t".join(cell(r, h) for h in headers) + "\n")

    # Pretty summary table.
    print()
    print("=" * 110)
    print("BULK READINESS SWEEP — all locally-available corpora")
    print("=" * 110)
    print(
        f"{'corpus':<22} {'n_q':>6} {'R@10':>5} {'BB%':>5} {'BP%':>5} {'SCHS':>5} "
        f"{'n_bags':>6} {'pred_rs':>7} {'meas_rs':>7} {'real_Δ':>7} {'verdict':<12}"
    )
    print("-" * 110)
    rows.sort(key=lambda r: -r["realistic_lift_pp"])
    for r in rows:
        pred_s = f"{r['predicted_rescue_pp']:.1f}" if r["predicted_rescue_pp"] is not None else "—"
        meas_s = f"{r['measured_rescue_pp']:.1f}" if r["measured_rescue_pp"] is not None else "—"
        schs_s = f"{r['schs']:.2f}" if r["schs"] == r["schs"] else "—"
        print(
            f"{r['label']:<22} {r['n_queries']:>6,} {r['base_R10']:>5.3f} "
            f"{r['base_blind_pct']:>5.1f} {r['base_perfect_pct']:>5.1f} {schs_s:>5} "
            f"{r['n_bags']:>6,} {pred_s:>7} {meas_s:>7} "
            f"{r['realistic_lift_pp']:>+7.1f} {r['verdict']:<12}"
        )
    print(f"\nFull TSV at {out_path}")


if __name__ == "__main__":
    main()
