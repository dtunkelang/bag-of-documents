"""Tests for evaluation/bod_readiness_report.py.

Covers the pure prediction logic (predict_lift, verdict). The full CLI
needs a corpus + encoder, so we exercise it via the import smoke test in
test_imports.py and skip integration here.
"""

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "bod_readiness_report",
        ROOT / "evaluation" / "bod_readiness_report.py",
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["bod_readiness_report"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_predict_lift_increases_with_base_blind_size():
    """Bigger base-blind subset → bigger predicted lift in every scenario."""
    mod = _load_module()
    small = mod.predict_lift(base_blind=0.20, base_perfect=0.05)
    large = mod.predict_lift(base_blind=0.60, base_perfect=0.05)
    for k in small:
        assert large[k] > small[k]


def test_predict_lift_decreases_with_base_perfect_size():
    """Bigger base-perfect subset → smaller predicted lift (more tax)."""
    mod = _load_module()
    low_tax = mod.predict_lift(base_blind=0.40, base_perfect=0.05)
    high_tax = mod.predict_lift(base_blind=0.40, base_perfect=0.30)
    for k in low_tax:
        assert low_tax[k] > high_tax[k]


def test_verdict_skip_on_low_schs():
    """SCHS below the floor overrides everything to SKIP, even with great headroom."""
    mod = _load_module()
    predicted = mod.predict_lift(base_blind=0.50, base_perfect=0.05)
    label, reason = mod.verdict(schs=0.30, base_blind=0.50, base_perfect=0.05, predicted=predicted)
    assert label == "SKIP"
    assert "0.30" in reason or "SCHS" in reason


def test_verdict_go_with_strong_predictions():
    """High SCHS + big base-blind subset → GO."""
    mod = _load_module()
    predicted = mod.predict_lift(base_blind=0.50, base_perfect=0.05)
    label, _reason = mod.verdict(schs=0.55, base_blind=0.50, base_perfect=0.05, predicted=predicted)
    assert label == "GO"


def test_verdict_skip_when_optimistic_too_small():
    """Tiny base-blind, high base-perfect → SKIP (lift can't justify the pipeline)."""
    mod = _load_module()
    predicted = mod.predict_lift(base_blind=0.05, base_perfect=0.40)
    label, _reason = mod.verdict(schs=0.55, base_blind=0.05, base_perfect=0.40, predicted=predicted)
    assert label == "SKIP"


def test_verdict_thresholds_match_calibration():
    """The SCHS_FLOOR matches the documented 0.40 cutoff in CHS_RESULTS.md."""
    mod = _load_module()
    assert mod.SCHS_FLOOR == 0.40
    assert set(mod.RESCUE_BANDS) == {"pessimistic", "realistic", "optimistic"}
    assert mod.RESCUE_BANDS["pessimistic"] < mod.RESCUE_BANDS["realistic"]
    assert mod.RESCUE_BANDS["realistic"] < mod.RESCUE_BANDS["optimistic"]
    assert mod.TAX_K["pessimistic"] > mod.TAX_K["realistic"]
    assert mod.TAX_K["realistic"] > mod.TAX_K["optimistic"]


def test_architecture_recommendation_rerank_when_bm25_stronger():
    """BM25 ≥ base + threshold → recommend rerank (Pattern 10)."""
    mod = _load_module()
    arch, msg = mod.architecture_recommendation(bm25_r=0.40, base_r=0.20)
    assert arch == "rerank"
    assert "rerank" in msg.lower()


def test_architecture_recommendation_retrieve_when_bm25_weaker():
    """BM25 ≤ base − threshold → recommend retrieve."""
    mod = _load_module()
    arch, msg = mod.architecture_recommendation(bm25_r=0.20, base_r=0.40)
    assert arch == "retrieve"
    assert "retriever" in msg.lower() or "retrieve" in msg.lower()


def test_architecture_recommendation_either_when_close():
    """BM25 ≈ base within ±2pp → either architecture works."""
    mod = _load_module()
    arch, msg = mod.architecture_recommendation(bm25_r=0.31, base_r=0.30)
    assert arch == "either"


def test_architecture_recommendation_handles_none():
    """When bm25s isn't available, return None gracefully."""
    mod = _load_module()
    arch, msg = mod.architecture_recommendation(bm25_r=None, base_r=0.30)
    assert arch is None
    assert "BM25" in msg or "bm25" in msg.lower()


def test_predict_lift_scales_tax_with_base_r10():
    """Tax shrinks as base R@10 grows toward 1.0 (Pattern 9 calibration)."""
    mod = _load_module()
    # Same buckets, low base R@10 (lots of headroom).
    low_base = mod.predict_lift(base_blind=0.30, base_perfect=0.30, base_overall_r10=0.20)
    # Same buckets, high base R@10 (almost no headroom).
    high_base = mod.predict_lift(base_blind=0.30, base_perfect=0.30, base_overall_r10=0.95)
    # Predicted lift should be HIGHER (less negative tax) at high base R@10.
    assert high_base["realistic"] > low_base["realistic"]
    assert high_base["optimistic"] > low_base["optimistic"]


def test_predict_lift_falls_back_when_base_r10_unset():
    """Backward-compat: when base_overall_r10 is None, behave like v1 fixed-tax."""
    mod = _load_module()
    new = mod.predict_lift(base_blind=0.30, base_perfect=0.30)
    # Equivalent to v1: rescue × BB - tax × BP (using TAX_K as the v1 constants).
    expected_real = 0.30 * mod.RESCUE_BANDS["realistic"] - 0.30 * mod.TAX_K["realistic"]
    assert abs(new["realistic"] - expected_real) < 1e-9


def test_predict_rescue_rate_none_when_too_few_bags():
    """Below n_bags=10, the regression isn't trustworthy → return None."""
    mod = _load_module()
    assert mod.predict_rescue_rate(None) is None
    assert mod.predict_rescue_rate({"n_bags": 5, "median_size": 10, "median_spec": 0.5}) is None


def test_predict_rescue_rate_gated_above_base_r10_threshold():
    """Above RESCUE_BASE_R10_MAX, the predictor is out of regime → None."""
    mod = _load_module()
    stats = {"n_bags": 1000, "median_size": 8, "median_spec": 0.55}
    # Below threshold: returns a value.
    assert mod.predict_rescue_rate(stats, base_r10=0.30) is not None
    # At/above threshold: returns None.
    assert mod.predict_rescue_rate(stats, base_r10=mod.RESCUE_BASE_R10_MAX) is None
    assert mod.predict_rescue_rate(stats, base_r10=0.95) is None
    # base_r10=None falls back to ungated behavior (backward-compatible).
    assert mod.predict_rescue_rate(stats, base_r10=None) is not None


def test_predict_rescue_rate_matches_formula():
    """Returned value should match the documented Pattern 8a formula."""
    import numpy as np

    mod = _load_module()
    stats = {"n_bags": 1000, "median_size": 8, "median_spec": 0.55}
    expected_pp = (
        mod.RESCUE_W_LOG_N * np.log10(stats["n_bags"])
        + mod.RESCUE_W_SIZE * stats["median_size"]
        + mod.RESCUE_W_SPEC * stats["median_spec"]
        + mod.RESCUE_INTERCEPT
    )
    expected_frac = max(0.0, min(0.40, expected_pp / 100.0))
    assert abs(mod.predict_rescue_rate(stats) - expected_frac) < 1e-9


def test_predict_rescue_rate_clamps_to_plausible_range():
    """Output is clamped to [0, 0.40] regardless of pathological inputs."""
    mod = _load_module()
    # Pathologically high inputs would otherwise blow past 40pp.
    huge = mod.predict_rescue_rate({"n_bags": 10**9, "median_size": 1, "median_spec": 0.99})
    assert 0.0 <= huge <= 0.40
    # Pathologically low (near-zero spec, single-doc bags).
    tiny = mod.predict_rescue_rate({"n_bags": 10, "median_size": 100, "median_spec": 0.01})
    assert tiny == 0.0


def test_predict_lift_collapses_band_with_predicted_rescue():
    """When `predicted_rescue` is set, the rescue band is point ± RMSE/100."""
    mod = _load_module()
    out = mod.predict_lift(
        base_blind=0.40,
        base_perfect=0.10,
        base_overall_r10=0.30,
        predicted_rescue=0.12,
    )
    rmse = mod.RESCUE_RMSE_PP / 100.0
    headroom = 1.0 - 0.30
    # realistic uses rescue=0.12 exactly.
    expected_real = 0.40 * 0.12 - 0.10 * (mod.TAX_K["realistic"] * headroom)
    assert abs(out["realistic"] - expected_real) < 1e-9
    # optimistic uses rescue=0.12 + rmse.
    expected_opt = 0.40 * (0.12 + rmse) - 0.10 * (mod.TAX_K["optimistic"] * headroom)
    assert abs(out["optimistic"] - expected_opt) < 1e-9
    # pessimistic uses max(0, 0.12 - rmse).
    expected_pess = 0.40 * max(0.0, 0.12 - rmse) - 0.10 * (mod.TAX_K["pessimistic"] * headroom)
    assert abs(out["pessimistic"] - expected_pess) < 1e-9


def test_compute_bag_stats_on_toy_corpus():
    """Tight bags (high cosine within bag) should produce high median_spec."""
    import numpy as np

    mod = _load_module()
    # 12 docs, 6 bags of 2 docs each. Each pair shares an axis → cosine ~ 1.
    rng = np.random.default_rng(0)
    pids = [f"d{i}" for i in range(12)]
    pid_to_idx = {p: i for i, p in enumerate(pids)}
    base_pv = np.zeros((12, 8), dtype=np.float32)
    for bag_idx in range(6):
        v = rng.normal(size=8).astype(np.float32)
        v /= np.linalg.norm(v)
        base_pv[2 * bag_idx] = v
        # Slightly perturb the partner so it isn't identical but still tight.
        v2 = v + 0.01 * rng.normal(size=8).astype(np.float32)
        v2 /= np.linalg.norm(v2)
        base_pv[2 * bag_idx + 1] = v2
    qrels_full = {f"q{b}": {pids[2 * b]: 1, pids[2 * b + 1]: 1} for b in range(6)}
    stats = mod.compute_bag_stats(qrels_full, pid_to_idx, base_pv, min_relevance=1)
    assert stats["n_bags"] == 6
    assert stats["median_size"] == 2
    # Tight bags → median spec should be very near 1.
    assert stats["median_spec"] > 0.99


def test_compute_bag_stats_skips_singleton_bags():
    """Bags with <2 positive docs should be excluded (can't compute spec)."""
    import numpy as np

    mod = _load_module()
    pids = ["a", "b", "c"]
    pid_to_idx = {p: i for i, p in enumerate(pids)}
    base_pv = np.eye(3, dtype=np.float32)
    qrels_full = {
        "q1": {"a": 1},  # singleton → skipped
        "q2": {"b": 1, "c": 1},  # kept
    }
    stats = mod.compute_bag_stats(qrels_full, pid_to_idx, base_pv, min_relevance=1)
    assert stats["n_bags"] == 1
