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
    assert mod.TAX_BANDS["pessimistic"] > mod.TAX_BANDS["realistic"]
    assert mod.TAX_BANDS["realistic"] > mod.TAX_BANDS["optimistic"]


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
