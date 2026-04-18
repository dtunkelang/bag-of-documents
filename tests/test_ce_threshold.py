"""Tests for cross-encoder threshold logic."""

import numpy as np


def test_threshold_filters_low_scores():
    """Only candidates with CE score >= threshold should be kept."""
    candidates = ["good product A", "bad product B", "good product C", "bad product D"]
    ce_scores = np.array([0.8, 0.1, 0.6, 0.2])
    threshold = 0.3

    ce_ranked = sorted(zip(ce_scores, candidates), key=lambda x: -x[0])
    passing = [t for s, t in ce_ranked if s >= threshold]

    assert passing == ["good product A", "good product C"]


def test_threshold_respects_k_limit():
    """Even if many pass threshold, only top K should be kept."""
    candidates = [f"product {i}" for i in range(100)]
    ce_scores = np.random.RandomState(42).uniform(0.3, 1.0, size=100)
    threshold = 0.3
    K = 50

    ce_ranked = sorted(zip(ce_scores, candidates), key=lambda x: -x[0])
    passing = [t for s, t in ce_ranked if s >= threshold][:K]

    assert len(passing) == K


def test_threshold_returns_empty_when_none_pass():
    ce_scores = np.array([0.1, 0.05, 0.2])
    candidates = ["a", "b", "c"]
    threshold = 0.3

    ce_ranked = sorted(zip(ce_scores, candidates), key=lambda x: -x[0])
    passing = [t for s, t in ce_ranked if s >= threshold]

    assert passing == []


def test_threshold_sorts_by_score_descending():
    candidates = ["C", "A", "B"]
    ce_scores = np.array([0.5, 0.9, 0.7])
    threshold = 0.3

    ce_ranked = sorted(zip(ce_scores, candidates), key=lambda x: -x[0])
    passing = [t for s, t in ce_ranked if s >= threshold]

    assert passing == ["A", "B", "C"]
