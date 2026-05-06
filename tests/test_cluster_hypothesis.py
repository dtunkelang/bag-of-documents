"""Tests for the Cluster Hypothesis Score (CHS) metric.

Validates the metric formulas on synthetic corpora with known properties:
    - Perfect clustering: in-bag pairs are identical, negatives are random.
        Expect SCHS close to 1, HCHS close to 1, strong_inv_rate = 0.
    - No structure: positives are random; cluster hypothesis fails.
        Expect SCHS close to 0.
    - Edge cases: too few queries, no explicit negs, etc.

These tests use precomputed embeddings (cache_vecs) to avoid loading any
sentence-transformer model — keeps the test fast and offline.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bagofdocs.cluster_hypothesis import compute_chs, schs_verdict  # noqa: E402


def _make_corpus(n_queries, pos_per_query, n_distractors, intra_noise, dim=32, seed=0):
    """Build a synthetic corpus with controlled cluster structure.

    For each query q:
      - centroid_q is a random unit vector.
      - pos products = centroid_q + small Gaussian noise (controlled by intra_noise).
      - explicit negative product = a random unit vector (uncorrelated with centroid).
      - n_distractors random products are added to the catalog.

    With small intra_noise, in-bag pairs are very tight; SCHS should be near 1.
    """
    rng = np.random.default_rng(seed)
    pids = []
    titles = []
    qrels = {}
    vecs = []

    def add_item(prefix, vec):
        pid = f"{prefix}{len(pids)}"
        pids.append(pid)
        titles.append(pid)
        vecs.append(vec)
        return pid

    # Distractors first (background catalog)
    for _ in range(n_distractors):
        v = rng.normal(size=dim).astype(np.float32)
        v /= np.linalg.norm(v) + 1e-8
        add_item("d", v)

    # Then per-query bags
    for q in range(n_queries):
        centroid = rng.normal(size=dim).astype(np.float32)
        centroid /= np.linalg.norm(centroid) + 1e-8
        qid = f"q{q}"
        qrels[qid] = {}

        for _ in range(pos_per_query):
            noise = rng.normal(size=dim).astype(np.float32) * intra_noise
            v = centroid + noise
            v /= np.linalg.norm(v) + 1e-8
            pid = add_item("p", v)
            qrels[qid][pid] = 3

        # one explicit negative per query — random, uncorrelated with centroid
        v_neg = rng.normal(size=dim).astype(np.float32)
        v_neg /= np.linalg.norm(v_neg) + 1e-8
        pid = add_item("n", v_neg)
        qrels[qid][pid] = 0

    cache_vecs = np.stack(vecs).astype(np.float32)
    return qrels, pids, titles, cache_vecs


def test_perfect_clustering_is_high_chs():
    """Positives nearly identical, negatives random -> SCHS ~ 1, HCHS ~ 1."""
    qrels, pids, titles, vecs = _make_corpus(
        n_queries=100, pos_per_query=4, n_distractors=400, intra_noise=0.01, seed=42
    )
    res = compute_chs(
        qrels,
        pids,
        titles,
        encoder_name="UNUSED",
        partition="strict",
        cache_vecs=vecs,
        verbose=False,
    )
    assert res.n_pos_bearing == 100
    assert res.n_explicit_neg == 100
    assert res.has_explicit_negs
    assert res.schs > 0.9, f"expected near-1 SCHS, got {res.schs}"
    assert res.hchs > 0.8, f"expected near-1 HCHS, got {res.hchs}"
    assert res.strong_inv_rate == 0, "perfect clusters should never invert"


def test_no_clustering_is_low_chs():
    """Positives random (high noise) and negatives random -> SCHS near 0."""
    qrels, pids, titles, vecs = _make_corpus(
        n_queries=100, pos_per_query=4, n_distractors=400, intra_noise=10.0, seed=7
    )
    res = compute_chs(
        qrels,
        pids,
        titles,
        encoder_name="UNUSED",
        partition="strict",
        cache_vecs=vecs,
        verbose=False,
    )
    # When intra_noise dominates the centroid, the cluster signal is destroyed:
    # in-bag pairs aren't meaningfully closer than random pairs.
    assert -0.1 < res.schs < 0.2, f"expected near-0 SCHS, got {res.schs}"


def test_chs_is_monotone_in_intra_noise():
    """Lower intra noise -> higher SCHS. Smoke check the directional property."""
    schs_values = []
    for noise in [0.05, 0.5, 5.0]:
        qrels, pids, titles, vecs = _make_corpus(
            n_queries=80, pos_per_query=4, n_distractors=300, intra_noise=noise, seed=1
        )
        res = compute_chs(
            qrels,
            pids,
            titles,
            encoder_name="UNUSED",
            partition="strict",
            cache_vecs=vecs,
            verbose=False,
        )
        schs_values.append(res.schs)
    # SCHS should monotonically decrease as intra noise grows.
    assert schs_values[0] > schs_values[1] > schs_values[2], (
        f"non-monotone SCHS in intra noise: {schs_values}"
    )


def test_too_few_queries_returns_nan():
    """Fewer than min_pos_bearing queries -> NaN scores, no encoding attempted."""
    qrels, pids, titles, vecs = _make_corpus(
        n_queries=10, pos_per_query=4, n_distractors=50, intra_noise=0.1, seed=3
    )
    res = compute_chs(
        qrels,
        pids,
        titles,
        encoder_name="UNUSED",
        partition="strict",
        cache_vecs=vecs,
        verbose=False,
        min_pos_bearing=50,
    )
    assert res.n_pos_bearing == 10
    assert np.isnan(res.schs)
    assert np.isnan(res.hchs)


def test_no_explicit_negs_returns_nan_hchs():
    """Corpus with positives only -> SCHS computed, HCHS NaN."""
    rng = np.random.default_rng(11)
    n_queries = 60
    pos_per = 3
    dim = 16
    pids, titles, vecs = [], [], []
    qrels = {}
    for q in range(n_queries):
        centroid = rng.normal(size=dim).astype(np.float32)
        centroid /= np.linalg.norm(centroid) + 1e-8
        qid = f"q{q}"
        qrels[qid] = {}
        for _ in range(pos_per):
            v = centroid + rng.normal(size=dim).astype(np.float32) * 0.1
            v /= np.linalg.norm(v) + 1e-8
            pid = f"p{len(pids)}"
            pids.append(pid)
            titles.append(pid)
            vecs.append(v)
            qrels[qid][pid] = 1
    # add some random distractors
    for _ in range(200):
        v = rng.normal(size=dim).astype(np.float32)
        v /= np.linalg.norm(v) + 1e-8
        pid = f"d{len(pids)}"
        pids.append(pid)
        titles.append(pid)
        vecs.append(v)
    cache_vecs = np.stack(vecs).astype(np.float32)

    res = compute_chs(
        qrels,
        pids,
        titles,
        encoder_name="UNUSED",
        partition="strict",
        pos_grade=1,
        cache_vecs=cache_vecs,
        verbose=False,
    )
    assert not res.has_explicit_negs
    assert np.isfinite(res.schs)
    assert np.isnan(res.hchs)
    assert np.isnan(res.strong_inv_rate)


def test_schs_verdict_thresholds():
    """The verdict mapping uses the calibrated SCHS thresholds."""
    assert "GREEN" in schs_verdict(0.55, 5000)
    assert "YELLOW" in schs_verdict(0.45, 5000)
    assert "RED" in schs_verdict(0.30, 5000)
    assert "UNDETERMINED" in schs_verdict(float("nan"), 5000)
    # Low n_pos_bearing should be flagged even at high SCHS.
    assert "low n_pos_bearing" in schs_verdict(0.70, 200)


def test_hchs_apples_to_apples():
    """HCHS should use mean_intra over the explicit-neg subset, not over all
    pos-bearing queries. Test by mixing queries with and without negatives:
    HCHS should reflect the explicit-neg subset's intra-bag tightness only.
    """
    rng = np.random.default_rng(99)
    dim = 16
    pids, titles, vecs = [], [], []
    qrels = {}

    # Subset A: 60 queries with explicit negatives, TIGHT clusters (low noise)
    for q in range(60):
        centroid = rng.normal(size=dim).astype(np.float32)
        centroid /= np.linalg.norm(centroid) + 1e-8
        qid = f"qa{q}"
        qrels[qid] = {}
        for _ in range(3):
            v = centroid + rng.normal(size=dim).astype(np.float32) * 0.05
            v /= np.linalg.norm(v) + 1e-8
            pid = f"pa{len(pids)}"
            pids.append(pid)
            titles.append(pid)
            vecs.append(v)
            qrels[qid][pid] = 3
        # explicit negative
        vn = rng.normal(size=dim).astype(np.float32)
        vn /= np.linalg.norm(vn) + 1e-8
        pid = f"na{len(pids)}"
        pids.append(pid)
        titles.append(pid)
        vecs.append(vn)
        qrels[qid][pid] = 0

    # Subset B: 60 queries WITHOUT explicit negatives, LOOSE clusters (high noise)
    for q in range(60):
        centroid = rng.normal(size=dim).astype(np.float32)
        centroid /= np.linalg.norm(centroid) + 1e-8
        qid = f"qb{q}"
        qrels[qid] = {}
        for _ in range(3):
            v = centroid + rng.normal(size=dim).astype(np.float32) * 1.0
            v /= np.linalg.norm(v) + 1e-8
            pid = f"pb{len(pids)}"
            pids.append(pid)
            titles.append(pid)
            vecs.append(v)
            qrels[qid][pid] = 3

    # Random distractors for the random-pair baseline
    for _ in range(300):
        v = rng.normal(size=dim).astype(np.float32)
        v /= np.linalg.norm(v) + 1e-8
        pid = f"d{len(pids)}"
        pids.append(pid)
        titles.append(pid)
        vecs.append(v)

    cache_vecs = np.stack(vecs).astype(np.float32)
    res = compute_chs(
        qrels,
        pids,
        titles,
        encoder_name="UNUSED",
        partition="strict",
        cache_vecs=cache_vecs,
        verbose=False,
    )
    # Subset A is tight, B is loose. SCHS averages over both -> should be
    # noticeably lower than HCHS (which uses only the tight subset A).
    assert res.has_explicit_negs
    assert res.hchs > res.schs, (
        f"HCHS ({res.hchs}) should exceed SCHS ({res.schs}) when explicit-neg "
        f"subset is tighter than the overall pos-bearing population"
    )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
