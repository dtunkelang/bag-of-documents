"""Tests for centroid computation and specificity."""

import faiss
import numpy as np


def test_centroid_is_normalized():
    rng = np.random.RandomState(42)
    vecs = rng.randn(20, 384).astype(np.float32)
    faiss.normalize_L2(vecs)

    centroid = vecs.mean(axis=0)
    centroid = centroid / np.linalg.norm(centroid)

    assert abs(np.linalg.norm(centroid) - 1.0) < 1e-5


def test_specificity_is_mean_cosine():
    rng = np.random.RandomState(42)
    vecs = rng.randn(20, 384).astype(np.float32)
    faiss.normalize_L2(vecs)

    centroid = vecs.mean(axis=0)
    centroid = centroid / np.linalg.norm(centroid)

    specificity = float(np.mean([centroid @ v for v in vecs]))
    assert 0.0 < specificity < 1.0


def test_tight_cluster_has_high_specificity():
    """Vectors that are nearly identical should produce high specificity."""
    base = np.random.RandomState(42).randn(384).astype(np.float32)
    base = base / np.linalg.norm(base)
    # Add tiny noise
    vecs = np.array(
        [base + np.random.RandomState(i).randn(384) * 0.01 for i in range(20)], dtype=np.float32
    )
    faiss.normalize_L2(vecs)

    centroid = vecs.mean(axis=0)
    centroid = centroid / np.linalg.norm(centroid)
    specificity = float(np.mean([centroid @ v for v in vecs]))

    assert specificity > 0.98


def test_diverse_cluster_has_lower_specificity():
    """Random vectors should produce lower specificity."""
    rng = np.random.RandomState(42)
    vecs = rng.randn(20, 384).astype(np.float32)
    faiss.normalize_L2(vecs)

    centroid = vecs.mean(axis=0)
    centroid = centroid / np.linalg.norm(centroid)
    specificity = float(np.mean([centroid @ v for v in vecs]))

    assert specificity < 0.5


def test_centroid_retrieval_finds_members(sample_vectors, sample_faiss_index):
    """Centroid of a subset should retrieve those members."""
    # Take first 10 vectors as "bag members"
    members = sample_vectors[:10]
    centroid = members.mean(axis=0)
    centroid = centroid / np.linalg.norm(centroid)
    centroid = centroid.reshape(1, -1).astype(np.float32)
    faiss.normalize_L2(centroid)

    D, I = sample_faiss_index.search(centroid, 20)
    retrieved = set(I[0].tolist())
    member_indices = set(range(10))

    overlap = len(retrieved & member_indices)
    assert overlap >= 5, f"Expected >=5 members in top-20, got {overlap}"
