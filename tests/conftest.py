"""Shared test fixtures for bag-of-documents tests."""

import json

import faiss
import numpy as np
import pytest


@pytest.fixture
def sample_titles():
    """100 synthetic product titles across a few categories."""
    titles = []
    for i in range(25):
        titles.append(f"Wireless Bluetooth Keyboard Model {i}")
    for i in range(25):
        titles.append(f"iPhone {12 + i % 4} Case Protective Cover Style {i}")
    for i in range(25):
        titles.append(f"Acoustic Guitar Strings Set {i} Bronze Wound")
    for i in range(25):
        titles.append(f"Watercolor Paint Set {i} Colors Professional Art")
    return titles


@pytest.fixture
def sample_vectors(sample_titles):
    """Normalized 384-dim vectors for sample titles."""
    rng = np.random.RandomState(42)
    vecs = rng.randn(len(sample_titles), 384).astype(np.float32)
    faiss.normalize_L2(vecs)
    return vecs


@pytest.fixture
def sample_faiss_index(sample_vectors):
    """FAISS flat index for sample vectors."""
    index = faiss.IndexFlatL2(384)
    index.add(sample_vectors)
    return index


@pytest.fixture
def sample_bag():
    """A single well-formed bag."""
    rng = np.random.RandomState(42)
    centroid = rng.randn(384).astype(np.float32)
    centroid = centroid / np.linalg.norm(centroid)
    return {
        "query": "wireless keyboard",
        "num_results": 10,
        "query_vector": centroid.tolist(),
        "specificity": 0.95,
        "results": [{"title": f"Wireless Keyboard Model {i}"} for i in range(10)],
    }


@pytest.fixture
def sample_bags_jsonl(tmp_path):
    """JSONL file with a few sample bags (for finetune tests)."""
    rng = np.random.RandomState(42)
    queries = [
        ("wireless keyboard", 10, 0.95),
        ("iphone case", 20, 0.92),
        ("guitar strings", 5, 0.88),
        ("empty query", 0, 0.0),
    ]
    path = tmp_path / "bags.jsonl"
    with open(path, "w") as f:
        for query, n_results, spec in queries:
            centroid = rng.randn(384).astype(np.float32)
            centroid = centroid / np.linalg.norm(centroid)
            bag = {
                "query": query,
                "num_results": n_results,
                "query_vector": centroid.tolist(),
                "specificity": spec,
                "results": [{"title": f"{query} product {i}"} for i in range(n_results)],
            }
            f.write(json.dumps(bag) + "\n")
    return path
