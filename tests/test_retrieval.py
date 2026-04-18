"""Tests for FAISS retrieval logic."""


def test_faiss_retrieval_returns_k_results(sample_vectors, sample_faiss_index):
    query = sample_vectors[0:1]
    k = 10
    D, I = sample_faiss_index.search(query, k)
    assert I.shape == (1, k)
    assert all(idx >= 0 for idx in I[0])


def test_faiss_self_retrieval(sample_vectors, sample_faiss_index):
    """Each vector's nearest neighbor should be itself."""
    for i in [0, 25, 50, 75]:
        query = sample_vectors[i : i + 1]
        D, I = sample_faiss_index.search(query, 1)
        assert I[0][0] == i


def test_faiss_dedup_by_title(sample_titles, sample_vectors, sample_faiss_index):
    """Retrieval + dedup should produce unique titles."""
    query = sample_vectors[0:1]
    D, I = sample_faiss_index.search(query, 20)

    seen = {}
    results = []
    for dist, idx in zip(D[0], I[0]):
        if idx < 0:
            continue
        title = sample_titles[idx]
        score = float(1 - dist / 2)
        if title not in seen or score > seen[title]:
            seen[title] = score
    results = sorted(seen.items(), key=lambda x: -x[1])

    titles = [t for t, _ in results]
    assert len(titles) == len(set(titles))


def test_normalized_vectors_cosine_range(sample_vectors, sample_faiss_index):
    """L2 distances on normalized vectors should give cosine in [0, 1]."""
    query = sample_vectors[0:1]
    D, I = sample_faiss_index.search(query, 10)
    for dist in D[0]:
        cosine = 1 - dist / 2
        assert -0.01 <= cosine <= 1.01, f"Cosine {cosine} out of range"
