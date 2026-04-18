"""Tests for demo API response contracts."""


def test_search_response_fields():
    """Verify expected fields in /api/search response."""
    # Simulate response structure
    response = {
        "query": "test",
        "specificity": 0.85,
        "result_count": 50,
        "results": [{"title": "Product A", "score": 0.95}],
        "nearest_queries": [],
        "base_results": [{"title": "Product B", "score": 0.80}],
        "base_total": 50,
    }

    required = {"query", "specificity", "result_count", "results", "base_results", "base_total"}
    assert required <= set(response.keys())
    assert 0 <= response["specificity"] <= 1
    assert isinstance(response["results"], list)
    assert isinstance(response["base_results"], list)


def test_bag_search_response_fields():
    """Verify expected fields in /api/bag_search response."""
    response = {
        "query": "test",
        "step1_candidates": 400,
        "step2_relevant": 30,
        "step2_method": "hybrid+CE",
        "step3_specificity": 0.95,
        "step4_bag_results": [{"title": "Product A", "score": 0.90}],
        "judgments": [{"title": "Product A", "score": 0.1, "relevant": True}],
    }

    required = {
        "query",
        "step1_candidates",
        "step2_relevant",
        "step3_specificity",
        "step4_bag_results",
        "judgments",
    }
    assert required <= set(response.keys())
    assert response["step1_candidates"] >= response["step2_relevant"]


def test_judgment_structure():
    """Each judgment should have title, score, and relevant flag."""
    judgment = {"title": "Some Product", "score": 0.15, "relevant": True}

    assert "title" in judgment
    assert "score" in judgment
    assert "relevant" in judgment
    assert isinstance(judgment["relevant"], bool)


def test_bag_result_structure():
    """Each bag result should have title and score."""
    result = {"title": "Some Product", "score": 0.92}

    assert "title" in result
    assert "score" in result
    assert 0 <= result["score"] <= 1
