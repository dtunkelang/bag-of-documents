"""Tests for bag JSON structure and invariants."""

import json

import numpy as np


def test_bag_has_required_fields(sample_bag):
    required = {"query", "num_results", "query_vector", "specificity", "results"}
    assert required <= set(sample_bag.keys())


def test_bag_vector_dimension(sample_bag):
    vec = np.array(sample_bag["query_vector"])
    assert vec.shape == (384,)


def test_bag_vector_is_normalized(sample_bag):
    vec = np.array(sample_bag["query_vector"])
    norm = np.linalg.norm(vec)
    assert abs(norm - 1.0) < 1e-5


def test_bag_specificity_range(sample_bag):
    assert 0.0 <= sample_bag["specificity"] <= 1.0


def test_bag_results_count_matches(sample_bag):
    assert len(sample_bag["results"]) == sample_bag["num_results"]


def test_bag_results_have_titles(sample_bag):
    for r in sample_bag["results"]:
        assert "title" in r
        assert isinstance(r["title"], str)
        assert len(r["title"]) > 0


def test_empty_bag_has_zero_specificity(sample_bags_jsonl):
    with open(sample_bags_jsonl) as f:
        for line in f:
            bag = json.loads(line)
            if bag["query"] == "empty query":
                assert bag["num_results"] == 0
                assert bag["specificity"] == 0.0
                assert bag["results"] == []
                return
    raise AssertionError("empty query bag not found")


def test_bags_jsonl_no_duplicates(sample_bags_jsonl):
    queries = []
    with open(sample_bags_jsonl) as f:
        for line in f:
            bag = json.loads(line)
            queries.append(bag["query"])
    assert len(queries) == len(set(queries))


def test_bags_jsonl_all_valid(sample_bags_jsonl):
    required = {"query", "num_results", "query_vector", "specificity", "results"}
    with open(sample_bags_jsonl) as f:
        for line in f:
            bag = json.loads(line)
            assert required <= set(bag.keys())
            assert isinstance(bag["query"], str)
            assert isinstance(bag["num_results"], int)
            assert len(bag["results"]) == bag["num_results"]
