"""Tests for fine-tuning data loading."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from finetune_query_model import load_bags, split_train_val


def test_load_bags_skips_empty(sample_bags_jsonl):
    pairs = load_bags(str(sample_bags_jsonl))
    queries = [p["query"] for p in pairs]
    assert "empty query" not in queries


def test_load_bags_returns_vectors(sample_bags_jsonl):
    pairs = load_bags(str(sample_bags_jsonl))
    for p in pairs:
        assert p["vector"].shape == (384,)
        assert isinstance(p["specificity"], float)
        assert isinstance(p["query"], str)


def test_split_preserves_all_data(sample_bags_jsonl):
    pairs = load_bags(str(sample_bags_jsonl))
    train, val = split_train_val(pairs, val_fraction=0.5, seed=42)
    assert len(train) + len(val) == len(pairs)


def test_split_is_deterministic(sample_bags_jsonl):
    pairs = load_bags(str(sample_bags_jsonl))
    train1, val1 = split_train_val(pairs, val_fraction=0.2, seed=42)
    train2, val2 = split_train_val(pairs, val_fraction=0.2, seed=42)
    assert [p["query"] for p in train1] == [p["query"] for p in train2]
    assert [p["query"] for p in val1] == [p["query"] for p in val2]


def test_split_different_seeds_differ(sample_bags_jsonl):
    pairs = load_bags(str(sample_bags_jsonl))
    train1, _ = split_train_val(pairs, val_fraction=0.5, seed=1)
    train2, _ = split_train_val(pairs, val_fraction=0.5, seed=2)
    q1 = [p["query"] for p in train1]
    q2 = [p["query"] for p in train2]
    assert len(q1) > 0
    assert len(q2) > 0
