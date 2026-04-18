"""Tests for shared utilities."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from utils import fmt_duration, generate_keyword_combos, l2_to_cosine, tokenize_query


def test_fmt_duration_seconds():
    assert fmt_duration(45) == "45s"


def test_fmt_duration_minutes():
    assert fmt_duration(125) == "2m 05s"


def test_fmt_duration_hours():
    assert fmt_duration(3725) == "1h 02m"


def test_tokenize_query_basic():
    assert tokenize_query("wireless keyboard") == ["wireless", "keyboard"]


def test_tokenize_query_mixed_case():
    assert tokenize_query("iPhone 12 Case") == ["iphone", "12", "case"]


def test_tokenize_query_single_char_filtered():
    assert tokenize_query("usb c cable") == ["usb", "cable"]


def test_tokenize_query_special_chars():
    assert tokenize_query("hp laptop 16gb") == ["hp", "laptop", "16gb"]


def test_tokenize_query_empty():
    assert tokenize_query("") == []


def test_generate_combos_full_match():
    combos = generate_keyword_combos(["wireless", "keyboard"])
    assert combos[0] == (2, [["wireless", "keyboard"]])


def test_generate_combos_relaxation():
    combos = generate_keyword_combos(["hp", "laptop", "16gb"])
    # First: all 3 words
    assert combos[0] == (3, [["hp", "laptop", "16gb"]])
    # Second: 2-word combos, sorted by total length, max 3
    assert combos[1][0] == 2
    assert len(combos[1][1]) <= 3
    # Third: single words
    assert combos[2][0] == 1


def test_generate_combos_single_word():
    combos = generate_keyword_combos(["keyboard"])
    assert combos == [(1, [["keyboard"]])]


def test_generate_combos_empty():
    assert generate_keyword_combos([]) == []


def test_generate_combos_prefers_longer_words():
    combos = generate_keyword_combos(["hp", "laptop", "ram"])
    two_word = combos[1][1]
    # First combo should contain "laptop" (longest word)
    assert "laptop" in two_word[0]


def test_l2_to_cosine_identical():
    # L2 distance 0 = cosine 1
    assert l2_to_cosine(0.0) == 1.0


def test_l2_to_cosine_orthogonal():
    # L2 distance 2 on unit vectors = cosine 0
    assert l2_to_cosine(2.0) == 0.0


def test_l2_to_cosine_opposite():
    # L2 distance 4 on unit vectors = cosine -1
    assert l2_to_cosine(4.0) == -1.0


def test_l2_to_cosine_typical():
    # L2 distance 0.5 = cosine 0.75
    assert l2_to_cosine(0.5) == 0.75
