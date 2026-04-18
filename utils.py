"""Shared utilities."""

import re
from itertools import combinations


def fmt_duration(seconds):
    """Format seconds as human-readable duration."""
    h, rem = divmod(int(seconds), 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h {m:02d}m"
    if m:
        return f"{m}m {s:02d}s"
    return f"{s}s"


def tokenize_query(query):
    """Extract lowercase alphanumeric tokens (length > 1) from a query."""
    return [w for w in re.findall(r"[a-z0-9]+", query.lower()) if len(w) > 1]


def generate_keyword_combos(words, max_relaxation_combos=3):
    """Generate keyword combinations for tantivy AND-matching with relaxation.

    Tries full AND first, then drops words one at a time, returning the
    longest combos first. Returns a list of (n_required, combos) pairs.
    """
    if not words:
        return []
    result = []
    for n_required in range(len(words), 0, -1):
        if n_required == len(words):
            combos = [words]
        else:
            combos = [list(c) for c in combinations(words, n_required)]
            combos.sort(key=lambda c: -sum(len(w) for w in c))
            combos = combos[:max_relaxation_combos]
        result.append((n_required, combos))
    return result


def l2_to_cosine(l2_dist):
    """Convert L2 distance on normalized vectors to cosine similarity."""
    return 1 - l2_dist / 2
