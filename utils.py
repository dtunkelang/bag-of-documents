"""Shared utilities."""

import math
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


def generate_keyword_combos(words, max_relaxation_combos=3, idf=None, n_docs=None):
    """Generate keyword combinations for tantivy AND-matching with relaxation.

    Tries full AND first, then progressively drops tokens. Returns a list of
    (n_required, combos) pairs.

    Combos at each relaxation level are ranked and capped at
    max_relaxation_combos. If `idf` (token -> doc_frequency dict) and
    `n_docs` are provided, ranks by IDF-sum (keeps rare/distinctive tokens,
    drops common ones first). Otherwise falls back to token-length sum.
    """
    if not words:
        return []

    if idf is not None and n_docs is not None:

        def score(combo):
            return sum(math.log((n_docs + 1) / (idf.get(w, 0) + 1)) for w in combo)
    else:

        def score(combo):
            return sum(len(w) for w in combo)

    result = []
    for n_required in range(len(words), 0, -1):
        if n_required == len(words):
            combos = [words]
        else:
            combos = [list(c) for c in combinations(words, n_required)]
            combos.sort(key=lambda c: -score(c))
            combos = combos[:max_relaxation_combos]
        result.append((n_required, combos))
    return result


def l2_to_cosine(l2_dist):
    """Convert L2 distance on normalized vectors to cosine similarity."""
    return 1 - l2_dist / 2
