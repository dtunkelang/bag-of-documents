#!/usr/bin/env python3
"""Classify the strongly-inverted queries (pn_max > pp_max under rerank_A) into:

  Query-INDEPENDENT (better product rep would fix):
    * Negation: "without X", "no X", "non-X", "X-free", "minus X"
    * Brand/model-number disambiguation: query contains alphanumeric model token
      and the distinguishing axis is product-internal (e.g., "kindle paperwhite"
      vs "kindle fire")

  Query-DEPENDENT (CE attention earns its keep):
    * Compound spec: 4+ tokens, multiple modifiers (constraints AND'd)
    * Abstract / open-ended: "essentials", "gifts", "ideas", "things for X",
      "stuff", "decor", "for [demographic]"
    * Other (ambiguous / generic single-word like "floral" — leave for sample)

Heuristics are precision-leaning where possible (regex for negation/brand);
a residual "ambiguous" bucket holds the rest. Manual sample of 15 ambiguous
queries printed for inspection.
"""

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

NEGATION = re.compile(
    r"\b(without|no|non[\s-]|minus|free of|free from|except|excluding)\b|\b\w+-free\b",
    re.IGNORECASE,
)

ABSTRACT = re.compile(
    r"\b(essentials?|gifts?|ideas?|things?|stuff|presents?|gear|accessories|"
    r"basics?|kits?|sets?|bundles?|starters?|decor|decoration|"
    r"for (women|men|kids|teens|girls|boys|babies|toddlers|her|him|mom|dad|grandma|grandpa|"
    r"the kitchen|the bedroom|the bathroom|christmas|halloween|easter|"
    r"birthdays?|anniversaries|weddings?|holidays?))\b",
    re.IGNORECASE,
)

# alphanumeric model number: a token of length >=3 with digits and letters mixed,
# OR a digit-heavy token of length >=3
MODEL_TOKEN = re.compile(r"\b(?=\w*\d)(?=\w*[a-z])[a-z0-9]{3,}\b|\b\d{3,}\b", re.IGNORECASE)


def classify(q):
    if NEGATION.search(q):
        return "negation"
    if ABSTRACT.search(q):
        return "abstract"
    tokens = q.split()
    has_model = bool(MODEL_TOKEN.search(q))
    n_tokens = len(tokens)
    if has_model:
        return "model_number"
    if n_tokens >= 4:
        return "compound_spec"
    return "ambiguous"


def report(label, queries):
    print(f"\n=== {label} (n={len(queries)}) ===")
    counts = {}
    examples = {}
    for r in queries:
        c = classify(r["query"])
        counts[c] = counts.get(c, 0) + 1
        examples.setdefault(c, []).append(r)
    for c in ["negation", "model_number", "abstract", "compound_spec", "ambiguous"]:
        if c not in counts:
            continue
        n = counts[c]
        print(f"  {c:<14} {n:>5}  ({n / len(queries):.1%})")

    # Print 5 examples per category
    for c in ["negation", "model_number", "abstract", "compound_spec", "ambiguous"]:
        if c not in examples:
            continue
        print(f"\n  -- {c} examples --")
        for r in sorted(examples[c], key=lambda x: -x["gap"])[:5]:
            print(f"    qid={r['qid']:<10} gap={r['gap']:+.3f}  query={r['query']!r}")


def categorize_query_dependent_vs_independent(queries):
    qd = 0  # query-dependent (CE)
    qi = 0  # query-independent (product rep)
    amb = 0
    by_cause = {}
    for r in queries:
        c = classify(r["query"])
        by_cause[c] = by_cause.get(c, 0) + 1
        if c in ("abstract", "compound_spec"):
            qd += 1
        elif c in ("negation", "model_number"):
            qi += 1
        else:
            amb += 1
    return qd, qi, amb, by_cause


def main():
    with open("/tmp/embedding_separation.json") as f:
        data = json.load(f)

    # Filter to strong inversions: pn_max > pp_max. The saved file has pp_min and pp_max
    # in the per-record fields.
    strong = [r for r in data if r["pn_max"] > r["pp_max"]]
    all_inv = [r for r in data if r["pn_max"] > r["pp_min"]]

    print(f"loaded {len(data)} eligible queries")
    print(f"  pn_max > pp_min (any inversion):     {len(all_inv)} ({len(all_inv) / len(data):.1%})")
    print(f"  pn_max > pp_max (strong inversion):  {len(strong)} ({len(strong) / len(data):.1%})")

    print("\n" + "=" * 72)
    print("DISTRIBUTION OF FAILURE CAUSES")
    print("=" * 72)
    report("Strong inversions (pn_max > pp_max, encoder cannot fix)", strong)
    report("Any inversions (pn_max > pp_min)", all_inv)

    print("\n" + "=" * 72)
    print("ROLLED UP: query-dependent (CE) vs query-independent (product rep)")
    print("=" * 72)
    qd_s, qi_s, amb_s, _ = categorize_query_dependent_vs_independent(strong)
    qd_a, qi_a, amb_a, _ = categorize_query_dependent_vs_independent(all_inv)
    n_s = len(strong)
    n_a = len(all_inv)
    print(f"\nstrong inversions (n={n_s}):")
    print(f"  query-dependent (compound + abstract):     {qd_s:>5}  ({qd_s / n_s:.1%})")
    print(f"  query-independent (negation + model_num):  {qi_s:>5}  ({qi_s / n_s:.1%})")
    print(f"  ambiguous (short / generic):                {amb_s:>5}  ({amb_s / n_s:.1%})")
    print(f"\nany inversions (n={n_a}):")
    print(f"  query-dependent (compound + abstract):     {qd_a:>5}  ({qd_a / n_a:.1%})")
    print(f"  query-independent (negation + model_num):  {qi_a:>5}  ({qi_a / n_a:.1%})")
    print(f"  ambiguous (short / generic):                {amb_a:>5}  ({amb_a / n_a:.1%})")

    # Sample of ambiguous to inspect manually
    print("\n=== AMBIGUOUS sample for manual review (15 strong inversions) ===")
    amb_strong = [r for r in strong if classify(r["query"]) == "ambiguous"]
    amb_strong.sort(key=lambda x: -x["gap"])
    for r in amb_strong[:15]:
        print(f"  qid={r['qid']:<10} gap={r['gap']:+.3f}  query={r['query']!r}")


if __name__ == "__main__":
    main()
