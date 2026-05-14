#!/usr/bin/env python3
"""Three-way union-oracle analysis: BoD + HyDE + Doc2Query.

Joins per-query R@10 hit data across all three methods (and base) on
each corpus. Computes:

  - per-method overall R@10 and bucket breakdown
  - UNION-oracle ceiling: best-per-query across the three methods
    (lower bound on what perfect routing could achieve)
  - method-disjointness: of queries where the union beats the best
    single method, how many?

This is the cheap form of fusion analysis — uses cached hit-counts only,
no recomputation. A LOWER bound on the true union-oracle because we can't
tell from hit-counts whether two methods retrieved the *same* gold doc
or different gold docs on a multi-gold query.
"""

import json
import sys
from pathlib import Path


def load_jsonl(p):
    with open(p) as f:
        return [json.loads(line) for line in f]


def main(corpus_label: str, bod_file: Path, hyde_file: Path, d2q_file: Path):
    bod = {r["query_id"]: r for r in load_jsonl(bod_file)}
    hyde = {r["query_id"]: r for r in load_jsonl(hyde_file)}
    d2q = {r["query_id"]: r for r in load_jsonl(d2q_file)}

    shared_qids = set(bod) & set(hyde) & set(d2q)
    print(f"\n=== {corpus_label} ===")
    print(
        f"queries: bod={len(bod):,}  hyde={len(hyde):,}  d2q={len(d2q):,}  "
        f"shared={len(shared_qids):,}"
    )

    rows = []
    for qid in shared_qids:
        b = bod[qid]
        h = hyde[qid]
        d = d2q[qid]
        if not (b["n_gold"] == h["n_gold"] == d["n_gold"]):
            continue  # skip if gold count disagrees (different qrels filter)
        n_gold = b["n_gold"]
        if n_gold == 0:
            continue
        rows.append(
            {
                "qid": qid,
                "n_gold": n_gold,
                "base": b["base_hit"],
                "bod": b["bod_hit"],
                "hyde": h["hyde_hit"],
                "d2q": d["d2q_hit"],
            }
        )

    n = len(rows)
    print(f"evaluated: n={n:,} (positive-bearing, gold-counts agree)")
    if n == 0:
        return

    def mean_r(key):
        return sum(r[key] / r["n_gold"] for r in rows) / n

    base_r = mean_r("base")
    bod_r = mean_r("bod")
    hyde_r = mean_r("hyde")
    d2q_r = mean_r("d2q")
    union_r = sum(max(r["bod"], r["hyde"], r["d2q"]) / r["n_gold"] for r in rows) / n

    print(f"\n  {'method':<14} {'R@10':>7} {'Δ vs base':>10}")
    print(f"  {'base':<14} {base_r:>7.4f} {0.0:>+10.4f}")
    print(f"  {'BoD':<14} {bod_r:>7.4f} {bod_r - base_r:>+10.4f}")
    print(f"  {'HyDE':<14} {hyde_r:>7.4f} {hyde_r - base_r:>+10.4f}")
    print(f"  {'Doc2Query':<14} {d2q_r:>7.4f} {d2q_r - base_r:>+10.4f}")
    print(f"  {'UNION-oracle':<14} {union_r:>7.4f} {union_r - base_r:>+10.4f}")
    print(f"  (oracle headroom over best-single: {union_r - max(bod_r, hyde_r, d2q_r):+.4f})")

    # Disjointness: how many queries does UNION rescue beyond any single method?
    union_beats_each = sum(
        1
        for r in rows
        if max(r["bod"], r["hyde"], r["d2q"]) > r["bod"]
        and max(r["bod"], r["hyde"], r["d2q"]) > r["hyde"]
        and max(r["bod"], r["hyde"], r["d2q"]) > r["d2q"]
    )
    # NOTE: above is impossible — max of {a,b,c} can't be strictly > all three.
    # Instead: count queries where the best method differs across the three.
    # Useful metric: # queries where exactly one method finds the gold.
    only_bod = sum(1 for r in rows if r["bod"] > 0 and r["hyde"] == 0 and r["d2q"] == 0)
    only_hyde = sum(1 for r in rows if r["hyde"] > 0 and r["bod"] == 0 and r["d2q"] == 0)
    only_d2q = sum(1 for r in rows if r["d2q"] > 0 and r["bod"] == 0 and r["hyde"] == 0)
    all_three = sum(1 for r in rows if r["bod"] > 0 and r["hyde"] > 0 and r["d2q"] > 0)
    none = sum(1 for r in rows if r["bod"] == 0 and r["hyde"] == 0 and r["d2q"] == 0)
    print("\n  exclusivity (any-hit per method, base-blind subset):")
    base_blind = [r for r in rows if r["base"] == 0]
    if base_blind:
        ob = sum(1 for r in base_blind if r["bod"] > 0 and r["hyde"] == 0 and r["d2q"] == 0)
        oh = sum(1 for r in base_blind if r["hyde"] > 0 and r["bod"] == 0 and r["d2q"] == 0)
        od = sum(1 for r in base_blind if r["d2q"] > 0 and r["bod"] == 0 and r["hyde"] == 0)
        at = sum(1 for r in base_blind if r["bod"] > 0 and r["hyde"] > 0 and r["d2q"] > 0)
        nn = sum(1 for r in base_blind if r["bod"] == 0 and r["hyde"] == 0 and r["d2q"] == 0)
        print(f"    base-blind queries: {len(base_blind):,}")
        print(f"    only BoD: {ob}   only HyDE: {oh}   only Doc2Query: {od}")
        print(f"    all three: {at}   none: {nn}")
    _ = (
        union_beats_each,
        only_bod,
        only_hyde,
        only_d2q,
        all_three,
        none,
    )  # all queries view computed above

    # Bucket by base ratio
    buckets = [
        ("0.0 (base miss)", lambda r: r["base"] == 0),
        ("0.0-0.5", lambda r: 0 < r["base"] / r["n_gold"] <= 0.5),
        ("0.5-1.0", lambda r: 0.5 < r["base"] / r["n_gold"] < 1.0),
        ("1.0 (perfect)", lambda r: r["base"] == r["n_gold"]),
    ]
    print("\n  per-bucket Δ vs base:")
    print(f"    {'bucket':<18} {'n':>5} {'BoD':>8} {'HyDE':>8} {'D2Q':>8} {'UNION':>8}")
    for label, pred in buckets:
        rb = [r for r in rows if pred(r)]
        if not rb:
            continue
        nbk = len(rb)
        bb = sum(r["base"] / r["n_gold"] for r in rb) / nbk
        bbod = sum(r["bod"] / r["n_gold"] for r in rb) / nbk
        bhyde = sum(r["hyde"] / r["n_gold"] for r in rb) / nbk
        bd2q = sum(r["d2q"] / r["n_gold"] for r in rb) / nbk
        buni = sum(max(r["bod"], r["hyde"], r["d2q"]) / r["n_gold"] for r in rb) / nbk
        print(
            f"    {label:<18} {nbk:>5} "
            f"{bbod - bb:>+8.3f} {bhyde - bb:>+8.3f} {bd2q - bb:>+8.3f} "
            f"{buni - bb:>+8.3f}"
        )


if __name__ == "__main__":
    corpora = [
        (
            "scifact",
            "scifact_data/bod_per_query_scifact.jsonl",
            "scifact_data/hyde_per_query_scifact.jsonl",
            "scifact_data/doc2query_per_query_scifact_d2q_full.jsonl",
        ),
        (
            "nfcorpus",
            "nfcorpus_data/bod_per_query_nfcorpus.jsonl",
            "nfcorpus_data/hyde_per_query_nfcorpus.jsonl",
            "nfcorpus_data/doc2query_per_query_nfcorpus_d2q_oracle_vecavg_fixed.jsonl",
        ),
        (
            "fiqa",
            "fiqa_data/bod_per_query_fiqa.jsonl",
            "fiqa_data/hyde_per_query_fiqa.jsonl",
            "fiqa_data/doc2query_per_query_fiqa_d2q_oracle.jsonl",
        ),
        (
            "programmers",
            "cqadupstack_programmers_data/bod_per_query_programmers.jsonl",
            "cqadupstack_programmers_data/hyde_per_query_programmers.jsonl",
            "cqadupstack_programmers_data/doc2query_per_query_programmers_d2q_oracle.jsonl",
        ),
        (
            "english",
            "cqadupstack_english_data/bod_per_query_english.jsonl",
            "cqadupstack_english_data/hyde_per_query_english.jsonl",
            "cqadupstack_english_data/doc2query_per_query_english_d2q_oracle.jsonl",
        ),
    ]
    for label, b, h, d in corpora:
        main(label, Path(b), Path(h), Path(d))
        sys.stdout.flush()
