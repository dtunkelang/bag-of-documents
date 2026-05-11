#!/usr/bin/env python3
"""Per-query overlap analysis: does HyDE rescue the same queries as BoD?

Reads two per-query JSONLs (produced by `diagnose_lift.py` and
`eval_hyde.py` respectively) and reports:

  - aggregate R@10 for base, BoD, HyDE
  - rescue rate on the base-blind subset (queries where base R@10 = 0)
    per method
  - the contingency table on the base-blind subset:
        BoD rescued y/n × HyDE rescued y/n
    With cell counts and a few example queries per cell.

A query "rescued" = method hit ≥ 1 of the gold positives in top-k.

Usage:
    python evaluation/diagnose_hyde_vs_bod.py \\
        --bod-per-query scifact_data/bod_per_query_scifact.jsonl \\
        --hyde-per-query scifact_data/hyde_per_query_scifact_hyde.jsonl \\
        --queries scifact_data/test_queries.jsonl \\
        --label scifact
"""

import argparse
import json


def load_per_query(path):
    rows = {}
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            rows[d["query_id"]] = d
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--bod-per-query", required=True)
    ap.add_argument("--hyde-per-query", required=True)
    ap.add_argument("--queries", default=None, help="optional queries.jsonl for example text")
    ap.add_argument("--label", default="corpus")
    ap.add_argument("--example-cells", type=int, default=3, help="examples per contingency cell")
    args = ap.parse_args()

    bod = load_per_query(args.bod_per_query)
    hyde = load_per_query(args.hyde_per_query)
    common = sorted(set(bod) & set(hyde))
    print(f"BoD per-query rows: {len(bod):,}")
    print(f"HyDE per-query rows: {len(hyde):,}")
    print(f"intersection: {len(common):,}")
    if not common:
        print("ERROR: no overlapping query ids — abort.")
        return

    qtext = {}
    if args.queries:
        with open(args.queries) as f:
            for line in f:
                d = json.loads(line)
                qtext[d["query_id"]] = d["query"]

    def hit_rate(rows, col):
        return sum(r[col] / r["n_gold"] for r in rows) / len(rows)

    rows = [{"qid": q, **bod[q], **hyde[q]} for q in common]
    # bod[q]: base_hit, bod_hit, n_gold (base_hit duplicated from hyde[q] which is fine)
    print(f"\noverall R@10 (n={len(rows):,}):")
    print(f"  base:  {hit_rate(rows, 'base_hit'):.3f}")
    print(f"  BoD:   {hit_rate(rows, 'bod_hit'):.3f}")
    print(f"  HyDE:  {hit_rate(rows, 'hyde_hit'):.3f}")

    blind = [r for r in rows if r["base_hit"] == 0]
    print(
        f"\nbase-blind subset (base hit 0): n={len(blind):,} "
        f"({100 * len(blind) / len(rows):.1f}% of pos-bearing)"
    )
    if not blind:
        print("  no base-blind queries — overlap analysis n/a")
        return

    # Rescue rate per method on the blind subset.
    bod_rescued = [r for r in blind if r["bod_hit"] > 0]
    hyde_rescued = [r for r in blind if r["hyde_hit"] > 0]
    both = [r for r in blind if r["bod_hit"] > 0 and r["hyde_hit"] > 0]
    bod_only = [r for r in blind if r["bod_hit"] > 0 and r["hyde_hit"] == 0]
    hyde_only = [r for r in blind if r["bod_hit"] == 0 and r["hyde_hit"] > 0]
    neither = [r for r in blind if r["bod_hit"] == 0 and r["hyde_hit"] == 0]

    print("\nrescue rate on base-blind subset (fraction of gold recovered):")
    print(
        f"  BoD:   {sum(r['bod_hit'] / r['n_gold'] for r in blind) / len(blind):.3f} "
        f"(rescues {len(bod_rescued)}/{len(blind)} = "
        f"{100 * len(bod_rescued) / len(blind):.1f}% of base-blind queries)"
    )
    print(
        f"  HyDE:  {sum(r['hyde_hit'] / r['n_gold'] for r in blind) / len(blind):.3f} "
        f"(rescues {len(hyde_rescued)}/{len(blind)} = "
        f"{100 * len(hyde_rescued) / len(blind):.1f}% of base-blind queries)"
    )

    print("\ncontingency on base-blind subset:")
    print(f"  {'':<20} {'HyDE-rescues':>15} {'HyDE-misses':>15}")
    print(f"  {'BoD-rescues':<20} {len(both):>15,} {len(bod_only):>15,}")
    print(f"  {'BoD-misses':<20} {len(hyde_only):>15,} {len(neither):>15,}")

    # Examples per cell.
    if qtext and args.example_cells > 0:
        print(f"\nexample queries per contingency cell (up to {args.example_cells} each):")
        for cell_name, cell in (
            ("both rescued", both),
            ("BoD only", bod_only),
            ("HyDE only", hyde_only),
            ("neither rescued", neither),
        ):
            if not cell:
                continue
            print(f"\n  {cell_name} (n={len(cell):,}):")
            for r in cell[: args.example_cells]:
                q = qtext.get(r["qid"], "")
                print(f"    [{r['qid']}] {q[:80]}{'…' if len(q) > 80 else ''}")

    # Print a summary line useful for CHS_RESULTS markdown.
    print(
        f"\nSUMMARY: corpus={args.label}  base-blind={len(blind)}  "
        f"BoD-rescues={len(bod_rescued)} ({100 * len(bod_rescued) / len(blind):.1f}%)  "
        f"HyDE-rescues={len(hyde_rescued)} ({100 * len(hyde_rescued) / len(blind):.1f}%)  "
        f"overlap={len(both)}"
    )


if __name__ == "__main__":
    main()
