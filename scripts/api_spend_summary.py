#!/usr/bin/env python3
"""Summarize the local API spend ledger (.api_spend.jsonl, gitignored).

For the user's own bookkeeping only; provider dashboard is authoritative.
"""

import json
from collections import defaultdict
from pathlib import Path

LEDGER = Path(__file__).resolve().parent.parent / ".api_spend.jsonl"


def main():
    if not LEDGER.exists():
        print(f"no spend ledger at {LEDGER}")
        return
    by_provider = defaultdict(lambda: {"tokens": 0, "cost": 0.0, "calls": 0})
    by_model = defaultdict(lambda: {"tokens": 0, "cost": 0.0, "calls": 0})
    total = {"tokens": 0, "cost": 0.0, "calls": 0}
    with open(LEDGER) as f:
        for line in f:
            r = json.loads(line)
            for d in (by_provider[r["provider"]], by_model[r["model"]], total):
                d["tokens"] += r["tokens"]
                d["cost"] += r["cost_usd"]
                d["calls"] += 1
    print(f"=== API spend ledger ({LEDGER.name}) ===\n")
    print("by provider:")
    for p, d in sorted(by_provider.items()):
        print(
            f"  {p:<12} calls={d['calls']:>3}  tokens={d['tokens']:>12,}  cost=${d['cost']:>8.4f}"
        )
    print("\nby model:")
    for m, d in sorted(by_model.items()):
        print(
            f"  {m:<32} calls={d['calls']:>3}  tokens={d['tokens']:>12,}  cost=${d['cost']:>8.4f}"
        )
    print(f"\nTOTAL: {total['calls']} calls, {total['tokens']:,} tokens, ${total['cost']:.4f}")


if __name__ == "__main__":
    main()
