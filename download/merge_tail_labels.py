"""Merge ChatGPT-labeled unclassified-tail slugs into slug_industry_overrides.csv.

Skips rows with industry=other or confidence=low (stashes them in a review CSV
for follow-up). Refuses to add slugs that already exist in the override CSV
(no overlaps expected since the tail is post-round-2 unclassified)."""

import argparse
import csv
from pathlib import Path

OVERRIDES = Path("unified_jobs/slug_industry_overrides.csv")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--labeled", type=Path, required=True, help="CSV with slug,industry,confidence columns"
    )
    ap.add_argument(
        "--skipped",
        type=Path,
        required=True,
        help="output CSV for other/low rows requiring review",
    )
    ap.add_argument(
        "--note-tag",
        default="llm_tail_v1",
        help="provenance tag written into the overrides notes column",
    )
    args = ap.parse_args()

    existing: set[str] = set()
    with OVERRIDES.open() as f:
        for row in csv.DictReader(f):
            existing.add(row["slug"])

    keep: list[tuple[str, str, str]] = []
    skip: list[dict] = []
    with args.labeled.open() as f:
        for row in csv.DictReader(f):
            if row["industry"] == "other" or row["confidence"] == "low":
                skip.append(row)
                continue
            if row["slug"] in existing:
                print(f"  duplicate, skipping: {row['slug']}")
                continue
            keep.append((row["slug"], row["industry"], row["confidence"]))

    with OVERRIDES.open("a", newline="") as f:
        w = csv.writer(f)
        for slug, industry, conf in keep:
            w.writerow([slug, industry, f"{args.note_tag} ({conf})"])

    with args.skipped.open("w", newline="") as f:
        if skip:
            w = csv.DictWriter(f, fieldnames=list(skip[0].keys()))
            w.writeheader()
            w.writerows(skip)

    print(f"appended {len(keep)} rows to {OVERRIDES}")
    print(f"stashed {len(skip)} other/low rows in {args.skipped}")


if __name__ == "__main__":
    main()
