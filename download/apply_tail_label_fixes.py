"""Replace existing slug labels in slug_industry_overrides.csv with corrected ones.

Differs from merge_tail_labels.py — that script refuses to touch existing rows;
this one rewrites them in place with a new label and a new note tag. Used after
auditing an earlier LLM-labeled batch.
"""

import argparse
import csv
from pathlib import Path

OVERRIDES = Path("unified_jobs/slug_industry_overrides.csv")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--fixes",
        type=Path,
        action="append",
        required=True,
        help="CSV(s) with slug,industry,confidence columns; can pass --fixes multiple times",
    )
    ap.add_argument("--note-tag", default="llm_tail_v1_fix")
    ap.add_argument(
        "--backup", type=Path, default=Path("unified_jobs/slug_industry_overrides.bak.csv")
    )
    args = ap.parse_args()

    fix_map: dict[str, tuple[str, str]] = {}
    for path in args.fixes:
        with path.open() as f:
            for row in csv.DictReader(f):
                slug = row["slug"].strip()
                fix_map[slug] = (row["industry"].strip(), row["confidence"].strip())
    print(f"loaded {len(fix_map):,} fixes from {len(args.fixes)} file(s)")

    rows = []
    with OVERRIDES.open() as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        for row in reader:
            rows.append(row)

    OVERRIDES.replace(args.backup)
    print(f"backed up to {args.backup}")

    applied = 0
    not_found = set(fix_map.keys())
    with OVERRIDES.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            slug = row["slug"]
            if slug in fix_map:
                new_label, conf = fix_map[slug]
                w.writerow(
                    {
                        "slug": slug,
                        "industry": new_label,
                        "note": f"{args.note_tag} ({conf})",
                    }
                )
                applied += 1
                not_found.discard(slug)
            else:
                w.writerow(row)

    print(f"applied {applied} fixes; {len(not_found)} fix-slugs not found in overrides")
    if not_found:
        for s in sorted(not_found)[:20]:
            print(f"  missing: {s}")


if __name__ == "__main__":
    main()
