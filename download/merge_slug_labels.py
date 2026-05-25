"""Merge the relabeled 'other' batch back into the master labeled CSV."""

import csv
from pathlib import Path

MASTER = Path("unified_jobs/top500_slugs_labeled.csv")
RELABEL = Path("unified_jobs/other140_reclassified.csv")
OUT = Path("unified_jobs/top500_slugs_labeled_merged.csv")


def main() -> None:
    new_labels: dict[str, tuple[str, str]] = {}
    with RELABEL.open() as f:
        for r in csv.DictReader(f):
            new_labels[r["slug"]] = (r["industry"], r["confidence"])

    rows = []
    swapped = 0
    with MASTER.open() as f:
        for r in csv.DictReader(f):
            if r["industry"] == "other" and r["slug"] in new_labels:
                ind, conf = new_labels[r["slug"]]
                r["industry"] = ind
                r["confidence"] = conf
                swapped += 1
            rows.append(r)

    fields = list(rows[0].keys())
    with OUT.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, quoting=csv.QUOTE_ALL)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"swapped {swapped} rows; wrote {OUT}")


if __name__ == "__main__":
    main()
