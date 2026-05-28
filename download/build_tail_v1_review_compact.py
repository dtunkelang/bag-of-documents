"""Compact CSV of tail-v1 LLM-labeled slugs for re-review (audit the ChatGPT pass).

Filters slug_industry_overrides.csv for note ~ llm_tail_v1 and emits
{rank, slug, source, n_jobs, current_label, confidence, top_titles}
sorted by n_jobs descending. Default scope is tech_software_internet
since that was the attractor class in tail-v1.
"""

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

OVERRIDES = Path("unified_jobs/slug_industry_overrides.csv")
ROUND2 = Path("unified_jobs/slug_industry_labels_round2.csv")
META = Path("unified_jobs/metadata.jsonl")


def clean(s: str, n: int) -> str:
    s = re.sub(r"\s+", " ", s or "").strip()
    return s[:n]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--label",
        default="tech_software_internet",
        help="restrict to this current label (set to ALL to skip filter)",
    )
    ap.add_argument("--note-substring", default="llm_tail_v1")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    targets: dict[str, tuple[str, str]] = {}  # slug -> (current_label, confidence_tag)
    with OVERRIDES.open() as f:
        r = csv.DictReader(f)
        for row in r:
            note = row.get("note", "")
            if args.note_substring not in note:
                continue
            label = row["industry"]
            if args.label != "ALL" and label != args.label:
                continue
            targets[row["slug"]] = (label, note)
    print(
        f"matched {len(targets):,} slugs with note~'{args.note_substring}' and label={args.label}"
    )

    slug_njobs: dict[str, int] = {}
    slug_round2_src: dict[str, str] = {}
    with ROUND2.open() as f:
        r = csv.DictReader(f)
        for row in r:
            slug = row["slug"]
            if slug in targets:
                slug_njobs[slug] = int(row["n_jobs"])
                slug_round2_src[slug] = row.get("source", "?")

    slug_src: dict[str, Counter[str]] = defaultdict(Counter)
    slug_titles: dict[str, Counter[str]] = defaultdict(Counter)

    with META.open() as f:
        for line in f:
            d = json.loads(line)
            slug = d.get("source_slug")
            if not slug or slug not in targets:
                continue
            title = (d.get("title") or "").strip()
            src = d.get("source") or "?"
            slug_src[slug][src] += 1
            if title:
                slug_titles[slug][title] += 1

    rows = []
    for slug in targets:
        n = slug_njobs.get(slug, 0)
        rows.append((slug, n))
    rows.sort(key=lambda x: -x[1])

    with args.out.open("w", newline="") as f:
        w = csv.writer(f, quoting=csv.QUOTE_ALL)
        w.writerow(
            [
                "rank",
                "slug",
                "source",
                "n_jobs",
                "current_label",
                "confidence",
                "top_titles",
            ]
        )
        for rank, (slug, n) in enumerate(rows, 1):
            current_label, conf = targets[slug]
            src = (
                slug_src[slug].most_common(1)[0][0]
                if slug_src[slug]
                else slug_round2_src.get(slug, "?")
            )
            tt = slug_titles[slug].most_common(5)
            top_titles = " | ".join(clean(t, 80) for t, _ in tt)
            w.writerow([rank, slug, src, n, current_label, conf, top_titles])
    print(f"wrote {args.out} ({len(rows):,} rows)")


if __name__ == "__main__":
    main()
