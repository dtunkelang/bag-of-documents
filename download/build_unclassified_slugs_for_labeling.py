"""Top-N unclassified employer slugs (post round-2 propagation) for LLM labeling."""

import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

ROUND2 = Path("unified_jobs/slug_industry_labels_round2.csv")
META = Path("unified_jobs/metadata.jsonl")
OUT = Path("unified_jobs/tail1500_unclassified_for_labeling.csv")

PLACEHOLDER_SLUGS = {
    "private-advertiser",
    "company-unknown",
    "the-job-network",
    "agency",
    "confidential",
    "unknown",
    "company-confidential",
    "leverdemo",
}

TARGET = 1500


def clean(s: str, n: int) -> str:
    s = re.sub(r"\s+", " ", s or "").strip()
    return s[:n]


def main() -> None:
    unclassified: list[tuple[str, int]] = []
    with ROUND2.open() as f:
        r = csv.DictReader(f)
        for row in r:
            if row["industry"] != "unclassified":
                continue
            slug = row["slug"]
            if slug in PLACEHOLDER_SLUGS:
                continue
            unclassified.append((slug, int(row["n_jobs"])))
    unclassified.sort(key=lambda x: -x[1])
    top = unclassified[:TARGET]
    target_slugs = {s for s, _ in top}
    print(f"top {TARGET} unclassified slugs cover {sum(n for _, n in top):,} docs")

    slug_src: dict[str, Counter[str]] = defaultdict(Counter)
    slug_titles: dict[str, Counter[str]] = defaultdict(Counter)
    slug_sample_desc: dict[str, str] = {}

    with META.open() as f:
        for line in f:
            d = json.loads(line)
            slug = d.get("source_slug")
            if not slug or slug not in target_slugs:
                continue
            title = (d.get("title") or "").strip()
            desc = (d.get("description") or "").strip()
            src = d.get("source") or "?"
            slug_src[slug][src] += 1
            if title:
                slug_titles[slug][title] += 1
            cur = slug_sample_desc.get(slug, "")
            if 300 <= len(desc) <= 1500 and not (300 <= len(cur) <= 1500) or len(desc) > len(cur):
                slug_sample_desc[slug] = desc

    with OUT.open("w", newline="") as f:
        w = csv.writer(f, quoting=csv.QUOTE_ALL)
        w.writerow(["rank", "slug", "source", "n_jobs", "top_titles", "sample_description"])
        for rank, (slug, n) in enumerate(top, 1):
            src = slug_src[slug].most_common(1)[0][0] if slug_src[slug] else "?"
            tt = slug_titles[slug].most_common(5)
            top_titles = " | ".join(clean(t, 80) for t, _ in tt)
            desc = clean(slug_sample_desc.get(slug, ""), 400)
            w.writerow([rank, slug, src, n, top_titles, desc])
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
