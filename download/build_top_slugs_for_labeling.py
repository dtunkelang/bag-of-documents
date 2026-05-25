"""Top 500 employer slugs with sample titles + a sample description, for ChatGPT labeling."""

import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

META = Path("unified_jobs/metadata.jsonl")
OUT = Path("unified_jobs/top500_slugs_for_labeling.csv")

PLACEHOLDER_SLUGS = {
    "private-advertiser",
    "company-unknown",
    "the-job-network",
    "agency",
    "confidential",
    "unknown",
}

TARGET = 500


def clean(s: str, n: int) -> str:
    s = re.sub(r"\s+", " ", s or "").strip()
    return s[:n]


def main() -> None:
    slug_n: Counter[str] = Counter()
    slug_src: dict[str, Counter[str]] = defaultdict(Counter)
    slug_titles: dict[str, Counter[str]] = defaultdict(Counter)
    slug_sample_desc: dict[str, str] = {}

    with META.open() as f:
        for line in f:
            d = json.loads(line)
            slug = d.get("source_slug")
            if not slug or slug in PLACEHOLDER_SLUGS:
                continue
            title = (d.get("title") or "").strip()
            desc = (d.get("description") or "").strip()
            src = d.get("source") or "?"

            slug_n[slug] += 1
            slug_src[slug][src] += 1
            if title:
                slug_titles[slug][title] += 1

            cur = slug_sample_desc.get(slug, "")
            # prefer a description between 300 and 1500 chars (informative without being huge);
            # otherwise keep the longest we've seen
            if 300 <= len(desc) <= 1500 and not (300 <= len(cur) <= 1500) or len(desc) > len(cur):
                slug_sample_desc[slug] = desc

    top = slug_n.most_common(TARGET)
    covered = sum(n for _, n in top)
    total = sum(slug_n.values())
    print(f"top {TARGET} slugs cover {covered:,} of {total:,} docs = {covered / total:.1%}")

    with OUT.open("w", newline="") as f:
        w = csv.writer(f, quoting=csv.QUOTE_ALL)
        w.writerow(["rank", "slug", "source", "n_jobs", "top_titles", "sample_description"])
        for rank, (slug, n) in enumerate(top, 1):
            src = slug_src[slug].most_common(1)[0][0]
            tt = slug_titles[slug].most_common(5)
            top_titles = " | ".join(clean(t, 80) for t, _ in tt)
            desc = clean(slug_sample_desc.get(slug, ""), 400)
            w.writerow([rank, slug, src, n, top_titles, desc])
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
