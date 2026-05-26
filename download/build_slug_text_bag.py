"""Build per-slug text bag (slug tokens x3 + top-50 titles) for embedding-based classifiers.

Reads unified_jobs/metadata.jsonl and emits:
  unified_jobs/slug_text_bag.csv  -> slug,n_jobs,source,text

Matches the TF-IDF document format in propagate_industry_tfidf.py exactly,
so dense-embedding classifiers can be benchmarked on the same input.
"""

from __future__ import annotations

import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
UJ = REPO / "unified_jobs"
META = UJ / "metadata.jsonl"
OUT = UJ / "slug_text_bag.csv"

TOP_K_TITLES = 50


def slug_tokens(slug: str) -> str:
    return re.sub(r"[-_./]+", " ", slug).lower()


def main():
    slug_titles: dict[str, Counter[str]] = defaultdict(Counter)
    slug_n: Counter[str] = Counter()
    slug_src: dict[str, str] = {}

    print(f"Reading {META} …")
    with META.open() as f:
        for line in f:
            d = json.loads(line)
            slug = d.get("source_slug")
            if not slug:
                continue
            slug_n[slug] += 1
            slug_src[slug] = d.get("source", "?")
            title = (d.get("title") or "").strip()
            if title:
                slug_titles[slug][title.lower()] += 1
    print(f"  {len(slug_titles):,} unique slugs")

    with OUT.open("w", newline="") as fout:
        w = csv.writer(fout)
        w.writerow(["slug", "n_jobs", "source", "text"])
        for s in sorted(slug_titles.keys()):
            slug_text = (slug_tokens(s) + " ") * 3
            top_titles = [t for t, _ in slug_titles[s].most_common(TOP_K_TITLES)]
            text = slug_text + " ".join(top_titles)
            w.writerow([s, slug_n[s], slug_src.get(s, "?"), text])
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
