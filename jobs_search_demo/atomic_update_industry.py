"""Atomic-update the 348k Solr docs with their propagated `industry` label.

Reads:
  - unified_jobs/metadata.jsonl (to enumerate idx -> source_slug)
  - unified_jobs/slug_industry_labels_round2.csv (slug -> industry)
"""

import csv
import json
import time
from pathlib import Path

import requests

ROOT = Path("/Users/dtunkelang/bagofdocs")
META = ROOT / "unified_jobs/metadata.jsonl"
LABELS = ROOT / "unified_jobs/slug_industry_labels_round2.csv"
SOLR = "http://localhost:8983"
CORE = "jobs"
BATCH = 1000


def load_slug_labels() -> dict[str, str]:
    out: dict[str, str] = {}
    with LABELS.open() as f:
        for r in csv.DictReader(f):
            out[r["slug"]] = r["industry"]
    return out


def main() -> None:
    print("loading slug labels...")
    slug_to_label = load_slug_labels()
    print(f"  {len(slug_to_label):,} slugs labeled")

    print("posting atomic updates...")
    t0 = time.time()
    batch: list[dict] = []
    n = 0
    with META.open() as f:
        for i, line in enumerate(f):
            d = json.loads(line)
            slug = d.get("source_slug") or ""
            label = slug_to_label.get(slug, "unclassified")
            batch.append({"id": str(i), "industry": {"set": label}})
            if len(batch) >= BATCH:
                r = requests.post(f"{SOLR}/solr/{CORE}/update", json=batch, timeout=120)
                r.raise_for_status()
                n += len(batch)
                batch = []
                if n % 20000 == 0:
                    rate = n / (time.time() - t0)
                    print(f"  {n:,} ({rate:.0f}/s)", flush=True)
    if batch:
        r = requests.post(f"{SOLR}/solr/{CORE}/update", json=batch, timeout=120)
        r.raise_for_status()
        n += len(batch)
    print("committing...", flush=True)
    requests.get(
        f"{SOLR}/solr/{CORE}/update", params={"commit": "true"}, timeout=300
    ).raise_for_status()
    print(f"done: {n:,} docs in {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
