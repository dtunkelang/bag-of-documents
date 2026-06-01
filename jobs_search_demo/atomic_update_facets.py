#!/usr/bin/env python3
"""Atomic-update the existing 348k Solr docs with the 9 facet fields.

Reads facets.jsonl, batches docs in groups of 500, posts with `set` ops so
existing fields (title, vectors, description, etc.) are preserved.
"""

import json
import sys
import time
from pathlib import Path

import requests

FACETS = Path("/Users/dtunkelang/bagofdocs/jobs_search_demo/facets/facets.jsonl")
SOLR = "http://localhost:8983"
CORE = "jobs"
BATCH = 500

FIELDS = (
    "role_family",
    "seniority",
    "remote_mode",
    "location_country",
    "location_state",
    "location_city",
    "posted_bucket",
    "salary_band_usd_annual",
    "tech_stack",
)


def main() -> int:
    t0 = time.time()
    batch: list[dict] = []
    n = 0
    with open(FACETS) as f:
        for line in f:
            d = json.loads(line)
            idx = d["idx"]
            doc = {"id": str(idx)}
            for k in FIELDS:
                v = d.get(k)
                # tech_stack is a list (multi-value); rest are scalars.
                if k == "tech_stack":
                    doc[k] = {"set": v or []}
                else:
                    doc[k] = {"set": v or ""}
            batch.append(doc)
            if len(batch) >= BATCH:
                r = requests.post(f"{SOLR}/solr/{CORE}/update", json=batch, timeout=120)
                r.raise_for_status()
                n += len(batch)
                batch = []
                if n % 10000 == 0:
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
    print(f"done: {n:,} in {time.time() - t0:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
