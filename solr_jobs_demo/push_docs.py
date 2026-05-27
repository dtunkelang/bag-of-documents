#!/usr/bin/env python3
"""Push 347.9k jobs (title + metadata + dense_vec[e5-small] + te3_vec) into the Solr 'jobs' core.

Solr id = integer position in the catalog (0..347899). Lets the existing demo UI
keep using `idx` for click-through. The Solr field is named `bge_vec` for schema
backwards-compat but holds intfloat/e5-small-v2 vectors (also 384-dim).
"""

import json
import os
import sys
import time
from collections.abc import Iterator

import numpy as np
import requests

STAGE = "/Users/dtunkelang/bagofdocs/unified_jobs"
SOLR = os.environ.get("SOLR", "http://localhost:8983")
CORE = "jobs"
BATCH = 500


def stream_docs() -> Iterator[dict]:
    with open(os.path.join(STAGE, "titles.json")) as f:
        titles = json.load(f)
    with open(os.path.join(STAGE, "source_index.json")) as f:
        sources = json.load(f)["sources"]
    dense = np.load(os.path.join(STAGE, "e5_small_catalog.vecs.fp16.npy"), mmap_mode="r")
    te3 = np.load(os.path.join(STAGE, "te3_catalog.vecs.fp16.npy"), mmap_mode="r")
    assert len(titles) == len(sources) == dense.shape[0] == te3.shape[0], (
        f"length mismatch: titles={len(titles)} sources={len(sources)} dense={dense.shape[0]} te3={te3.shape[0]}"
    )

    # slug -> industry label from round-2 propagation (slug_industry_labels_round2.csv)
    industry_csv = os.path.join(STAGE, "slug_industry_labels_round2.csv")
    slug_to_industry: dict[str, str] = {}
    if os.path.exists(industry_csv):
        import csv as _csv

        with open(industry_csv) as f:
            for r in _csv.DictReader(f):
                slug_to_industry[r["slug"]] = r["industry"]
        print(f"  loaded {len(slug_to_industry):,} slug -> industry labels", flush=True)

    meta_path = os.path.join(STAGE, "metadata.jsonl")
    with open(meta_path) as mf:
        for i, line in enumerate(mf):
            rec = json.loads(line)
            title_display = (rec.get("title") or titles[i].split("\n", 1)[0]).strip()
            # BM25 in the demo indexes titles[i] = "title\n\ndescription" (full text).
            # Push that into 'title' to preserve BM25 ranking.
            slug = rec.get("source_slug") or ""
            doc = {
                "id": str(i),
                "title": titles[i],
                "title_display": title_display,
                "employer": slug,
                "industry": slug_to_industry.get(slug, "unclassified"),
                "locations": rec.get("locations") or [],
                "employment_type": rec.get("employment_type") or "",
                "salary_currency": rec.get("salary_currency") or "",
                "department": rec.get("department") or "",
                "posted_at": rec.get("posted_at") or "",
                "source_corpus": sources[i],
                "description": rec.get("description") or "",
                "bge_vec": dense[i].astype(np.float32).tolist(),
                "te3_vec": te3[i].astype(np.float32).tolist(),
            }
            if rec.get("salary_min") is not None:
                doc["salary_min"] = float(rec["salary_min"])
            if rec.get("salary_max") is not None:
                doc["salary_max"] = float(rec["salary_max"])
            yield doc


def post_batch(batch: list[dict]) -> None:
    r = requests.post(
        f"{SOLR}/solr/{CORE}/update/json/docs",
        params={"commit": "false"},
        json=batch,
        timeout=120,
    )
    r.raise_for_status()


def main() -> int:
    print("clearing core...", flush=True)
    requests.post(
        f"{SOLR}/solr/{CORE}/update",
        json={"delete": {"query": "*:*"}},
        params={"commit": "true"},
        timeout=120,
    ).raise_for_status()
    t0 = time.time()
    batch: list[dict] = []
    n = 0
    for doc in stream_docs():
        batch.append(doc)
        if len(batch) >= BATCH:
            post_batch(batch)
            n += len(batch)
            batch = []
            if n % 10000 == 0:
                rate = n / (time.time() - t0)
                print(f"  pushed {n:,} ({rate:.0f}/s)", flush=True)
    if batch:
        post_batch(batch)
        n += len(batch)
    print("committing...", flush=True)
    r = requests.get(f"{SOLR}/solr/{CORE}/update", params={"commit": "true"}, timeout=300)
    r.raise_for_status()
    print(f"done: {n:,} docs in {time.time() - t0:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
