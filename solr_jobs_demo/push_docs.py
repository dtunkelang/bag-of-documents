#!/usr/bin/env python3
"""One-shot full re-push: original fields (title, vectors, metadata) + the
9 facet fields from facets.jsonl. Replaces atomic_update_facets.py which
wiped the unstored title field. The dense vec field (`bge_vec` for schema
backwards-compat) now carries e5-small-v2 vectors.
"""

import json
import os
import sys
import time
from collections.abc import Iterator

import numpy as np
import requests

STAGE = "/Users/dtunkelang/bagofdocs/unified_jobs"
FACETS = "/Users/dtunkelang/bagofdocs/solr_jobs_demo/facets/facets.jsonl"
SOLR = os.environ.get("SOLR", "http://localhost:8983")
CORE = "jobs"
BATCH = 500

FACET_FIELDS = (
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


def load_facets() -> dict[int, dict]:
    print(f"loading facets from {FACETS} ...", flush=True)
    out: dict[int, dict] = {}
    with open(FACETS) as f:
        for line in f:
            d = json.loads(line)
            idx = d["idx"]
            out[idx] = {k: d[k] for k in FACET_FIELDS}
    print(f"  {len(out):,} facet records", flush=True)
    return out


def stream_docs(facets: dict[int, dict]) -> Iterator[dict]:
    with open(os.path.join(STAGE, "titles.json")) as f:
        titles = json.load(f)
    with open(os.path.join(STAGE, "source_index.json")) as f:
        sources = json.load(f)["sources"]
    dense = np.load(os.path.join(STAGE, "e5_small_catalog.vecs.fp16.npy"), mmap_mode="r")

    import csv as _csv

    industry_csv = os.path.join(STAGE, "slug_industry_labels_round2.csv")
    slug_to_industry: dict[str, str] = {}
    if os.path.exists(industry_csv):
        with open(industry_csv) as f:
            for r in _csv.DictReader(f):
                slug_to_industry[r["slug"]] = r["industry"]
        print(f"  loaded {len(slug_to_industry):,} slug -> industry labels", flush=True)

    # Per-doc industry override for staffing/employment-agency employers.
    import sys as _sys

    _sys.path.insert(0, os.path.dirname(__file__))
    from staffing_override import resolve_industry as _resolve_industry  # noqa: E402

    meta_path = os.path.join(STAGE, "metadata.jsonl")
    with open(meta_path) as mf:
        for i, line in enumerate(mf):
            rec = json.loads(line)
            title_display = (rec.get("title") or titles[i].split("\n", 1)[0]).strip()
            slug = rec.get("source_slug") or ""
            fac = facets.get(i, {})
            slug_ind = slug_to_industry.get(slug, "unclassified")
            doc_industry = _resolve_industry(
                slug, slug_ind, fac.get("role_family") or "", title_display
            )
            doc = {
                "id": str(i),
                "title": titles[i],  # full title + description for BM25
                "title_display": title_display,
                "employer": slug,
                "industry": doc_industry,
                "locations": rec.get("locations") or [],
                "employment_type": rec.get("employment_type") or "",
                "salary_currency": rec.get("salary_currency") or "",
                "department": rec.get("department") or "",
                "posted_at": rec.get("posted_at") or "",
                "source_corpus": sources[i],
                "description": rec.get("description") or "",
                "bge_vec": dense[i].astype(np.float32).tolist(),
            }
            if rec.get("salary_min") is not None:
                doc["salary_min"] = float(rec["salary_min"])
            if rec.get("salary_max") is not None:
                doc["salary_max"] = float(rec["salary_max"])

            # Layer in facets
            for k in FACET_FIELDS:
                v = fac.get(k)
                if k == "tech_stack":
                    doc[k] = v or []
                else:
                    doc[k] = v or ""
            yield doc


def main() -> int:
    facets = load_facets()
    print("clearing core ...", flush=True)
    requests.post(
        f"{SOLR}/solr/{CORE}/update",
        json={"delete": {"query": "*:*"}},
        params={"commit": "true"},
        timeout=120,
    ).raise_for_status()
    t0 = time.time()
    batch: list[dict] = []
    n = 0
    for doc in stream_docs(facets):
        batch.append(doc)
        if len(batch) >= BATCH:
            r = requests.post(
                f"{SOLR}/solr/{CORE}/update/json/docs",
                json=batch,
                params={"commit": "false"},
                timeout=120,
            )
            r.raise_for_status()
            n += len(batch)
            batch = []
            if n % 10000 == 0:
                rate = n / (time.time() - t0)
                print(f"  {n:,} ({rate:.0f}/s)", flush=True)
    if batch:
        requests.post(
            f"{SOLR}/solr/{CORE}/update/json/docs",
            json=batch,
            params={"commit": "false"},
            timeout=120,
        ).raise_for_status()
        n += len(batch)
    print("committing ...", flush=True)
    requests.get(
        f"{SOLR}/solr/{CORE}/update", params={"commit": "true"}, timeout=300
    ).raise_for_status()
    print(f"done: {n:,} in {time.time() - t0:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
