"""Dump the 9 facet fields from the current Solr index to facets.jsonl.

Used to rebuild facets.jsonl after it was lost on disk but the values are still
present in Solr (because they're stored). After this dump, push_docs.py
can do a clean full re-push to restore title + vectors that atomic-update wiped.
"""

import json
from pathlib import Path

import requests

SOLR = "http://localhost:8983"
CORE = "jobs"
OUT = Path(__file__).parent / "facets.jsonl"

FIELDS = [
    "role_family",
    "seniority",
    "remote_mode",
    "location_country",
    "location_state",
    "location_city",
    "posted_bucket",
    "salary_band_usd_annual",
    "tech_stack",
]


def main() -> None:
    n_total = requests.get(
        f"{SOLR}/solr/{CORE}/select",
        params={"q": "*:*", "rows": 0, "wt": "json"},
        timeout=30,
    ).json()["response"]["numFound"]
    print(f"total docs: {n_total:,}")

    fl = "id," + ",".join(FIELDS)
    rows = 5000
    start = 0
    with OUT.open("w") as out:
        while start < n_total:
            r = requests.get(
                f"{SOLR}/solr/{CORE}/select",
                params={
                    "q": "*:*",
                    "rows": rows,
                    "start": start,
                    "fl": fl,
                    "sort": "id asc",  # but id is string; we'll handle non-numeric ids OK
                    "wt": "json",
                },
                timeout=120,
            )
            r.raise_for_status()
            docs = r.json()["response"]["docs"]
            if not docs:
                break
            for d in docs:
                idx = int(d["id"])
                row = {"idx": idx}
                for f in FIELDS:
                    row[f] = d.get(f, [] if f == "tech_stack" else "")
                out.write(json.dumps(row) + "\n")
            start += len(docs)
            if start % 50000 == 0:
                print(f"  dumped {start:,}", flush=True)
    print(f"wrote {OUT} with {start:,} rows")


if __name__ == "__main__":
    main()
