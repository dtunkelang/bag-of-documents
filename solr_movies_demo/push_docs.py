#!/usr/bin/env python3
"""Push unified_movies/metadata.jsonl into the Solr 'movies' core.

Solr id = IMDb tconst (e.g. tt0002130).
"""

import json
import os
import sys
import time
from collections.abc import Iterator

import requests

STAGE = "/Users/dtunkelang/bagofdocs/unified_movies"
SOLR = os.environ.get("SOLR", "http://localhost:8984")
CORE = os.environ.get("CORE", "movies")
BATCH = 500


def decade_of(year) -> str | None:
    if not isinstance(year, int):
        return None
    return f"{(year // 10) * 10}s"


def stream_docs(path: str) -> Iterator[dict]:
    with open(path) as f:
        for line in f:
            rec = json.loads(line)
            year = rec.get("year")
            doc: dict = {
                "id": rec["tconst"],
                "title": rec.get("title") or "",
                "title_display": rec.get("title") or "",
                "type": rec.get("type") or "",
                "genres": rec.get("genres") or [],
                "is_adult": bool(rec.get("is_adult")),
                "cast_names": rec.get("cast_names") or [],
                "director_names": rec.get("director_names") or [],
                "corated_bag": rec.get("corated_bag") or [],
                "has_lead": bool(rec.get("has_lead")),
                "has_plot": bool(rec.get("has_plot")),
                "has_bag": bool(rec.get("has_bag")),
            }
            if rec.get("original_title"):
                doc["original_title"] = rec["original_title"]
            if rec.get("lead"):
                doc["lead"] = rec["lead"]
            if rec.get("plot"):
                doc["plot"] = rec["plot"]
            if rec.get("enwiki_title"):
                doc["enwiki_title"] = rec["enwiki_title"]
            if isinstance(year, int):
                doc["year"] = year
                d = decade_of(year)
                if d:
                    doc["decade"] = d
            if isinstance(rec.get("runtime"), int):
                doc["runtime"] = rec["runtime"]
            if isinstance(rec.get("rating"), (int, float)):
                doc["rating"] = float(rec["rating"])
            if isinstance(rec.get("votes"), int):
                doc["votes"] = rec["votes"]
            yield doc


def post_batch(batch: list[dict]) -> None:
    r = requests.post(
        f"{SOLR}/solr/{CORE}/update/json/docs",
        params={"commit": "false"},
        json=batch,
        timeout=180,
    )
    r.raise_for_status()


def main() -> int:
    src = os.path.join(STAGE, "metadata.jsonl")
    print(f"clearing core {CORE}...", flush=True)
    requests.post(
        f"{SOLR}/solr/{CORE}/update",
        json={"delete": {"query": "*:*"}},
        params={"commit": "true"},
        timeout=120,
    ).raise_for_status()

    t0 = time.time()
    batch: list[dict] = []
    n = 0
    for doc in stream_docs(src):
        batch.append(doc)
        if len(batch) >= BATCH:
            post_batch(batch)
            n += len(batch)
            batch = []
            if n % 20000 == 0:
                rate = n / (time.time() - t0)
                print(f"  pushed {n:,} ({rate:.0f}/s)", flush=True)
    if batch:
        post_batch(batch)
        n += len(batch)
    print("committing...", flush=True)
    r = requests.get(f"{SOLR}/solr/{CORE}/update", params={"commit": "true"}, timeout=600)
    r.raise_for_status()
    print(f"done: {n:,} docs in {time.time() - t0:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
