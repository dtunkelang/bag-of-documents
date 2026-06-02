#!/usr/bin/env python3
"""Fetch currently-posted jobs from the Jooble API.

Jooble is a global job-search aggregator (1M+ live postings), so it brings broad
aggregator inventory the OpenApply crawler (Greenhouse/Lever/Ashby), USAJOBS
(federal) and the ATS adapters do not cover. It has no "list everything"
endpoint, so this adapter sweeps a list of broad occupation keywords (see
jobs_search_demo/job_search_queries.txt) to approximate full coverage.

NOTE: Jooble only returns a short *snippet* per posting (no full description and
no detail endpoint), so its docs are thinner than the ATS / Reed / The Muse /
Findwork sources. Wire it in only if snippet-grade descriptions are acceptable.

Requires a free API key (https://jooble.org/api/about). The key goes in the URL
PATH:
  JOOBLE_API_KEY='...'
  search: POST https://jooble.org/api/{KEY}   body {"keywords","location","page"}

Writes parquet files whose schema matches download/fetch_adzuna.py / what
prep_open_apply.py expects (id, source_slug, title, description_html, department,
employment_type, remote, locations, salary_min, salary_max, salary_currency,
posted_at, source).

Usage:
  export JOOBLE_API_KEY='...'
  .venv/bin/python download/fetch_jooble.py \\
      --out-dir jobs_data_jooble/raw \\
      --queries-file jobs_search_demo/job_search_queries.txt \\
      --location "" --max-pages 5
"""

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import pyarrow as pa
import pyarrow.parquet as pq

SLUG_RE = re.compile(r"[^a-z0-9]+")
TAG_RE = re.compile(r"<[^>]+>")
DEFAULT_QUERIES = ("software engineer", "nurse", "sales", "accountant", "teacher")


def slugify(s: str) -> str:
    return SLUG_RE.sub("-", (s or "").lower()).strip("-")


def strip_tags(s: str) -> str:
    return TAG_RE.sub("", s or "").strip()


def _post(key: str, body: dict, retries: int = 4) -> Any:
    url = f"https://jooble.org/api/{key}"
    data = json.dumps(body).encode("utf-8")
    backoff = 2
    for attempt in range(retries):
        req = Request(
            url,
            data=data,
            method="POST",
            headers={
                "Content-Type": "application/json",
                "User-Agent": "bagofdocs-jobs-refresh",
                "Accept": "application/json",
            },
        )
        try:
            with urlopen(req, timeout=60) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except (HTTPError, URLError, TimeoutError) as e:
            if isinstance(e, HTTPError) and 400 <= e.code < 500 and e.code != 429:
                raise
            if attempt == retries - 1:
                raise
            sys.stderr.write(f"  {body} attempt {attempt + 1} failed ({e}); sleeping {backoff}s\n")
            time.sleep(backoff)
            backoff *= 2
    raise RuntimeError("unreachable")


def transform(job: dict) -> dict[str, Any]:
    job_id = str(job.get("id") or "")
    company = job.get("company") or ""
    title = strip_tags(job.get("title") or "")

    loc = job.get("location") or ""
    locations = [loc] if loc else []

    hay = f"{title} {loc} {job.get('snippet') or ''}".lower()
    remote = True if ("remote" in hay or "work from home" in hay) else None

    et = (job.get("type") or "").strip() or None

    return {
        "id": f"jooble:{job_id}",
        "source_slug": slugify(company) or "jooble",
        "title": title,
        # snippet is the only body Jooble exposes; prep strips HTML downstream.
        "description_html": job.get("snippet") or "",
        "department": None,
        "employment_type": et,
        "remote": remote,
        "locations": locations,
        "salary_min": None,  # salary is free-text (e.g. "$94.2k - $141.2k"), not parseable
        "salary_max": None,
        "salary_currency": None,
        "posted_at": job.get("updated") or None,  # ISO 8601
        "source": "jooble",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument(
        "--queries-file",
        default="jobs_search_demo/job_search_queries.txt",
        help="one keyword query per line (# comments allowed)",
    )
    ap.add_argument(
        "--location", default="", help="optional location filter applied to every query"
    )
    ap.add_argument("--max-pages", type=int, default=5, help="result pages per query")
    ap.add_argument("--rows-per-file", type=int, default=10_000)
    ap.add_argument("--sleep", type=float, default=0.3, help="seconds between calls (be polite)")
    args = ap.parse_args()

    key = os.environ.get("JOOBLE_API_KEY")
    if not key:
        sys.exit("ERROR: set JOOBLE_API_KEY env var")

    qpath = Path(args.queries_file)
    if qpath.exists():
        queries = [
            ln.strip()
            for ln in qpath.read_text().splitlines()
            if ln.strip() and not ln.lstrip().startswith("#")
        ]
    else:
        print(f"[jooble] queries file {qpath} missing; using built-in defaults", flush=True)
        queries = list(DEFAULT_QUERIES)
    if not queries:
        sys.exit(f"ERROR: no queries in {qpath}")

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    buf: list[dict] = []
    file_idx = 0
    n_written = 0
    seen_ids: set[str] = set()

    def flush():
        nonlocal buf, file_idx, n_written
        if not buf:
            return
        table = pa.Table.from_pylist(buf)
        path = out / f"jooble-{file_idx:04d}.parquet"
        pq.write_table(table, path)
        n_written += len(buf)
        print(f"  wrote {path.name}: {len(buf):,} rows (total {n_written:,})", flush=True)
        file_idx += 1
        buf = []

    for q in queries:
        q_rows = 0
        for page in range(1, args.max_pages + 1):
            payload = _post(key, {"keywords": q, "location": args.location, "page": page})
            time.sleep(args.sleep)
            jobs = payload.get("jobs") or []
            if not jobs:
                break
            for job in jobs:
                row = transform(job)
                if not row["title"] or row["id"] in seen_ids:
                    continue
                seen_ids.add(row["id"])
                buf.append(row)
                q_rows += 1
                if len(buf) >= args.rows_per_file:
                    flush()
        if q_rows:
            print(f"[{q}] collected {q_rows:,} new postings", flush=True)

    flush()
    print(
        f"done: {n_written:,} unique postings across {file_idx} parquet files -> {out}",
        flush=True,
    )


if __name__ == "__main__":
    main()
