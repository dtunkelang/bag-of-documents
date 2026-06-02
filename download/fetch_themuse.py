#!/usr/bin/env python3
"""Fetch currently-posted jobs from The Muse public jobs API.

The Muse aggregates curated postings from mid/large employers, with full HTML
job descriptions, so it brings inventory the OpenApply crawler (Greenhouse/Lever/
Ashby), USAJOBS (federal), Adzuna, SmartRecruiters and Workable do not cover.
The listing response already carries the full description (contents), so unlike
the ATS adapters this needs no per-posting detail fetch.

The API key is OPTIONAL (the endpoint works keyless; a key only lifts the rate
limit). Provide one via:
  THEMUSE_API_KEY='...'
  list: GET https://www.themuse.com/api/public/jobs?page=N[&api_key=KEY]

Writes parquet files whose schema matches download/fetch_adzuna.py / what
prep_open_apply.py expects (id, source_slug, title, description_html, department,
employment_type, remote, locations, salary_min, salary_max, salary_currency,
posted_at, source).

Usage:
  .venv/bin/python download/fetch_themuse.py \\
      --out-dir jobs_data_themuse/raw \\
      --max-pages 50 --max-days-old 30
"""

import argparse
import datetime as dt
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pyarrow as pa
import pyarrow.parquet as pq

API = "https://www.themuse.com/api/public/jobs"
SLUG_RE = re.compile(r"[^a-z0-9]+")
TAG_RE = re.compile(r"<[^>]+>")


def slugify(s: str) -> str:
    return SLUG_RE.sub("-", (s or "").lower()).strip("-")


def strip_tags(s: str) -> str:
    return TAG_RE.sub("", s or "").strip()


def _get(url: str, retries: int = 4) -> Any:
    req = Request(
        url, headers={"User-Agent": "bagofdocs-jobs-refresh", "Accept": "application/json"}
    )
    backoff = 2
    for attempt in range(retries):
        try:
            with urlopen(req, timeout=60) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except (HTTPError, URLError, TimeoutError) as e:
            # 400 once the page index runs past page_count -> caller stops.
            if isinstance(e, HTTPError) and e.code in (400, 404):
                return None
            if isinstance(e, HTTPError) and 400 <= e.code < 500 and e.code != 429:
                raise
            if attempt == retries - 1:
                raise
            sys.stderr.write(f"  {url} attempt {attempt + 1} failed ({e}); sleeping {backoff}s\n")
            time.sleep(backoff)
            backoff *= 2
    raise RuntimeError("unreachable")


def transform(job: dict) -> dict[str, Any]:
    job_id = str(job.get("id") or "")
    company = (job.get("company") or {}).get("name") or ""
    title = strip_tags(job.get("name") or "")

    # locations is a list of {name}; The Muse uses "Flexible / Remote" for remote.
    locs = [loc.get("name") for loc in (job.get("locations") or []) if loc.get("name")]
    remote = True if any("remote" in (loc or "").lower() for loc in locs) else None
    locations = [loc for loc in locs if "remote" not in loc.lower()] or locs

    cats = [c.get("name") for c in (job.get("categories") or []) if c.get("name")]
    department = cats[0] if cats else None

    # No employment_type field; "levels" is seniority, not type -> leave None.
    return {
        "id": f"themuse:{job_id}",
        "source_slug": slugify(company) or "the-muse",
        "title": title,
        "description_html": job.get("contents") or "",
        "department": department,
        "employment_type": None,
        "remote": remote,
        "locations": locations,
        "salary_min": None,  # not exposed
        "salary_max": None,
        "salary_currency": None,
        "posted_at": job.get("publication_date") or None,  # ISO 8601
        "source": "themuse",
    }


def _too_old(pub: str | None, cutoff: dt.datetime | None) -> bool:
    if cutoff is None or not pub:
        return False
    try:
        ts = dt.datetime.fromisoformat(pub.replace("Z", "+00:00"))
    except ValueError:
        return False
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=dt.timezone.utc)
    return ts < cutoff


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--max-pages", type=int, default=50, help="listing pages (20 jobs/page; 0=all)")
    ap.add_argument(
        "--max-days-old", type=int, default=30, help="skip postings older than this (0 = no limit)"
    )
    ap.add_argument("--rows-per-file", type=int, default=10_000)
    ap.add_argument("--sleep", type=float, default=0.3, help="seconds between calls (be polite)")
    args = ap.parse_args()

    api_key = os.environ.get("THEMUSE_API_KEY")  # optional

    cutoff = None
    if args.max_days_old > 0:
        cutoff = dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=args.max_days_old)

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
        path = out / f"themuse-{file_idx:04d}.parquet"
        pq.write_table(table, path)
        n_written += len(buf)
        print(f"  wrote {path.name}: {len(buf):,} rows (total {n_written:,})", flush=True)
        file_idx += 1
        buf = []

    page = 1
    page_count = None
    while args.max_pages == 0 or page <= args.max_pages:
        params = {"page": page}
        if api_key:
            params["api_key"] = api_key
        payload = _get(f"{API}?{urlencode(params)}")
        time.sleep(args.sleep)
        if payload is None:
            break
        page_count = int(payload.get("page_count", 0) or 0)
        results = payload.get("results") or []
        if not results:
            break
        for job in results:
            if _too_old(job.get("publication_date"), cutoff):
                continue
            row = transform(job)
            if not row["title"] or row["id"] in seen_ids:
                continue
            seen_ids.add(row["id"])
            buf.append(row)
            if len(buf) >= args.rows_per_file:
                flush()
        page += 1
        if page_count and page > page_count:
            break

    flush()
    print(
        f"done: {n_written:,} unique postings across {file_idx} parquet files -> {out}",
        flush=True,
    )


if __name__ == "__main__":
    main()
