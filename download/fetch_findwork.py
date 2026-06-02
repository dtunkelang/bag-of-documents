#!/usr/bin/env python3
"""Fetch currently-posted jobs from the Findwork.dev API.

Findwork aggregates (mostly remote / tech) postings with full HTML descriptions,
so it brings remote-first inventory the OpenApply crawler (Greenhouse/Lever/
Ashby), USAJOBS (federal), Adzuna, SmartRecruiters and Workable under-cover. The
listing response already carries the full description (text), so unlike the ATS
adapters this needs no per-posting detail fetch.

Requires a free API key (https://findwork.dev/developers/). Auth is a custom
token header (NOT Bearer):
  FINDWORK_API_KEY='...'
  list: GET https://findwork.dev/api/jobs/?sort_by=date   (paginate via the "next" url)
        header  Authorization: Token {KEY}

Writes parquet files whose schema matches download/fetch_adzuna.py / what
prep_open_apply.py expects (id, source_slug, title, description_html, department,
employment_type, remote, locations, salary_min, salary_max, salary_currency,
posted_at, source).

Usage:
  export FINDWORK_API_KEY='...'
  .venv/bin/python download/fetch_findwork.py \\
      --out-dir jobs_data_findwork/raw \\
      --max-pages 20 --max-days-old 30
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

API = "https://findwork.dev/api/jobs/"
SLUG_RE = re.compile(r"[^a-z0-9]+")
TAG_RE = re.compile(r"<[^>]+>")


def slugify(s: str) -> str:
    return SLUG_RE.sub("-", (s or "").lower()).strip("-")


def strip_tags(s: str) -> str:
    return TAG_RE.sub("", s or "").strip()


def _get(url: str, key: str, retries: int = 4) -> Any:
    req = Request(
        url,
        headers={
            "Authorization": f"Token {key}",  # Findwork uses Token, not Bearer
            "User-Agent": "bagofdocs-jobs-refresh",
            "Accept": "application/json",
        },
    )
    backoff = 2
    for attempt in range(retries):
        try:
            with urlopen(req, timeout=60) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except (HTTPError, URLError, TimeoutError) as e:
            if isinstance(e, HTTPError) and e.code == 404:
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
    company = job.get("company_name") or ""
    title = strip_tags(job.get("role") or "")

    loc = job.get("location") or ""
    locations = [loc] if loc else []
    remote = True if job.get("remote") else None

    et = (job.get("employment_type") or "").strip()
    employment_type = {
        "full time": "Full-time",
        "full-time": "Full-time",
        "part time": "Part-time",
        "part-time": "Part-time",
        "contract": "Contract",
        "internship": "Internship",
    }.get(et.lower(), et) or None

    return {
        "id": f"findwork:{job_id}",
        "source_slug": slugify(company) or "findwork",
        "title": title,
        "description_html": job.get("text") or "",
        "department": None,
        "employment_type": employment_type,
        "remote": remote,
        "locations": locations,
        "salary_min": None,  # not exposed
        "salary_max": None,
        "salary_currency": None,
        "posted_at": job.get("date_posted") or None,  # ISO 8601
        "source": "findwork",
    }


def _too_old(posted: str | None, cutoff: dt.datetime | None) -> bool:
    if cutoff is None or not posted:
        return False
    try:
        ts = dt.datetime.fromisoformat(posted.replace("Z", "+00:00"))
    except ValueError:
        return False
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=dt.timezone.utc)
    return ts < cutoff


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--search", default="", help="optional keyword filter (default: all jobs)")
    ap.add_argument("--max-pages", type=int, default=20, help="listing pages (0 = until exhausted)")
    ap.add_argument(
        "--max-days-old", type=int, default=30, help="skip postings older than this (0 = no limit)"
    )
    ap.add_argument("--rows-per-file", type=int, default=10_000)
    ap.add_argument("--sleep", type=float, default=0.3, help="seconds between calls (be polite)")
    args = ap.parse_args()

    key = os.environ.get("FINDWORK_API_KEY")
    if not key:
        sys.exit("ERROR: set FINDWORK_API_KEY env var")

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
        path = out / f"findwork-{file_idx:04d}.parquet"
        pq.write_table(table, path)
        n_written += len(buf)
        print(f"  wrote {path.name}: {len(buf):,} rows (total {n_written:,})", flush=True)
        file_idx += 1
        buf = []

    params = {"sort_by": "date"}
    if args.search:
        params["search"] = args.search
    url = f"{API}?{urlencode(params)}"
    page = 0
    while url and (args.max_pages == 0 or page < args.max_pages):
        payload = _get(url, key)
        time.sleep(args.sleep)
        if payload is None:
            break
        results = payload.get("results") or []
        if not results:
            break
        stop = False
        for job in results:
            # sort_by=date is newest-first, so the first too-old row ends the walk.
            if _too_old(job.get("date_posted"), cutoff):
                stop = True
                break
            row = transform(job)
            if not row["title"] or row["id"] in seen_ids:
                continue
            seen_ids.add(row["id"])
            buf.append(row)
            if len(buf) >= args.rows_per_file:
                flush()
        if stop:
            break
        url = payload.get("next")
        page += 1

    flush()
    print(
        f"done: {n_written:,} unique postings across {file_idx} parquet files -> {out}",
        flush=True,
    )


if __name__ == "__main__":
    main()
