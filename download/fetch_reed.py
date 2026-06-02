#!/usr/bin/env python3
"""Fetch currently-posted jobs from the Reed.co.uk API.

Reed is the UK's largest job board; its public API exposes the live national job
inventory (~100k postings), so it brings UK / non-US inventory that the OpenApply
crawler (Greenhouse/Lever/Ashby), USAJOBS (federal), Adzuna, SmartRecruiters and
Workable do not cover. The search response carries only a truncated description,
so we fetch the per-job detail for the full HTML jobDescription.

Requires a free API key (https://www.reed.co.uk/developers). Auth is HTTP Basic
with the key as the username and an empty password:
  REED_API_KEY='...'
  list:   GET https://www.reed.co.uk/api/1.0/search?resultsToTake=100&resultsToSkip=N
  detail: GET https://www.reed.co.uk/api/1.0/jobs/{jobId}   (full HTML jobDescription)

Writes parquet files whose schema matches download/fetch_adzuna.py / what
prep_open_apply.py expects (id, source_slug, title, description_html, department,
employment_type, remote, locations, salary_min, salary_max, salary_currency,
posted_at, source).

Usage:
  export REED_API_KEY='...'
  .venv/bin/python download/fetch_reed.py \\
      --out-dir jobs_data_reed/raw \\
      --max-pages 20 --max-days-old 30
"""

import argparse
import base64
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

API_BASE = "https://www.reed.co.uk/api/1.0"
PAGE = 100  # API max resultsToTake
SLUG_RE = re.compile(r"[^a-z0-9]+")
TAG_RE = re.compile(r"<[^>]+>")


def slugify(s: str) -> str:
    return SLUG_RE.sub("-", (s or "").lower()).strip("-")


def strip_tags(s: str) -> str:
    return TAG_RE.sub("", s or "").strip()


def _get(url: str, auth: str, retries: int = 4) -> Any:
    req = Request(
        url,
        headers={
            "Authorization": f"Basic {auth}",
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
            # 404 = job pulled since the search page; not fixed by retrying.
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


def _iso_date(ddmmyyyy: str | None) -> str | None:
    # Reed dates are dd/MM/yyyy; normalise to ISO so posted_at sorts like the
    # other sources.
    if not ddmmyyyy:
        return None
    try:
        return dt.datetime.strptime(ddmmyyyy.strip(), "%d/%m/%Y").date().isoformat()
    except ValueError:
        return None


def transform(item: dict, detail: dict | None) -> dict[str, Any]:
    d = detail or item  # detail is richer; fall back to the search row
    job_id = str(item.get("jobId") or d.get("jobId") or "")
    company = d.get("employerName") or item.get("employerName") or ""
    title = strip_tags(d.get("jobTitle") or item.get("jobTitle") or "")

    loc = d.get("locationName") or item.get("locationName") or ""
    locations = [loc] if loc else []

    hay = f"{title} {loc}".lower()
    remote = True if ("remote" in hay or "work from home" in hay or "wfh" in hay) else None

    bits = []
    if d.get("fullTime"):
        bits.append("Full-time")
    elif d.get("partTime"):
        bits.append("Part-time")
    if d.get("contractType"):
        bits.append(d["contractType"])
    employment_type = " / ".join(bits) or None

    def as_float(v):
        try:
            return float(v) if v is not None else None
        except (TypeError, ValueError):
            return None

    salary_min = as_float(d.get("minimumSalary"))
    salary_max = as_float(d.get("maximumSalary"))
    # Reed salaries are GBP; the API only sets currency on some rows.
    salary_currency = (d.get("currency") or "GBP") if (salary_min or salary_max) else None

    return {
        "id": f"reed:{job_id}",
        "source_slug": slugify(company) or "reed",
        "title": title,
        "description_html": d.get("jobDescription") or item.get("jobDescription") or "",
        "department": None,  # Reed has no category field
        "employment_type": employment_type,
        "remote": remote,
        "locations": locations,
        "salary_min": salary_min,
        "salary_max": salary_max,
        "salary_currency": salary_currency,
        "posted_at": _iso_date(d.get("datePosted") or item.get("date")),
        "source": "reed",
    }


def _too_old(item: dict, cutoff: dt.date | None) -> bool:
    if cutoff is None:
        return False
    iso = _iso_date(item.get("date"))
    if not iso:
        return False
    return dt.date.fromisoformat(iso) < cutoff


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--keywords", default="", help="optional search keywords (default: all jobs)")
    ap.add_argument("--location", default="", help="optional locationName filter")
    ap.add_argument("--max-pages", type=int, default=20, help="search pages (100 jobs/page; 0=all)")
    ap.add_argument(
        "--max-days-old", type=int, default=30, help="skip postings older than this (0 = no limit)"
    )
    ap.add_argument(
        "--no-detail",
        action="store_true",
        help="skip the per-job detail fetch (faster, but only truncated descriptions)",
    )
    ap.add_argument("--rows-per-file", type=int, default=10_000)
    ap.add_argument("--sleep", type=float, default=0.2, help="seconds between calls (be polite)")
    args = ap.parse_args()

    key = os.environ.get("REED_API_KEY")
    if not key:
        sys.exit("ERROR: set REED_API_KEY env var")
    auth = base64.b64encode(f"{key}:".encode()).decode()

    cutoff = None
    if args.max_days_old > 0:
        cutoff = dt.date.today() - dt.timedelta(days=args.max_days_old)

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
        path = out / f"reed-{file_idx:04d}.parquet"
        pq.write_table(table, path)
        n_written += len(buf)
        print(f"  wrote {path.name}: {len(buf):,} rows (total {n_written:,})", flush=True)
        file_idx += 1
        buf = []

    skip = 0
    page = 0
    total = None
    while args.max_pages == 0 or page < args.max_pages:
        params = {"resultsToTake": PAGE, "resultsToSkip": skip}
        if args.keywords:
            params["keywords"] = args.keywords
        if args.location:
            params["locationName"] = args.location
        payload = _get(f"{API_BASE}/search?{urlencode(params)}", auth)
        time.sleep(args.sleep)
        if payload is None:
            break
        total = int(payload.get("totalResults", 0) or 0)
        results = payload.get("results") or []
        if not results:
            break
        for item in results:
            if _too_old(item, cutoff):
                continue
            detail = None
            if not args.no_detail:
                jid = str(item.get("jobId") or "")
                if jid:
                    detail = _get(f"{API_BASE}/jobs/{jid}", auth)
                    time.sleep(args.sleep)
            row = transform(item, detail)
            if not row["title"] or row["id"] in seen_ids:
                continue
            seen_ids.add(row["id"])
            buf.append(row)
            if len(buf) >= args.rows_per_file:
                flush()
        skip += PAGE
        page += 1
        if total is not None and skip >= total:
            break

    flush()
    print(
        f"done: {n_written:,} unique postings across {file_idx} parquet files -> {out}",
        flush=True,
    )


if __name__ == "__main__":
    main()
