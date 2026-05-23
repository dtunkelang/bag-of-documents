#!/usr/bin/env python3
"""Fetch open federal job postings from the USAJOBS Search API.

Writes parquet files whose schema matches what download/prep_open_apply.py expects
(id, source_slug, title, description_html, department, employment_type, remote,
locations, salary_min, salary_max, salary_currency, posted_at, source).

Usage:
  export USAJOBS_EMAIL='you@example.com'
  export USAJOBS_API_KEY='...'
  .venv/bin/python download/fetch_usajobs.py \\
      --out-dir jobs_data_usajobs/raw \\
      --results-per-page 500

Register for a free API key at https://developer.usajobs.gov/APIRequest/Index
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
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pyarrow as pa
import pyarrow.parquet as pq

API_HOST = "data.usajobs.gov"
API_URL = f"https://{API_HOST}/api/search"
SLUG_RE = re.compile(r"[^a-z0-9]+")


def slugify(s: str) -> str:
    return SLUG_RE.sub("-", (s or "").lower()).strip("-")


def get_first(d: dict, key: str, default=None):
    v = d.get(key)
    if isinstance(v, list):
        return v[0] if v else default
    return v if v is not None else default


def fetch_page(email: str, api_key: str, page: int, per_page: int, retries: int = 4) -> dict:
    params = {"ResultsPerPage": per_page, "Page": page}
    url = f"{API_URL}?{urlencode(params)}"
    req = Request(
        url,
        headers={
            "Host": API_HOST,
            "User-Agent": email,
            "Authorization-Key": api_key,
        },
    )
    backoff = 2
    for attempt in range(retries):
        try:
            with urlopen(req, timeout=60) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except (HTTPError, URLError, TimeoutError) as e:
            if attempt == retries - 1:
                raise
            sys.stderr.write(
                f"  page {page} attempt {attempt + 1} failed ({e}); sleeping {backoff}s\n"
            )
            time.sleep(backoff)
            backoff *= 2
    raise RuntimeError("unreachable")


def transform(item: dict) -> dict[str, Any]:
    desc = item.get("MatchedObjectDescriptor", {}) or {}
    pos_id = item.get("MatchedObjectId") or desc.get("PositionID") or ""

    dept = desc.get("DepartmentName") or ""
    org = desc.get("OrganizationName") or ""
    source_slug = slugify(org or dept) or "usajobs"

    title = (desc.get("PositionTitle") or "").strip()

    user_area = desc.get("UserArea", {}) or {}
    details = user_area.get("Details", {}) or {}

    def stringify(v) -> str:
        if v is None:
            return ""
        if isinstance(v, list):
            return "\n".join(stringify(x) for x in v if x)
        return str(v)

    # USAJOBS returns these as HTML-bearing strings (or lists of strings for MajorDuties);
    # prep_open_apply will strip tags later.
    parts = [
        stringify(desc.get("QualificationSummary")),
        stringify(details.get("JobSummary")),
        stringify(details.get("MajorDuties")),
        stringify(details.get("Requirements")),
        stringify(details.get("Education")),
    ]
    description_html = "\n\n".join(p for p in parts if p).strip()

    schedule = get_first(desc, "PositionSchedule", {}) or {}
    employment_type = schedule.get("Name") if isinstance(schedule, dict) else None

    locations_list = desc.get("PositionLocation") or []
    locations = []
    for loc in locations_list:
        name = loc.get("LocationName") if isinstance(loc, dict) else None
        if name:
            locations.append(name)

    remote = None
    loc_display = (desc.get("PositionLocationDisplay") or "").lower()
    if (
        any(
            "remote" in (loc or "").lower() or "anywhere" in (loc or "").lower()
            for loc in locations
        )
        or "remote" in loc_display
        or "anywhere" in loc_display
    ):
        remote = True

    salary_min = None
    salary_max = None
    salary_currency = None
    rem = get_first(desc, "PositionRemuneration", {}) or {}
    if isinstance(rem, dict):
        try:
            if rem.get("MinimumRange"):
                salary_min = float(rem["MinimumRange"])
            if rem.get("MaximumRange"):
                salary_max = float(rem["MaximumRange"])
        except (TypeError, ValueError):
            pass
        salary_currency = rem.get("RateIntervalCode") and "USD" or None

    posted_at = desc.get("PublicationStartDate") or None

    return {
        "id": f"usajobs:{source_slug}:{pos_id}",
        "source_slug": source_slug,
        "title": title,
        "description_html": description_html,
        "department": dept,
        "employment_type": employment_type,
        "remote": remote,
        "locations": locations,
        "salary_min": salary_min,
        "salary_max": salary_max,
        "salary_currency": salary_currency,
        "posted_at": posted_at,
        "source": "usajobs",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--results-per-page", type=int, default=500)
    ap.add_argument("--max-pages", type=int, default=0, help="0 = all pages")
    ap.add_argument("--rows-per-file", type=int, default=10_000)
    args = ap.parse_args()

    email = os.environ.get("USAJOBS_EMAIL")
    api_key = os.environ.get("USAJOBS_API_KEY")
    if not email or not api_key:
        sys.exit("ERROR: set USAJOBS_EMAIL and USAJOBS_API_KEY env vars")

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # First page to discover total count
    first = fetch_page(email, api_key, page=1, per_page=args.results_per_page)
    sr = first.get("SearchResult", {}) or {}
    total = int(sr.get("SearchResultCountAll", 0) or 0)
    per_page = int(sr.get("SearchResultCount", 0) or args.results_per_page)
    print(f"USAJOBS total open postings: {total:,}; per_page: {per_page}", flush=True)

    if total == 0:
        sys.exit("no results returned; check credentials")

    total_pages = (total + per_page - 1) // per_page
    if args.max_pages and args.max_pages < total_pages:
        total_pages = args.max_pages
    print(f"fetching {total_pages} pages", flush=True)

    buf: list[dict] = []
    file_idx = 0
    n_written = 0
    seen_ids: set[str] = set()

    def flush():
        nonlocal buf, file_idx, n_written
        if not buf:
            return
        table = pa.Table.from_pylist(buf)
        path = out / f"usajobs-{file_idx:04d}.parquet"
        pq.write_table(table, path)
        n_written += len(buf)
        print(f"  wrote {path.name}: {len(buf):,} rows (total {n_written:,})", flush=True)
        file_idx += 1
        buf = []

    pages = [(1, first)]
    for page in range(2, total_pages + 1):
        pages.append((page, None))

    for page, payload in pages:
        if payload is None:
            payload = fetch_page(email, api_key, page=page, per_page=args.results_per_page)
            time.sleep(0.2)  # be polite
        items = (payload.get("SearchResult", {}) or {}).get("SearchResultItems", []) or []
        for item in items:
            row = transform(item)
            if not row["title"] or row["id"] in seen_ids:
                continue
            seen_ids.add(row["id"])
            buf.append(row)
        if len(buf) >= args.rows_per_file:
            flush()

    flush()
    print(f"done: {n_written:,} unique postings across {file_idx} parquet files", flush=True)


if __name__ == "__main__":
    main()
