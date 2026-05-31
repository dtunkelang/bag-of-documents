#!/usr/bin/env python3
"""Fetch currently-posted jobs from the Adzuna Search API.

Adzuna aggregates job-board, recruiter, and non-ATS employer listings, so it
brings inventory the OpenApply crawler (which polls Greenhouse/Lever/Ashby across
its committed slug lists) and USAJOBS (federal only) do not cover.

Writes parquet files whose schema matches what download/prep_open_apply.py expects
(id, source_slug, title, description_html, department, employment_type, remote,
locations, salary_min, salary_max, salary_currency, posted_at, source).

Recency-first by design: results are pulled with sort_by=date and bounded by
--max-days-old, so each run captures freshly-posted jobs and the stable per-posting
ids keep the downstream content-addressed delta encode + Xet dedup cheap.

Usage:
  export ADZUNA_APP_ID='...'
  export ADZUNA_APP_KEY='...'
  .venv/bin/python download/fetch_adzuna.py \\
      --out-dir jobs_data_adzuna/raw \\
      --countries us --max-pages 20 --max-days-old 7

Register for a free key at https://developer.adzuna.com/ (free tier: small daily
call quota + results_per_page<=50, so keep --max-pages modest per country).
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

API_BASE = "https://api.adzuna.com/v1/api/jobs"
SLUG_RE = re.compile(r"[^a-z0-9]+")
TAG_RE = re.compile(r"<[^>]+>")

# Adzuna runs one index per country; map each to its salary currency so the
# salary facet stays meaningful in a mixed-country corpus.
COUNTRY_CURRENCY = {
    "us": "USD",
    "gb": "GBP",
    "au": "AUD",
    "ca": "CAD",
    "de": "EUR",
    "fr": "EUR",
    "nl": "EUR",
    "at": "EUR",
    "it": "EUR",
    "es": "EUR",
    "pl": "PLN",
    "br": "BRL",
    "in": "INR",
    "mx": "MXN",
    "nz": "NZD",
    "sg": "SGD",
    "za": "ZAR",
    "ch": "CHF",
}


def slugify(s: str) -> str:
    return SLUG_RE.sub("-", (s or "").lower()).strip("-")


def strip_tags(s: str) -> str:
    # Adzuna wraps matched query terms in <strong> and the like; titles are used
    # as-is downstream (prep only strips HTML from the description), so clean here.
    return TAG_RE.sub("", s or "").strip()


def fetch_page(
    country: str,
    app_id: str,
    app_key: str,
    page: int,
    per_page: int,
    max_days_old: int,
    retries: int = 4,
) -> dict:
    params = {
        "app_id": app_id,
        "app_key": app_key,
        "results_per_page": per_page,
        "sort_by": "date",  # freshest first -> currently-posted jobs
        "content-type": "application/json",
    }
    if max_days_old > 0:
        params["max_days_old"] = max_days_old
    url = f"{API_BASE}/{country}/search/{page}?{urlencode(params)}"
    req = Request(url, headers={"User-Agent": "bagofdocs-jobs-refresh"})
    backoff = 2
    for attempt in range(retries):
        try:
            with urlopen(req, timeout=60) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except (HTTPError, URLError, TimeoutError) as e:
            # 4xx (bad key, exhausted quota) won't fix itself by retrying.
            if isinstance(e, HTTPError) and 400 <= e.code < 500:
                raise
            if attempt == retries - 1:
                raise
            sys.stderr.write(
                f"  {country} page {page} attempt {attempt + 1} failed ({e}); sleeping {backoff}s\n"
            )
            time.sleep(backoff)
            backoff *= 2
    raise RuntimeError("unreachable")


def transform(item: dict, country: str) -> dict[str, Any]:
    job_id = str(item.get("id") or "")
    company = (item.get("company") or {}).get("display_name") or ""
    source_slug = slugify(company) or f"adzuna-{country}"

    title = strip_tags(item.get("title") or "")

    loc = item.get("location") or {}
    # area is hierarchical: [country, region, city, ...]; drop the leading country
    # token so locations read like USAJobs (region/city), fall back to display_name.
    area = [a for a in (loc.get("area") or []) if a]
    locations = (
        area[1:] if len(area) > 1 else ([loc["display_name"]] if loc.get("display_name") else [])
    )

    # Adzuna has no explicit remote flag; infer from title/location like fetch_usajobs.
    hay = f"{title} {loc.get('display_name') or ''}".lower()
    remote = True if ("remote" in hay or "work from home" in hay or "anywhere" in hay) else None

    # contract_time (full_time/part_time) + contract_type (permanent/contract)
    bits = []
    ct_time = item.get("contract_time")
    ct_type = item.get("contract_type")
    if ct_time:
        bits.append({"full_time": "Full-time", "part_time": "Part-time"}.get(ct_time, ct_time))
    if ct_type:
        bits.append({"permanent": "Permanent", "contract": "Contract"}.get(ct_type, ct_type))
    employment_type = " / ".join(bits) or None

    def as_float(v):
        try:
            return float(v) if v is not None else None
        except (TypeError, ValueError):
            return None

    # salary_is_predicted == "1" means Adzuna estimated it, not employer-stated.
    predicted = str(item.get("salary_is_predicted") or "0") == "1"
    salary_min = None if predicted else as_float(item.get("salary_min"))
    salary_max = None if predicted else as_float(item.get("salary_max"))
    salary_currency = COUNTRY_CURRENCY.get(country) if (salary_min or salary_max) else None

    category = (item.get("category") or {}).get("label") or None

    return {
        "id": f"adzuna:{country}:{job_id}",
        "source_slug": source_slug,
        "title": title,
        "description_html": item.get("description") or "",
        "department": category,  # nearest categorical analog (e.g. "IT Jobs")
        "employment_type": employment_type,
        "remote": remote,
        "locations": locations,
        "salary_min": salary_min,
        "salary_max": salary_max,
        "salary_currency": salary_currency,
        "posted_at": item.get("created") or None,  # ISO 8601
        "source": "adzuna",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument(
        "--countries",
        default="us",
        help="comma-separated Adzuna country codes (us,gb,ca,...)",
    )
    ap.add_argument("--results-per-page", type=int, default=50, help="Adzuna max is 50")
    ap.add_argument("--max-pages", type=int, default=20, help="pages per country (0 = until empty)")
    ap.add_argument(
        "--max-days-old", type=int, default=7, help="only postings newer than this (0 = no limit)"
    )
    ap.add_argument("--rows-per-file", type=int, default=10_000)
    ap.add_argument("--sleep", type=float, default=0.3, help="seconds between calls (be polite)")
    args = ap.parse_args()

    app_id = os.environ.get("ADZUNA_APP_ID")
    app_key = os.environ.get("ADZUNA_APP_KEY")
    if not app_id or not app_key:
        sys.exit("ERROR: set ADZUNA_APP_ID and ADZUNA_APP_KEY env vars")

    per_page = min(args.results_per_page, 50)
    countries = [c.strip().lower() for c in args.countries.split(",") if c.strip()]

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
        path = out / f"adzuna-{file_idx:04d}.parquet"
        pq.write_table(table, path)
        n_written += len(buf)
        print(f"  wrote {path.name}: {len(buf):,} rows (total {n_written:,})", flush=True)
        file_idx += 1
        buf = []

    for country in countries:
        first = fetch_page(country, app_id, app_key, 1, per_page, args.max_days_old)
        total = int(first.get("count", 0) or 0)
        print(f"[{country}] count (<= {args.max_days_old}d old): {total:,}", flush=True)

        max_pages = args.max_pages or (total + per_page - 1) // per_page
        page = 1
        payload = first
        country_rows = 0
        while page <= max_pages:
            if payload is None:
                payload = fetch_page(country, app_id, app_key, page, per_page, args.max_days_old)
                time.sleep(args.sleep)
            results = payload.get("results") or []
            if not results:
                break
            for item in results:
                row = transform(item, country)
                if not row["title"] or row["id"] in seen_ids:
                    continue
                seen_ids.add(row["id"])
                buf.append(row)
                country_rows += 1
            if len(buf) >= args.rows_per_file:
                flush()
            payload = None
            page += 1
        print(f"[{country}] collected {country_rows:,} unique postings", flush=True)

    flush()
    print(
        f"done: {n_written:,} unique postings across {file_idx} parquet files -> {out}",
        flush=True,
    )


if __name__ == "__main__":
    main()
