#!/usr/bin/env python3
"""Fetch currently-posted jobs from the Careerjet Partner API (v4).

Careerjet is a global job-search aggregator, so it brings broad aggregator
inventory the OpenApply crawler (Greenhouse/Lever/Ashby), USAJOBS (federal) and
the ATS adapters do not cover. It has no "list everything" endpoint, so this
adapter sweeps a list of broad occupation keywords (see
jobs_search_demo/job_search_queries.txt) to approximate coverage.

NOTE: Careerjet only returns a short *snippet* per posting (~250 chars, no full
description and no detail endpoint), so its docs are thinner than the ATS / Reed /
The Muse / Findwork sources. It is also IP-pinned (see below), so it only works
for the LOCAL refresh model, not from an HF Space egress.

Requires a Publisher API key plus three access conditions, ALL mandatory:
  CAREERJET_API_KEY='...'
  endpoint: GET https://search.api.careerjet.net/v4/query   (Basic auth, key as user)
  (1) the egress IP must be whitelisted in the partner dashboard,
  (2) a valid public `user_ip` query param (--user-ip, defaults to CAREERJET_USER_IP),
  (3) a `Referer` header matching the declared domain (--referer, default bagofdocs.com).

Writes parquet files whose schema matches download/fetch_adzuna.py / what
prep_open_apply.py expects (id, source_slug, title, description_html, department,
employment_type, remote, locations, salary_min, salary_max, salary_currency,
posted_at, source).

Usage:
  export CAREERJET_API_KEY='...'
  .venv/bin/python download/fetch_careerjet.py \\
      --out-dir jobs_data_careerjet/raw \\
      --user-ip 172.10.233.112 --referer https://bagofdocs.com \\
      --location "" --max-pages 5
"""

import argparse
import base64
import json
import os
import re
import sys
import time
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pyarrow as pa
import pyarrow.parquet as pq

API = "https://search.api.careerjet.net/v4/query"
SLUG_RE = re.compile(r"[^a-z0-9]+")
TAG_RE = re.compile(r"<[^>]+>")
DEFAULT_QUERIES = ("software engineer", "nurse", "sales", "accountant", "teacher")
SALARY_CURRENCY = {  # Careerjet uses symbols/codes; map the common ones we see
    "USD": "USD",
    "GBP": "GBP",
    "EUR": "EUR",
    "CAD": "CAD",
    "AUD": "AUD",
}


def slugify(s: str) -> str:
    return SLUG_RE.sub("-", (s or "").lower()).strip("-")


def strip_tags(s: str) -> str:
    return TAG_RE.sub("", s or "").strip()


def _get(url: str, headers: dict, retries: int = 4) -> Any:
    backoff = 2
    for attempt in range(retries):
        req = Request(url, headers=headers)
        try:
            with urlopen(req, timeout=60) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except (HTTPError, URLError, TimeoutError) as e:
            if isinstance(e, HTTPError) and 400 <= e.code < 500 and e.code != 429:
                raise
            if attempt == retries - 1:
                raise
            sys.stderr.write(
                f"  {url[:80]} attempt {attempt + 1} failed ({e}); sleeping {backoff}s\n"
            )
            time.sleep(backoff)
            backoff *= 2
    raise RuntimeError("unreachable")


def _iso(rfc822: str | None) -> str | None:
    # Careerjet dates are RFC 822 ("Tue, 02 Jun 2026 01:22:23 GMT").
    if not rfc822:
        return None
    try:
        return parsedate_to_datetime(rfc822).isoformat()
    except (TypeError, ValueError):
        return None


def transform(job: dict) -> dict[str, Any]:
    # Careerjet has no stable id; derive a stable one from the apply url.
    url = job.get("url") or ""
    job_id = slugify(url.rsplit("/", 1)[-1]) if url else slugify(job.get("title") or "")
    company = job.get("company") or ""
    title = strip_tags(job.get("title") or "")

    loc = job.get("locations") or ""
    locations = [loc] if loc else []

    hay = f"{title} {loc} {job.get('description') or ''}".lower()
    remote = True if ("remote" in hay or "work from home" in hay) else None

    def as_float(v):
        try:
            return float(v) if v is not None else None
        except (TypeError, ValueError):
            return None

    salary_min = as_float(job.get("salary_min"))
    salary_max = as_float(job.get("salary_max"))
    cc = (job.get("salary_currency_code") or "").upper()
    salary_currency = SALARY_CURRENCY.get(cc, cc or None) if (salary_min or salary_max) else None

    return {
        "id": f"careerjet:{job_id}",
        "source_slug": slugify(company) or "careerjet",
        "title": title,
        # description is a short snippet; prep strips HTML downstream.
        "description_html": job.get("description") or "",
        "department": None,
        "employment_type": None,  # not exposed
        "remote": remote,
        "locations": locations,
        "salary_min": salary_min,
        "salary_max": salary_max,
        "salary_currency": salary_currency,
        "posted_at": _iso(job.get("date")),
        "source": "careerjet",
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
    ap.add_argument("--locale-code", default="en_US", help="Careerjet locale (e.g. en_US, en_GB)")
    ap.add_argument(
        "--user-ip",
        default=os.environ.get("CAREERJET_USER_IP", ""),
        help="public IP for the user_ip param (must be a real public IP)",
    )
    ap.add_argument("--referer", default="https://bagofdocs.com", help="declared Referer domain")
    ap.add_argument("--max-pages", type=int, default=5, help="result pages per query")
    ap.add_argument("--pagesize", type=int, default=99, help="results per page (API max 99)")
    ap.add_argument("--rows-per-file", type=int, default=10_000)
    ap.add_argument("--sleep", type=float, default=0.3, help="seconds between calls (be polite)")
    args = ap.parse_args()

    key = os.environ.get("CAREERJET_API_KEY")
    if not key:
        sys.exit("ERROR: set CAREERJET_API_KEY env var")
    if not args.user_ip:
        sys.exit("ERROR: --user-ip (or CAREERJET_USER_IP) is required and must be a public IP")

    auth = base64.b64encode(f"{key}:".encode()).decode()
    headers = {
        "Authorization": f"Basic {auth}",
        "Referer": args.referer,
        "User-Agent": "bagofdocs-jobs-refresh",
        "Accept": "application/json",
    }

    qpath = Path(args.queries_file)
    if qpath.exists():
        queries = [
            ln.strip()
            for ln in qpath.read_text().splitlines()
            if ln.strip() and not ln.lstrip().startswith("#")
        ]
    else:
        print(f"[careerjet] queries file {qpath} missing; using built-in defaults", flush=True)
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
        path = out / f"careerjet-{file_idx:04d}.parquet"
        pq.write_table(table, path)
        n_written += len(buf)
        print(f"  wrote {path.name}: {len(buf):,} rows (total {n_written:,})", flush=True)
        file_idx += 1
        buf = []

    pagesize = min(args.pagesize, 99)
    for q in queries:
        q_rows = 0
        for page in range(1, args.max_pages + 1):
            params = {
                "keywords": q,
                "location": args.location,
                "pagesize": pagesize,
                "page": page,
                "user_ip": args.user_ip,
                "user_agent": "bagofdocs-jobs-refresh",
                "locale_code": args.locale_code,
            }
            payload = _get(f"{API}?{urlencode(params)}", headers)
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
            if page >= int(payload.get("pages", 0) or 0):
                break
        if q_rows:
            print(f"[{q}] collected {q_rows:,} new postings", flush=True)

    flush()
    print(
        f"done: {n_written:,} unique postings across {file_idx} parquet files -> {out}",
        flush=True,
    )


if __name__ == "__main__":
    main()
