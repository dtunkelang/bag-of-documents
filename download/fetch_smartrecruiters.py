#!/usr/bin/env python3
"""Fetch currently-posted jobs from the SmartRecruiters public Posting API.

SmartRecruiters is a mainstream ATS whose live postings are exposed through an
unauthenticated public API (the same data that powers each company's hosted
careers page), so it brings inventory the OpenApply crawler (Greenhouse/Lever/
Ashby), USAJOBS (federal), and Adzuna (aggregator) do not cover. Like the
extra-ATS poller it is slug-driven: you list the company identifiers to poll.

No credentials required. The public endpoints are:
  list:   GET https://api.smartrecruiters.com/v1/companies/{slug}/postings?limit=100&offset=N
  detail: GET https://api.smartrecruiters.com/v1/companies/{slug}/postings/{id}
The list response carries title/location/department/dates; the per-posting detail
carries the HTML job ad (jobAd.sections), so we fetch detail to populate
description_html. A company's identifier is the slug in its careers URL, e.g.
careers.smartrecruiters.com/Square -> "Square".

Writes parquet files whose schema matches download/fetch_adzuna.py / what
prep_open_apply.py expects (id, source_slug, title, description_html, department,
employment_type, remote, locations, salary_min, salary_max, salary_currency,
posted_at, source).

Usage:
  .venv/bin/python download/fetch_smartrecruiters.py \\
      --out-dir jobs_data_smartrecruiters/raw \\
      --slugs-file jobs_search_demo/smartrecruiters_slugs.txt \\
      --max-days-old 30
"""

import argparse
import datetime as dt
import json
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

API_BASE = "https://api.smartrecruiters.com/v1/companies"
SLUG_RE = re.compile(r"[^a-z0-9]+")
TAG_RE = re.compile(r"<[^>]+>")
# jobAd.sections is an ordered-ish dict; render in the order a reader expects.
SECTION_ORDER = ("companyDescription", "jobDescription", "qualifications", "additionalInformation")


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
            # 404 = unknown/closed company slug; 403 = postings not public. Neither
            # is fixed by retrying, so surface as None and let the caller skip.
            if isinstance(e, HTTPError) and e.code in (403, 404):
                return None
            if isinstance(e, HTTPError) and 400 <= e.code < 500:
                raise
            if attempt == retries - 1:
                raise
            sys.stderr.write(f"  {url} attempt {attempt + 1} failed ({e}); sleeping {backoff}s\n")
            time.sleep(backoff)
            backoff *= 2
    raise RuntimeError("unreachable")


def _build_description_html(detail: dict) -> str:
    sections = ((detail.get("jobAd") or {}).get("sections")) or {}
    if not isinstance(sections, dict):
        return ""
    keys = [k for k in SECTION_ORDER if k in sections] + [
        k for k in sections if k not in SECTION_ORDER
    ]
    parts: list[str] = []
    for k in keys:
        sec = sections.get(k) or {}
        text = sec.get("text") or ""
        if not text:
            continue
        title = sec.get("title")
        if title:
            parts.append(f"<h3>{title}</h3>")
        parts.append(text)
    return "\n".join(parts)


def transform(posting: dict, detail: dict | None, slug: str) -> dict[str, Any]:
    posting_id = str(posting.get("id") or posting.get("uuid") or "")
    title = strip_tags(posting.get("name") or "")

    loc = posting.get("location") or {}
    # SmartRecruiters location is {city, region, country, remote}; mirror Adzuna by
    # dropping the country token and keeping the finer-grained parts.
    locations = [v for v in (loc.get("city"), loc.get("region")) if v]
    remote = True if loc.get("remote") else None

    dept = (
        (posting.get("department") or {}).get("label")
        or (posting.get("function") or {}).get("label")
        or None
    )
    employment_type = (posting.get("typeOfEmployment") or {}).get("label") or None

    return {
        "id": f"smartrecruiters:{slug}:{posting_id}",
        "source_slug": slugify(slug),
        "title": title,
        "description_html": _build_description_html(detail) if detail else "",
        "department": dept,
        "employment_type": employment_type,
        "remote": remote,
        "locations": locations,
        "salary_min": None,  # not exposed on public postings
        "salary_max": None,
        "salary_currency": None,
        "posted_at": posting.get("releasedDate") or None,  # ISO 8601
        "source": "smartrecruiters",
    }


def _too_old(released: str | None, cutoff: dt.datetime | None) -> bool:
    if cutoff is None or not released:
        return False
    try:
        ts = dt.datetime.fromisoformat(released.replace("Z", "+00:00"))
    except ValueError:
        return False
    return ts < cutoff


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument(
        "--slugs-file",
        default="jobs_search_demo/smartrecruiters_slugs.txt",
        help="one SmartRecruiters company identifier per line (# comments allowed)",
    )
    ap.add_argument("--limit", type=int, default=100, help="postings per list page (max 100)")
    ap.add_argument(
        "--max-days-old", type=int, default=30, help="skip postings older than this (0 = no limit)"
    )
    ap.add_argument("--max-postings-per-company", type=int, default=0, help="0 = all open postings")
    ap.add_argument(
        "--no-detail",
        action="store_true",
        help="skip the per-posting detail fetch (faster, but no description_html)",
    )
    ap.add_argument("--rows-per-file", type=int, default=10_000)
    ap.add_argument("--sleep", type=float, default=0.2, help="seconds between calls (be polite)")
    args = ap.parse_args()

    slugs_path = Path(args.slugs_file)
    if not slugs_path.exists():
        sys.exit(f"ERROR: slugs file not found: {slugs_path}")
    slugs = [
        ln.strip()
        for ln in slugs_path.read_text().splitlines()
        if ln.strip() and not ln.lstrip().startswith("#")
    ]
    if not slugs:
        sys.exit(f"ERROR: no company slugs in {slugs_path}")

    cutoff = None
    if args.max_days_old > 0:
        cutoff = dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=args.max_days_old)

    limit = min(args.limit, 100)
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
        path = out / f"smartrecruiters-{file_idx:04d}.parquet"
        pq.write_table(table, path)
        n_written += len(buf)
        print(f"  wrote {path.name}: {len(buf):,} rows (total {n_written:,})", flush=True)
        file_idx += 1
        buf = []

    for slug in slugs:
        offset = 0
        company_rows = 0
        while True:
            qs = urlencode({"limit": limit, "offset": offset})
            payload = _get(f"{API_BASE}/{slug}/postings?{qs}")
            time.sleep(args.sleep)
            if payload is None:
                print(f"[{slug}] no public postings (404/403); skipping", flush=True)
                break
            content = payload.get("content") or []
            if not content:
                break
            for posting in content:
                if _too_old(posting.get("releasedDate"), cutoff):
                    continue
                detail = None
                if not args.no_detail:
                    pid = str(posting.get("id") or "")
                    if pid:
                        detail = _get(f"{API_BASE}/{slug}/postings/{pid}")
                        time.sleep(args.sleep)
                row = transform(posting, detail, slug)
                if not row["title"] or row["id"] in seen_ids:
                    continue
                seen_ids.add(row["id"])
                buf.append(row)
                company_rows += 1
                if len(buf) >= args.rows_per_file:
                    flush()
                if args.max_postings_per_company and company_rows >= args.max_postings_per_company:
                    break
            offset += limit
            total_found = int(payload.get("totalFound", 0) or 0)
            if offset >= total_found or (
                args.max_postings_per_company and company_rows >= args.max_postings_per_company
            ):
                break
        if company_rows:
            print(f"[{slug}] collected {company_rows:,} postings", flush=True)

    flush()
    print(
        f"done: {n_written:,} unique postings across {file_idx} parquet files -> {out}",
        flush=True,
    )


if __name__ == "__main__":
    main()
