#!/usr/bin/env python3
"""Fetch currently-posted jobs from the Workable public "apply" API.

Workable is a mainstream ATS whose live postings are exposed through an
unauthenticated public API (the same JSON that backs each company's hosted
careers page), so it brings inventory the OpenApply crawler (Greenhouse/Lever/
Ashby), USAJOBS (federal), and Adzuna (aggregator) do not cover. Like the
extra-ATS poller it is slug-driven: you list the company subdomains to poll.

No credentials required. The public endpoints are:
  list:   POST https://apply.workable.com/api/v3/accounts/{subdomain}/jobs   (body {} ; paginate with {"token": nextPage})
  detail: GET  https://apply.workable.com/api/v1/accounts/{subdomain}/jobs/{shortcode}   (note: v1)
The list response carries title/location/department; the per-job detail carries
the HTML description (description + requirements + benefits), so we fetch detail
to populate description_html. A company's subdomain is the host in its careers
URL, e.g. apply.workable.com/acme -> "acme".

Writes parquet files whose schema matches download/fetch_adzuna.py / what
prep_open_apply.py expects (id, source_slug, title, description_html, department,
employment_type, remote, locations, salary_min, salary_max, salary_currency,
posted_at, source).

Usage:
  .venv/bin/python download/fetch_workable.py \\
      --out-dir jobs_data_workable/raw \\
      --slugs-file jobs_search_demo/workable_slugs.txt \\
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
from urllib.request import Request, urlopen

import pyarrow as pa
import pyarrow.parquet as pq

API_BASE = "https://apply.workable.com/api/v3/accounts"
DETAIL_BASE = "https://apply.workable.com/api/v1/accounts"  # detail lives on v1, not v3
SLUG_RE = re.compile(r"[^a-z0-9]+")
TAG_RE = re.compile(r"<[^>]+>")


def slugify(s: str) -> str:
    return SLUG_RE.sub("-", (s or "").lower()).strip("-")


def strip_tags(s: str) -> str:
    return TAG_RE.sub("", s or "").strip()


def _request(url: str, body: dict | None, retries: int = 4) -> Any:
    data = json.dumps(body).encode("utf-8") if body is not None else None
    req = Request(
        url,
        data=data,
        method="POST" if body is not None else "GET",
        headers={
            "User-Agent": "bagofdocs-jobs-refresh",
            "Accept": "application/json",
            **({"Content-Type": "application/json"} if body is not None else {}),
        },
    )
    backoff = 2
    for attempt in range(retries):
        try:
            with urlopen(req, timeout=60) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except (HTTPError, URLError, TimeoutError) as e:
            # 404 = unknown subdomain / closed job; 403 = postings not public.
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
    parts: list[str] = []
    for key, heading in (
        ("description", None),
        ("requirements", "Requirements"),
        ("benefits", "Benefits"),
    ):
        text = detail.get(key) or ""
        if not text:
            continue
        if heading:
            parts.append(f"<h3>{heading}</h3>")
        parts.append(text)
    return "\n".join(parts)


def transform(job: dict, detail: dict | None, slug: str) -> dict[str, Any]:
    d = detail or job  # detail is richer; fall back to the list item
    shortcode = str(job.get("shortcode") or d.get("shortcode") or job.get("id") or "")
    title = strip_tags(d.get("title") or job.get("title") or "")

    loc = d.get("location") or job.get("location") or {}
    locations = [v for v in (loc.get("city"), loc.get("region")) if v]
    workplace = (loc.get("workplace") or d.get("workplace") or "").lower()
    # workplace is one of remote / hybrid / on_site; only flag the fully-remote case
    # as True (matches the conservative True/None convention in fetch_adzuna).
    remote = True if workplace == "remote" or d.get("remote") is True else None

    dept = d.get("department") or job.get("department") or None
    if isinstance(dept, list):
        dept = dept[0] if dept else None

    # Workable calls this "type" on both list and detail (e.g. "full", "part", "contract").
    et = d.get("type") or job.get("type") or d.get("employment_type")
    employment_type = {
        "full": "Full-time",
        "full_time": "Full-time",
        "part": "Part-time",
        "part_time": "Part-time",
        "contract": "Contract",
        "temporary": "Temporary",
        "internship": "Internship",
    }.get((et or "").lower(), et) or None

    return {
        "id": f"workable:{slug}:{shortcode}",
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
        # Workable's field is "published" (ISO 8601); keep published_on as a fallback.
        "posted_at": d.get("published")
        or job.get("published")
        or d.get("published_on")
        or job.get("published_on")
        or None,
        "source": "workable",
    }


def _too_old(published: str | None, cutoff: dt.datetime | None) -> bool:
    if cutoff is None or not published:
        return False
    try:
        ts = dt.datetime.fromisoformat(published.replace("Z", "+00:00"))
    except ValueError:
        return False
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=dt.timezone.utc)
    return ts < cutoff


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument(
        "--slugs-file",
        default="jobs_search_demo/workable_slugs.txt",
        help="one Workable subdomain per line (# comments allowed)",
    )
    ap.add_argument(
        "--max-days-old", type=int, default=30, help="skip postings older than this (0 = no limit)"
    )
    ap.add_argument("--max-postings-per-company", type=int, default=0, help="0 = all open postings")
    ap.add_argument(
        "--no-detail",
        action="store_true",
        help="skip the per-job detail fetch (faster, but no description_html)",
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
        sys.exit(f"ERROR: no company subdomains in {slugs_path}")

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
        path = out / f"workable-{file_idx:04d}.parquet"
        pq.write_table(table, path)
        n_written += len(buf)
        print(f"  wrote {path.name}: {len(buf):,} rows (total {n_written:,})", flush=True)
        file_idx += 1
        buf = []

    for slug in slugs:
        token = None
        company_rows = 0
        while True:
            body = {"token": token} if token else {}
            payload = _request(f"{API_BASE}/{slug}/jobs", body)
            time.sleep(args.sleep)
            if payload is None:
                print(f"[{slug}] no public postings (404/403); skipping", flush=True)
                break
            results = payload.get("results") or payload.get("jobs") or []
            if not results:
                break
            for job in results:
                if _too_old(job.get("published") or job.get("published_on"), cutoff):
                    continue
                detail = None
                if not args.no_detail:
                    sc = str(job.get("shortcode") or "")
                    if sc:
                        detail = _request(f"{DETAIL_BASE}/{slug}/jobs/{sc}", None)
                        time.sleep(args.sleep)
                row = transform(job, detail, slug)
                if not row["title"] or row["id"] in seen_ids:
                    continue
                seen_ids.add(row["id"])
                buf.append(row)
                company_rows += 1
                if len(buf) >= args.rows_per_file:
                    flush()
                if args.max_postings_per_company and company_rows >= args.max_postings_per_company:
                    break
            token = payload.get("nextPage") or (payload.get("paging") or {}).get("next")
            if not token or (
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
