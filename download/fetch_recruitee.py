#!/usr/bin/env python3
"""Fetch currently-posted jobs from Recruitee-hosted career sites.

Recruitee is a keyless, slug-driven ATS (each company is a {slug}.recruitee.com
subdomain). Its public offers endpoint already carries the full HTML description,
so unlike SmartRecruiters/Workable this needs no per-posting detail fetch. The
companies are not in OpenApply's Greenhouse/Lever/Ashby crawl, so they bring
net-new inventory.

  list: GET https://{slug}.recruitee.com/api/offers/  -> {"offers": [...]}

Slugs are read from --slugs-file (one per line, # = comment); harvest via
site-scoped search of *.recruitee.com careers URLs, then verify each returns
offers before adding.

Writes parquet files whose schema matches download/fetch_adzuna.py / what
prep_open_apply.py expects (id, source_slug, title, description_html, department,
employment_type, remote, locations, salary_min, salary_max, salary_currency,
posted_at, source).

Usage:
  .venv/bin/python download/fetch_recruitee.py \\
      --out-dir jobs_data_recruitee/raw \\
      --slugs-file jobs_search_demo/recruitee_slugs.txt \\
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
        except (HTTPError, URLError, TimeoutError, ValueError) as e:
            if isinstance(e, HTTPError) and e.code in (404, 410):
                return None  # company removed / no public board
            if isinstance(e, HTTPError) and 400 <= e.code < 500 and e.code != 429:
                raise
            if attempt == retries - 1:
                return None
            sys.stderr.write(f"  {url} attempt {attempt + 1} failed ({e}); sleeping {backoff}s\n")
            time.sleep(backoff)
            backoff *= 2
    return None


def _posted_iso(published_at: str | None) -> str | None:
    # Recruitee format: "2026-05-28 20:36:05 UTC".
    if not published_at:
        return None
    try:
        ts = dt.datetime.strptime(published_at.replace(" UTC", ""), "%Y-%m-%d %H:%M:%S")
        return ts.replace(tzinfo=dt.timezone.utc).isoformat()
    except ValueError:
        return published_at


def transform(offer: dict, slug: str) -> dict[str, Any]:
    company = offer.get("company_name") or slug
    city = (offer.get("city") or "").strip()
    country = (offer.get("country") or "").strip()
    if city and country:
        primary = f"{city}, {country}"
    else:
        primary = city or country or ("Remote" if offer.get("remote") else "")
    locations = [primary] if primary else []

    sal = offer.get("salary") or {}
    smin, smax = sal.get("min"), sal.get("max")

    return {
        "id": f"recruitee:{slug}:{offer.get('id')}",
        "source_slug": slugify(company) or slug,
        "title": strip_tags(offer.get("title") or ""),
        "description_html": offer.get("description") or "",
        "department": offer.get("department") or None,
        "employment_type": offer.get("employment_type_code") or None,
        "remote": True if offer.get("remote") else None,
        "locations": locations,
        "salary_min": int(smin) if isinstance(smin, (int, float)) and smin else None,
        "salary_max": int(smax) if isinstance(smax, (int, float)) and smax else None,
        "salary_currency": sal.get("currency") if (smin or smax) else None,
        "posted_at": _posted_iso(offer.get("published_at")),
        "source": "recruitee",
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
    ap.add_argument("--slugs-file", default="jobs_search_demo/recruitee_slugs.txt")
    ap.add_argument(
        "--max-days-old", type=int, default=30, help="skip postings older than this (0 = no limit)"
    )
    ap.add_argument("--max-postings-per-company", type=int, default=0, help="0 = all open postings")
    ap.add_argument("--rows-per-file", type=int, default=10_000)
    ap.add_argument("--sleep", type=float, default=0.3, help="seconds between calls (be polite)")
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

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    seen_ids: set[str] = set()
    for slug in slugs:
        payload = _get(f"https://{slug}.recruitee.com/api/offers/")
        time.sleep(args.sleep)
        offers = (payload or {}).get("offers") or []
        kept = 0
        for offer in offers:
            if args.max_postings_per_company and kept >= args.max_postings_per_company:
                break
            if _too_old(_posted_iso(offer.get("published_at")), cutoff):
                continue
            row = transform(offer, slug)
            if not row["title"] or row["id"] in seen_ids:
                continue
            seen_ids.add(row["id"])
            rows.append(row)
            kept += 1
        print(f"[{slug}] kept {kept} of {len(offers)} offers", flush=True)

    n_written = 0
    for i in range(0, len(rows), args.rows_per_file):
        chunk = rows[i : i + args.rows_per_file]
        path = out / f"recruitee-{i // args.rows_per_file:04d}.parquet"
        pq.write_table(pa.Table.from_pylist(chunk), path)
        n_written += len(chunk)
        print(f"  wrote {path.name}: {len(chunk):,} rows (total {n_written:,})", flush=True)

    print(f"done: {n_written:,} unique postings -> {out}", flush=True)


if __name__ == "__main__":
    main()
