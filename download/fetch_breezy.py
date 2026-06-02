#!/usr/bin/env python3
"""Fetch currently-posted jobs from Breezy-HR-hosted career sites.

Breezy is a keyless, slug-driven ATS (each company is a {slug}.breezy.hr subdomain,
SMB-heavy). The public board JSON lists positions but WITHOUT the description, so
we fetch each position's public page and extract the schema.org JobPosting JSON-LD,
which carries the full HTML description. Companies are not in OpenApply's Greenhouse/
Lever/Ashby crawl, so they bring net-new inventory.

  list:   GET https://{slug}.breezy.hr/json            -> [ {position}, ... ]
  detail: GET <position url>  (HTML)  -> <script type=application/ld+json> JobPosting

Slugs are read from --slugs-file (one per line, # = comment); harvest via
site-scoped search of *.breezy.hr URLs, then verify each returns positions.

Writes parquet files whose schema matches download/fetch_adzuna.py / what
prep_open_apply.py expects (id, source_slug, title, description_html, department,
employment_type, remote, locations, salary_min, salary_max, salary_currency,
posted_at, source).

Usage:
  .venv/bin/python download/fetch_breezy.py \\
      --out-dir jobs_data_breezy/raw \\
      --slugs-file jobs_search_demo/breezy_slugs.txt \\
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
LD_RE = re.compile(r"<script[^>]*application/ld\+json[^>]*>(.*?)</script>", re.S)


def slugify(s: str) -> str:
    return SLUG_RE.sub("-", (s or "").lower()).strip("-")


def strip_tags(s: str) -> str:
    return TAG_RE.sub("", s or "").strip()


def _get_json(url: str, retries: int = 4) -> Any:
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
                return None
            if isinstance(e, HTTPError) and 400 <= e.code < 500 and e.code != 429:
                raise
            if attempt == retries - 1:
                return None
            time.sleep(backoff)
            backoff *= 2
    return None


def _jobposting_ld(url: str) -> dict | None:
    """Fetch a position's public page and return its JobPosting JSON-LD (or None)."""
    req = Request(url, headers={"User-Agent": "bagofdocs-jobs-refresh"})
    try:
        with urlopen(req, timeout=60) as resp:
            html = resp.read().decode("utf-8", "replace")
    except (HTTPError, URLError, TimeoutError):
        return None
    for block in LD_RE.findall(html):
        try:
            d = json.loads(block)
        except ValueError:
            continue
        if isinstance(d, dict) and d.get("@type") == "JobPosting":
            return d
    return None


def _primary_location(pos: dict) -> list[str]:
    loc = pos.get("location") or {}
    city = (loc.get("city") or "").strip()
    country = ((loc.get("country") or {}).get("name") or "").strip()
    if city and country:
        return [f"{city}, {country}"]
    return [city or country] if (city or country) else []


def transform(pos: dict, ld: dict | None, slug: str) -> dict[str, Any]:
    company = (pos.get("company") or {}).get("name") or slug
    desc = (ld or {}).get("description") or ""
    etype = (pos.get("type") or {}).get("name") or (ld or {}).get("employmentType") or None
    city = ((pos.get("location") or {}).get("city") or "").lower()
    remote = (
        True if ("remote" in city or (ld or {}).get("jobLocationType") == "TELECOMMUTE") else None
    )

    return {
        "id": f"breezy:{slug}:{pos.get('id')}",
        "source_slug": slugify(company) or slug,
        "title": strip_tags(pos.get("name") or ""),
        "description_html": desc,
        "department": pos.get("department") or None,
        "employment_type": etype,
        "remote": remote,
        "locations": _primary_location(pos),
        "salary_min": None,  # not exposed in board JSON / JSON-LD
        "salary_max": None,
        "salary_currency": None,
        "posted_at": pos.get("published_date") or (ld or {}).get("datePosted") or None,
        "source": "breezy",
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
    ap.add_argument("--slugs-file", default="jobs_search_demo/breezy_slugs.txt")
    ap.add_argument(
        "--max-days-old", type=int, default=30, help="skip postings older than this (0 = no limit)"
    )
    ap.add_argument("--max-postings-per-company", type=int, default=0, help="0 = all open postings")
    ap.add_argument(
        "--no-detail",
        action="store_true",
        help="skip the JSON-LD detail fetch (no description_html)",
    )
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
        sys.exit(f"ERROR: no company subdomains in {slugs_path}")

    cutoff = None
    if args.max_days_old > 0:
        cutoff = dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=args.max_days_old)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    seen_ids: set[str] = set()
    for slug in slugs:
        positions = _get_json(f"https://{slug}.breezy.hr/json") or []
        time.sleep(args.sleep)
        kept = 0
        for pos in positions:
            if args.max_postings_per_company and kept >= args.max_postings_per_company:
                break
            if _too_old(pos.get("published_date"), cutoff):
                continue
            ld = None
            if not args.no_detail and pos.get("url"):
                ld = _jobposting_ld(pos["url"])
                time.sleep(args.sleep)
            row = transform(pos, ld, slug)
            if not row["title"] or row["id"] in seen_ids:
                continue
            seen_ids.add(row["id"])
            rows.append(row)
            kept += 1
        print(f"[{slug}] kept {kept} of {len(positions)} positions", flush=True)

    n_written = 0
    for i in range(0, len(rows), args.rows_per_file):
        chunk = rows[i : i + args.rows_per_file]
        path = out / f"breezy-{i // args.rows_per_file:04d}.parquet"
        pq.write_table(pa.Table.from_pylist(chunk), path)
        n_written += len(chunk)
        print(f"  wrote {path.name}: {len(chunk):,} rows (total {n_written:,})", flush=True)

    print(f"done: {n_written:,} unique postings -> {out}", flush=True)


if __name__ == "__main__":
    main()
