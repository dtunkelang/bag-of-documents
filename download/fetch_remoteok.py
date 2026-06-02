#!/usr/bin/env python3
"""Fetch currently-posted remote jobs from the RemoteOK public API.

RemoteOK aggregates remote-first postings (tech-heavy, but also design, marketing,
sales, support) with full HTML descriptions, so it brings remote inventory the
OpenApply crawler (Greenhouse/Lever/Ashby), USAJOBS, Adzuna, Reed, The Muse and
Findwork do not all cover. The single public endpoint returns the most recent
~100 active postings in one shot (no pagination, no auth); stable per-posting ids
let the daily --delta refresh accumulate new ones over time.

Per RemoteOK's API terms we send an identifying User-Agent and link back to each
posting's RemoteOK url. The endpoint's first array element is a legal/metadata
notice (no `id`) and is skipped.

  list: GET https://remoteok.com/api  -> JSON array [{legal...}, {job}, {job}, ...]

Writes parquet files whose schema matches download/fetch_adzuna.py / what
prep_open_apply.py expects (id, source_slug, title, description_html, department,
employment_type, remote, locations, salary_min, salary_max, salary_currency,
posted_at, source).

Usage:
  .venv/bin/python download/fetch_remoteok.py \\
      --out-dir jobs_data_remoteok/raw \\
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

API = "https://remoteok.com/api"
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
            if isinstance(e, HTTPError) and 400 <= e.code < 500 and e.code != 429:
                raise
            if attempt == retries - 1:
                raise
            sys.stderr.write(f"  {url} attempt {attempt + 1} failed ({e}); sleeping {backoff}s\n")
            time.sleep(backoff)
            backoff *= 2
    raise RuntimeError("unreachable")


def _posted_iso(job: dict) -> str | None:
    # `date` is already ISO 8601; fall back to the epoch seconds if absent.
    if job.get("date"):
        return str(job["date"])
    epoch = job.get("epoch")
    if epoch:
        try:
            return dt.datetime.fromtimestamp(int(epoch), dt.timezone.utc).isoformat()
        except (ValueError, OverflowError, OSError):
            return None
    return None


def transform(job: dict) -> dict[str, Any]:
    job_id = str(job.get("id") or job.get("slug") or "")
    company = job.get("company") or ""
    title = strip_tags(job.get("position") or "")

    # RemoteOK postings are remote by definition; `location` is a region restriction
    # (e.g. "Worldwide", "US only") rather than a city.
    loc = (job.get("location") or "").strip()
    locations = [loc] if loc else ["Remote"]

    smin = job.get("salary_min")
    smax = job.get("salary_max")

    return {
        "id": f"remoteok:{job_id}",
        "source_slug": slugify(company) or "remoteok",
        "title": title,
        "description_html": job.get("description") or "",
        "department": None,  # only tech keyword `tags`; no real department taxonomy
        "employment_type": None,  # not reliably exposed
        "remote": True,
        "locations": locations,
        "salary_min": int(smin) if isinstance(smin, (int, float)) and smin else None,
        "salary_max": int(smax) if isinstance(smax, (int, float)) and smax else None,
        "salary_currency": "USD" if (smin or smax) else None,  # RemoteOK salaries are USD
        "posted_at": _posted_iso(job),
        "source": "remoteok",
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
    ap.add_argument(
        "--max-days-old", type=int, default=30, help="skip postings older than this (0 = no limit)"
    )
    ap.add_argument("--rows-per-file", type=int, default=10_000)
    args = ap.parse_args()

    cutoff = None
    if args.max_days_old > 0:
        cutoff = dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=args.max_days_old)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    payload = _get(API)
    # The endpoint returns a flat array; the first element is a legal notice (no id).
    rows: list[dict] = []
    seen_ids: set[str] = set()
    for job in payload if isinstance(payload, list) else []:
        if not job.get("id") and not job.get("slug"):
            continue  # legal/metadata notice
        if _too_old(_posted_iso(job), cutoff):
            continue
        row = transform(job)
        if not row["title"] or row["id"] in seen_ids:
            continue
        seen_ids.add(row["id"])
        rows.append(row)

    n_written = 0
    file_idx = 0
    for i in range(0, len(rows), args.rows_per_file):
        chunk = rows[i : i + args.rows_per_file]
        table = pa.Table.from_pylist(chunk)
        path = out / f"remoteok-{file_idx:04d}.parquet"
        pq.write_table(table, path)
        n_written += len(chunk)
        print(f"  wrote {path.name}: {len(chunk):,} rows (total {n_written:,})", flush=True)
        file_idx += 1

    print(
        f"done: {n_written:,} unique postings across {file_idx} parquet files -> {out}", flush=True
    )


if __name__ == "__main__":
    main()
