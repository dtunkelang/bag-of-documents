#!/usr/bin/env python3
"""Fetch currently-posted Swedish jobs from the JobTech (Arbetsförmedlingen) open API.

JobTech Dev is the open-data arm of the Swedish Public Employment Service
(Arbetsförmedlingen). Its JobStream API publishes every ad in Platsbanken with a
full free-text description AND a pre-attached occupation taxonomy
(occupation / occupation_group / occupation_field, each a stable concept id +
label). That taxonomy is the Swedish analogue of France Travail's ROME codes, but
it ships inside every posting -- no separate nomenclature download or
title->code reconstruction is needed to ground role/related-search later.

This brings Swedish (North-Germanic) national inventory the other sources don't
cover. No auth, no key. The JobStream `/stream` endpoint returns every ad
created, updated OR removed since a timestamp, so it maps directly onto the daily
--delta refresh: ask for `date = now - max_days_old`, keep the live ones, skip the
records flagged `removed`. Stable per-ad ids let --delta accumulate over time.

  stream: GET https://jobstream.api.jobtechdev.se/stream?date=<ISO8601>
          -> JSON array [{ad}, {ad}, ...]   (each ad has a `removed` bool)

The response is large (tens of MB per day), so it is streamed to a temp file
rather than held in memory as a string before parsing.

Writes parquet files whose schema matches download/fetch_remoteok.py /
fetch_adzuna.py / what prep_open_apply.py expects (id, source_slug, title,
description_html, department, employment_type, remote, locations, salary_min,
salary_max, salary_currency, posted_at, source).

Usage:
  .venv/bin/python download/fetch_jobtech_se.py \\
      --out-dir jobs_data_jobtech/raw \\
      --max-days-old 7
"""

import argparse
import datetime as dt
import json
import re
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import pyarrow as pa
import pyarrow.parquet as pq

STREAM = "https://jobstream.api.jobtechdev.se/stream"
SLUG_RE = re.compile(r"[^a-z0-9]+")
TAG_RE = re.compile(r"<[^>]+>")


def slugify(s: str) -> str:
    return SLUG_RE.sub("-", (s or "").lower()).strip("-")


def strip_tags(s: str) -> str:
    return TAG_RE.sub("", s or "").strip()


def _fetch_stream(date_iso: str, retries: int = 4) -> list:
    """Download the JobStream delta since `date_iso` and return the parsed array.

    The endpoint can deliver a very large response (tens to hundreds of MB), so it is
    streamed to a temp file rather than held in memory. A connection dropped mid-flight
    yields a truncated body that still arrived with HTTP 200; we parse INSIDE the retry
    loop so that JSONDecodeError (the signature of a truncated download) triggers a
    retry rather than crashing the caller."""
    url = f"{STREAM}?date={date_iso}"
    req = Request(
        url, headers={"User-Agent": "bagofdocs-jobs-refresh", "Accept": "application/json"}
    )
    backoff = 5
    for attempt in range(retries):
        tmp = Path(tempfile.mkstemp(prefix="jobtech-", suffix=".json")[1])
        try:
            with urlopen(req, timeout=900) as resp, open(tmp, "wb") as fh:
                shutil.copyfileobj(resp, fh, length=1 << 20)
            with open(tmp, encoding="utf-8") as fh:
                return json.load(fh)
        except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as e:
            if isinstance(e, HTTPError) and 400 <= e.code < 500 and e.code != 429:
                raise
            if attempt == retries - 1:
                raise
            sys.stderr.write(
                f"  {url} attempt {attempt + 1} failed ({type(e).__name__}: {e}); "
                f"sleeping {backoff}s\n"
            )
            time.sleep(backoff)
            backoff *= 2
        finally:
            tmp.unlink(missing_ok=True)
    raise RuntimeError("unreachable")


def _label(d: dict | None, *fallback: str) -> str | None:
    if isinstance(d, dict) and d.get("label"):
        return d["label"]
    for f in fallback:
        if f:
            return f
    return None


def _locations(ad: dict) -> list[str]:
    addr = ad.get("workplace_address") or {}
    city = addr.get("municipality")
    region = addr.get("region")
    country = addr.get("country")
    place = ", ".join(p for p in (city, region) if p)
    if place:
        return [place]
    return [country] if country else []


def transform(ad: dict) -> dict[str, Any]:
    ad_id = str(ad.get("id") or "")
    title = strip_tags(ad.get("headline") or "")
    desc = strip_tags((ad.get("description") or {}).get("text") or "")
    employer = (ad.get("employer") or {}).get("name") or ""

    # occupation_field is the broad taxonomy bucket (e.g. "Hälso- och sjukvård"),
    # the closest analogue to the `department` field the other sources populate.
    department = _label(ad.get("occupation_field"))

    # Combine contract + working-hours labels, mirroring France Travail's
    # "{typeContrat}/{dureeTravail}" convention.
    et = _label(ad.get("employment_type"))
    wt = _label(ad.get("working_hours_type"))
    employment_type = " / ".join(p for p in (et, wt) if p) or None

    return {
        "id": f"jobtech:{ad_id}",
        "source_slug": slugify(employer) or "jobtech",
        "title": title,
        "description_html": desc,
        "department": department,
        "employment_type": employment_type,
        # Swedish salaries are free-text only; remote is not a reliable structured flag.
        "remote": None,
        "locations": _locations(ad),
        "salary_min": None,
        "salary_max": None,
        "salary_currency": None,
        "posted_at": ad.get("publication_date"),
        "source": "jobtech",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument(
        "--max-days-old",
        type=int,
        default=7,
        help="fetch the JobStream delta covering the last N days",
    )
    ap.add_argument(
        "--since",
        default=None,
        help="explicit ISO8601 cutoff (overrides --max-days-old) for incremental runs",
    )
    ap.add_argument("--rows-per-file", type=int, default=10_000)
    args = ap.parse_args()

    if args.since:
        date_iso = args.since
    else:
        cutoff = dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=args.max_days_old)
        date_iso = cutoff.strftime("%Y-%m-%dT%H:%M:%S")

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"  streaming JobTech ads since {date_iso} ...", flush=True)
    payload = _fetch_stream(date_iso)

    rows: list[dict] = []
    seen_ids: set[str] = set()
    n_removed = 0
    for ad in payload if isinstance(payload, list) else []:
        if ad.get("removed"):
            n_removed += 1
            continue  # expired / taken down since the cutoff
        row = transform(ad)
        if not row["title"] or not row["description_html"] or row["id"] in seen_ids:
            continue
        seen_ids.add(row["id"])
        rows.append(row)

    n_written = 0
    file_idx = 0
    for i in range(0, len(rows), args.rows_per_file):
        chunk = rows[i : i + args.rows_per_file]
        table = pa.Table.from_pylist(chunk)
        path = out / f"jobtech-{file_idx:04d}.parquet"
        pq.write_table(table, path)
        n_written += len(chunk)
        print(f"  wrote {path.name}: {len(chunk):,} rows (total {n_written:,})", flush=True)
        file_idx += 1

    print(
        f"done: {n_written:,} active postings "
        f"({n_removed:,} removed skipped) across {file_idx} parquet files -> {out}",
        flush=True,
    )


if __name__ == "__main__":
    main()
