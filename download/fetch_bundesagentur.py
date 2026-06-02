#!/usr/bin/env python3
"""Fetch currently-posted jobs from the German Bundesagentur fuer Arbeit API.

The Bundesagentur fuer Arbeit (German Federal Employment Agency) "Jobsuche" API
exposes Germany's national job inventory (~1M live postings), so it brings German
/ non-English inventory the OpenApply crawler (Greenhouse/Lever/Ashby), USAJOBS
(federal US), Adzuna and the ATS adapters do not cover.

No registration: the documented static header key is used as-is.
  list: GET https://rest.arbeitsagentur.de/jobboerse/jobsuche-service/pc/v4/jobs
        header  X-API-Key: jobboerse-jobsuche
        params  size, page, was (keywords), wo (location), veroeffentlichtseit (days)

IMPORTANT: the keyless list endpoint returns NO job description (the per-posting
detail endpoint requires a separate registered OAuth credential and 403s with the
static key). So this adapter populates description_html with the occupation label
(beruf) as a minimal stand-in -- these docs are title-only and much thinner than
the ATS / Reed / The Muse / Findwork sources. Wire in only if that is acceptable.

Writes parquet files whose schema matches download/fetch_adzuna.py / what
prep_open_apply.py expects (id, source_slug, title, description_html, department,
employment_type, remote, locations, salary_min, salary_max, salary_currency,
posted_at, source).

Usage:
  .venv/bin/python download/fetch_bundesagentur.py \\
      --out-dir jobs_data_bundesagentur/raw \\
      --max-pages 20 --published-since 7
"""

import argparse
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

API = "https://rest.arbeitsagentur.de/jobboerse/jobsuche-service/pc/v4/jobs"
API_KEY = "jobboerse-jobsuche"  # documented static header key
PAGE = 100  # API max size
SLUG_RE = re.compile(r"[^a-z0-9]+")
TAG_RE = re.compile(r"<[^>]+>")


def slugify(s: str) -> str:
    return SLUG_RE.sub("-", (s or "").lower()).strip("-")


def strip_tags(s: str) -> str:
    return TAG_RE.sub("", s or "").strip()


def _get(url: str, retries: int = 4) -> Any:
    req = Request(
        url,
        headers={
            "X-API-Key": API_KEY,
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
            # 404 once the page index runs past the result set -> caller stops.
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


def transform(job: dict) -> dict[str, Any]:
    refnr = str(job.get("refnr") or "")
    company = job.get("arbeitgeber") or ""
    title = strip_tags(job.get("titel") or job.get("beruf") or "")
    beruf = job.get("beruf") or ""

    ort = job.get("arbeitsort") or {}
    locations = [v for v in (ort.get("ort"), ort.get("region")) if v and v != "null"]

    hay = f"{title} {beruf}".lower()
    remote = True if ("homeoffice" in hay or "remote" in hay or "telearbeit" in hay) else None

    return {
        "id": f"bundesagentur:{slugify(refnr) or slugify(title)}",
        "source_slug": slugify(company) or "bundesagentur",
        "title": title,
        # No description on the keyless endpoint; occupation label is the only body.
        "description_html": beruf,
        "department": beruf or None,  # occupation is the nearest categorical analog
        "employment_type": None,
        "remote": remote,
        "locations": locations,
        "salary_min": None,  # not exposed
        "salary_max": None,
        "salary_currency": None,
        "posted_at": job.get("aktuelleVeroeffentlichungsdatum") or None,  # ISO date
        "source": "bundesagentur",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--was", default="", help="optional keyword filter (default: all jobs)")
    ap.add_argument("--wo", default="", help="optional location filter")
    ap.add_argument(
        "--published-since",
        type=int,
        default=7,
        help="only offers published within the last N days (0 = no limit)",
    )
    ap.add_argument(
        "--max-pages", type=int, default=20, help="listing pages (100 jobs/page; 0=all)"
    )
    ap.add_argument("--rows-per-file", type=int, default=10_000)
    ap.add_argument("--sleep", type=float, default=0.3, help="seconds between calls (be polite)")
    args = ap.parse_args()

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
        path = out / f"bundesagentur-{file_idx:04d}.parquet"
        pq.write_table(table, path)
        n_written += len(buf)
        print(f"  wrote {path.name}: {len(buf):,} rows (total {n_written:,})", flush=True)
        file_idx += 1
        buf = []

    page = 1
    total = None
    while args.max_pages == 0 or page <= args.max_pages:
        params: dict[str, Any] = {"size": PAGE, "page": page}
        if args.was:
            params["was"] = args.was
        if args.wo:
            params["wo"] = args.wo
        if args.published_since > 0:
            params["veroeffentlichtseit"] = args.published_since
        payload = _get(f"{API}?{urlencode(params)}")
        time.sleep(args.sleep)
        if payload is None:
            break
        total = int(payload.get("maxErgebnisse", 0) or 0)
        results = payload.get("stellenangebote") or []
        if not results:
            break
        for job in results:
            row = transform(job)
            if not row["title"] or row["id"] in seen_ids:
                continue
            seen_ids.add(row["id"])
            buf.append(row)
            if len(buf) >= args.rows_per_file:
                flush()
        page += 1
        if total is not None and (page - 1) * PAGE >= total:
            break

    flush()
    print(
        f"done: {n_written:,} unique postings across {file_idx} parquet files -> {out}",
        flush=True,
    )


if __name__ == "__main__":
    main()
