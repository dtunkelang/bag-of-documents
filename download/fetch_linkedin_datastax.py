#!/usr/bin/env python3
"""Fetch datastax/linkedin_job_listings from HuggingFace and emit parquet.

The output schema matches what download/prep_open_apply.py expects, so prep
can be reused unchanged. The dataset is a single CSV (~124k rows) covering
broad sectors (healthcare, trades, hospitality, retail, plus tech).

Usage:
  .venv/bin/python download/fetch_linkedin_datastax.py \\
      --out-dir jobs_data_linkedin/raw \\
      --rows-per-file 25000
"""

import argparse
import csv
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

REPO = "datastax/linkedin_job_listings"
FILENAME = "postings.csv"
SLUG_RE = re.compile(r"[^a-z0-9]+")


def slugify(s: str) -> str:
    return SLUG_RE.sub("-", (s or "").lower()).strip("-")


def parse_float(v: str):
    if v is None or v == "":
        return None
    try:
        return float(v)
    except ValueError:
        return None


def parse_epoch_ms(v: str):
    f = parse_float(v)
    if f is None:
        return None
    try:
        return datetime.fromtimestamp(f / 1000.0, tz=timezone.utc).isoformat()
    except (ValueError, OSError, OverflowError):
        return None


def parse_remote(v: str):
    if v is None or v == "":
        return None
    try:
        return bool(int(float(v)))
    except ValueError:
        return None


def transform(row: dict[str, str]) -> dict[str, Any]:
    company = (row.get("company_name") or "").strip()
    company_id = (row.get("company_id") or "").strip().rstrip(".0") or "unknown"
    source_slug = slugify(company) or f"company-{company_id}"

    title = (row.get("title") or "").strip()
    description = (row.get("description") or "").strip()
    skills = (row.get("skills_desc") or "").strip()
    # Concatenate skills onto description; prep_open_apply.strip_html handles tags if any.
    description_html = description + (f"\n\nSkills: {skills}" if skills else "")

    locations = []
    loc = (row.get("location") or "").strip()
    if loc:
        locations.append(loc)

    return {
        "id": f"linkedin:{source_slug}:{row.get('job_id', '')}",
        "source_slug": source_slug,
        "title": title,
        "description_html": description_html,
        "department": None,
        "employment_type": (row.get("formatted_work_type") or "").strip() or None,
        "remote": parse_remote(row.get("remote_allowed")),
        "locations": locations,
        "salary_min": parse_float(row.get("min_salary")),
        "salary_max": parse_float(row.get("max_salary")),
        "salary_currency": (row.get("currency") or "").strip() or None,
        "posted_at": parse_epoch_ms(row.get("listed_time") or row.get("original_listed_time")),
        "source": "linkedin",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--rows-per-file", type=int, default=25_000)
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"downloading {REPO}/{FILENAME} from HuggingFace...", flush=True)
    csv_path = hf_hub_download(repo_id=REPO, filename=FILENAME, repo_type="dataset")
    print(f"  cached at: {csv_path}", flush=True)

    buf: list[dict] = []
    file_idx = 0
    n_in = 0
    n_out = 0
    n_skipped_no_title = 0

    def flush():
        nonlocal buf, file_idx, n_out
        if not buf:
            return
        table = pa.Table.from_pylist(buf)
        path = out / f"linkedin-{file_idx:04d}.parquet"
        pq.write_table(table, path)
        n_out += len(buf)
        print(f"  wrote {path.name}: {len(buf):,} rows (total {n_out:,})", flush=True)
        file_idx += 1
        buf = []

    # CSV has very long fields (descriptions up to ~23k chars); raise the limit.
    csv.field_size_limit(sys.maxsize)

    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            n_in += 1
            r = transform(row)
            if not r["title"]:
                n_skipped_no_title += 1
                continue
            buf.append(r)
            if len(buf) >= args.rows_per_file:
                flush()

    flush()
    print(
        f"done: read {n_in:,} CSV rows, wrote {n_out:,} parquet rows "
        f"({n_skipped_no_title:,} skipped for empty title)",
        flush=True,
    )


if __name__ == "__main__":
    main()
