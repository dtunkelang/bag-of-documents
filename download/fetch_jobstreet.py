#!/usr/bin/env python3
"""Fetch azrai99/job-dataset (JobStreet Malaysia/SEA) from HuggingFace and emit parquet.

Schema matches what download/prep_open_apply.py expects, so prep can be reused
unchanged. Covers ~59k SEA postings spanning Manufacturing/Logistics, Accounting,
Banking, Construction, Engineering, ICT, HR, Sales — the non-US non-tech tranche.

Usage:
  .venv/bin/python download/fetch_jobstreet.py --out-dir jobs_data_jobstreet/raw
"""

import argparse
import csv
import re
import sys
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

REPO = "azrai99/job-dataset"
FILENAME = "jobstreet_all_job_dataset.csv"
SLUG_RE = re.compile(r"[^a-z0-9]+")

# Currency markers seen in JobStreet salary strings.
CURRENCY_MAP = {
    "RM": "MYR",
    "MYR": "MYR",
    "SGD": "SGD",
    "S$": "SGD",
    "USD": "USD",
    "US$": "USD",
    "HKD": "HKD",
    "HK$": "HKD",
    "IDR": "IDR",
    "Rp": "IDR",
    "PHP": "PHP",
    "₱": "PHP",
    "THB": "THB",
    "฿": "THB",
    "VND": "VND",
    "₫": "VND",
}
# Match e.g. "RM 2,800 – RM 3,200" or "RM2800-3200" or "SGD 5,000 per month"
SALARY_RE = re.compile(
    r"(RM|MYR|SGD|S\$|USD|US\$|HKD|HK\$|IDR|Rp|PHP|₱|THB|฿|VND|₫)\s*([\d,]+(?:\.\d+)?)"
    r"(?:\s*[–\-to]+\s*(?:(?:RM|MYR|SGD|S\$|USD|US\$|HKD|HK\$|IDR|Rp|PHP|₱|THB|฿|VND|₫)\s*)?([\d,]+(?:\.\d+)?))?",
    re.IGNORECASE,
)


def slugify(s: str) -> str:
    return SLUG_RE.sub("-", (s or "").lower()).strip("-")


def parse_salary(raw: str):
    """Returns (min, max, currency) — any/all can be None."""
    if not raw:
        return None, None, None
    m = SALARY_RE.search(raw)
    if not m:
        return None, None, None
    cur_token = m.group(1).upper()
    cur = CURRENCY_MAP.get(cur_token, CURRENCY_MAP.get(cur_token.replace("$", "$")))
    try:
        lo = float(m.group(2).replace(",", ""))
    except (ValueError, AttributeError):
        lo = None
    hi = None
    if m.group(3):
        try:
            hi = float(m.group(3).replace(",", ""))
        except ValueError:
            hi = None
    return lo, hi, cur


def transform(row: dict[str, str]) -> dict[str, Any]:
    company = (row.get("company") or "").strip()
    source_slug = slugify(company) or "unknown-company"

    title = (row.get("job_title") or "").strip()
    desc = (row.get("descriptions") or "").strip()
    category = (row.get("category") or "").strip()
    subcategory = (row.get("subcategory") or "").strip()

    # Tack category/subcategory onto the text so retrievers can use it.
    tag_line = ""
    if category or subcategory:
        tag_line = f"[Category: {category}" + (f" / {subcategory}" if subcategory else "") + "]\n\n"
    description_html = (tag_line + desc).strip()

    loc = (row.get("location") or "").strip()
    locations = [loc] if loc else []

    salary_min, salary_max, salary_currency = parse_salary(row.get("salary") or "")

    return {
        "id": f"jobstreet:{source_slug}:{row.get('job_id', '')}",
        "source_slug": source_slug,
        "title": title,
        "description_html": description_html,
        "department": category or None,
        "employment_type": (row.get("type") or "").strip() or None,
        "remote": None,  # not in schema
        "locations": locations,
        "salary_min": salary_min,
        "salary_max": salary_max,
        "salary_currency": salary_currency,
        "posted_at": (row.get("listingDate") or "").strip() or None,
        "source": "jobstreet",
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
    n_skipped = 0

    def flush():
        nonlocal buf, file_idx, n_out
        if not buf:
            return
        table = pa.Table.from_pylist(buf)
        path = out / f"jobstreet-{file_idx:04d}.parquet"
        pq.write_table(table, path)
        n_out += len(buf)
        print(f"  wrote {path.name}: {len(buf):,} rows (total {n_out:,})", flush=True)
        file_idx += 1
        buf = []

    csv.field_size_limit(sys.maxsize)

    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            n_in += 1
            r = transform(row)
            if not r["title"]:
                n_skipped += 1
                continue
            buf.append(r)
            if len(buf) >= args.rows_per_file:
                flush()

    flush()
    print(
        f"done: read {n_in:,} CSV rows, wrote {n_out:,} parquet rows "
        f"({n_skipped:,} skipped for empty title)",
        flush=True,
    )


if __name__ == "__main__":
    main()
