#!/usr/bin/env python3
"""Fetch IMDb non-commercial TSV dumps from datasets.imdbws.com.

Licensed for personal and non-commercial use only. See https://www.imdb.com/interfaces/.

Streams each .tsv.gz to disk and verifies the gzip is readable. Total download
is ~1 GB compressed across 7 files. Output:

  <out-dir>/title.basics.tsv.gz
  <out-dir>/title.ratings.tsv.gz
  <out-dir>/title.akas.tsv.gz
  <out-dir>/title.crew.tsv.gz
  <out-dir>/title.principals.tsv.gz
  <out-dir>/title.episode.tsv.gz
  <out-dir>/name.basics.tsv.gz

Usage:
  .venv/bin/python download/fetch_imdb_dumps.py --out-dir movies_data/imdb_raw
"""

import argparse
import gzip
import sys
import time
import urllib.request
from pathlib import Path

BASE = "https://datasets.imdbws.com"
FILES = [
    "title.basics.tsv.gz",
    "title.ratings.tsv.gz",
    "title.akas.tsv.gz",
    "title.crew.tsv.gz",
    "title.principals.tsv.gz",
    "title.episode.tsv.gz",
    "name.basics.tsv.gz",
]


def download(url: str, dest: Path, chunk: int = 1 << 20) -> int:
    tmp = dest.with_suffix(dest.suffix + ".part")
    n = 0
    t0 = time.time()
    with urllib.request.urlopen(url) as r, open(tmp, "wb") as f:
        while True:
            buf = r.read(chunk)
            if not buf:
                break
            f.write(buf)
            n += len(buf)
            if n % (32 * chunk) == 0:
                mb = n / (1 << 20)
                rate = mb / max(time.time() - t0, 1e-6)
                print(f"  ... {mb:.0f} MB ({rate:.1f} MB/s)", flush=True)
    tmp.rename(dest)
    return n


def verify_gzip(path: Path, sniff_bytes: int = 1 << 16) -> bool:
    with gzip.open(path, "rb") as f:
        return len(f.read(sniff_bytes)) > 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="movies_data/imdb_raw")
    ap.add_argument("--force", action="store_true", help="re-download even if present")
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    for name in FILES:
        dest = out / name
        if dest.exists() and not args.force:
            print(f"[skip] {name} already at {dest} ({dest.stat().st_size / (1 << 20):.0f} MB)")
            continue
        url = f"{BASE}/{name}"
        print(f"[fetch] {url} -> {dest}")
        size = download(url, dest)
        mb = size / (1 << 20)
        if not verify_gzip(dest):
            print(f"  ERROR: gzip verification failed for {dest}", file=sys.stderr)
            return 1
        print(f"  done: {mb:.0f} MB, gzip OK")
    print("all dumps present")
    return 0


if __name__ == "__main__":
    sys.exit(main())
