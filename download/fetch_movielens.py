#!/usr/bin/env python3
"""Fetch MovieLens 25M dataset and unzip.

Downloads ~265 MB zip from files.grouplens.org and extracts to <out-dir>/.
After unzip, you'll have:
  <out-dir>/ml-25m/ratings.csv     ~720 MB (userId, movieId, rating, timestamp)
  <out-dir>/ml-25m/movies.csv      ~3 MB (movieId, title, genres)
  <out-dir>/ml-25m/links.csv       ~2 MB (movieId, imdbId, tmdbId)
  <out-dir>/ml-25m/tags.csv        ~38 MB
  ...

Usage:
  .venv/bin/python download/fetch_movielens.py --out-dir movies_data/movielens
"""

import argparse
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

URL = "https://files.grouplens.org/datasets/movielens/ml-25m.zip"


def download(url: str, dest: Path, chunk: int = 1 << 20) -> int:
    n = 0
    t0 = time.time()
    tmp = dest.with_suffix(dest.suffix + ".part")
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="movies_data/movielens")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    zip_path = out / "ml-25m.zip"
    extracted = out / "ml-25m"

    if extracted.exists() and not args.force:
        print(f"[skip] already extracted at {extracted}")
        return 0

    if not zip_path.exists() or args.force:
        print(f"[fetch] {URL} -> {zip_path}")
        size = download(URL, zip_path)
        print(f"  done: {size / (1 << 20):.0f} MB")

    print(f"[unzip] {zip_path} -> {out}")
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(out)
    print(f"[done] extracted at {extracted}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
