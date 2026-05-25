#!/usr/bin/env python3
"""Build a unified per-title JSONL from IMDb non-commercial TSV dumps.

Joins title.basics + title.ratings + title.crew + title.principals + name.basics
into one row per title, filtered by titleType. Output:

  <out-path>: JSONL with fields:
    tconst, title, original_title, year, type, genres (list),
    runtime, is_adult, rating, votes,
    director_names (list), cast_names (list)

The output is the entity layer of the movies/TV demo. Plot text comes from
Wikipedia (Phase 3) and collaborative signal from MovieLens (Phase 5); see
project_movies_demo_data_stack.md.

Usage:
  .venv/bin/python download/build_imdb_titles.py \\
      --imdb-dir movies_data/imdb_raw \\
      --out-path movies_data/titles_imdb.jsonl
"""

import argparse
import csv
import gzip
import json
import sys
import time
from pathlib import Path

DEFAULT_TYPES = ["movie", "tvSeries", "tvMiniSeries", "tvMovie"]
CAST_CATEGORIES = {"actor", "actress", "self"}


def parse_null(v: str) -> str | None:
    return None if v == r"\N" else v


def parse_int(v: str) -> int | None:
    v = parse_null(v)
    if v is None:
        return None
    try:
        return int(v)
    except ValueError:
        return None


def parse_float(v: str) -> float | None:
    v = parse_null(v)
    if v is None:
        return None
    try:
        return float(v)
    except ValueError:
        return None


def parse_csv_field(v: str) -> list[str]:
    v = parse_null(v)
    return v.split(",") if v else []


def tsv_reader(f):
    return csv.reader(f, delimiter="\t", quoting=csv.QUOTE_NONE)


def open_tsv_gz(path: Path):
    return gzip.open(path, "rt", encoding="utf-8", newline="")


def log(msg: str) -> None:
    print(msg, flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--imdb-dir", default="movies_data/imdb_raw")
    ap.add_argument("--out-path", default="movies_data/titles_imdb.jsonl")
    ap.add_argument("--types", nargs="+", default=DEFAULT_TYPES)
    ap.add_argument("--top-cast", type=int, default=10)
    args = ap.parse_args()

    imdb = Path(args.imdb_dir)
    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    type_set = set(args.types)

    # Pass 1: title.basics → keep filtered titles
    t0 = time.time()
    kept: dict[str, dict] = {}
    with open_tsv_gz(imdb / "title.basics.tsv.gz") as f:
        r = tsv_reader(f)
        next(r)  # header
        for i, row in enumerate(r):
            if len(row) < 9:
                continue
            (
                tconst,
                titleType,
                primaryTitle,
                originalTitle,
                isAdult,
                startYear,
                _endYear,
                runtime,
                genres,
            ) = row
            if titleType not in type_set:
                continue
            kept[tconst] = {
                "tconst": tconst,
                "title": primaryTitle,
                "original_title": originalTitle if originalTitle != primaryTitle else None,
                "year": parse_int(startYear),
                "type": titleType,
                "genres": parse_csv_field(genres),
                "runtime": parse_int(runtime),
                "is_adult": isAdult == "1",
            }
            if (i + 1) % 1_000_000 == 0:
                log(f"  basics: scanned {i + 1:,} rows, kept {len(kept):,}")
    log(f"[basics] kept {len(kept):,} titles in {time.time() - t0:.1f}s")

    # Pass 2: title.ratings → merge ratings
    t0 = time.time()
    n_rated = 0
    with open_tsv_gz(imdb / "title.ratings.tsv.gz") as f:
        r = tsv_reader(f)
        next(r)
        for row in r:
            if len(row) < 3:
                continue
            tconst, avg, votes = row
            doc = kept.get(tconst)
            if doc is None:
                continue
            doc["rating"] = parse_float(avg)
            doc["votes"] = parse_int(votes)
            n_rated += 1
    log(f"[ratings] merged {n_rated:,} ratings in {time.time() - t0:.1f}s")

    # Pass 3: title.crew → collect director nconsts
    t0 = time.time()
    director_nconsts: dict[str, list[str]] = {}
    with open_tsv_gz(imdb / "title.crew.tsv.gz") as f:
        r = tsv_reader(f)
        next(r)
        for row in r:
            if len(row) < 3:
                continue
            tconst, directors, _writers = row
            if tconst not in kept:
                continue
            dirs = parse_csv_field(directors)
            if dirs:
                director_nconsts[tconst] = dirs
    log(
        f"[crew] collected directors for {len(director_nconsts):,} titles in {time.time() - t0:.1f}s"
    )

    # Pass 4: title.principals → collect top-N cast nconsts per kept title
    t0 = time.time()
    cast_buf: dict[str, list[tuple[int, str]]] = {}
    n_rows = 0
    with open_tsv_gz(imdb / "title.principals.tsv.gz") as f:
        r = tsv_reader(f)
        next(r)
        for row in r:
            n_rows += 1
            if len(row) < 4:
                continue
            tconst, ordering, nconst, category = row[0], row[1], row[2], row[3]
            if tconst not in kept:
                continue
            if category not in CAST_CATEGORIES:
                continue
            try:
                ord_i = int(ordering)
            except ValueError:
                continue
            buf = cast_buf.setdefault(tconst, [])
            buf.append((ord_i, nconst))
            if n_rows % 10_000_000 == 0:
                log(f"  principals: scanned {n_rows:,} rows")
    # Top-K by ordering per title
    cast_nconsts: dict[str, list[str]] = {}
    for tconst, buf in cast_buf.items():
        buf.sort()
        cast_nconsts[tconst] = [nc for _, nc in buf[: args.top_cast]]
    del cast_buf
    log(f"[principals] collected cast for {len(cast_nconsts):,} titles in {time.time() - t0:.1f}s")

    # Pass 5: name.basics → name lookup for referenced nconsts only
    t0 = time.time()
    needed: set[str] = set()
    for ncs in director_nconsts.values():
        needed.update(ncs)
    for ncs in cast_nconsts.values():
        needed.update(ncs)
    log(f"[names] need {len(needed):,} nconsts")
    name_lookup: dict[str, str] = {}
    with open_tsv_gz(imdb / "name.basics.tsv.gz") as f:
        r = tsv_reader(f)
        next(r)
        for row in r:
            if len(row) < 2:
                continue
            nconst, primaryName = row[0], row[1]
            if nconst in needed:
                name_lookup[nconst] = primaryName
    log(f"[names] resolved {len(name_lookup):,} names in {time.time() - t0:.1f}s")

    # Final write
    t0 = time.time()
    n_written = 0
    with open(out_path, "w") as out:
        for tconst, doc in kept.items():
            dirs = [name_lookup[n] for n in director_nconsts.get(tconst, []) if n in name_lookup]
            cast = [name_lookup[n] for n in cast_nconsts.get(tconst, []) if n in name_lookup]
            doc["director_names"] = dirs
            doc["cast_names"] = cast
            out.write(json.dumps(doc, ensure_ascii=False) + "\n")
            n_written += 1
    log(f"[write] {n_written:,} rows -> {out_path} in {time.time() - t0:.1f}s")
    log(f"output size: {out_path.stat().st_size / (1 << 20):.1f} MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
