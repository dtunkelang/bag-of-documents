#!/usr/bin/env python3
"""Build BoD-style co-rated bags from MovieLens 25M.

For each movie M, the bag is the top-K movies most frequently co-rated highly
(rating >= --min-rating) by users who also rated M highly. This is the
collaborative-signal analog to BestBuy's "users who clicked X also clicked Y"
bags that drove the +17.5pp R@10 lift validation.

Joins MovieLens movieId → IMDb tconst via links.csv. Only emits bags for
movies whose tconst is present in --imdb-titles (so the bag-source side stays
aligned with the index side).

Output:
  <out-path>: JSONL with one row per movie M:
    {
      "tconst": "tt...",
      "movielens_id": <int>,
      "n_high_raters": <int>,
      "bag": [{"tconst": "tt...", "co_count": <int>}, ...top K, sorted desc]
    }

Usage:
  .venv/bin/python download/build_movielens_corated_bags.py \\
      --ml-dir movies_data/movielens/ml-25m \\
      --imdb-titles movies_data/titles_imdb.jsonl \\
      --out-path movies_data/movielens_corated_bags.jsonl \\
      --min-rating 4.0 --top-k 50
"""

import argparse
import csv
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path


def log(msg: str) -> None:
    print(msg, flush=True)


def load_imdb_tconsts(path: Path) -> set[str]:
    out: set[str] = set()
    with open(path) as f:
        for line in f:
            out.add(json.loads(line)["tconst"])
    return out


def load_links(path: Path, keep_tconsts: set[str]) -> dict[int, str]:
    """movieId (int) → tconst (with 'tt' prefix), filtered to keep_tconsts."""
    out: dict[int, str] = {}
    with open(path, newline="") as f:
        r = csv.reader(f)
        header = next(r)
        i_mid = header.index("movieId")
        i_imdb = header.index("imdbId")
        for row in r:
            try:
                mid = int(row[i_mid])
            except (ValueError, IndexError):
                continue
            imdb_raw = row[i_imdb] if i_imdb < len(row) else ""
            if not imdb_raw:
                continue
            tc = f"tt{imdb_raw.zfill(7)}"
            if tc in keep_tconsts:
                out[mid] = tc
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ml-dir", default="movies_data/movielens/ml-25m")
    ap.add_argument("--imdb-titles", default="movies_data/titles_imdb.jsonl")
    ap.add_argument("--out-path", default="movies_data/movielens_corated_bags.jsonl")
    ap.add_argument(
        "--min-rating",
        type=float,
        default=4.0,
        help="threshold for 'high' rating (default 4.0/5.0)",
    )
    ap.add_argument("--top-k", type=int, default=50, help="max bag size")
    ap.add_argument(
        "--min-bag-coverage",
        type=int,
        default=5,
        help="minimum co_count to keep a bag entry",
    )
    args = ap.parse_args()

    ml_dir = Path(args.ml_dir)
    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    log("[step 1] load IMDb tconst set")
    tconsts = load_imdb_tconsts(Path(args.imdb_titles))
    log(f"  {len(tconsts):,} tconsts loaded")

    log("[step 2] load MovieLens links (movieId -> tconst), filtered")
    mid2tc = load_links(ml_dir / "links.csv", tconsts)
    log(f"  {len(mid2tc):,} MovieLens movies matched to IMDb tconsts")

    log("[step 3] scan ratings.csv → user-high-rated movie lists")
    user_high: dict[int, list[int]] = defaultdict(list)
    n_rows = 0
    n_kept = 0
    with open(ml_dir / "ratings.csv", newline="") as f:
        r = csv.reader(f)
        header = next(r)
        i_u = header.index("userId")
        i_m = header.index("movieId")
        i_r = header.index("rating")
        for row in r:
            n_rows += 1
            try:
                rating = float(row[i_r])
            except (ValueError, IndexError):
                continue
            if rating < args.min_rating:
                continue
            try:
                mid = int(row[i_m])
            except (ValueError, IndexError):
                continue
            if mid not in mid2tc:
                continue
            try:
                uid = int(row[i_u])
            except (ValueError, IndexError):
                continue
            user_high[uid].append(mid)
            n_kept += 1
            if n_rows % 5_000_000 == 0:
                log(f"  ratings scanned: {n_rows:,} / kept {n_kept:,}")
    log(
        f"  scanned {n_rows:,} ratings, kept {n_kept:,} high ratings across {len(user_high):,} users"
    )

    log("[step 4] build per-movie co-rating counters")
    co: dict[int, Counter] = defaultdict(Counter)
    n_high_raters: Counter = Counter()
    for _uid, mids in user_high.items():
        unique_mids = set(mids)
        for m in unique_mids:
            n_high_raters[m] += 1
        # all-pairs increment (symmetric)
        for m1 in unique_mids:
            cm = co[m1]
            for m2 in unique_mids:
                if m1 == m2:
                    continue
                cm[m2] += 1
    del user_high
    log(f"  co-rating tables built for {len(co):,} movies in {time.time() - t0:.0f}s elapsed")

    log("[step 5] emit bags")
    n_written = 0
    with open(out_path, "w") as out:
        for mid, counter in co.items():
            tc = mid2tc.get(mid)
            if tc is None:
                continue
            top = counter.most_common(args.top_k)
            bag = [
                {"tconst": mid2tc[m2], "co_count": c}
                for m2, c in top
                if c >= args.min_bag_coverage and m2 in mid2tc
            ]
            if not bag:
                continue
            out.write(
                json.dumps(
                    {
                        "tconst": tc,
                        "movielens_id": mid,
                        "n_high_raters": n_high_raters[mid],
                        "bag": bag,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            n_written += 1
    log(f"  wrote {n_written:,} bags -> {out_path}")
    log(f"[done] elapsed {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
