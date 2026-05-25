#!/usr/bin/env python3
"""Left-join the four movies ingest sources on tconst.

Inputs (movies_data/):
  titles_imdb.jsonl              IMDb base catalog (filtered to movie/tvMovie/
                                 tvSeries/tvMiniSeries)
  wikidata_bridge.jsonl          tconst -> qid + enwiki_title (nullable)
  wikipedia_plots.jsonl.gz       tconst -> lead, plot (subset w/ enwiki_title)
  movielens_corated_bags.jsonl   tconst -> bag[] of {tconst, co_count}

Output: unified_movies/metadata.jsonl
  One JSON row per IMDb tconst, with:
    - all IMDb base fields
    - enwiki_title         (str | null)
    - lead                 (str | null)
    - plot                 (str | null)
    - corated_bag          (list[str], possibly empty)  -- tconst-only
    - has_lead, has_plot, has_bag (bool)

Usage:
  .venv/bin/python download/unify_movies_catalogs.py
"""

import argparse
import gzip
import json
import sys
from pathlib import Path


def load_bridge(path: Path) -> dict[str, str]:
    """tconst -> enwiki_title (only rows that have one)."""
    out: dict[str, str] = {}
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            t = r.get("enwiki_title")
            if t:
                out[r["tconst"]] = t
    return out


def load_plots(path: Path) -> dict[str, dict]:
    """tconst -> {lead, plot} (each nullable)."""
    out: dict[str, dict] = {}
    with gzip.open(path, "rt") as f:
        for line in f:
            r = json.loads(line)
            out[r["tconst"]] = {"lead": r.get("lead"), "plot": r.get("plot")}
    return out


def load_bags(path: Path) -> dict[str, list[str]]:
    """tconst -> list[tconst] (bag members, in original order)."""
    out: dict[str, list[str]] = {}
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            out[r["tconst"]] = [b["tconst"] for b in r.get("bag", [])]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="movies_data")
    ap.add_argument("--out", default="unified_movies")
    args = ap.parse_args()

    data = Path(args.data)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    print("loading bridge...", flush=True)
    bridge = load_bridge(data / "wikidata_bridge.jsonl")
    print(f"  {len(bridge):,} tconst -> enwiki_title", flush=True)

    print("loading wikipedia plots...", flush=True)
    plots = load_plots(data / "wikipedia_plots.jsonl.gz")
    print(f"  {len(plots):,} tconst -> {{lead, plot}}", flush=True)

    print("loading movielens bags...", flush=True)
    bags = load_bags(data / "movielens_corated_bags.jsonl")
    print(f"  {len(bags):,} tconst -> bag", flush=True)

    out_path = out / "metadata.jsonl"
    n_total = 0
    n_lead = 0
    n_plot = 0
    n_bag = 0

    print(f"joining onto IMDb base -> {out_path}...", flush=True)
    with open(data / "titles_imdb.jsonl") as fin, open(out_path, "w") as fout:
        for line in fin:
            row = json.loads(line)
            tc = row["tconst"]

            enwiki_title = bridge.get(tc)
            plot_rec = plots.get(tc) or {}
            lead = plot_rec.get("lead")
            plot = plot_rec.get("plot")
            bag = bags.get(tc, [])

            row["enwiki_title"] = enwiki_title
            row["lead"] = lead
            row["plot"] = plot
            row["corated_bag"] = bag
            row["has_lead"] = bool(lead)
            row["has_plot"] = bool(plot)
            row["has_bag"] = bool(bag)

            fout.write(json.dumps(row) + "\n")

            n_total += 1
            if lead:
                n_lead += 1
            if plot:
                n_plot += 1
            if bag:
                n_bag += 1

    size_mb = out_path.stat().st_size / (1024 * 1024)
    print(f"\nwrote {out_path} ({size_mb:.1f} MB)", flush=True)
    print(f"  total rows:      {n_total:>10,}", flush=True)
    print(f"  with lead:       {n_lead:>10,}  ({100 * n_lead / n_total:.1f}%)", flush=True)
    print(f"  with plot:       {n_plot:>10,}  ({100 * n_plot / n_total:.1f}%)", flush=True)
    print(f"  with bag:        {n_bag:>10,}  ({100 * n_bag / n_total:.1f}%)", flush=True)


if __name__ == "__main__":
    sys.exit(main())
