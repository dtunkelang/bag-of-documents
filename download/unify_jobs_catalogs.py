#!/usr/bin/env python3
"""Concatenate te3 catalogs + doc_ids + titles + metadata across the 4 jobs
corpora into a single unified index for the demo.

Output: unified_jobs/
  te3_catalog.vecs.fp16.npy      (n, 1024) fp16
  doc_ids.json                    list[str], source-prefixed (already unique)
  titles.json                     list[str], aligned to doc_ids
  metadata.jsonl                  one JSON per doc, with 'source_corpus'
  source_index.json               {'starts': {...}, 'sources': [...]}
                                  starts maps corpus -> first index;
                                  sources is per-doc corpus label.

Usage:
  .venv/bin/python download/unify_jobs_catalogs.py --out unified_jobs
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

# (corpus_dir, te3_vecs_filename)
CORPORA = [
    ("jobs_data", "te3_large_1024.vecs.fp16.npy"),
    ("jobs_data_linkedin", "te3_large_1024.vecs.fp16.npy"),
    ("jobs_data_jobstreet", "openai_te3large_1024.vecs.fp16.npy"),
    ("jobs_data_usajobs", "te3_large_1024.vecs.fp16.npy"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="unified_jobs")
    ap.add_argument("--root", default=".")
    args = ap.parse_args()

    root = Path(args.root)
    out = root / args.out
    out.mkdir(parents=True, exist_ok=True)

    # First pass: compute total size
    sizes = {}
    for d, vname in CORPORA:
        v = np.load(root / d / vname, mmap_mode="r")
        sizes[d] = v.shape[0]
    total = sum(sizes.values())
    print(f"unifying {len(CORPORA)} corpora, total {total:,} docs", flush=True)

    # Allocate output
    out_vecs_path = out / "te3_catalog.vecs.fp16.npy"
    np.save(out_vecs_path, np.zeros((total, 1024), dtype=np.float16))
    vecs = np.lib.format.open_memmap(out_vecs_path, mode="r+")

    all_ids: list[str] = []
    all_titles: list[str] = []
    all_sources: list[str] = []
    starts: dict[str, int] = {}
    cursor = 0
    with open(out / "metadata.jsonl", "w") as meta_out:
        for d, vname in CORPORA:
            n = sizes[d]
            starts[d] = cursor
            print(f"  {d}: {n:,} docs starting at {cursor:,}...", flush=True)

            # vecs
            src_vecs = np.load(root / d / vname, mmap_mode="r")
            vecs[cursor : cursor + n] = src_vecs[:]
            vecs.flush()

            # ids + titles
            with open(root / d / "doc_ids.json") as f:
                ids = json.load(f)
            with open(root / d / "titles.json") as f:
                titles = json.load(f)
            if len(ids) != n or len(titles) != n:
                raise SystemExit(f"size mismatch in {d}")
            all_ids.extend(ids)
            all_titles.extend(titles)
            all_sources.extend([d] * n)

            # metadata
            with open(root / d / "metadata.jsonl") as fin:
                for line in fin:
                    rec = json.loads(line)
                    rec["source_corpus"] = d
                    meta_out.write(json.dumps(rec) + "\n")

            cursor += n
    with open(out / "doc_ids.json", "w") as f:
        json.dump(all_ids, f)
    with open(out / "titles.json", "w") as f:
        json.dump(all_titles, f)
    with open(out / "source_index.json", "w") as f:
        json.dump({"starts": starts, "sources": all_sources}, f)

    print(f"\nwrote unified catalog at {out}:", flush=True)
    print(f"  te3_catalog.vecs.fp16.npy  shape=({total:,}, 1024)", flush=True)
    print(f"  doc_ids.json               {len(all_ids):,} ids", flush=True)
    print(f"  titles.json                {len(all_titles):,} titles", flush=True)
    with open(out / "metadata.jsonl") as fin:
        n_meta = sum(1 for _ in fin)
    print(f"  metadata.jsonl             {n_meta:,} records", flush=True)


if __name__ == "__main__":
    main()
