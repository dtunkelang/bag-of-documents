#!/usr/bin/env python3
"""Push encoded bge_vec vectors into Solr movies core via atomic update.

Reads shard_*.npz from bge_vecs/, posts batches of {"id": ..., "bge_vec":
{"set": [...]}} to /solr/movies/update?commit=false. Final commit at end.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import requests

DEFAULT_IN = "/Users/dtunkelang/bagofdocs/solr_movies_demo/bge_vecs"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", default=DEFAULT_IN)
    ap.add_argument("--solr", default="http://localhost:8984")
    ap.add_argument("--core", default="movies")
    ap.add_argument("--batch", type=int, default=500)
    args = ap.parse_args()

    in_dir = Path(args.in_dir)
    shards = sorted(in_dir.glob("shard_*.npz"))
    if not shards:
        print(f"no shards in {in_dir}", file=sys.stderr)
        sys.exit(1)

    update_url = f"{args.solr}/solr/{args.core}/update"
    sess = requests.Session()
    total = 0
    t0 = time.time()
    buf: list[dict] = []

    def flush() -> None:
        nonlocal total
        if not buf:
            return
        r = sess.post(
            update_url,
            params={"commit": "false"},
            headers={"Content-Type": "application/json"},
            data=json.dumps(buf),
            timeout=120,
        )
        r.raise_for_status()
        total += len(buf)
        buf.clear()

    for shard in shards:
        with np.load(shard, allow_pickle=False) as z:
            ids = z["ids"]
            vecs = z["vecs"].astype(np.float32)
        for tconst, v in zip(ids, vecs):
            buf.append({"id": str(tconst), "bge_vec": {"set": v.tolist()}})
            if len(buf) >= args.batch:
                flush()
        dt = time.time() - t0
        rate = total / dt if dt > 0 else 0
        print(f"{shard.name}: cumulative {total:,} | {rate:.0f}/s")

    flush()
    print(f"committing... total {total:,}")
    r = sess.get(f"{update_url}", params={"commit": "true"}, timeout=300)
    r.raise_for_status()
    print(f"done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
