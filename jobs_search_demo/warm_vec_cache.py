#!/usr/bin/env python3
"""Seed the content-addressed vector cache from an already-encoded positional vecs
file, WITHOUT re-encoding.

stage_encode (refresh.py) keeps a persistent {title-hash -> vector} cache so each
nightly refresh only encodes titles new since the last run. A *full* encode (e.g. the
first run after this feature shipped, or a manual one-off) produces the positional
`{EMBED_OUT_NAME}.vecs.fp16.npy` but -- if it ran old code -- no cache. This tool
builds the cache from that vecs file in seconds, so the next refresh starts warm and
finishes in minutes instead of bootstrapping with another full ~90min encode.

Usage: python jobs_search_demo/warm_vec_cache.py [--out-dir DIR]
"""

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("refresh", HERE / "refresh.py")
refresh = importlib.util.module_from_spec(spec)
spec.loader.exec_module(refresh)

ap = argparse.ArgumentParser()
ap.add_argument("--out-dir", default=str(HERE.parent / "unified_jobs_daily"))
args = ap.parse_args()

out = Path(args.out_dir)
name = refresh.EMBED_OUT_NAME
raw = (out / "titles.json").read_bytes()
titles = json.loads(raw)
vec = np.load(out / f"{name}.vecs.fp16.npy", mmap_mode="r")
if vec.shape[0] != len(titles):
    raise SystemExit(f"row/title mismatch: {vec.shape[0]} vecs vs {len(titles)} titles")

idx: dict[str, int] = {}
keys: list[str] = []
vecs: list = []
for i, t in enumerate(titles):
    h = refresh._title_hash(t)
    if h not in idx:
        idx[h] = len(vecs)
        keys.append(h)
        vecs.append(np.array(vec[i], dtype=np.float16))

np.save(out / f"{name}.cache.vecs.fp16.npy", np.asarray(vecs, dtype=np.float16))
with open(out / f"{name}.cache_keys.json", "w") as f:
    json.dump(keys, f)
(out / f"{name}.titles.sha").write_text(hashlib.blake2b(raw, digest_size=16).hexdigest())
print(f"warmed cache: {len(keys):,} unique vectors from {len(titles):,} titles", flush=True)
