#!/usr/bin/env python3
"""Encode title + lead for movies with has_lead=true using bge-small-en-v1.5.

Streams unified_movies/metadata.jsonl, picks rows with has_lead=true, builds
"title. lead" text, encodes on MPS, writes fp16 vectors + tconst ids in
checkpointed shards so the run can resume.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

DEFAULT_META = "/Users/dtunkelang/bagofdocs/unified_movies/metadata.jsonl"
DEFAULT_OUT = "/Users/dtunkelang/bagofdocs/solr_movies_demo/bge_vecs"
MODEL_NAME = "BAAI/bge-small-en-v1.5"
SHARD_SIZE = 20_000


def iter_targets(meta_path: str):
    with open(meta_path) as f:
        for line in f:
            d = json.loads(line)
            if not d.get("has_lead"):
                continue
            title = d.get("title") or ""
            lead = d.get("lead") or ""
            text = (title + ". " + lead).strip()
            if not text:
                continue
            yield d["tconst"], text


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta", default=DEFAULT_META)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--shard-size", type=int, default=SHARD_SIZE)
    ap.add_argument("--max-seq", type=int, default=384)
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    done_ids: set[str] = set()
    for shard in sorted(out_dir.glob("shard_*.npz")):
        with np.load(shard, allow_pickle=False) as z:
            done_ids.update(z["ids"].tolist())
    print(
        f"resume: {len(done_ids):,} already encoded across "
        f"{len(list(out_dir.glob('shard_*.npz')))} shard(s)"
    )

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"device={device} model={MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME, device=device)
    model.max_seq_length = args.max_seq

    next_shard = len(list(out_dir.glob("shard_*.npz")))
    buf_ids: list[str] = []
    buf_texts: list[str] = []
    total_done = len(done_ids)
    t0 = time.time()

    def flush_shard() -> None:
        nonlocal next_shard, total_done
        if not buf_ids:
            return
        vecs = model.encode(
            buf_texts,
            batch_size=args.batch,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        ).astype(np.float16)
        path = out_dir / f"shard_{next_shard:04d}.npz"
        np.savez(path, ids=np.array(buf_ids), vecs=vecs)
        total_done += len(buf_ids)
        dt = time.time() - t0
        rate = total_done / dt if dt > 0 else 0
        print(
            f"shard {next_shard:04d}: {len(buf_ids):,} vecs → {path.name} "
            f"| cumulative {total_done:,} | {rate:.0f}/s"
        )
        next_shard += 1
        buf_ids.clear()
        buf_texts.clear()

    for tconst, text in iter_targets(args.meta):
        if tconst in done_ids:
            continue
        buf_ids.append(tconst)
        buf_texts.append(text)
        if len(buf_ids) >= args.shard_size:
            flush_shard()
    flush_shard()
    print(f"done. total {total_done:,} vecs in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
