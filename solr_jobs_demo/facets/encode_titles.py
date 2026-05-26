#!/usr/bin/env python3
"""Encode just the title field (not title+description) with bge-small.

The default catalog embeds title+description, but descriptions like
'... AI Trainer - Freelance - 8-20hrs/week ...' dominate the embedding
and break zero-shot role classification. Title-only embeddings are
much cleaner for role inference.
"""

import json
import time
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

META = Path("/Users/dtunkelang/bagofdocs/unified_jobs/metadata.jsonl")
OUT = Path("/Users/dtunkelang/bagofdocs/solr_jobs_demo/facets/title_bge.vecs.fp16.npy")


def main() -> int:
    print("loading titles ...", flush=True)
    titles: list[str] = []
    with open(META) as f:
        for line in f:
            titles.append((json.loads(line).get("title") or "").strip())
    print(f"  {len(titles):,} titles", flush=True)

    print("loading bge-small on mps ...", flush=True)
    t0 = time.time()
    m = SentenceTransformer("BAAI/bge-small-en-v1.5", device="mps")
    print(f"  loaded in {time.time() - t0:.1f}s", flush=True)

    print("encoding (batch=512) ...", flush=True)
    t0 = time.time()
    vecs = m.encode(
        titles,
        batch_size=512,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    print(
        f"  encoded {vecs.shape} in {time.time() - t0:.1f}s "
        f"({len(titles) / (time.time() - t0):.0f}/s)",
        flush=True,
    )

    print(f"saving to {OUT} (fp16) ...", flush=True)
    np.save(OUT, vecs.astype(np.float16))
    print("done.", flush=True)
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
