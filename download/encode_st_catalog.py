#!/usr/bin/env python3
"""Encode a corpus catalog with a sentence-transformers model.

Mirrors the .vecs.fp16.npy output format used by eval_alt_encoder.py and
evaluation/eval_openai_embeddings.py, so existing eval harnesses can consume
the cached vectors.

Resumable via a sidecar progress.json (records completed chunks).

Usage:
  .venv/bin/python download/encode_st_catalog.py \\
      --data-dir jobs_data \\
      --model sentence-transformers/all-MiniLM-L6-v2 \\
      --out-name base_catalog
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--titles-file", default="titles.json")
    ap.add_argument("--model", required=True)
    ap.add_argument("--out-name", required=True, help="stem for .vecs.fp16.npy")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--chunk-size", type=int, default=50_000)
    ap.add_argument("--device", default="mps", help="mps / cuda / cpu")
    ap.add_argument(
        "--max-seq-length",
        type=int,
        default=0,
        help="override model.max_seq_length (0 = keep model default)",
    )
    ap.add_argument(
        "--doc-prefix", default="", help="optional prefix prepended to each doc (e.g., 'passage: ')"
    )
    ap.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="enable for models with custom architectures (e.g., gte-v1.5, nomic-embed)",
    )
    args = ap.parse_args()

    from sentence_transformers import SentenceTransformer

    data = Path(args.data_dir).resolve()
    with open(data / args.titles_file) as f:
        titles = json.load(f)
    print(f"loaded {len(titles):,} docs from {data / args.titles_file}", flush=True)

    if args.doc_prefix:
        titles = [args.doc_prefix + t for t in titles]
        print(f"applied doc-prefix {args.doc_prefix!r}", flush=True)

    print(f"loading {args.model} on {args.device}...", flush=True)
    t0 = time.time()
    st_kwargs = {"device": args.device}
    if args.trust_remote_code:
        st_kwargs["trust_remote_code"] = True
    model = SentenceTransformer(args.model, **st_kwargs)
    if args.max_seq_length > 0:
        model.max_seq_length = args.max_seq_length
    dim = model.get_sentence_embedding_dimension()
    print(f"  loaded in {time.time() - t0:.1f}s; dim={dim}", flush=True)

    vec_path = data / f"{args.out_name}.vecs.fp16.npy"
    progress_path = data / f"{args.out_name}.progress.json"

    n = len(titles)
    n_chunks = (n + args.chunk_size - 1) // args.chunk_size
    vecs = np.lib.format.open_memmap(
        vec_path, mode="w+" if not vec_path.exists() else "r+", dtype=np.float16, shape=(n, dim)
    )

    done = set()
    if progress_path.exists():
        try:
            with open(progress_path) as f:
                done = set(json.load(f))
            print(f"resuming: {len(done)}/{n_chunks} chunks already encoded", flush=True)
        except Exception:
            done = set()

    t_start = time.time()
    for ci in range(n_chunks):
        if ci in done:
            continue
        s = ci * args.chunk_size
        e = min(s + args.chunk_size, n)
        t1 = time.time()
        chunk_vecs = model.encode(
            titles[s:e],
            normalize_embeddings=True,
            batch_size=args.batch_size,
            show_progress_bar=False,
        ).astype(np.float16)
        vecs[s:e] = chunk_vecs
        vecs.flush()
        done.add(ci)
        with open(progress_path, "w") as f:
            json.dump(sorted(done), f)
        dur = time.time() - t1
        elapsed = time.time() - t_start
        remaining = n_chunks - len(done)
        eta_s = (elapsed / max(len(done), 1)) * remaining if remaining > 0 else 0
        print(
            f"  chunk {ci + 1}/{n_chunks} ({e:,}/{n:,}) "
            f"{dur:.1f}s ({(e - s) / dur:.0f} docs/s) ETA {eta_s / 60:.1f}min",
            flush=True,
        )

    print(f"\nwrote {vec_path} shape={vecs.shape} dtype={vecs.dtype}", flush=True)


if __name__ == "__main__":
    main()
