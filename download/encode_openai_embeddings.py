#!/usr/bin/env python3
"""Encode a corpus catalog with an OpenAI embeddings model (text-embedding-3-*).

Mirrors the .vecs.fp16.npy output format of evaluation/eval_alt_encoder.py so
existing eval harnesses can consume the cached vectors. Uses Matryoshka
dimensionality truncation (`--dim` flag) to keep storage manageable.

Batched + checkpointed: writes a progress.json after each successful batch
so a kill mid-run can resume cleanly.

Cost reference (2026-05, $/M tokens):
- text-embedding-3-large: $0.13
- text-embedding-3-small: $0.02
Typical product title is ~25-40 tokens; 1M titles → ~$3-5 for large, $0.50-0.80 for small.

Usage:
    .venv/bin/python download/encode_openai_embeddings.py \\
        --data-dir bestbuy_acm_data \\
        --model text-embedding-3-large \\
        --dim 1024 \\
        --out-name openai_te3large_1024
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse  # noqa: E402
import datetime  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
from dotenv import load_dotenv  # noqa: E402

# override=True so .env always wins over stale shell exports
# (a stray `export OPENAI_API_KEY=...` in ~/.zshrc would otherwise shadow .env)
load_dotenv(override=True)

from openai import OpenAI  # noqa: E402

PRICES_PER_M_TOKENS = {
    "text-embedding-3-large": 0.13,
    "text-embedding-3-small": 0.02,
    "text-embedding-ada-002": 0.10,
}

SPEND_LEDGER = Path(__file__).resolve().parent.parent / ".api_spend.jsonl"


def record_spend(provider: str, model: str, tokens: int, cost_usd: float, purpose: str):
    """Append a single completed-run record to the local spend ledger.

    The ledger is gitignored. This is for the user's own bookkeeping only;
    the authoritative record is the provider's dashboard.
    """
    rec = {
        "ts": datetime.datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "provider": provider,
        "model": model,
        "tokens": int(tokens),
        "cost_usd": round(float(cost_usd), 4),
        "purpose": purpose,
    }
    with open(SPEND_LEDGER, "a") as f:
        f.write(json.dumps(rec) + "\n")


def encode_batch(client: OpenAI, texts: list[str], model: str, dim: int | None):
    kwargs = {"model": model, "input": texts}
    if dim is not None:
        kwargs["dimensions"] = dim
    resp = client.embeddings.create(**kwargs)
    vecs = np.array([d.embedding for d in resp.data], dtype=np.float32)
    # L2-normalize (text-embedding-3 returns normalized at full dim but post-truncation
    # the slice is not guaranteed normalized; cheap insurance).
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    vecs = vecs / np.maximum(norms, 1e-9)
    usage = resp.usage.total_tokens if hasattr(resp, "usage") else 0
    return vecs.astype(np.float16), usage


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument(
        "--titles-file",
        default="titles.json",
        help="JSON list of texts to encode (default: catalog titles.json)",
    )
    ap.add_argument("--model", default="text-embedding-3-large")
    ap.add_argument(
        "--dim",
        type=int,
        default=None,
        help="output dimensionality (Matryoshka truncation). Default: full model dim "
        "(3072 for -large, 1536 for -small). Set 1024 for storage efficiency.",
    )
    ap.add_argument("--batch-size", type=int, default=512, help="texts per API request")
    ap.add_argument("--out-name", required=True, help="stem for .vecs.fp16.npy")
    ap.add_argument(
        "--max-texts",
        type=int,
        default=0,
        help="cap for testing (0 = all)",
    )
    args = ap.parse_args()

    data = Path(args.data_dir).resolve()
    with open(data / args.titles_file) as f:
        texts = json.load(f)
    if args.max_texts and args.max_texts < len(texts):
        texts = texts[: args.max_texts]
    n = len(texts)
    print(f"corpus: {data.name}  texts: {n:,}  model: {args.model}  dim: {args.dim}", flush=True)

    # Probe one batch to discover the model's output dim, then allocate.
    client = OpenAI()
    print("probing model output dim...", flush=True)
    probe_vecs, probe_tokens = encode_batch(client, texts[:1], args.model, args.dim)
    out_dim = probe_vecs.shape[1]
    print(f"  output dim: {out_dim}", flush=True)

    out_vecs_path = data / f"{args.out_name}.vecs.fp16.npy"
    progress_path = data / f"{args.out_name}.progress.json"

    if not out_vecs_path.exists():
        np.save(out_vecs_path, np.zeros((n, out_dim), dtype=np.float16))
    vecs = np.lib.format.open_memmap(out_vecs_path, mode="r+")
    if vecs.shape != (n, out_dim):
        raise SystemExit(f"existing {out_vecs_path} has shape {vecs.shape}; want ({n},{out_dim})")

    done = set()
    if progress_path.exists():
        with open(progress_path) as f:
            done = set(json.load(f).get("done_batches", []))

    n_batches = (n + args.batch_size - 1) // args.batch_size
    print(f"\nencoding {n:,} texts in {n_batches} batches of {args.batch_size}", flush=True)
    print(f"  resume: {len(done):,}/{n_batches:,} batches already done", flush=True)

    t0 = time.time()
    total_tokens = 0
    new_batches = 0
    for bi in range(n_batches):
        if bi in done:
            continue
        start = bi * args.batch_size
        end = min(start + args.batch_size, n)
        batch_texts = texts[start:end]
        # Empty / very short text guards
        batch_texts = [t if t and t.strip() else " " for t in batch_texts]
        # Retry on transient errors
        for attempt in range(5):
            try:
                v, tokens = encode_batch(client, batch_texts, args.model, args.dim)
                break
            except Exception as e:
                print(
                    f"  batch {bi} attempt {attempt + 1} failed: {type(e).__name__}: {str(e)[:120]}",
                    flush=True,
                )
                if attempt == 4:
                    raise
                time.sleep(2**attempt)
        vecs[start:end] = v
        vecs.flush()
        done.add(bi)
        total_tokens += tokens
        new_batches += 1
        with open(progress_path, "w") as f:
            json.dump({"done_batches": sorted(done), "tokens_this_run": total_tokens}, f)
        if new_batches % 20 == 0 or bi == n_batches - 1:
            elapsed = time.time() - t0
            rate = new_batches / max(elapsed, 1e-3)
            remaining = (n_batches - len(done)) / max(rate, 1e-3) / 60
            cost_so_far = total_tokens * PRICES_PER_M_TOKENS.get(args.model, 0) / 1e6
            print(
                f"  batch {bi + 1}/{n_batches}  ({rate:.1f} batch/s)  "
                f"tokens={total_tokens:,}  cost=${cost_so_far:.3f}  eta {remaining:.1f}m",
                flush=True,
            )

    elapsed = time.time() - t0
    cost = total_tokens * PRICES_PER_M_TOKENS.get(args.model, 0) / 1e6
    print(
        f"\ndone: {n:,} texts in {elapsed / 60:.1f}m  tokens={total_tokens:,}  cost=${cost:.3f}",
        flush=True,
    )
    print(f"saved {out_vecs_path}  shape={vecs.shape}", flush=True)
    record_spend(
        provider="openai",
        model=args.model,
        tokens=total_tokens,
        cost_usd=cost,
        purpose=f"encode {data.name}/{args.titles_file} dim={args.dim} -> {args.out_name}",
    )
    print(f"appended spend record to {SPEND_LEDGER}", flush=True)


if __name__ == "__main__":
    main()
