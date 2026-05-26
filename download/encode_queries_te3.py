#!/usr/bin/env python3
"""Encode a JSONL of queries with te3-large, save to .npy + .ids.json.

Output two files side-by-side:
  <out>.vecs.fp16.npy   shape=(n, dim), dtype=float16
  <out>.ids.json        list of query strings in row order

Usage:
  .venv/bin/python download/encode_queries_te3.py \\
      --queries-file jobs_data/train_queries.jsonl \\
      --model text-embedding-3-large --dim 1024 \\
      --batch-size 512 \\
      --out jobs_data/train_queries_te3_1024
"""

import argparse
import datetime
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
from dotenv import load_dotenv  # noqa: E402

load_dotenv(override=True)

SPEND_LEDGER = Path(__file__).resolve().parent.parent / ".api_spend.jsonl"
PRICES = {"text-embedding-3-large": 0.13, "text-embedding-3-small": 0.02}


def record_spend(model, tokens, cost, purpose):
    rec = {
        "ts": datetime.datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "provider": "openai",
        "model": model,
        "tokens": int(tokens),
        "cost_usd": round(float(cost), 4),
        "purpose": purpose,
    }
    with open(SPEND_LEDGER, "a") as f:
        f.write(json.dumps(rec) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--queries-file", required=True)
    ap.add_argument("--model", default="text-embedding-3-large")
    ap.add_argument("--dim", type=int, default=1024)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--out", required=True, help="stem (no extension)")
    args = ap.parse_args()

    queries = []
    with open(args.queries_file) as f:
        for line in f:
            queries.append(json.loads(line)["query"])
    print(f"loaded {len(queries):,} queries", flush=True)

    from openai import OpenAI

    client = OpenAI()
    out_vecs = []
    tokens = 0
    t0 = time.time()
    for i in range(0, len(queries), args.batch_size):
        sub = queries[i : i + args.batch_size]
        resp = client.embeddings.create(
            model=args.model, input=sub, dimensions=args.dim if args.dim else None
        )
        out_vecs.extend([d.embedding for d in resp.data])
        tokens += resp.usage.total_tokens
        rate = (i + len(sub)) / max(time.time() - t0, 1e-3)
        if (i // args.batch_size) % 4 == 0 or i + args.batch_size >= len(queries):
            print(
                f"  {min(i + args.batch_size, len(queries)):,}/{len(queries):,}  "
                f"tokens={tokens:,}  ({rate:.0f}/s)",
                flush=True,
            )

    vecs = np.array(out_vecs, dtype=np.float16)
    cost = tokens * PRICES[args.model] / 1e6
    record_spend(args.model, tokens, cost, f"te3 query encode {args.queries_file}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path.with_suffix(".vecs.fp16.npy"), vecs)
    with open(out_path.with_suffix(".ids.json"), "w") as f:
        json.dump(queries, f)
    print(
        f"\nwrote {vecs.shape} to {out_path}.vecs.fp16.npy  (tokens={tokens:,} cost=${cost:.4f})",
        flush=True,
    )


if __name__ == "__main__":
    main()
