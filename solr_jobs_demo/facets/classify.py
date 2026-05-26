#!/usr/bin/env python3
"""Run gpt-4o-mini structured-output classification on a sample of jobs.

Usage:
  python classify.py --sample 200 --out pilot.jsonl
  python classify.py --all --out full.jsonl --workers 32
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from openai import OpenAI

sys.path.insert(0, str(Path(__file__).parent))
from taxonomy import SYSTEM_PROMPT, json_schema  # noqa: E402

META = Path("/Users/dtunkelang/bagofdocs/unified_jobs/metadata.jsonl")
MODEL = "gpt-4o-mini"
DESC_CHARS = 2000  # title + first 2000 chars of description = ~500 input tokens


def build_prompt(rec: dict) -> str:
    title = (rec.get("title") or "").strip()
    desc = (rec.get("description") or "").strip()[:DESC_CHARS]
    return f"Title: {title}\n\nDescription excerpt:\n{desc}"


def classify_one(client: OpenAI, idx: int, rec: dict) -> tuple[int, dict | None, str | None]:
    try:
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_prompt(rec)},
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "job_facets",
                    "strict": True,
                    "schema": json_schema(),
                },
            },
            temperature=0.0,
            max_completion_tokens=400,
        )
        data = json.loads(resp.choices[0].message.content)
        return idx, data, None
    except Exception as e:
        return idx, None, repr(e)


def stream_records(line_indices: list[int] | None = None):
    """Yield (idx, rec). If line_indices given, return only those by line number."""
    want = set(line_indices) if line_indices else None
    with open(META) as f:
        for i, line in enumerate(f):
            if want is not None and i not in want:
                continue
            yield i, json.loads(line)
            if want is not None and len(want) == 1:
                # Tiny optimization for single-line lookups; not common path.
                return


def main() -> int:
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--sample", type=int, help="random sample size for pilot")
    g.add_argument("--all", action="store_true", help="classify all jobs")
    ap.add_argument("--out", required=True, help="output JSONL path")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--resume", action="store_true", help="skip indices already present in --out")
    args = ap.parse_args()

    client = OpenAI()

    # Count total docs (cheap; just scan headers).
    n_total = 0
    with open(META) as f:
        for _ in f:
            n_total += 1
    print(f"corpus: {n_total:,} docs", flush=True)

    if args.sample:
        rng = random.Random(args.seed)
        indices = sorted(rng.sample(range(n_total), args.sample))
    else:
        indices = list(range(n_total))

    done: set[int] = set()
    if args.resume and os.path.exists(args.out):
        with open(args.out) as f:
            for line in f:
                try:
                    done.add(json.loads(line)["idx"])
                except Exception:
                    pass
        print(f"resume: {len(done):,} already done", flush=True)
        indices = [i for i in indices if i not in done]

    records = list(stream_records(indices))
    print(f"to classify: {len(records):,}", flush=True)

    t0 = time.time()
    written = 0
    errors = 0
    with open(args.out, "a") as out_f, ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(classify_one, client, idx, rec): idx for idx, rec in records}
        for fut in as_completed(futs):
            idx, data, err = fut.result()
            if err is not None:
                errors += 1
                out_f.write(json.dumps({"idx": idx, "error": err}) + "\n")
            else:
                out_f.write(json.dumps({"idx": idx, **data}) + "\n")
                written += 1
            if (written + errors) % 50 == 0:
                out_f.flush()
                rate = (written + errors) / (time.time() - t0)
                print(f"  {written + errors:,} done ({rate:.1f}/s, errors={errors})", flush=True)
    print(f"done: wrote {written:,} ({errors} errors) in {time.time() - t0:.1f}s", flush=True)
    return 0 if errors == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
