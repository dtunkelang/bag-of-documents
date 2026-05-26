#!/usr/bin/env python3
"""Industry-only classification via local Ollama (qwen2.5:7b-instruct).

Fills the one facet field the regex heuristic can't produce: employer
industry. Output JSONL is one record per doc:
    {"idx": <line in metadata.jsonl>, "industry": "<enum>"}
or {"idx": ..., "error": "..."} on failure.

Usage:
  # smoke test
  python classify_industry_ollama.py --sample 100 --out industry.smoke.jsonl
  # full run, resumable
  python classify_industry_ollama.py --all --out industry.jsonl --resume
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from taxonomy import INDUSTRY  # noqa: E402

META = Path("/Users/dtunkelang/bagofdocs/unified_jobs/metadata.jsonl")
OLLAMA_URL = "http://localhost:11434/api/chat"
DEFAULT_MODEL = "qwen2.5:7b-instruct"
DESC_CHARS = 1500  # title + slug + first 1500 chars of description

SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["industry"],
    "properties": {
        "industry": {"type": "string", "enum": INDUSTRY},
    },
}

SYSTEM_PROMPT = """You classify the EMPLOYER's industry for a job listing.

You are given a job title, an ATS slug that is usually the employer's name or
abbreviation, and an excerpt of the job description. Output one JSON object
with a single field `industry`, picking from the allowed enum.

Rules:
- Classify the EMPLOYER's industry, NOT the role's function. A software
  engineer at a bank is `finance_banking`, not `tech_software_internet`.
- For staffing agencies / recruiting firms posting on behalf of another
  company, use the actual hiring company's industry when clearly stated,
  otherwise `consulting_professional_services`.
- Use `other` only when the description gives no usable signal.
- Pick a single best match. Be strict about enum values."""


def build_user_prompt(rec: dict) -> str:
    title = (rec.get("title") or "").strip()
    slug = (rec.get("source_slug") or "").strip()
    desc = (rec.get("description") or "").strip()[:DESC_CHARS]
    return f"Title: {title}\nEmployer slug: {slug}\n\nDescription excerpt:\n{desc}"


def classify_one(rec: dict, model: str, timeout: int = 60) -> tuple[dict | None, str | None]:
    body = {
        "model": model,
        "stream": False,
        "format": SCHEMA,
        "options": {"temperature": 0.0, "num_predict": 50},
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(rec)},
        ],
    }
    req = urllib.request.Request(
        OLLAMA_URL,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            payload = json.loads(r.read())
    except urllib.error.URLError as e:
        return None, f"URLError: {e}"
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"
    content = payload.get("message", {}).get("content", "")
    try:
        data = json.loads(content)
    except json.JSONDecodeError as e:
        return None, f"JSONDecodeError: {e}; raw={content[:200]!r}"
    if data.get("industry") not in INDUSTRY:
        return None, f"out-of-enum: {data.get('industry')!r}"
    return data, None


def count_lines(p: Path) -> int:
    n = 0
    with open(p) as f:
        for _ in f:
            n += 1
    return n


def load_done_indices(out_path: str) -> set[int]:
    done: set[int] = set()
    if not os.path.exists(out_path):
        return done
    with open(out_path) as f:
        for line in f:
            try:
                done.add(json.loads(line)["idx"])
            except Exception:
                pass
    return done


def stream_records(want: set[int]):
    with open(META) as f:
        for i, line in enumerate(f):
            if i not in want:
                continue
            yield i, json.loads(line)


def main() -> int:
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--sample", type=int, help="random sample size")
    g.add_argument("--all", action="store_true", help="classify all jobs")
    ap.add_argument("--out", required=True, help="output JSONL path")
    ap.add_argument("--model", default=DEFAULT_MODEL, help="ollama model tag")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--resume", action="store_true", help="skip indices already present in --out")
    ap.add_argument("--flush-every", type=int, default=25)
    ap.add_argument("--progress-every", type=int, default=50)
    args = ap.parse_args()

    n_total = count_lines(META)
    print(f"corpus: {n_total:,} docs in {META}", flush=True)

    if args.sample:
        rng = random.Random(args.seed)
        indices = set(rng.sample(range(n_total), args.sample))
    else:
        indices = set(range(n_total))

    done = load_done_indices(args.out) if args.resume else set()
    if done:
        print(f"resume: {len(done):,} already done", flush=True)
    indices -= done
    print(f"to classify: {len(indices):,}", flush=True)
    if not indices:
        print("nothing to do; exiting", flush=True)
        return 0

    t0 = time.time()
    written = errors = 0
    with open(args.out, "a") as out_f:
        for idx, rec in stream_records(indices):
            data, err = classify_one(rec, args.model)
            if err is not None:
                errors += 1
                out_f.write(json.dumps({"idx": idx, "error": err}) + "\n")
            else:
                out_f.write(json.dumps({"idx": idx, **data}) + "\n")
                written += 1
            total = written + errors
            if total % args.flush_every == 0:
                out_f.flush()
                os.fsync(out_f.fileno())
            if total % args.progress_every == 0:
                rate = total / (time.time() - t0)
                remaining = (len(indices) - total) / rate if rate else 0
                print(
                    f"  {total:,}/{len(indices):,} "
                    f"({rate:.2f}/s, errors={errors}, "
                    f"eta={remaining / 3600:.1f}h)",
                    flush=True,
                )
    print(
        f"done: wrote {written:,} ({errors} errors) in {time.time() - t0:.1f}s",
        flush=True,
    )
    return 0 if errors == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
