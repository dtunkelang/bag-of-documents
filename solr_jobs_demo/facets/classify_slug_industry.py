#!/usr/bin/env python3
"""Slug-aggregate industry classifier.

For each employer slug, concatenate text from up to N of its postings and
classify the EMPLOYER's industry once. Output JSONL one record per slug:
    {"slug": "...", "industry": "<enum>", "n_used": N, "n_total": M}

Usage:
  # validate on 50 random slugs from the top-1000 by post count
  python classify_slug_industry.py --sample 50 --min-posts 5 \
      --out slug_validate.jsonl --model qwen2.5:3b-instruct

  # full top-K run
  python classify_slug_industry.py --top 10000 \
      --out slug_industry.jsonl --resume --model qwen2.5:3b-instruct
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
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from taxonomy import INDUSTRY  # noqa: E402

META = Path("/Users/dtunkelang/bagofdocs/unified_jobs/metadata.jsonl")
OLLAMA_URL = "http://localhost:11434/api/chat"
DEFAULT_MODEL = "qwen2.5:3b-instruct"
POSTS_PER_SLUG = 5
CHARS_PER_POST = 400  # ~80-100 tokens; 5 posts -> ~400-500 tokens total

SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["industry"],
    "properties": {"industry": {"type": "string", "enum": INDUSTRY}},
}

SYSTEM_PROMPT = """You classify the EMPLOYER's industry given multiple job
listings from the same employer.

You are given several postings (title + description excerpt) from a single
employer. Use the combined signal to identify the employer's industry. Output
one JSON object with a single field `industry` from the allowed enum.

Rules:
- Classify the EMPLOYER's industry, NOT any individual role's function. A
  software engineer at a bank is `finance_banking`, not `tech_software_internet`.
- For staffing agencies / recruiting firms posting on behalf of multiple
  clients, use `consulting_professional_services`.
- Use `other` only when the postings give no usable industry signal.
- Be strict about enum values. Pick exactly one."""


def build_user_prompt(slug: str, posts: list[dict]) -> str:
    parts = [f"Employer slug: {slug}", f"({len(posts)} postings shown)\n"]
    for i, p in enumerate(posts, 1):
        title = (p.get("title") or "").strip()
        desc = (p.get("description") or "").strip()[:CHARS_PER_POST]
        parts.append(f"--- Posting {i} ---")
        parts.append(f"Title: {title}")
        parts.append(f"Description: {desc}\n")
    return "\n".join(parts)


def classify_one(
    slug: str, posts: list[dict], model: str, timeout: int = 90
) -> tuple[dict | None, str | None]:
    body = {
        "model": model,
        "stream": False,
        "format": SCHEMA,
        "options": {"temperature": 0.0, "num_predict": 50},
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(slug, posts)},
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


def build_slug_index() -> dict[str, list[dict]]:
    """Group postings by slug. Each posting kept as dict with title+desc.

    Memory: ~1 GB peak for 348k docs at ~3KB each truncated. To stay light,
    we only retain title + truncated description per posting.
    """
    by_slug: dict[str, list[dict]] = defaultdict(list)
    with open(META) as f:
        for line in f:
            d = json.loads(line)
            slug = (d.get("source_slug") or "").strip()
            if not slug:
                continue
            by_slug[slug].append(
                {
                    "title": d.get("title") or "",
                    "description": (d.get("description") or "")[:CHARS_PER_POST],
                }
            )
    return by_slug


def pick_posts(posts: list[dict], k: int, rng: random.Random) -> list[dict]:
    if len(posts) <= k:
        return posts
    return rng.sample(posts, k)


def load_done_slugs(out_path: str) -> set[str]:
    done: set[str] = set()
    if not os.path.exists(out_path):
        return done
    with open(out_path) as f:
        for line in f:
            try:
                done.add(json.loads(line)["slug"])
            except Exception:
                pass
    return done


def main() -> int:
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--sample", type=int, help="random sample of N slugs (use with --min-posts)")
    g.add_argument("--top", type=int, help="top K slugs by post count")
    g.add_argument("--all", action="store_true", help="classify all slugs")
    ap.add_argument(
        "--min-posts",
        type=int,
        default=2,
        help="only consider slugs with at least this many postings",
    )
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--flush-every", type=int, default=25)
    ap.add_argument("--progress-every", type=int, default=25)
    args = ap.parse_args()

    print("indexing postings by slug...", flush=True)
    by_slug = build_slug_index()
    print(f"  {len(by_slug):,} unique slugs", flush=True)

    eligible = [s for s, posts in by_slug.items() if len(posts) >= args.min_posts]
    eligible.sort(key=lambda s: -len(by_slug[s]))
    print(f"  {len(eligible):,} slugs with >= {args.min_posts} posts", flush=True)

    rng = random.Random(args.seed)
    if args.sample:
        # Sample from the top-1000-eligible bucket for stability of validation.
        pool = eligible[: min(len(eligible), 1000)]
        target_slugs = rng.sample(pool, min(args.sample, len(pool)))
    elif args.top:
        target_slugs = eligible[: args.top]
    else:  # --all
        target_slugs = eligible
    print(f"target slugs: {len(target_slugs):,}", flush=True)

    done = load_done_slugs(args.out) if args.resume else set()
    if done:
        print(f"resume: {len(done):,} already done", flush=True)
    target_slugs = [s for s in target_slugs if s not in done]
    print(f"to classify: {len(target_slugs):,}", flush=True)
    if not target_slugs:
        print("nothing to do", flush=True)
        return 0

    t0 = time.time()
    written = errors = 0
    with open(args.out, "a") as out_f:
        for slug in target_slugs:
            posts = by_slug[slug]
            chosen = pick_posts(posts, POSTS_PER_SLUG, rng)
            data, err = classify_one(slug, chosen, args.model)
            rec: dict = {"slug": slug, "n_total": len(posts), "n_used": len(chosen)}
            if err is not None:
                errors += 1
                rec["error"] = err
            else:
                rec["industry"] = data["industry"]
                written += 1
            out_f.write(json.dumps(rec) + "\n")
            total = written + errors
            if total % args.flush_every == 0:
                out_f.flush()
                os.fsync(out_f.fileno())
            if total % args.progress_every == 0:
                rate = total / (time.time() - t0)
                remaining = (len(target_slugs) - total) / rate if rate else 0
                print(
                    f"  {total:,}/{len(target_slugs):,} "
                    f"({rate:.2f} slugs/s, errors={errors}, "
                    f"eta={remaining / 3600:.2f}h)",
                    flush=True,
                )
    print(
        f"done: wrote {written:,} ({errors} errors) in {time.time() - t0:.1f}s",
        flush=True,
    )
    return 0 if errors == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
