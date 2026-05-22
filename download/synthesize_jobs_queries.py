#!/usr/bin/env python3
"""Synthesize job-search queries for the jobs_data corpus.

Three-source blend matching the user-articulated query-distribution goal
(see project_jobs_track.md):

1. HEAD queries from title frequency distribution of the catalog itself.
   Top-N most-frequent (normalized) titles are essentially the head of the
   real query distribution — no LLM needed, no scraping.

2. DISTILLED queries: LLM-generated from a sample of jobs, with a
   distribution-aware prompt covering the styles a real candidate would type:
   short title-only, title+attribute, multi-constraint, conversational NL.

3. AUTOCOMPLETE (not implemented here — would require live API calls to
   Indeed/Google; skipped in this script).

Train/eval split:
  - Job IDs are partitioned into TRAIN_JOBS and EVAL_JOBS (disjoint).
  - DISTILLED queries are sourced from the corresponding split.
  - HEAD queries are also split (deterministically by query slug hash) so
    no query string appears in both train and eval.

Outputs:
  jobs_data/train_queries.jsonl       {query_id, query, source: "head"|"distilled", source_doc_id}
  jobs_data/eval_queries.jsonl        same
  jobs_data/job_split.json            {"train_jobs": [...], "eval_jobs": [...]}

Usage:
  .venv/bin/python download/synthesize_jobs_queries.py \\
      --data-dir jobs_data \\
      --n-distilled-train 10000 \\
      --n-distilled-eval 900 \\
      --n-head 200 \\
      --seed 42
"""

import argparse
import asyncio
import datetime
import hashlib
import json
import os
import random
import re
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(override=True)

SPEND_LEDGER = Path(__file__).resolve().parent.parent / ".api_spend.jsonl"

# gpt-4o-mini pricing (2026-05)
PRICES_PER_M_TOKENS = {"gpt-4o-mini": {"in": 0.15, "out": 0.60}}

DISTILL_PROMPT = """You are generating realistic search queries that a job-seeking candidate would type into Indeed, LinkedIn, or Google to find this specific job.

JOB:
{title}

DESCRIPTION (first ~300 chars):
{desc}

Generate exactly 1 realistic search query that a candidate looking for this job might type. Vary across calls — sometimes short, sometimes with constraints, sometimes natural language.

Style cues for realism:
- Short queries (2-5 words) are most common: "software engineer", "data scientist remote", "senior swe python"
- With attributes: location, level (junior/senior/staff/principal), key skill, "remote", "hybrid"
- Multi-constraint: combine 2-3 attributes ("senior ML engineer remote python")
- Natural-language conversational (less common): "looking for a remote ML role at a startup"
- Use lowercase mostly, occasional capitalization for proper nouns
- Use real skill names (Python, SQL, machine learning, NOT "Java Enterprise Edition")
- NO "I am looking for..." — be direct
- Do not just copy the job title verbatim — generate a *plausible search query* a candidate would type, which may rephrase or abbreviate
- Constraints can come from any field in the job (title, description, location implied)

Output: exactly one line, just the query text, no quotes, no prefix.
"""

WS_RE = re.compile(r"\s+")


def normalize_title(t: str) -> str:
    """Normalize a job title to its 'query form' — lowercase, collapse whitespace,
    strip common punctuation suffixes."""
    t = t.lower().strip()
    t = WS_RE.sub(" ", t)
    # strip trailing parens (often locations) and dashes
    t = re.sub(r"\s*[-–—|]\s*[^,]*$", "", t)  # "Senior SWE - SF" → "senior swe"
    t = re.sub(r"\s*\([^)]*\)\s*$", "", t)  # "Senior SWE (Remote)" → "senior swe"
    t = re.sub(r"[,.;:]+$", "", t).strip()
    return t


def record_spend(model: str, tokens_in: int, tokens_out: int, cost: float, purpose: str):
    rec = {
        "ts": datetime.datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "provider": "openai",
        "model": model,
        "tokens": int(tokens_in + tokens_out),
        "tokens_in": int(tokens_in),
        "tokens_out": int(tokens_out),
        "cost_usd": round(float(cost), 4),
        "purpose": purpose,
    }
    with open(SPEND_LEDGER, "a") as f:
        f.write(json.dumps(rec) + "\n")


async def distill_one(client, sem, rec, model, max_retries=5):
    title = rec["title"]
    desc = (rec.get("description") or "")[:600]
    prompt = DISTILL_PROMPT.format(title=title, desc=desc)
    backoff = 1.0
    async with sem:
        for _ in range(max_retries):
            try:
                resp = await client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=40,
                    temperature=0.7,
                )
                break
            except Exception:
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 30)
        else:
            return {
                "source_doc_id": rec["id"],
                "query": "",
                "error": True,
                "tokens_in": 0,
                "tokens_out": 0,
            }
        raw = (resp.choices[0].message.content or "").strip()
        # Take only the first line
        query = raw.splitlines()[0].strip().strip('"').strip("'")
        usage = resp.usage
        return {
            "source_doc_id": rec["id"],
            "query": query,
            "tokens_in": int(usage.prompt_tokens or 0),
            "tokens_out": int(usage.completion_tokens or 0),
        }


async def run_distillation(jobs: list, n: int, model: str, concurrency: int, purpose: str):
    from openai import AsyncOpenAI

    client = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    sem = asyncio.Semaphore(concurrency)

    target = jobs[:n]
    print(f"  distilling {len(target):,} queries with {model}...", flush=True)

    out = []
    tokens_in = 0
    tokens_out = 0
    t_start = time.time()
    chunk = 100
    for i in range(0, len(target), chunk):
        batch = target[i : i + chunk]
        results = await asyncio.gather(*[distill_one(client, sem, r, model) for r in batch])
        for r in results:
            tokens_in += r["tokens_in"]
            tokens_out += r["tokens_out"]
            if not r.get("error") and r["query"]:
                out.append(r)
        elapsed = time.time() - t_start
        rate = len(out) / elapsed if elapsed > 0 else 0
        cost = (
            tokens_in * PRICES_PER_M_TOKENS[model]["in"]
            + tokens_out * PRICES_PER_M_TOKENS[model]["out"]
        ) / 1e6
        print(
            f"    [{len(out):,}/{len(target):,}] rate={rate * 60:.0f}/min cost=${cost:.3f}",
            flush=True,
        )
    cost = (
        tokens_in * PRICES_PER_M_TOKENS[model]["in"]
        + tokens_out * PRICES_PER_M_TOKENS[model]["out"]
    ) / 1e6
    record_spend(model, tokens_in, tokens_out, cost, purpose)
    print(
        f"  done: {len(out):,} distilled queries, "
        f"tokens in={tokens_in:,} out={tokens_out:,} cost=${cost:.4f}",
        flush=True,
    )
    return out


def slug_hash(s: str) -> int:
    """Stable hash for splitting head queries deterministically."""
    return int(hashlib.md5(s.encode()).hexdigest()[:8], 16)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--metadata-file", default="metadata.jsonl")
    ap.add_argument(
        "--eval-job-frac",
        type=float,
        default=0.10,
        help="fraction of jobs to put in EVAL_JOBS (rest is TRAIN_JOBS)",
    )
    ap.add_argument("--n-distilled-train", type=int, default=10000)
    ap.add_argument("--n-distilled-eval", type=int, default=900)
    ap.add_argument(
        "--n-head",
        type=int,
        default=200,
        help="top-N most-frequent normalized titles to use as head queries",
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--model", default="gpt-4o-mini")
    ap.add_argument("--concurrency", type=int, default=24)
    args = ap.parse_args()

    data = Path(args.data_dir)
    rng = random.Random(args.seed)

    # Load jobs metadata
    jobs = []
    with open(data / args.metadata_file) as f:
        for line in f:
            jobs.append(json.loads(line))
    print(f"loaded {len(jobs):,} jobs from {data / args.metadata_file}", flush=True)

    # Train/eval split
    ids = [j["id"] for j in jobs]
    rng.shuffle(ids)
    n_eval = int(len(ids) * args.eval_job_frac)
    eval_jobs = set(ids[:n_eval])
    train_jobs = set(ids[n_eval:])
    print(f"  train_jobs: {len(train_jobs):,}  eval_jobs: {len(eval_jobs):,}", flush=True)
    with open(data / "job_split.json", "w") as f:
        json.dump({"train_jobs": sorted(train_jobs), "eval_jobs": sorted(eval_jobs)}, f)

    # Index jobs by id
    by_id = {j["id"]: j for j in jobs}

    # ---------- HEAD queries: title frequency distribution ----------
    title_counts = Counter()
    for j in jobs:
        t = normalize_title(j["title"])
        if t and 5 <= len(t) <= 60:  # avoid singletons / huge titles
            title_counts[t] += 1
    head_queries_sorted = [(t, n) for t, n in title_counts.most_common(args.n_head)]
    print(
        f"\nHEAD queries: top-{args.n_head} normalized titles "
        f"(coverage: {sum(n for _, n in head_queries_sorted):,}/{len(jobs):,} jobs)",
        flush=True,
    )
    if head_queries_sorted:
        print("  examples:", flush=True)
        for t, n in head_queries_sorted[:8]:
            print(f"    [{n:>5}] {t}", flush=True)

    # Split head queries deterministically by hash
    head_train = []
    head_eval = []
    n_head_eval_target = max(1, int(args.n_head * args.eval_job_frac))
    head_sorted_by_freq = sorted(head_queries_sorted, key=lambda x: -x[1])
    for t, _ in head_sorted_by_freq:
        h = slug_hash(t) % 1000
        if h < 1000 * args.eval_job_frac and len(head_eval) < n_head_eval_target:
            head_eval.append(t)
        else:
            head_train.append(t)
    print(f"  head split: {len(head_train)} train, {len(head_eval)} eval", flush=True)

    # ---------- DISTILLED queries: LLM-generated ----------
    train_pool = [by_id[i] for i in train_jobs if by_id[i].get("title")]
    eval_pool = [by_id[i] for i in eval_jobs if by_id[i].get("title")]
    rng.shuffle(train_pool)
    rng.shuffle(eval_pool)

    print("\nDistilling TRAIN queries from sampled TRAIN_JOBS...", flush=True)
    train_distilled = asyncio.run(
        run_distillation(
            train_pool,
            args.n_distilled_train,
            args.model,
            args.concurrency,
            f"jobs query distillation TRAIN n={args.n_distilled_train}",
        )
    )
    print("\nDistilling EVAL queries from sampled EVAL_JOBS...", flush=True)
    eval_distilled = asyncio.run(
        run_distillation(
            eval_pool,
            args.n_distilled_eval,
            args.model,
            args.concurrency,
            f"jobs query distillation EVAL n={args.n_distilled_eval}",
        )
    )

    # ---------- Write outputs ----------
    def write_queries(path, head_qs, distilled_qs, prefix):
        n = 0
        seen_q = set()
        with open(path, "w") as f:
            for t in head_qs:
                if t in seen_q:
                    continue
                seen_q.add(t)
                qid = f"{prefix}_head_{n:05d}"
                f.write(json.dumps({"query_id": qid, "query": t, "source": "head"}) + "\n")
                n += 1
            for r in distilled_qs:
                q = r["query"]
                if q in seen_q:
                    continue
                seen_q.add(q)
                qid = f"{prefix}_dist_{n:05d}"
                f.write(
                    json.dumps(
                        {
                            "query_id": qid,
                            "query": q,
                            "source": "distilled",
                            "source_doc_id": r["source_doc_id"],
                        }
                    )
                    + "\n"
                )
                n += 1
        return n

    n_train = write_queries(data / "train_queries.jsonl", head_train, train_distilled, "train")
    n_eval = write_queries(data / "eval_queries.jsonl", head_eval, eval_distilled, "eval")
    print(
        f"\nwrote {n_train:,} train queries to {data / 'train_queries.jsonl'}\n"
        f"wrote {n_eval:,} eval queries to {data / 'eval_queries.jsonl'}",
        flush=True,
    )


if __name__ == "__main__":
    main()
