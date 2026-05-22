#!/usr/bin/env python3
"""Prepare Open-Apply jobs slice for retrieval evaluation.

Reads the raw Parquet snapshots downloaded from huggingface.co/datasets/edwarddgao/open-apply-jobs,
strips HTML from descriptions, lightly dedupes on (source_slug, title), and writes:

  jobs_data/doc_ids.json      list[str] of canonical job IDs (parallel to titles)
  jobs_data/titles.json       list[str] of "{title}\n\n{description}" — the text used for
                              retrieval encoding and LLM-judge
  jobs_data/metadata.jsonl    one JSON row per job with all original fields

The "titles.json" naming is a misnomer for jobs (a job is title+description, not just title),
but we use it for consistency with the rest of the bagofdocs catalog conventions.

Usage:
  .venv/bin/python download/prep_open_apply.py \\
      --raw-dir jobs_data/raw \\
      --out-dir jobs_data \\
      --sample-n 100000 --seed 42
"""

import argparse
import html
import json
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pyarrow.parquet as pq  # noqa: E402

TAG_RE = re.compile(r"<[^>]+>")
WS_RE = re.compile(r"\s+")


def strip_html(s: str) -> str:
    if not s:
        return ""
    # Open-Apply's description_html is HTML-entity-encoded (e.g., &lt;p&gt; not <p>),
    # so we must unescape BEFORE stripping tags, otherwise the regex sees no tags.
    out = html.unescape(s)
    out = TAG_RE.sub(" ", out)
    out = WS_RE.sub(" ", out).strip()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--sample-n", type=int, default=100_000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--max-text-chars",
        type=int,
        default=2400,
        help="truncate combined title+description to this many characters",
    )
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    rows = []
    raw = Path(args.raw_dir)
    files = sorted(raw.rglob("*.parquet"))
    print(f"reading {len(files)} parquet files...", flush=True)
    for p in files:
        t = pq.read_table(p)
        d = t.to_pylist()
        print(f"  {p.name}: {len(d):,} rows", flush=True)
        rows.extend(d)
    print(f"total raw rows: {len(rows):,}", flush=True)

    # Strip HTML, build text, drop empties
    n_empty_desc = 0
    n_empty_title = 0
    cleaned = []
    for r in rows:
        title = (r.get("title") or "").strip()
        desc = strip_html(r.get("description_html") or "")
        if not title:
            n_empty_title += 1
            continue
        if not desc:
            n_empty_desc += 1
            # keep the row but text will be title only
        text = f"{title}\n\n{desc}".strip()[: args.max_text_chars]
        cleaned.append(
            {
                "id": r["id"],
                "source_slug": r.get("source_slug") or "",
                "title": title,
                "description": desc[: args.max_text_chars - len(title) - 2],
                "text": text,
                "department": r.get("department"),
                "employment_type": r.get("employment_type"),
                "remote": r.get("remote"),
                "locations": r.get("locations") or [],
                "salary_min": r.get("salary_min"),
                "salary_max": r.get("salary_max"),
                "salary_currency": r.get("salary_currency"),
                "posted_at": r.get("posted_at"),
                "source": r.get("source") or "",
            }
        )
    print(
        f"  after empty-title drop: {len(cleaned):,} "
        f"({n_empty_title:,} empty title; {n_empty_desc:,} empty desc kept)",
        flush=True,
    )

    # Light dedupe on (source_slug, title) — same job often posted twice via different ATS routes
    seen = set()
    deduped = []
    for r in cleaned:
        key = (r["source_slug"].lower(), r["title"].lower())
        if key in seen:
            continue
        seen.add(key)
        deduped.append(r)
    print(f"  after (source_slug, title) dedupe: {len(deduped):,}", flush=True)

    # Sample
    rng = random.Random(args.seed)
    if args.sample_n > 0 and len(deduped) > args.sample_n:
        rng.shuffle(deduped)
        deduped = deduped[: args.sample_n]
        print(f"  sampled to: {len(deduped):,}", flush=True)

    # Write outputs
    doc_ids = [r["id"] for r in deduped]
    texts = [r["text"] for r in deduped]

    with open(out / "doc_ids.json", "w") as f:
        json.dump(doc_ids, f)
    with open(out / "titles.json", "w") as f:
        json.dump(texts, f)
    with open(out / "metadata.jsonl", "w") as f:
        for r in deduped:
            f.write(json.dumps(r) + "\n")

    # Stats
    title_lens = [len(r["title"]) for r in deduped]
    text_lens = [len(r["text"]) for r in deduped]
    src_counts = {}
    for r in deduped:
        src_counts[r["source"]] = src_counts.get(r["source"], 0) + 1
    print(
        f"\nwrote {len(deduped):,} jobs to {out}/\n"
        f"  doc_ids.json    {sum(len(d) for d in doc_ids):,} bytes (rough)\n"
        f"  titles.json     {sum(text_lens):,} chars\n"
        f"  metadata.jsonl  ~{len(deduped):,} rows\n"
        f"  title len: median={sorted(title_lens)[len(title_lens) // 2]} "
        f"max={max(title_lens)}\n"
        f"  text len:  median={sorted(text_lens)[len(text_lens) // 2]} "
        f"max={max(text_lens)}\n"
        f"  source counts: {sorted(src_counts.items())}",
        flush=True,
    )


if __name__ == "__main__":
    main()
