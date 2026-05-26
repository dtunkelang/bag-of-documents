#!/usr/bin/env python3
"""Union per-corpus eval_queries.jsonl into one unified file.

Keeps only rows with `source_doc_id` (distilled / single-positive gold).
Resolves gold against unified_jobs/doc_ids.json and reports coverage.
"""

import json
from pathlib import Path

PER_CORPUS = [
    "jobs_data/eval_queries.jsonl",
    "jobs_data_linkedin/eval_queries.jsonl",
    "jobs_data_jobstreet/eval_queries.jsonl",
    "jobs_data_usajobs/eval_queries.jsonl",
]
OUT = Path("unified_jobs/eval_queries_unified.jsonl")
DOC_IDS = Path("unified_jobs/doc_ids.json")


def main():
    with open(DOC_IDS) as f:
        doc_ids = json.load(f)
    pid_set = set(doc_ids)
    rows = []
    per_src_stats = {}
    for path in PER_CORPUS:
        src = path.split("/")[0]
        kept = resolved = 0
        with open(path) as f:
            for line in f:
                r = json.loads(line)
                if "source_doc_id" not in r:
                    continue
                r["corpus"] = src
                rows.append(r)
                kept += 1
                if r["source_doc_id"] in pid_set:
                    resolved += 1
        per_src_stats[src] = (kept, resolved)
    with open(OUT, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {OUT}  total_rows={len(rows):,}")
    for src, (kept, resolved) in per_src_stats.items():
        print(f"  {src:24s}  kept={kept:5d}  resolved={resolved:5d}")
    total_resolved = sum(v[1] for v in per_src_stats.values())
    print(f"  overall resolved: {total_resolved:,} / {len(rows):,}")


if __name__ == "__main__":
    main()
