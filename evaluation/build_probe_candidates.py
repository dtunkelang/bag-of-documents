#!/usr/bin/env python3
"""Build dedup'd (query, doc) candidate pool from probe_topk.jsonl for in-session judging."""

import argparse
import json
import re
from pathlib import Path


def split_title_body(text: str) -> tuple[str, str]:
    """titles.json entry begins with the job title, then '\\n\\n' then body."""
    parts = text.split("\n\n", 1)
    title = parts[0].strip()
    body = parts[1].strip() if len(parts) > 1 else ""
    return title, body


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--topk-file", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--body-chars", type=int, default=300, help="body snippet length")
    args = ap.parse_args()

    data = Path(args.data_dir)
    with open(data / "doc_ids.json") as f:
        doc_ids = json.load(f)
    with open(data / "source_index.json") as f:
        src = json.load(f)
    sources = src["sources"]
    assert len(sources) == len(doc_ids), f"{len(sources)} != {len(doc_ids)}"
    print(
        f"loading titles.json ({(data / 'titles.json').stat().st_size / 1e6:.0f} MB)...", flush=True
    )
    with open(data / "titles.json") as f:
        titles = json.load(f)
    print(f"  loaded {len(titles):,} doc texts", flush=True)

    with open(args.topk_file) as f:
        rows = [json.loads(line) for line in f]

    seen: set[tuple[str, int]] = set()
    out_rows = []
    for r in rows:
        qid = r["query_id"]
        for _retr_name, retr in r["retrievers"].items():
            for _rank, idx in enumerate(retr["doc_indices"]):
                key = (qid, idx)
                if key in seen:
                    continue
                seen.add(key)
                title, body = split_title_body(titles[idx])
                body_snip = re.sub(r"\s+", " ", body[: args.body_chars]).strip()
                out_rows.append(
                    {
                        "query_id": qid,
                        "query": r["query"],
                        "archetype": r["archetype"],
                        "doc_idx": idx,
                        "doc_id": doc_ids[idx],
                        "source_corpus": sources[idx],
                        "title": title[:200],
                        "body": body_snip,
                    }
                )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for r in out_rows:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {len(out_rows):,} (query, doc) pairs -> {out_path}", flush=True)
    # quick stats
    per_q = {}
    for r in out_rows:
        per_q.setdefault(r["query_id"], 0)
        per_q[r["query_id"]] += 1
    nq = len(per_q)
    avg = sum(per_q.values()) / nq if nq else 0
    print(
        f"  {nq} unique queries; mean {avg:.1f} candidates/query "
        f"(min {min(per_q.values())}, max {max(per_q.values())})"
    )


if __name__ == "__main__":
    main()
