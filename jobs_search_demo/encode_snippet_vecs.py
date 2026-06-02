#!/usr/bin/env python3
"""Build the inputs for index-time snippet passage vectors.

For every staged doc, split its description into candidate passages (the SAME
snippet_lib.passages_for the serving path uses), dedup across the whole corpus, and
write:

  * snippet_passages.json     — the unique passage strings (encoder input; the encoder
                                adds the "passage: " prefix and writes
                                snippet_passages.vecs.fp16.npy, one normalized fp16 row
                                per unique passage)
  * snippet_doc_rows.json     — {"<position>": [unique-row, ...]} so push_docs can gather
                                each doc's passage vectors (in passage order) and pack
                                them into the stored snippet_vecs field.

Dedup matters: boilerplate ("Equal Opportunity Employer ...") and reposts repeat heavily,
so unique passages are far fewer than total — that's the bulk of the encode savings.

Usage:
  encode_snippet_vecs.py --stage /path/to/unified_jobs [--positions pos.json] [--limit N]
then encode:
  python download/encode_st_catalog.py --data-dir <stage> \\
      --titles-file snippet_passages.json --out-name snippet_passages \\
      --doc-prefix 'passage: ' --device mps
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "space"))
from snippet_lib import passages_for  # noqa: E402


def build_passages(stage, keep=None, limit=0):
    """Walk metadata.jsonl and split each doc's description into candidate passages,
    deduping across the corpus. Returns (unique_passages, doc_rows, stats):
      unique_passages — list[str], one per distinct passage, in first-seen order
      doc_rows        — {"<metadata position>": [unique-row index, ...]} in passage order
      stats           — dict(n_docs, n_with, total_passages)
    `keep` (a set of positions) restricts to a subset; `limit` caps positions (debug).
    The serving path re-derives the same passages via snippet_lib.passages_for, so
    unique_passages[doc_rows[i][k]] is the vector for passage k of doc i."""
    meta_path = os.path.join(stage, "metadata.jsonl")
    unique: dict[str, int] = {}
    doc_rows: dict[str, list[int]] = {}
    t0 = time.time()
    n_docs = n_with = total_passages = 0
    with open(meta_path) as mf:
        for i, line in enumerate(mf):
            if keep is not None and i not in keep:
                continue
            if limit and n_docs >= limit:
                break
            n_docs += 1
            rec = json.loads(line)
            ps = passages_for(rec.get("description") or "")
            if not ps:
                continue
            rows = []
            for p in ps:
                idx = unique.get(p)
                if idx is None:
                    idx = len(unique)
                    unique[p] = idx
                rows.append(idx)
            doc_rows[str(i)] = rows
            n_with += 1
            total_passages += len(ps)
            if n_docs % 20000 == 0:
                print(
                    f"  {n_docs:,} docs, {len(unique):,} unique passages ({time.time() - t0:.0f}s)",
                    flush=True,
                )
    stats = {"n_docs": n_docs, "n_with": n_with, "total_passages": total_passages}
    return list(unique.keys()), doc_rows, stats


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--stage", default=os.environ.get("JOBS_STAGE", "/Users/dtunkelang/bagofdocs/unified_jobs")
    )
    ap.add_argument("--positions", help="optional JSON list of metadata positions to include")
    ap.add_argument("--limit", type=int, default=0, help="cap positions processed (debug)")
    args = ap.parse_args()

    keep = None
    if args.positions:
        with open(args.positions) as f:
            keep = set(json.load(f))

    unique, doc_rows, stats = build_passages(args.stage, keep=keep, limit=args.limit)

    passages_out = os.path.join(args.stage, "snippet_passages.json")
    rows_out = os.path.join(args.stage, "snippet_doc_rows.json")
    with open(passages_out, "w") as f:
        json.dump(unique, f)
    with open(rows_out, "w") as f:
        json.dump(doc_rows, f)

    dedup = 1 - len(unique) / max(stats["total_passages"], 1)
    print(
        f"done: {stats['n_docs']:,} docs ({stats['n_with']:,} with passages), "
        f"{stats['total_passages']:,} total passages -> {len(unique):,} unique ({dedup:.0%} deduped)",
        flush=True,
    )
    print(f"wrote {passages_out} + {rows_out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
