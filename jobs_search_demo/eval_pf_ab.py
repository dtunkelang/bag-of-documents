#!/usr/bin/env python3
"""A/B test BM25 title-only vs BM25 + phrase-boost variants on the jobs
distilled eval. Hits the same Solr core the demo uses.

Reports R@10 and E@1 per variant, with overall + per-corpus splits.
"""

import json
import time
from collections import Counter
from pathlib import Path

import requests

SOLR = "http://localhost:8983/solr/jobs"
EVAL_CORPORA = [
    ("jobs_data", "/Users/dtunkelang/bagofdocs/jobs_data/eval_queries.jsonl"),
    ("jobs_data_linkedin", "/Users/dtunkelang/bagofdocs/jobs_data_linkedin/eval_queries.jsonl"),
    ("jobs_data_jobstreet", "/Users/dtunkelang/bagofdocs/jobs_data_jobstreet/eval_queries.jsonl"),
    ("jobs_data_usajobs", "/Users/dtunkelang/bagofdocs/jobs_data_usajobs/eval_queries.jsonl"),
]
METADATA = Path("/Users/dtunkelang/bagofdocs/unified_jobs/metadata.jsonl")

VARIANTS = [
    ("baseline", "{!edismax qf=title v=$user_q}"),
    ("pf_title_2", "{!edismax qf=title pf=title^2 v=$user_q}"),
    ("pf_title_3", "{!edismax qf=title pf=title^3 v=$user_q}"),
    ("pf_title_5", "{!edismax qf=title pf=title^5 v=$user_q}"),
    ("pf_title_10", "{!edismax qf=title pf=title^10 v=$user_q}"),
]


def load_id_map() -> dict[str, str]:
    """orig source-id (ashby:...) -> Solr numeric doc-id (row index, as str)."""
    m: dict[str, str] = {}
    with METADATA.open() as f:
        for i, line in enumerate(f):
            d = json.loads(line)
            m[d["id"]] = str(i)
    return m


def load_eval_queries(path: Path):
    out = []
    with path.open() as f:
        for line in f:
            q = json.loads(line)
            if q.get("source") == "distilled" and q.get("source_doc_id"):
                out.append(q)
    return out


def topk_ids(q_handler: str, query: str, k: int = 10) -> list[str]:
    r = requests.get(
        f"{SOLR}/select",
        params=[
            ("q", q_handler),
            ("user_q", query),
            ("rows", str(k)),
            ("fl", "id"),
        ],
        timeout=10,
    )
    r.raise_for_status()
    return [d["id"] for d in r.json()["response"]["docs"]]


def main():
    t0 = time.time()
    id_map = load_id_map()
    print(f"loaded id_map ({len(id_map):,}) in {time.time() - t0:.1f}s")

    all_evals: list[tuple[str, dict]] = []  # (corpus, query)
    for corpus, path in EVAL_CORPORA:
        qs = load_eval_queries(Path(path))
        # only keep queries whose gold doc actually made it into the unified index
        kept = [q for q in qs if q["source_doc_id"] in id_map]
        print(f"  {corpus}: {len(qs)} distilled, {len(kept)} mapped")
        for q in kept:
            all_evals.append((corpus, q))
    print(f"total eval queries: {len(all_evals)}")

    # per-variant per-corpus tallies
    hit10: dict[tuple[str, str], int] = Counter()
    hit1: dict[tuple[str, str], int] = Counter()
    tot: dict[str, int] = Counter()

    for i, (corpus, q) in enumerate(all_evals):
        if i % 500 == 0:
            print(f"  ... {i}/{len(all_evals)} ({time.time() - t0:.1f}s)")
        gold = id_map[q["source_doc_id"]]
        tot[corpus] += 1
        for name, q_handler in VARIANTS:
            try:
                top = topk_ids(q_handler, q["query"], k=10)
            except Exception:
                continue
            if gold in top:
                hit10[(corpus, name)] += 1
            if top and top[0] == gold:
                hit1[(corpus, name)] += 1

    print(f"\nfinished in {time.time() - t0:.1f}s\n")

    # Per-corpus + overall
    rows: list[str] = []
    all_corp = [c for c, _ in EVAL_CORPORA] + ["__ALL__"]
    rows.append(f"{'corpus':24} {'variant':14} {'R@10':>8} {'E@1':>8} {'n':>6}")
    rows.append("-" * 64)
    for corpus in all_corp:
        for name, _ in VARIANTS:
            if corpus == "__ALL__":
                n = sum(tot.values())
                h10 = sum(hit10[(c, name)] for c in tot)
                h1 = sum(hit1[(c, name)] for c in tot)
            else:
                n = tot[corpus]
                h10 = hit10[(corpus, name)]
                h1 = hit1[(corpus, name)]
            if n == 0:
                continue
            rows.append(f"{corpus:24} {name:14} {h10 / n * 100:7.2f}% {h1 / n * 100:7.2f}% {n:>6}")
        rows.append("")
    print("\n".join(rows))

    out_path = Path("/tmp/eval_pf_ab.json")
    out_path.write_text(
        json.dumps(
            {
                "variants": [n for n, _ in VARIANTS],
                "totals": dict(tot),
                "hit10": {f"{c}|{v}": n for (c, v), n in hit10.items()},
                "hit1": {f"{c}|{v}": n for (c, v), n in hit1.items()},
            },
            indent=2,
        )
    )
    print(f"\nfull tallies → {out_path}")


if __name__ == "__main__":
    main()
