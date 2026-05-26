#!/usr/bin/env python3
"""Compare top-10 results between the in-memory demo (7862) and the Solr shim (7864).

For each query, prints both top-10 lists and overlap@10.
"""

import sys

import requests

DEMO = "http://127.0.0.1:7862"
SOLR_SHIM = "http://127.0.0.1:7864"

QUERIES = [
    "registered nurse",  # cached, common
    "software engineer",  # cached, very common
    "remote python developer",  # likely uncached
    "machine learning engineer",  # cached
    "data scientist",  # cached
    "kubernetes site reliability engineer",  # uncached/specific
]


def hit_demo(q: str) -> list[dict]:
    r = requests.get(
        f"{DEMO}/api/search", params={"q": q, "retriever": "rrf_bm25_bge_te3"}, timeout=30
    )
    r.raise_for_status()
    return r.json()


def hit_solr(q: str) -> list[dict]:
    r = requests.get(f"{SOLR_SHIM}/api/search", params={"q": q}, timeout=30)
    r.raise_for_status()
    return r.json()


def fmt(res: list[dict]) -> str:
    return "\n".join(
        f"    {r['rank']:>2}  {r['score']:.4f}  [{r['idx']:>6}]  {r['title'][:80]}" for r in res
    )


def main() -> int:
    for q in QUERIES:
        print(f"\n=== {q!r}")
        d = hit_demo(q)
        s = hit_solr(q)
        d_ids = [r["idx"] for r in d["results"]]
        s_ids = [r["idx"] for r in s["results"]]
        overlap = set(d_ids) & set(s_ids)
        print(
            f"  demo  ({d.get('ms', '?')} ms, cached={d['cached']}, mode={d.get('served_with', '')}):"
        )
        print(fmt(d["results"]))
        print(
            f"  solr  ({s.get('ms', '?')} ms, cached={s['cached']}, mode={s.get('served_with', '')}):"
        )
        print(fmt(s["results"]))
        print(f"  overlap@10 = {len(overlap)}/10  (top-1 same: {d_ids[0] == s_ids[0]})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
