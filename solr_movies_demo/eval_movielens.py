#!/usr/bin/env python3
"""MovieLens mask-one-out eval for the movies demo search lanes.

Ground truth: each of the ~18.3K bag-rich titles ships with a `corated_bag`
of IMDb tconsts (movies co-rated by users who liked the query title). We
use the title's own text as the query and measure recall@K of bag members
in the result list. The bag content is NOT in the BM25 qf and NOT in the
KNN vector, so the eval is unleaked.

Caveat: with query = bare title, this measures lexical co-occurrence (e.g.,
franchises, sequels) more than thematic similarity. BM25 tends to win this
proxy. The fusion tuning that helps concept queries ("Italian neorealism
postwar") slightly hurts this score because KNN dilutes BM25's sequel hits.
Use this harness as a floor / regression check, not as a primary tuning
signal for semantic-query quality.

Usage:
  python eval_movielens.py --lane hybrid --sample 500 --k 20
  W_KNN=3.0 python eval_movielens.py --lane hybrid --sample 500

Compare lanes:
  for L in bm25 knn hybrid; do python eval_movielens.py --lane $L --sample 500 --seed 0; done
"""

import argparse
import random
import statistics
import sys
import time

import requests
from app import search

SOLR = "http://localhost:8984/solr/movies"


def load_bag_rich_titles(sample: int, seed: int) -> list[dict]:
    """Fetch a uniform random sample of bag-rich titles."""
    rng = random.Random(seed)
    r = requests.get(
        f"{SOLR}/select",
        params={
            "q": "has_bag:true AND has_lead:true",
            "fl": "id,title,corated_bag,votes",
            "rows": 30000,
            "wt": "json",
        },
        timeout=30,
    )
    r.raise_for_status()
    docs = r.json()["response"]["docs"]
    rng.shuffle(docs)
    return docs[:sample]


def recall_at_k(retrieved_ids: list[str], gold_ids: set[str], k: int) -> float:
    if not gold_ids:
        return 0.0
    hit = sum(1 for did in retrieved_ids[:k] if did in gold_ids)
    return hit / min(k, len(gold_ids))


def run(args: argparse.Namespace) -> None:
    titles = load_bag_rich_titles(args.sample, args.seed)
    print(
        f"# lane={args.lane} sample={len(titles)} k={args.k} seed={args.seed}",
        file=sys.stderr,
    )

    recalls = []
    latencies = []
    t0 = time.time()
    skipped = 0

    for i, t in enumerate(titles):
        q = t.get("title")
        qid = t.get("id")
        gold = set(t.get("corated_bag") or [])
        if not q or not gold:
            skipped += 1
            continue
        s = time.time()
        out = search(q, rows=args.k + 1, lane=args.lane)
        latencies.append(time.time() - s)
        # Exclude the query itself from results.
        retrieved = [d["id"] for d in out.get("docs", []) if d.get("id") != qid][: args.k]
        r = recall_at_k(retrieved, gold, args.k)
        recalls.append(r)
        if args.verbose and i < 10:
            print(f"  [{i}] {q!r} bag={len(gold)} recall@{args.k}={r:.3f}", file=sys.stderr)

    dt = time.time() - t0
    if not recalls:
        print("no scorable queries", file=sys.stderr)
        return

    mean = statistics.mean(recalls)
    median = statistics.median(recalls)
    p90 = statistics.quantiles(recalls, n=10)[-1] if len(recalls) >= 10 else max(recalls)
    nonzero = sum(1 for r in recalls if r > 0) / len(recalls)
    print(f"lane={args.lane}")
    print(f"  n={len(recalls)} (skipped={skipped})")
    print(f"  recall@{args.k} mean={mean:.4f} median={median:.4f} p90={p90:.4f}")
    print(f"  nonzero_rate={nonzero:.3f}")
    print(f"  total_time={dt:.1f}s mean_latency={statistics.mean(latencies) * 1000:.0f}ms")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lane", choices=["bm25", "knn", "hybrid"], default="hybrid")
    ap.add_argument("--sample", type=int, default=500)
    ap.add_argument("--k", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--verbose", action="store_true")
    run(ap.parse_args())


if __name__ == "__main__":
    main()
