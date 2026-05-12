#!/usr/bin/env python3
"""BoQ-SCHS — Cluster Hypothesis Score on the QUERY side.

Standard SCHS measures: do documents relevant to the same query cluster
tightly in doc-embedding space?

This is the dual: for each document with multiple queries pointing at it
(via qrels or click data), do those QUERIES cluster tightly in
query-embedding space? A high BoQ-SCHS means the corpus's query-side
cluster hypothesis holds — different surface forms of the same intent
land close together in the encoder's view of query space. That's the
necessary condition for BoQ-style sparse retrieval (cluster top frequent
queries, tag docs with cluster ids, look up at query time).

Implementation: invert qrels (doc -> queries), filter to multi-query
docs, and call the existing `compute_chs` with queries-as-items. Same
score formula, opposite role assignment.

Usage:
    python evaluation/boq_cluster_hypothesis.py \\
        --qrels bestbuy_acm_data/test_qrels.jsonl --min-relevance 1 \\
        --queries bestbuy_acm_data/test_queries.jsonl \\
        --encoder all-MiniLM-L6-v2 \\
        --label bestbuy
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bagofdocs.cluster_hypothesis import compute_chs  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--qrels", required=True)
    ap.add_argument("--queries", required=True, help="queries.jsonl with query_id, query")
    ap.add_argument("--min-relevance", type=int, default=1)
    ap.add_argument("--encoder", default="all-MiniLM-L6-v2")
    ap.add_argument("--label", default="corpus")
    args = ap.parse_args()

    # Load query texts.
    queries_by_qid = {}
    with open(args.queries) as f:
        for line in f:
            d = json.loads(line)
            queries_by_qid[d["query_id"]] = d["query"]

    # Build inverted qrels: doc_id -> {qid: relevance}.
    inv_qrels = defaultdict(dict)
    field = None
    n_rows = 0
    skipped_unknown_qid = 0
    with open(args.qrels) as f:
        for line in f:
            r = json.loads(line)
            if field is None:
                field = "product_id" if "product_id" in r else "doc_id"
            if r["relevance"] < args.min_relevance:
                continue
            qid = r["query_id"]
            if qid not in queries_by_qid:
                skipped_unknown_qid += 1
                continue
            doc_id = r[field]
            inv_qrels[doc_id][qid] = r["relevance"]
            n_rows += 1

    print(
        f"loaded {n_rows:,} qrels rows ({skipped_unknown_qid:,} skipped "
        f"due to missing query text), {len(inv_qrels):,} unique docs",
        flush=True,
    )

    # Filter to docs with >= 2 queries (singletons are useless for cluster-tightness).
    multi_query_docs = {d: qs for d, qs in inv_qrels.items() if len(qs) >= 2}
    print(
        f"  multi-query docs (>= 2 queries pointing at them): {len(multi_query_docs):,} "
        f"({100 * len(multi_query_docs) / max(len(inv_qrels), 1):.1f}%)",
        flush=True,
    )
    if not multi_query_docs:
        print(
            "ERROR: no multi-query docs — corpus doesn't have the redundancy needed "
            "for BoQ-side cluster-hypothesis analysis.",
            flush=True,
        )
        sys.exit(1)

    qids_used = sorted({q for qs in multi_query_docs.values() for q in qs})
    queries_used = [queries_by_qid[q] for q in qids_used]
    print(
        f"  unique queries appearing in those bags: {len(qids_used):,}",
        flush=True,
    )

    # Call compute_chs with roles swapped:
    #   - "qrels" outer key = doc_id (the bag-id, formerly query_id)
    #   - "pids" = list of query_ids (formerly doc_ids)
    #   - "titles" = query texts (formerly doc texts)
    # The function treats them symmetrically: it groups items by outer key
    # and measures intra-group cosine vs random-pair cosine.
    print(
        "\ncomputing BoQ-SCHS by reusing compute_chs with role-swap...",
        flush=True,
    )
    chs = compute_chs(
        dict(multi_query_docs),
        qids_used,
        queries_used,
        args.encoder,
        partition="strict",
    )

    print(f"\n=== BoQ-SCHS for {args.label} ===")
    print(f"  multi-query docs (bag count):  {chs.n_pos_bearing:,}")
    print(f"  mean_intra (queries-in-same-doc-bag): {chs.mean_intra:.3f}")
    print(f"  mean_random (random query pairs):     {chs.mean_random:.3f}")
    print(f"  BoQ-SCHS:                             {chs.schs:.3f}")
    print()
    if chs.schs >= 0.50:
        verdict = (
            "GREEN — queries pointing at the same doc cluster tightly; "
            "BoQ-style cluster building should work on this corpus."
        )
    elif chs.schs >= 0.40:
        verdict = "YELLOW — moderate query-side clustering; BoQ may give smaller lift; pilot first."
    else:
        verdict = (
            "RED — queries pointing at the same doc don't cluster much "
            "more than random pairs; BoQ-style query clustering is "
            "unlikely to produce coherent buckets."
        )
    print(f"  verdict: {verdict}")


if __name__ == "__main__":
    main()
