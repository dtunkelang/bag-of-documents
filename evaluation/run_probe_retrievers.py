#!/usr/bin/env python3
"""Run te3 / e5-base / BM25 over the hand-curated probe set; dump top-K per retriever.

Probe queries have NO gold labels (they will be judged in-session). Output is a
labels-free top-K dump suitable for downstream candidate-pool building and judging.

Usage:
  .venv/bin/python evaluation/run_probe_retrievers.py \\
      --data-dir unified_jobs \\
      --queries-file probe_queries.jsonl \\
      --te3-vecs probe_queries_te3_1024.vecs.fp16.npy \\
      --te3-ids  probe_queries_te3_1024.ids.json \\
      --k 10 \\
      --output evaluation/results/probe_topk.jsonl
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
from eval_jobs_retrievers import (  # noqa: E402
    bm25_topk,
    build_bm25,
    load_doc_ids,
    load_titles,
    st_topk_with_argsort,
)


def preenc_topk_from_files(probe_queries, vecs_path: Path, ids_path: Path, cat_path: Path, k: int):
    with open(ids_path) as f:
        cached_ids = json.load(f)
    cached_vecs = np.load(vecs_path).astype(np.float32)
    qmap = dict(zip(cached_ids, cached_vecs))
    miss = [q for q in probe_queries if q not in qmap]
    if miss:
        raise SystemExit(f"te3 preenc: {len(miss)} queries missing; first: {miss[0]!r}")
    qv = np.stack([qmap[q] for q in probe_queries], axis=0).astype(np.float32)
    qv = qv / np.maximum(np.linalg.norm(qv, axis=1, keepdims=True), 1e-12)
    cat = np.load(cat_path, mmap_mode="r").astype(np.float32)
    n = np.linalg.norm(cat, axis=1, keepdims=True)
    n[n == 0] = 1.0
    cat = cat / n
    scores = qv @ cat.T
    top_idx = np.argpartition(-scores, kth=k, axis=1)[:, :k]
    out_idx = []
    out_scores = []
    for r, idx in enumerate(top_idx):
        order = idx[np.argsort(-scores[r, idx])]
        out_idx.append(order)
        out_scores.append(scores[r, order])
    return np.array(out_idx), np.array(out_scores)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--queries-file", required=True)
    ap.add_argument("--te3-vecs", required=True, help="probe te3 query vecs (filename in data-dir)")
    ap.add_argument("--te3-ids", required=True, help="probe te3 query ids (filename in data-dir)")
    ap.add_argument("--te3-catalog", default="te3_catalog.vecs.fp16.npy")
    ap.add_argument("--e5-catalog", default="e5_base_catalog.vecs.fp16.npy")
    ap.add_argument("--e5-model", default="intfloat/e5-base-v2")
    ap.add_argument("--e5-prefix", default="query: ")
    ap.add_argument(
        "--bge-base-catalog", default=None, help="If set, also run bge-base over the probe set."
    )
    ap.add_argument("--bge-base-model", default="BAAI/bge-base-en-v1.5")
    ap.add_argument("--bge-base-prefix", default="")
    ap.add_argument(
        "--e5-small-catalog", default=None, help="If set, also run e5-small over the probe set."
    )
    ap.add_argument("--e5-small-model", default="intfloat/e5-small-v2")
    ap.add_argument("--e5-small-prefix", default="query: ")
    ap.add_argument(
        "--bge-small-catalog", default=None, help="If set, also run bge-small over the probe set."
    )
    ap.add_argument("--bge-small-model", default="BAAI/bge-small-en-v1.5")
    ap.add_argument("--bge-small-prefix", default="")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    data = Path(args.data_dir)
    titles = load_titles(data / "titles.json")
    doc_ids = load_doc_ids(data / "doc_ids.json")
    assert len(titles) == len(doc_ids), f"{len(titles)} != {len(doc_ids)}"

    queries = []
    with open(data / args.queries_file) as f:
        for line in f:
            queries.append(json.loads(line))
    qtexts = [q["query"] for q in queries]
    print(f"corpus={data} docs={len(titles):,} probe_queries={len(queries)}", flush=True)

    out_rows = {
        q["query_id"]: {
            "query_id": q["query_id"],
            "query": q["query"],
            "archetype": q.get("archetype", "?"),
            "retrievers": {},
        }
        for q in queries
    }

    # --- BM25 ---
    print("\n=== bm25 ===", flush=True)
    t0 = time.time()
    idx, stem = build_bm25(titles)
    print(f"  built in {time.time() - t0:.1f}s", flush=True)
    t0 = time.time()
    top = bm25_topk(qtexts, idx, stem, args.k)
    for qi, q in enumerate(queries):
        out_rows[q["query_id"]]["retrievers"]["bm25"] = {
            "doc_indices": [int(x) for x in top[qi].tolist()],
            "doc_ids": [doc_ids[int(x)] for x in top[qi].tolist()],
        }
    print(f"  bm25 done in {time.time() - t0:.1f}s", flush=True)

    # --- te3-large (preenc) ---
    print("\n=== te3_large_1024 ===", flush=True)
    t0 = time.time()
    vecs_p = data / args.te3_vecs
    ids_p = data / args.te3_ids
    cat_p = data / args.te3_catalog
    top, sc = preenc_topk_from_files(qtexts, vecs_p, ids_p, cat_p, args.k)
    for qi, q in enumerate(queries):
        out_rows[q["query_id"]]["retrievers"]["te3_large_1024"] = {
            "doc_indices": [int(x) for x in top[qi].tolist()],
            "doc_ids": [doc_ids[int(x)] for x in top[qi].tolist()],
            "scores": [float(x) for x in sc[qi].tolist()],
        }
    print(f"  te3 done in {time.time() - t0:.1f}s", flush=True)

    # --- e5-base (st, query-prefix) ---
    print("\n=== e5_base ===", flush=True)
    t0 = time.time()
    e5_cat = data / args.e5_catalog if not os.path.isabs(args.e5_catalog) else Path(args.e5_catalog)
    top = st_topk_with_argsort(
        qtexts, e5_cat, args.e5_model, args.k, device=args.device, query_prefix=args.e5_prefix
    )
    for qi, q in enumerate(queries):
        out_rows[q["query_id"]]["retrievers"]["e5_base"] = {
            "doc_indices": [int(x) for x in top[qi].tolist()],
            "doc_ids": [doc_ids[int(x)] for x in top[qi].tolist()],
        }
    print(f"  e5_base done in {time.time() - t0:.1f}s", flush=True)

    # --- e5-small (st, optional) ---
    if args.e5_small_catalog:
        print("\n=== e5_small ===", flush=True)
        t0 = time.time()
        e5s_cat = (
            data / args.e5_small_catalog
            if not os.path.isabs(args.e5_small_catalog)
            else Path(args.e5_small_catalog)
        )
        top = st_topk_with_argsort(
            qtexts,
            e5s_cat,
            args.e5_small_model,
            args.k,
            device=args.device,
            query_prefix=args.e5_small_prefix,
        )
        for qi, q in enumerate(queries):
            out_rows[q["query_id"]]["retrievers"]["e5_small"] = {
                "doc_indices": [int(x) for x in top[qi].tolist()],
                "doc_ids": [doc_ids[int(x)] for x in top[qi].tolist()],
            }
        print(f"  e5_small done in {time.time() - t0:.1f}s", flush=True)

    # --- bge-small (st, optional) ---
    if args.bge_small_catalog:
        print("\n=== bge_small ===", flush=True)
        t0 = time.time()
        bges_cat = (
            data / args.bge_small_catalog
            if not os.path.isabs(args.bge_small_catalog)
            else Path(args.bge_small_catalog)
        )
        top = st_topk_with_argsort(
            qtexts,
            bges_cat,
            args.bge_small_model,
            args.k,
            device=args.device,
            query_prefix=args.bge_small_prefix,
        )
        for qi, q in enumerate(queries):
            out_rows[q["query_id"]]["retrievers"]["bge_small"] = {
                "doc_indices": [int(x) for x in top[qi].tolist()],
                "doc_ids": [doc_ids[int(x)] for x in top[qi].tolist()],
            }
        print(f"  bge_small done in {time.time() - t0:.1f}s", flush=True)

    # --- bge-base (st, optional) ---
    if args.bge_base_catalog:
        print("\n=== bge_base ===", flush=True)
        t0 = time.time()
        bge_cat = (
            data / args.bge_base_catalog
            if not os.path.isabs(args.bge_base_catalog)
            else Path(args.bge_base_catalog)
        )
        top = st_topk_with_argsort(
            qtexts,
            bge_cat,
            args.bge_base_model,
            args.k,
            device=args.device,
            query_prefix=args.bge_base_prefix,
        )
        for qi, q in enumerate(queries):
            out_rows[q["query_id"]]["retrievers"]["bge_base"] = {
                "doc_indices": [int(x) for x in top[qi].tolist()],
                "doc_ids": [doc_ids[int(x)] for x in top[qi].tolist()],
            }
        print(f"  bge_base done in {time.time() - t0:.1f}s", flush=True)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for q in queries:
            f.write(json.dumps(out_rows[q["query_id"]]) + "\n")
    print(f"\nwrote {len(queries)} rows -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
