#!/usr/bin/env python3
"""Residualized Rocchio (pseudo-relevance feedback) eval for jobs retrieval.

Tests whether blending the query with the mean of its top-k initial-pass
documents materially improves R@K. Standard one-pass PRF:

    q' = (1 - lambda) * q + lambda * c                       (then normalized)

where c is the top-k centroid, optionally residualized via --residualize:
    none        : c = centroid                                (plain Rocchio)
    q_perp      : c = centroid - (centroid . q) * q           (drop the q-parallel
                                                               component before blend)
    corpus_mean : c = centroid - mean(catalog)                (drop generic
                                                               popular-doc directions)

then re-score the full catalog with q'. lambda=0 reproduces the base retriever.

Supports two query sources, matching eval_jobs_retrievers.py:
  * st: live-encode queries with a sentence-transformers model
  * preenc: load pre-encoded query vectors from per-corpus dirs

Usage:
  .venv/bin/python evaluation/eval_rocchio.py \\
      --data-dir unified_jobs \\
      --queries-file eval_queries_unified.jsonl \\
      --catalog-vecs bge_base_catalog.vecs.fp16.npy \\
      --name bge_base \\
      --query-encoder BAAI/bge-base-en-v1.5 \\
      --sweep \\
      --output evaluation/results/rocchio_bge_base.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402


def load_titles(path: Path) -> list[str]:
    return json.load(open(path))


def load_doc_ids(path: Path) -> list[str]:
    return json.load(open(path))


def load_eval_queries(path: Path) -> list[dict]:
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def normalize_rows(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return x / n


def encode_queries_st(queries, model_id, device, query_prefix, trust_remote_code):
    from sentence_transformers import SentenceTransformer

    st_kwargs = {"device": device}
    if trust_remote_code:
        st_kwargs["trust_remote_code"] = True
    model = SentenceTransformer(model_id, **st_kwargs)
    if query_prefix:
        queries = [query_prefix + q for q in queries]
    qv = model.encode(
        queries,
        normalize_embeddings=True,
        batch_size=64,
        show_progress_bar=False,
    ).astype(np.float32)
    return qv


def encode_queries_preenc(queries, query_vec_dirs):
    qmap: dict[str, np.ndarray] = {}
    for d in query_vec_dirs:
        with open(Path(d) / "eval_queries_te3_1024.ids.json") as f:
            ids = json.load(f)
        vecs = np.load(Path(d) / "eval_queries_te3_1024.vecs.fp16.npy").astype(np.float32)
        for q, v in zip(ids, vecs):
            qmap[q] = v
    miss = [q for q in queries if q not in qmap]
    if miss:
        raise SystemExit(f"preenc: {len(miss)} queries missing; first: {miss[0]!r}")
    qv = np.stack([qmap[q] for q in queries], axis=0).astype(np.float32)
    return normalize_rows(qv)


QUERY_BATCH = 256


def _topk_in_batches(qv_norm: np.ndarray, cat_norm: np.ndarray, k: int) -> np.ndarray:
    """Compute argpartition top-k via query-batched matmul to bound memory.
    Returns indices sorted within top-k by descending score."""
    n_q = qv_norm.shape[0]
    out = np.empty((n_q, k), dtype=np.int64)
    for s in range(0, n_q, QUERY_BATCH):
        e = min(s + QUERY_BATCH, n_q)
        sc = qv_norm[s:e] @ cat_norm.T  # (bs, n_doc)
        idx = np.argpartition(-sc, kth=k, axis=1)[:, :k]
        for r in range(idx.shape[0]):
            out[s + r] = idx[r][np.argsort(-sc[r, idx[r]])]
    return out


def topk_indices(qv_norm: np.ndarray, cat_norm: np.ndarray, k: int) -> np.ndarray:
    return _topk_in_batches(qv_norm, cat_norm, k)


def rocchio_topk(
    qv_norm: np.ndarray,
    cat_norm: np.ndarray,
    k_eval: int,
    k_prf: int,
    lam: float,
    residualize: str = "none",
    corpus_mean: np.ndarray | None = None,
) -> np.ndarray:
    """Two-pass PRF. First pass: top-k_prf for centroid. Second pass: re-score full
    catalog with blended query. Batched to keep score-matrix memory bounded.

    residualize:
      * "none"        — standard Rocchio: q' = (1-lam)*q + lam*centroid
      * "q_perp"      — drop the component of centroid parallel to q before blending,
                        so PRF can only move q in directions it doesn't already hold
      * "corpus_mean" — subtract the global catalog mean from centroid before blending,
                        cancelling generic popular-doc directions
    """
    if lam == 0.0:
        return _topk_in_batches(qv_norm, cat_norm, k_eval)
    if residualize == "corpus_mean" and corpus_mean is None:
        raise ValueError("residualize=corpus_mean requires corpus_mean")

    n_q = qv_norm.shape[0]
    out = np.empty((n_q, k_eval), dtype=np.int64)
    for s in range(0, n_q, QUERY_BATCH):
        e = min(s + QUERY_BATCH, n_q)
        q_batch = qv_norm[s:e]
        sc1 = q_batch @ cat_norm.T
        top_prf = np.argpartition(-sc1, kth=k_prf, axis=1)[:, :k_prf]
        centroids = np.stack(
            [cat_norm[top_prf[r]].mean(axis=0) for r in range(top_prf.shape[0])], axis=0
        )
        centroids = normalize_rows(centroids)
        if residualize == "q_perp":
            proj = np.sum(centroids * q_batch, axis=1, keepdims=True)
            centroids = normalize_rows(centroids - proj * q_batch)
        elif residualize == "corpus_mean":
            centroids = normalize_rows(centroids - corpus_mean)
        q_adapted = normalize_rows((1.0 - lam) * q_batch + lam * centroids)
        sc2 = q_adapted @ cat_norm.T
        idx = np.argpartition(-sc2, kth=k_eval, axis=1)[:, :k_eval]
        for r in range(idx.shape[0]):
            out[s + r] = idx[r][np.argsort(-sc2[r, idx[r]])]
    return out


def hit_metrics(top_idx: np.ndarray, golds: list[int], k_eval: int) -> dict:
    h1 = h5 = h10 = n = 0
    for r, gi in enumerate(golds):
        if gi < 0:
            continue
        n += 1
        row = top_idx[r]
        if row[0] == gi:
            h1 += 1
        if gi in row[:5]:
            h5 += 1
        if gi in row[:10]:
            h10 += 1
    return {
        "n": n,
        "r_at_1": round(h1 / max(n, 1), 4),
        "r_at_5": round(h5 / max(n, 1), 4),
        "r_at_10": round(h10 / max(n, 1), 4),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--queries-file", default="eval_queries.jsonl")
    ap.add_argument("--name", required=True, help="retriever name for output")
    ap.add_argument("--catalog-vecs", required=True, help="filename under data-dir or abs path")
    ap.add_argument("--query-encoder", default=None, help="ST model id for live encoding")
    ap.add_argument(
        "--query-vec-dirs",
        default=None,
        help="comma-separated dirs holding eval_queries_te3_1024.{ids.json,vecs.fp16.npy}",
    )
    ap.add_argument("--query-prefix", default="", help="prefix for queries when live-encoding")
    ap.add_argument("--trust-remote-code", action="store_true")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--top-k-prf", type=int, default=10)
    ap.add_argument("--lambdas", default="0.0,0.1,0.2,0.3,0.5,0.7")
    ap.add_argument(
        "--residualize",
        default="none",
        choices=["none", "q_perp", "corpus_mean"],
        help="PRF centroid residualization mode; see rocchio_topk docstring",
    )
    ap.add_argument(
        "--sweep",
        action="store_true",
        help="also sweep top-k-prf in {5,10,20}; otherwise only the --top-k-prf value",
    )
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    if not (args.query_encoder or args.query_vec_dirs):
        raise SystemExit("must pass --query-encoder or --query-vec-dirs")
    if args.query_encoder and args.query_vec_dirs:
        raise SystemExit("pass only one of --query-encoder or --query-vec-dirs")

    data = Path(args.data_dir)
    titles = load_titles(data / "titles.json")
    doc_ids = load_doc_ids(data / "doc_ids.json")
    pid2idx = {p: i for i, p in enumerate(doc_ids)}
    queries = load_eval_queries(data / args.queries_file)
    qtexts = [q["query"] for q in queries]
    golds = [pid2idx.get(q["source_doc_id"], -1) for q in queries]
    print(
        f"corpus={args.data_dir} docs={len(titles):,} queries={len(queries):,}",
        flush=True,
    )

    vec_arg = Path(args.catalog_vecs)
    vec_path = vec_arg if vec_arg.is_absolute() else data / vec_arg
    print(f"loading catalog {vec_path}", flush=True)
    cat = np.load(vec_path, mmap_mode="r").astype(np.float32)
    cat_norm = normalize_rows(cat)
    print(f"  catalog shape={cat.shape}", flush=True)

    t0 = time.time()
    if args.query_encoder:
        print(f"live-encoding queries with {args.query_encoder}", flush=True)
        qv = encode_queries_st(
            qtexts, args.query_encoder, args.device, args.query_prefix, args.trust_remote_code
        )
    else:
        dirs = args.query_vec_dirs.split(",")
        print(f"loading preenc query vectors from {len(dirs)} dirs", flush=True)
        qv = encode_queries_preenc(qtexts, dirs)
    qv = normalize_rows(qv)
    print(f"  queries encoded in {time.time() - t0:.1f}s; shape={qv.shape}", flush=True)

    lambdas = [float(x) for x in args.lambdas.split(",")]
    k_prfs = [5, 10, 20] if args.sweep else [args.top_k_prf]

    corpus_mean = None
    if args.residualize == "corpus_mean":
        t_cm = time.time()
        corpus_mean = cat_norm.mean(axis=0, keepdims=True).astype(np.float32)
        print(f"corpus mean computed in {time.time() - t_cm:.1f}s", flush=True)

    print(f"residualize mode: {args.residualize}", flush=True)

    results = []
    for k_prf in k_prfs:
        for lam in lambdas:
            t1 = time.time()
            top = rocchio_topk(
                qv,
                cat_norm,
                args.k,
                k_prf,
                lam,
                residualize=args.residualize,
                corpus_mean=corpus_mean,
            )
            m = hit_metrics(top, golds, args.k)
            elapsed = time.time() - t1
            row = {
                "name": args.name,
                "lambda": lam,
                "k_prf": k_prf,
                "residualize": args.residualize,
                "elapsed_s": round(elapsed, 2),
                **m,
            }
            results.append(row)
            print(
                f"  lam={lam:.2f} k_prf={k_prf:>2} | "
                f"R@1={m['r_at_1']:.4f} R@5={m['r_at_5']:.4f} R@10={m['r_at_10']:.4f} "
                f"({elapsed:.1f}s)",
                flush=True,
            )

    summary = {
        "data_dir": args.data_dir,
        "n_queries": len(queries),
        "name": args.name,
        "catalog_vecs": str(vec_path),
        "residualize": args.residualize,
        "results": results,
    }
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\nwrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
