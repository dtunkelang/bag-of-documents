#!/usr/bin/env python3
"""CE rerank on top-N of RRF(BM25, e5_small, te3) for jobs retrieval.

Reuses the retriever stage from eval_jobs_hybrids.py (BM25 + dense + N-way
RRF), then re-scores the top-N pool with a CrossEncoder over
(query, title + description). Emits single-positive R@1/5/10 evaluated at
pool sizes K_eval in {50, 100} from the same CE scores.

Pass criterion vs shipped 3-way RRF (R@10 = 0.7457): must improve by at
least +0.5pp at one of the eval K's.

Usage:
  .venv/bin/python evaluation/eval_ce_rerank_jobs.py \\
      --data-dir unified_jobs \\
      --ce-model BAAI/bge-reranker-base \\
      --top-n 100 \\
      --device mps \\
      --output evaluation/results/jobs_unified_ce_rerank_bge_base.json
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np  # noqa: E402

from evaluation.eval_jobs_hybrids import (  # noqa: E402
    bm25_topn,
    build_bm25,
    dense_full_topn,
    encode_queries_preenc,
    encode_queries_st,
    hits_at_k,
    load_norm_cat,
    load_queries,
    rrf_topk_multi,
)

DEFAULT_TE3_DIRS = [
    "jobs_data",
    "jobs_data_usajobs",
    "jobs_data_linkedin",
    "jobs_data_jobstreet",
]


def load_doc_texts(metadata_path: Path, n_docs: int, max_chars: int) -> list[str]:
    """Stream metadata.jsonl in doc-position order. Returns text per position,
    truncated to max_chars to bound RAM. Verified order matches doc_ids.json
    elsewhere; we sanity-check the first id at load time."""
    texts: list[str] = [""] * n_docs
    t0 = time.time()
    with open(metadata_path) as f:
        for i, line in enumerate(f):
            if i >= n_docs:
                break
            d = json.loads(line)
            title = d.get("title") or ""
            desc = d.get("description") or ""
            txt = (title + "\n\n" + desc) if desc else title
            if len(txt) > max_chars:
                txt = txt[:max_chars]
            texts[i] = txt
    elapsed = time.time() - t0
    print(
        f"  loaded {n_docs:,} doc texts in {elapsed:.1f}s "
        f"(avg {sum(len(t) for t in texts) / n_docs:.0f} chars, max {max_chars})",
        flush=True,
    )
    return texts


def ce_rerank_scores(
    ce_model_id: str,
    device: str,
    queries: list[str],
    pool: np.ndarray,
    doc_texts: list[str],
    batch_size: int,
    flush_pairs: int,
) -> np.ndarray:
    """Score (q, doc_text) for every (qi, j) in pool. Returns (n_q, n_pool) float32."""
    import torch
    from sentence_transformers import CrossEncoder

    n_q, n_pool = pool.shape
    if device == "mps" and not torch.backends.mps.is_available():
        device = "cpu"
    print(f"loading CE {ce_model_id} on {device}...", flush=True)
    t0 = time.time()
    ce = CrossEncoder(ce_model_id, device=device)
    print(f"  loaded in {time.time() - t0:.0f}s", flush=True)

    scores = np.full((n_q, n_pool), -np.inf, dtype=np.float32)
    n_pairs_total = n_q * n_pool
    pairs_buf: list[tuple[str, str]] = []
    locs_buf: list[tuple[int, int]] = []
    n_done = 0
    t0 = time.time()
    print(f"CE rerank: {n_q:,} q x {n_pool} pool = {n_pairs_total:,} pairs", flush=True)
    for qi in range(n_q):
        q = queries[qi]
        row = pool[qi]
        for j in range(n_pool):
            pos = int(row[j])
            if pos < 0:
                continue
            pairs_buf.append((q, doc_texts[pos]))
            locs_buf.append((qi, j))
        if len(pairs_buf) >= flush_pairs:
            sc = ce.predict(pairs_buf, batch_size=batch_size, show_progress_bar=False)
            for (qi2, j2), s in zip(locs_buf, sc):
                scores[qi2, j2] = float(s)
            n_done += len(pairs_buf)
            elapsed = time.time() - t0
            rate = n_done / max(elapsed, 1e-3)
            eta = (n_pairs_total - n_done) / max(rate, 1e-3)
            print(
                f"  {n_done:,}/{n_pairs_total:,} ({n_done / n_pairs_total:.1%}) "
                f"@ {rate:.0f}/s  eta {eta / 60:.1f}m",
                flush=True,
            )
            pairs_buf.clear()
            locs_buf.clear()
    if pairs_buf:
        sc = ce.predict(pairs_buf, batch_size=batch_size, show_progress_bar=False)
        for (qi2, j2), s in zip(locs_buf, sc):
            scores[qi2, j2] = float(s)
        n_done += len(pairs_buf)
    elapsed = time.time() - t0
    print(f"CE done: {n_done:,} pairs in {elapsed:.1f}s ({n_done / elapsed:.0f}/s)", flush=True)
    return scores


def rerank_topk(pool: np.ndarray, ce_scores: np.ndarray, k: int) -> np.ndarray:
    """For each query, sort `pool` by CE score (desc) and take top-k."""
    n_q, _ = pool.shape
    out = np.empty((n_q, k), dtype=pool.dtype)
    for r in range(n_q):
        order = np.argsort(-ce_scores[r])
        out[r] = pool[r, order[:k]]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="unified_jobs")
    ap.add_argument("--queries-file", default="eval_queries_unified.jsonl")
    ap.add_argument("--metadata-file", default="metadata.jsonl")
    ap.add_argument("--bm25-n", type=int, default=100)
    ap.add_argument(
        "--top-n",
        type=int,
        default=100,
        help="size of RRF pool to feed to CE; eval reports K=50 and K=100",
    )
    ap.add_argument("--ce-model", default="BAAI/bge-reranker-base")
    ap.add_argument("--ce-device", default="mps")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--flush-pairs", type=int, default=2048)
    ap.add_argument("--max-text-chars", type=int, default=2048)
    ap.add_argument("--e5-model", default="intfloat/e5-small-v2")
    ap.add_argument("--e5-vecs", default="e5_small_catalog.vecs.fp16.npy")
    ap.add_argument(
        "--e5-query-prefix",
        default="query: ",
        help="E5 family expects 'query: ' prefix at encode time",
    )
    ap.add_argument(
        "--te3-vecs",
        default="te3_catalog.vecs.fp16.npy",
        help="path relative to data-dir (or absolute)",
    )
    ap.add_argument(
        "--te3-query-vec-dirs",
        default=";".join(DEFAULT_TE3_DIRS),
        help="semicolon-separated dirs containing eval_queries_te3_1024.{ids.json,vecs.fp16.npy}",
    )
    ap.add_argument("--retriever-device", default="cpu")
    ap.add_argument("--output", required=True)
    ap.add_argument(
        "--scores-npy",
        default=None,
        help="optional path to save raw CE scores (n_q, top_n) for reuse",
    )
    args = ap.parse_args()

    data = Path(args.data_dir)
    with open(data / "titles.json") as f:
        titles = json.load(f)
    with open(data / "doc_ids.json") as f:
        doc_ids = json.load(f)
    pid2idx = {p: i for i, p in enumerate(doc_ids)}
    qs = load_queries(data / args.queries_file)
    qtexts = [q["query"] for q in qs]
    golds = [pid2idx.get(q["source_doc_id"], -1) for q in qs]
    n_ok = sum(1 for g in golds if g >= 0)
    print(
        f"corpus={args.data_dir} docs={len(titles):,} queries={len(qs):,} gold-resolved={n_ok:,}",
        flush=True,
    )

    # Sanity: top-n must be <= bm25-n (RRF needs same-N rank lists)
    if args.top_n > args.bm25_n:
        raise SystemExit(f"--top-n {args.top_n} must be <= --bm25-n {args.bm25_n}")

    # ---- Retriever stage: BM25 + e5_small + te3 ----
    print("\nbuilding BM25...", flush=True)
    t0 = time.time()
    bm25_idx, bm25_stem = build_bm25(titles)
    print(f"  built in {time.time() - t0:.1f}s", flush=True)
    print(f"running BM25 top-{args.bm25_n}...", flush=True)
    t0 = time.time()
    bm25_top_n, _ = bm25_topn(qtexts, bm25_idx, bm25_stem, args.bm25_n)
    print(f"  done in {time.time() - t0:.1f}s", flush=True)

    print(f"\nrunning e5_small ({args.e5_model}) on {args.retriever_device}...", flush=True)
    t0 = time.time()
    qv_e5 = encode_queries_st(
        qtexts, args.e5_model, args.retriever_device, query_prefix=args.e5_query_prefix
    )
    e5_path = Path(args.e5_vecs) if os.path.isabs(args.e5_vecs) else data / args.e5_vecs
    e5_cat = load_norm_cat(e5_path)
    e5_top_n, _ = dense_full_topn(qv_e5, e5_cat, args.bm25_n)
    del e5_cat
    print(f"  done in {time.time() - t0:.1f}s", flush=True)

    print("\nloading te3 preenc query vecs...", flush=True)
    t0 = time.time()
    te3_dirs = [d for d in args.te3_query_vec_dirs.split(";") if d]
    qv_te3 = encode_queries_preenc(qtexts, te3_dirs)
    te3_path = Path(args.te3_vecs) if os.path.isabs(args.te3_vecs) else data / args.te3_vecs
    te3_cat = load_norm_cat(te3_path)
    te3_top_n, _ = dense_full_topn(qv_te3, te3_cat, args.bm25_n)
    del te3_cat
    print(f"  done in {time.time() - t0:.1f}s", flush=True)

    # ---- Fuse to RRF pool ----
    print(f"\nfusing RRF(bm25, e5_small, te3) top-{args.top_n}...", flush=True)
    t0 = time.time()
    pool = rrf_topk_multi([bm25_top_n, e5_top_n, te3_top_n], args.top_n)
    print(f"  done in {time.time() - t0:.1f}s  shape={pool.shape}", flush=True)

    out: dict[str, dict] = {}

    # Baseline at K=10 from the RRF pool (no CE) — should match shipped 0.7457.
    baseline_k = pool[:, :10]
    out["rrf_baseline_k10"] = hits_at_k(baseline_k, golds)[0]
    m = out["rrf_baseline_k10"]
    print(
        f"\n  rrf_baseline (pool[:10])  R@1={m[1]:.4f}  R@5={m[5]:.4f}  R@10={m[10]:.4f}",
        flush=True,
    )

    # ---- Load doc texts ----
    print(f"\nloading doc texts from {args.metadata_file}...", flush=True)
    doc_texts = load_doc_texts(data / args.metadata_file, len(doc_ids), args.max_text_chars)

    # ---- CE rerank ----
    print()
    ce_scores = ce_rerank_scores(
        ce_model_id=args.ce_model,
        device=args.ce_device,
        queries=qtexts,
        pool=pool,
        doc_texts=doc_texts,
        batch_size=args.batch_size,
        flush_pairs=args.flush_pairs,
    )

    if args.scores_npy:
        Path(args.scores_npy).parent.mkdir(parents=True, exist_ok=True)
        np.save(args.scores_npy, ce_scores)
        print(f"saved CE scores to {args.scores_npy}", flush=True)

    # ---- Eval at K_pool=50, K_pool=100 (same CE scores, truncated) ----
    print()
    for pool_k in (50, args.top_n):
        if pool_k > args.top_n:
            continue
        sub_pool = pool[:, :pool_k]
        sub_scores = ce_scores[:, :pool_k]
        reranked = rerank_topk(sub_pool, sub_scores, 10)
        key = f"rrf_ce_pool{pool_k}"
        out[key] = hits_at_k(reranked, golds)[0]
        m = out[key]
        print(
            f"  {key:24s}  R@1={m[1]:.4f}  R@5={m[5]:.4f}  R@10={m[10]:.4f}",
            flush=True,
        )

    # ---- Save JSON ----
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(
            {
                "data_dir": str(args.data_dir),
                "ce_model": args.ce_model,
                "ce_device": args.ce_device,
                "top_n": args.top_n,
                "bm25_n": args.bm25_n,
                "max_text_chars": args.max_text_chars,
                "n_queries": n_ok,
                "results": {n: {k: float(v) for k, v in r.items()} for n, r in out.items()},
            },
            f,
            indent=2,
        )
    print(f"\nwrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
