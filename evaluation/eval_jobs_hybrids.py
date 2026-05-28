#!/usr/bin/env python3
"""Score cascade and RRF hybrids of BM25 + dense retrievers on a jobs corpus.

Single-positive R@K eval (same gold logic as eval_jobs_retrievers.py): each
distilled query has source_doc_id; a retriever hits if that doc is in top-K.

Cascade(bm25 -> dense): BM25 top-N -> dense scores on that pool -> top-K.
RRF(bm25, dense): fuse top-N ranks from both, score 1/(c+rank), take top-K.

Dense source specs are repeatable. Two kinds:
  Live encode:  vecs=...,model=<st_model_or_path>,name=<label>
  Pre-encoded:  vecs=...,kind=preenc,query_vec_dirs=<dir1;dir2;...>,name=<label>
    Each query_vec_dir must contain eval_queries_te3_1024.{ids.json,vecs.fp16.npy};
    rows union across dirs by query string (matches eval_jobs_retrievers.py).

Usage:
  .venv/bin/python evaluation/eval_jobs_hybrids.py \\
      --data-dir jobs_data \\
      --bm25-n 100 \\
      --dense vecs=jobs_bod_catalog.vecs.fp16.npy,model=query_model_jobs_bod,name=bod_minilm \\
      --dense vecs=jobs_bge_bod_catalog.vecs.fp16.npy,model=query_model_jobs_bge_bod,name=bge_bod \\
      --dense vecs=bge_small_en_catalog.vecs.fp16.npy,model=BAAI/bge-small-en-v1.5,name=bge \\
      --device cpu --k 10 --output /tmp/jobs_data_hybrids.json
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

RRF_C = 60


def load_queries(path: Path):
    out = []
    with open(path) as f:
        for line in f:
            q = json.loads(line)
            if q.get("source") == "distilled" and q.get("source_doc_id"):
                out.append(q)
    return out


def build_bm25(titles):
    import bm25s

    try:
        from Stemmer import Stemmer  # type: ignore

        stemmer = Stemmer("english")
    except Exception:
        stemmer = None
    tok = bm25s.tokenize(titles, stopwords="en", stemmer=stemmer, show_progress=False)
    idx = bm25s.BM25(k1=0.9, b=0.4)
    idx.index(tok, show_progress=False)
    return idx, stemmer


def bm25_topn(queries, idx, stemmer, n):
    import bm25s

    qtok = bm25s.tokenize(queries, stopwords="en", stemmer=stemmer, show_progress=False)
    out_idx, out_scores = idx.retrieve(qtok, k=n, show_progress=False)
    return np.asarray(out_idx), np.asarray(out_scores)


def encode_queries_st(queries, model_id, device, query_prefix=""):
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_id, device=device)
    if query_prefix:
        queries = [query_prefix + q for q in queries]
    qv = model.encode(
        queries,
        normalize_embeddings=True,
        batch_size=64,
        show_progress_bar=False,
    ).astype(np.float32)
    return qv


def load_norm_cat(path: Path) -> np.ndarray:
    cat = np.load(path, mmap_mode="r").astype(np.float32)
    norms = np.linalg.norm(cat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return cat / norms


def dense_full_topn(qv, cat, n):
    scores = qv @ cat.T  # (n_q, n_doc)
    top = np.argpartition(-scores, kth=n - 1, axis=1)[:, :n]
    # sort
    out = np.empty_like(top)
    for r, row in enumerate(top):
        order = np.argsort(-scores[r, row])
        out[r] = row[order]
    return out, scores  # full scores returned for cascade reranking


def cascade_topk(bm25_top_n, dense_scores, k):
    """For each query, take BM25's top-N doc ids and re-sort by dense scores."""
    n_q = bm25_top_n.shape[0]
    out = np.empty((n_q, k), dtype=bm25_top_n.dtype)
    for r in range(n_q):
        pool = bm25_top_n[r]
        s = dense_scores[r, pool]
        order = np.argsort(-s)
        out[r] = pool[order[:k]]
    return out


def rrf_topk(top_n_a, top_n_b, k, c=RRF_C):
    """RRF over two ranked id arrays."""
    return rrf_topk_multi([top_n_a, top_n_b], k, c=c)


def rrf_topk_multi(top_ns, k, c=RRF_C):
    """RRF over N ranked id arrays, each shape (n_q, n_per_query)."""
    n_q = top_ns[0].shape[0]
    out = np.empty((n_q, k), dtype=top_ns[0].dtype)
    for r in range(n_q):
        rank_score: dict[int, float] = {}
        for top_n in top_ns:
            for rank, did in enumerate(top_n[r]):
                rank_score[int(did)] = rank_score.get(int(did), 0.0) + 1.0 / (c + rank + 1)
        ranked = sorted(rank_score.items(), key=lambda x: -x[1])
        out[r] = [d for d, _ in ranked[:k]]
    return out


def hits_at_k(top, golds, k_list=(1, 5, 10)):
    n = sum(1 for g in golds if g >= 0)
    counters = {k: 0 for k in k_list}
    for r, gi in enumerate(golds):
        if gi < 0:
            continue
        row = list(top[r])
        try:
            rank = row.index(gi) + 1
        except ValueError:
            continue
        for k in k_list:
            if rank <= k:
                counters[k] += 1
    return {k: counters[k] / n for k in k_list}, n


def parse_dense(spec: str) -> dict:
    """vecs=...,name=...,[model=...,query_prefix=... | kind=preenc,query_vec_dirs=d1;d2;...]"""
    parts = dict(p.split("=", 1) for p in spec.split(","))
    if "vecs" not in parts or "name" not in parts:
        raise SystemExit(f"--dense missing vecs/name: {spec}")
    if parts.get("kind") == "preenc":
        if "query_vec_dirs" not in parts:
            raise SystemExit(f"--dense kind=preenc requires query_vec_dirs: {spec}")
    elif "model" not in parts:
        raise SystemExit(f"--dense missing model (or kind=preenc): {spec}")
    return parts


def load_preenc_query_map(query_vec_dirs):
    qmap = {}
    for d in query_vec_dirs:
        ids_path = Path(d) / "eval_queries_te3_1024.ids.json"
        vec_path = Path(d) / "eval_queries_te3_1024.vecs.fp16.npy"
        with open(ids_path) as f:
            ids = json.load(f)
        vecs = np.load(vec_path).astype(np.float32)
        for q, v in zip(ids, vecs):
            qmap[q] = v
    return qmap


def encode_queries_preenc(queries, query_vec_dirs):
    qmap = load_preenc_query_map(query_vec_dirs)
    miss = [q for q in queries if q not in qmap]
    if miss:
        raise SystemExit(f"preenc: {len(miss)} queries missing; first miss: {miss[0]!r}")
    qv = np.stack([qmap[q] for q in queries], axis=0).astype(np.float32)
    n = np.linalg.norm(qv, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return qv / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--queries-file", default="eval_queries.jsonl")
    ap.add_argument("--dense", action="append", required=True, help="repeatable; see docstring")
    ap.add_argument("--bm25-n", type=int, default=100)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--output", default=None)
    ap.add_argument(
        "--rrf-combo",
        action="append",
        default=[],
        help="repeatable; comma-separated retriever names to RRF-fuse, e.g. bm25,e5_small,te3. "
        "Reserved name 'bm25' refers to BM25; other names must match --dense ...,name=X.",
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

    # Build BM25
    print("building BM25...", flush=True)
    t0 = time.time()
    bm25_idx, bm25_stem = build_bm25(titles)
    print(f"  built in {time.time() - t0:.1f}s", flush=True)
    print(f"running BM25 top-{args.bm25_n}...", flush=True)
    t0 = time.time()
    bm25_top_n, _ = bm25_topn(qtexts, bm25_idx, bm25_stem, args.bm25_n)
    bm25_top_k = bm25_top_n[:, : args.k]
    print(f"  done in {time.time() - t0:.1f}s  shape={bm25_top_n.shape}", flush=True)

    # Run each dense retriever -> top_n and full scores
    dense_results = {}
    for spec in args.dense:
        d = parse_dense(spec)
        name = d["name"]
        vp = Path(d["vecs"]) if os.path.isabs(d["vecs"]) else data / d["vecs"]
        if d.get("kind") == "preenc":
            qvdirs = d["query_vec_dirs"].split(";")
            print(f"\nrunning dense '{name}' (preenc dirs={qvdirs})...", flush=True)
            t0 = time.time()
            qv = encode_queries_preenc(qtexts, qvdirs)
        else:
            qp = d.get("query_prefix", "")
            print(f"\nrunning dense '{name}' (model={d['model']} prefix={qp!r})...", flush=True)
            t0 = time.time()
            qv = encode_queries_st(qtexts, d["model"], args.device, query_prefix=qp)
        cat = load_norm_cat(vp)
        top_n, scores = dense_full_topn(qv, cat, args.bm25_n)
        elapsed = time.time() - t0
        print(f"  done in {elapsed:.1f}s", flush=True)
        dense_results[name] = {"top_n": top_n, "scores": scores}

    # Compute metrics: BM25 alone, each dense alone, each cascade(bm25,dense), each rrf(bm25,dense)
    out = {}
    out["bm25"] = hits_at_k(bm25_top_k, golds)[0]
    print(
        f"\n  bm25            R@1={out['bm25'][1]:.4f}  R@5={out['bm25'][5]:.4f}  R@10={out['bm25'][10]:.4f}"
    )

    for name, dr in dense_results.items():
        dense_top_k = dr["top_n"][:, : args.k]
        out[name] = hits_at_k(dense_top_k, golds)[0]
        print(
            f"  {name:15s} R@1={out[name][1]:.4f}  R@5={out[name][5]:.4f}  R@10={out[name][10]:.4f}"
        )

        cascade_top = cascade_topk(bm25_top_n, dr["scores"], args.k)
        out[f"cascade_bm25_{name}"] = hits_at_k(cascade_top, golds)[0]
        m = out[f"cascade_bm25_{name}"]
        print(f"  cascade_bm25_{name:8s}  R@1={m[1]:.4f}  R@5={m[5]:.4f}  R@10={m[10]:.4f}")

        rrf_top = rrf_topk(bm25_top_n, dr["top_n"], args.k)
        out[f"rrf_bm25_{name}"] = hits_at_k(rrf_top, golds)[0]
        m = out[f"rrf_bm25_{name}"]
        print(f"  rrf_bm25_{name:12s} R@1={m[1]:.4f}  R@5={m[5]:.4f}  R@10={m[10]:.4f}")

    name_to_top_n = {"bm25": bm25_top_n}
    for name, dr in dense_results.items():
        name_to_top_n[name] = dr["top_n"]

    for combo_spec in args.rrf_combo:
        names = [n.strip() for n in combo_spec.split(",") if n.strip()]
        if len(names) < 2:
            raise SystemExit(f"--rrf-combo needs >=2 names: {combo_spec!r}")
        missing = [n for n in names if n not in name_to_top_n]
        if missing:
            raise SystemExit(
                f"--rrf-combo {combo_spec!r}: unknown names {missing}; "
                f"available: {sorted(name_to_top_n)}"
            )
        combo_top_ns = [name_to_top_n[n] for n in names]
        combo_top = rrf_topk_multi(combo_top_ns, args.k)
        key = "rrf_" + "_".join(names)
        out[key] = hits_at_k(combo_top, golds)[0]
        m = out[key]
        print(f"  {key:30s} R@1={m[1]:.4f}  R@5={m[5]:.4f}  R@10={m[10]:.4f}")

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(
                {
                    "data_dir": str(args.data_dir),
                    "bm25_n": args.bm25_n,
                    "k": args.k,
                    "n_queries": n_ok,
                    "results": {n: {k: float(v) for k, v in r.items()} for n, r in out.items()},
                },
                f,
                indent=2,
            )
        print(f"\nwrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
