#!/usr/bin/env python3
"""Multi-retriever R@K eval on a jobs corpus.

Single-positive eval: each distilled query has a source_doc_id; a retriever
"hits" if that doc is in its top-K. Head queries (no source_doc_id) are
skipped.

Retrievers are specified by --retriever and ranked side-by-side:
  bm25                       -> BM25 over titles.json
  st:<vecs.npy>:<model_id>   -> sentence-transformers query encode + cosine
  openai:<vecs.npy>:<model>:<dim>
                              -> OpenAI embedding + cosine

Usage:
  .venv/bin/python evaluation/eval_jobs_retrievers.py \\
      --data-dir jobs_data_usajobs \\
      --retriever name=bm25:bm25 \\
      --retriever name=base:st:base_catalog.vecs.fp16.npy:sentence-transformers/all-MiniLM-L6-v2 \\
      --retriever name=bge:st:bge_small_en_catalog.vecs.fp16.npy:BAAI/bge-small-en-v1.5 \\
      --retriever name=te3:openai:te3_large_1024.vecs.fp16.npy:text-embedding-3-large:1024 \\
      --k 10 --output /tmp/usajobs_eval.json
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from dotenv import load_dotenv

load_dotenv(override=True)


def load_eval_queries(path: Path):
    """Distilled queries only (need source_doc_id)."""
    out = []
    with open(path) as f:
        for line in f:
            q = json.loads(line)
            if q.get("source") == "distilled" and q.get("source_doc_id"):
                out.append(q)
    return out


def load_titles(path: Path) -> list[str]:
    return json.load(open(path))


def load_doc_ids(path: Path) -> list[str]:
    return json.load(open(path))


def parse_retriever(spec: str) -> dict:
    """name=<name>:<kind>[:...] -> dict"""
    parts = spec.split(":")
    head = parts[0]
    if "=" not in head or not head.startswith("name="):
        raise SystemExit(f"retriever spec must start with name=...: {spec}")
    name = head[len("name=") :]
    kind = parts[1]
    rest = parts[2:]
    r = {"name": name, "kind": kind}
    if kind == "bm25":
        return r
    if kind == "st":
        if len(rest) < 2:
            raise SystemExit(f"st requires vecs and model_id: {spec}")
        r["vecs"] = rest[0]
        r["model_id"] = ":".join(rest[1:])
        return r
    if kind == "openai":
        if len(rest) < 3:
            raise SystemExit(f"openai requires vecs:model:dim: {spec}")
        r["vecs"] = rest[0]
        r["model"] = rest[1]
        r["dim"] = int(rest[2])
        return r
    raise SystemExit(f"unknown retriever kind: {kind}")


def build_bm25(titles):
    import bm25s
    from bm25s.tokenization import Tokenizer  # noqa: F401

    try:
        from Stemmer import Stemmer  # type: ignore

        stemmer = Stemmer("english")
    except Exception:
        stemmer = None
    tok = bm25s.tokenize(titles, stopwords="en", stemmer=stemmer, show_progress=False)
    idx = bm25s.BM25(k1=0.9, b=0.4)
    idx.index(tok, show_progress=False)
    return idx, stemmer


def bm25_topk(queries, idx, stemmer, k):
    import bm25s

    qtok = bm25s.tokenize(queries, stopwords="en", stemmer=stemmer, show_progress=False)
    out_idx, _ = idx.retrieve(qtok, k=k, show_progress=False)
    return out_idx  # (n, k) row-array of doc indices


def st_topk(queries, vecs_path: Path, model_id: str, k: int, device: str = "mps"):
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_id, device=device)
    qv = model.encode(
        queries,
        normalize_embeddings=True,
        batch_size=64,
        show_progress_bar=False,
    ).astype(np.float32)
    cat = np.load(vecs_path, mmap_mode="r").astype(np.float32)
    # normalize catalog
    n = np.linalg.norm(cat, axis=1, keepdims=True)
    n[n == 0] = 1.0
    cat = cat / n
    scores = qv @ cat.T  # (n_q, n_doc)
    return np.argpartition(-scores, kth=k, axis=1)[:, :k], scores


def st_topk_with_argsort(queries, vecs_path: Path, model_id: str, k: int, device: str = "mps"):
    top_idx, scores = st_topk(queries, vecs_path, model_id, k, device)
    # sort within top-k by score desc
    out = []
    for r, idx in enumerate(top_idx):
        order = idx[np.argsort(-scores[r, idx])]
        out.append(order)
    return np.array(out)


def openai_query_encode(queries, model, dim, purpose):
    from openai import OpenAI

    client = OpenAI()
    chunks = []
    batch = 256
    for i in range(0, len(queries), batch):
        sub = queries[i : i + batch]
        resp = client.embeddings.create(model=model, input=sub, dimensions=dim if dim else None)
        chunks.extend([d.embedding for d in resp.data])
    return np.array(chunks, dtype=np.float32)


def openai_topk(queries, vecs_path: Path, model: str, dim: int, k: int):
    qv = openai_query_encode(queries, model, dim, purpose=f"jobs eval {model}")
    # L2 normalize query
    qv = qv / np.maximum(np.linalg.norm(qv, axis=1, keepdims=True), 1e-12)
    cat = np.load(vecs_path, mmap_mode="r").astype(np.float32)
    n = np.linalg.norm(cat, axis=1, keepdims=True)
    n[n == 0] = 1.0
    cat = cat / n
    scores = qv @ cat.T
    top_idx = np.argpartition(-scores, kth=k, axis=1)[:, :k]
    out = []
    for r, idx in enumerate(top_idx):
        order = idx[np.argsort(-scores[r, idx])]
        out.append(order)
    return np.array(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--queries-file", default="eval_queries.jsonl")
    ap.add_argument(
        "--retriever",
        action="append",
        required=True,
        help="repeatable spec; see file docstring",
    )
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    data = Path(args.data_dir)
    titles = load_titles(data / "titles.json")
    doc_ids = load_doc_ids(data / "doc_ids.json")
    pid2idx = {p: i for i, p in enumerate(doc_ids)}
    queries = load_eval_queries(data / args.queries_file)
    print(
        f"corpus={args.data_dir} docs={len(titles):,} eval_queries={len(queries):,}",
        flush=True,
    )

    # Gold: source_doc_id -> doc index
    qtexts = [q["query"] for q in queries]
    golds = []
    for q in queries:
        idx = pid2idx.get(q["source_doc_id"], -1)
        golds.append(idx)
    n_gold_ok = sum(1 for g in golds if g >= 0)
    print(f"  gold-doc resolution: {n_gold_ok:,}/{len(queries):,}", flush=True)

    specs = [parse_retriever(s) for s in args.retriever]

    # Pre-build BM25 once if needed
    bm25_idx = bm25_stem = None
    if any(s["kind"] == "bm25" for s in specs):
        print("building BM25 index...", flush=True)
        t0 = time.time()
        bm25_idx, bm25_stem = build_bm25(titles)
        print(f"  built in {time.time() - t0:.1f}s", flush=True)

    results = {}
    for s in specs:
        name = s["name"]
        print(f"\n=== {name} ({s['kind']}) ===", flush=True)
        t0 = time.time()
        if s["kind"] == "bm25":
            top = bm25_topk(qtexts, bm25_idx, bm25_stem, args.k)
        elif s["kind"] == "st":
            vp = data / s["vecs"] if not os.path.isabs(s["vecs"]) else Path(s["vecs"])
            top = st_topk_with_argsort(qtexts, vp, s["model_id"], args.k, args.device)
        elif s["kind"] == "openai":
            vp = data / s["vecs"] if not os.path.isabs(s["vecs"]) else Path(s["vecs"])
            top = openai_topk(qtexts, vp, s["model"], s["dim"], args.k)
        else:
            continue
        elapsed = time.time() - t0

        # Hit-at-K metrics
        hits1 = hits5 = hits10 = 0
        ranks = []
        for r, gi in enumerate(golds):
            if gi < 0:
                continue
            row = list(top[r])
            try:
                rank = row.index(gi) + 1
            except ValueError:
                rank = None
            ranks.append(rank)
            if rank == 1:
                hits1 += 1
            if rank and rank <= 5:
                hits5 += 1
            if rank and rank <= args.k:
                hits10 += 1
        n = n_gold_ok
        results[name] = {
            "kind": s["kind"],
            "elapsed_s": round(elapsed, 2),
            "r_at_1": round(hits1 / n, 4),
            "r_at_5": round(hits5 / n, 4),
            f"r_at_{args.k}": round(hits10 / n, 4),
            "n": n,
        }
        print(
            f"  R@1={hits1 / n:.4f}  R@5={hits5 / n:.4f}  "
            f"R@{args.k}={hits10 / n:.4f}  ({elapsed:.1f}s)",
            flush=True,
        )

    print("\n=== summary ===", flush=True)
    for name, r in results.items():
        print(
            f"  {name:20s}  R@1={r['r_at_1']:.4f}  R@5={r['r_at_5']:.4f}  "
            f"R@{args.k}={r[f'r_at_{args.k}']:.4f}  ({r['elapsed_s']:.1f}s)",
            flush=True,
        )

    if args.output:
        out = {
            "data_dir": str(args.data_dir),
            "n_queries": n_gold_ok,
            "k": args.k,
            "results": results,
        }
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\nwrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
