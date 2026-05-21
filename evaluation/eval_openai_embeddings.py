#!/usr/bin/env python3
"""Evaluate a corpus drop-in retrieval R@10/E@1/nDCG@10 using an OpenAI
text-embedding-3-* catalog encoded by download/encode_openai_embeddings.py.

Encodes the test queries via API at eval time (cheap; queries are small).
Then top-K retrieval against the cached catalog via dense cosine.

Compatible with the relevance-threshold and prefix conventions used in
eval_alt_encoder.py and eval_cc_cross_lingual.py.

Usage:
    .venv/bin/python evaluation/eval_openai_embeddings.py \\
        --data-dir nfcorpus_data \\
        --catalog-vecs nfcorpus_data/openai_te3large_1024.vecs.fp16.npy \\
        --model text-embedding-3-large --dim 1024
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse  # noqa: E402
import datetime  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import time  # noqa: E402
from collections import defaultdict  # noqa: E402

import numpy as np  # noqa: E402
from dotenv import load_dotenv  # noqa: E402

# override=True so .env always wins over stale shell exports
load_dotenv(override=True)

from openai import OpenAI  # noqa: E402

PRICES_PER_M_TOKENS = {
    "text-embedding-3-large": 0.13,
    "text-embedding-3-small": 0.02,
}

SPEND_LEDGER = Path(__file__).resolve().parent.parent / ".api_spend.jsonl"


def record_spend(provider, model, tokens, cost_usd, purpose):
    rec = {
        "ts": datetime.datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "provider": provider,
        "model": model,
        "tokens": int(tokens),
        "cost_usd": round(float(cost_usd), 4),
        "purpose": purpose,
    }
    with open(SPEND_LEDGER, "a") as f:
        f.write(json.dumps(rec) + "\n")


def encode_queries(queries, model, dim, batch_size=512):
    client = OpenAI()
    out_chunks = []
    total_tokens = 0
    for i in range(0, len(queries), batch_size):
        batch = [q if q and q.strip() else " " for q in queries[i : i + batch_size]]
        kwargs = {"model": model, "input": batch}
        if dim is not None:
            kwargs["dimensions"] = dim
        for attempt in range(5):
            try:
                r = client.embeddings.create(**kwargs)
                break
            except Exception as e:
                print(
                    f"  query batch attempt {attempt + 1} failed: {type(e).__name__}: {str(e)[:120]}",
                    flush=True,
                )
                if attempt == 4:
                    raise
                time.sleep(2**attempt)
        v = np.array([d.embedding for d in r.data], dtype=np.float32)
        v = v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-9)
        out_chunks.append(v)
        total_tokens += r.usage.total_tokens
    return np.concatenate(out_chunks, axis=0), total_tokens


def per_query_metrics(retrieved_pids, qrels_q, k=10, min_rel=2, exact_rel=3):
    pos_e = {pid for pid, g in qrels_q.items() if g >= exact_rel}
    pos_es = {pid for pid, g in qrels_q.items() if g >= min_rel}
    if not pos_es:
        return None
    top_k = retrieved_pids[:k]
    recall = sum(1 for p in top_k if p in pos_es) / len(pos_es)
    gains = [1.0 if p in pos_e else (0.1 if p in pos_es else 0.0) for p in top_k]
    dcg = sum(g / math.log2(i + 2) for i, g in enumerate(gains))
    ideal = sorted((1.0 if p in pos_e else 0.1 for p in pos_es), reverse=True)[:k]
    idcg = sum(g / math.log2(i + 2) for i, g in enumerate(ideal))
    ndcg = dcg / idcg if idcg > 0 else 0.0
    e1 = 1.0 if pos_e and top_k and top_k[0] in pos_e else 0.0 if pos_e else float("nan")
    e3 = sum(1 for p in top_k[:3] if p in pos_e) / min(3, len(pos_e)) if pos_e else float("nan")
    return recall, ndcg, e1, e3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--queries-file", default="test_queries.jsonl")
    ap.add_argument("--qrels-file", default="test_qrels.jsonl")
    ap.add_argument(
        "--ids-file", default=None, help="defaults to doc_ids.json then product_ids.json"
    )
    ap.add_argument("--catalog-vecs", required=True, help="path to cached catalog .vecs.fp16.npy")
    ap.add_argument("--model", required=True, help="OpenAI model name (for query encoding)")
    ap.add_argument("--dim", type=int, default=None, help="Matryoshka dim (must match catalog)")
    ap.add_argument("--min-relevance", type=int, default=2)
    ap.add_argument("--exact-relevance", type=int, default=3)
    ap.add_argument("--k", type=int, default=10)
    args = ap.parse_args()

    data = Path(args.data_dir).resolve()
    print(
        f"corpus: {data.name}  catalog: {args.catalog_vecs}  model: {args.model}  dim: {args.dim}",
        flush=True,
    )

    # Load qrels + queries
    qrels = defaultdict(dict)
    with open(data / args.qrels_file) as f:
        for line in f:
            r = json.loads(line)
            qrels[r["query_id"]][r["product_id"]] = r["relevance"]
    queries_all = {}
    with open(data / args.queries_file) as f:
        for line in f:
            d = json.loads(line)
            queries_all[d["query_id"]] = d["query"]
    eval_qids = [
        qid
        for qid in queries_all
        if qid in qrels and any(g >= args.min_relevance for g in qrels[qid].values())
    ]
    queries = [queries_all[qid] for qid in eval_qids]
    print(f"  {len(eval_qids):,} eval queries", flush=True)

    # Load catalog vecs + IDs
    ids_file = args.ids_file
    if not ids_file:
        for c in ("doc_ids.json", "product_ids.json"):
            if (data / c).exists():
                ids_file = c
                break
    with open(data / ids_file) as f:
        pids_arr = json.load(f)
    catalog = np.load(args.catalog_vecs, mmap_mode="r")
    if catalog.shape[0] != len(pids_arr):
        raise SystemExit(f"catalog rows ({catalog.shape[0]}) != ids ({len(pids_arr)})")
    print(f"  catalog: {catalog.shape}  dtype={catalog.dtype}", flush=True)

    # Encode queries
    print(f"\nencoding {len(queries):,} queries via API...", flush=True)
    t0 = time.time()
    qv, tokens = encode_queries(queries, args.model, args.dim)
    cost = tokens * PRICES_PER_M_TOKENS.get(args.model, 0) / 1e6
    print(f"  done in {time.time() - t0:.1f}s  tokens={tokens:,}  cost=${cost:.4f}", flush=True)
    record_spend(
        "openai",
        args.model,
        tokens,
        cost,
        f"query encode {data.name} (n={len(queries)})",
    )

    # Dense retrieval — batch queries so we don't allocate a (Nq, N_cat) matrix.
    # For 22k queries x 1.2M catalog at fp32 that'd be ~108 GB; instead score
    # 1024 queries at a time (~5 GB per chunk).
    print("\ntop-K retrieval...", flush=True)
    t0 = time.time()
    cat = np.asarray(catalog).astype(np.float32)
    n_q = qv.shape[0]
    top_pos = np.zeros((n_q, args.k), dtype=np.int64)
    chunk = 1024
    for start in range(0, n_q, chunk):
        end = min(start + chunk, n_q)
        sims_chunk = qv[start:end] @ cat.T  # (chunk, N_cat) fp32
        # argpartition gives unordered top-k, then sort within
        tk = np.argpartition(-sims_chunk, args.k - 1, axis=1)[:, : args.k]
        for i in range(end - start):
            tk[i] = tk[i][np.argsort(-sims_chunk[i, tk[i]])]
        top_pos[start:end] = tk
        if (start // chunk) % 5 == 0 or end == n_q:
            print(
                f"  retrieved {end:,}/{n_q:,}  ({(end / max(time.time() - t0, 1e-3)):.0f} q/s)",
                flush=True,
            )
    print(f"  done in {time.time() - t0:.1f}s", flush=True)

    # Eval
    rs, ns, e1s, e3s = [], [], [], []
    for i, qid in enumerate(eval_qids):
        ordering = [pids_arr[int(p)] for p in top_pos[i]]
        m = per_query_metrics(
            ordering,
            qrels[qid],
            k=args.k,
            min_rel=args.min_relevance,
            exact_rel=args.exact_relevance,
        )
        if m is None:
            continue
        r, nd, e1, e3 = m
        rs.append(r)
        ns.append(nd)
        if not math.isnan(e1):
            e1s.append(e1)
            e3s.append(e3)

    print(
        f"\n{args.model} (dim={args.dim}) R@{args.k} {np.mean(rs):.4f}  "
        f"nDCG@{args.k} {np.mean(ns):.4f}  "
        f"E@1 {np.mean(e1s):.4f}  E@3 {np.mean(e3s):.4f}  (n={len(rs):,})",
        flush=True,
    )


if __name__ == "__main__":
    main()
