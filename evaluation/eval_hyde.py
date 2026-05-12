#!/usr/bin/env python3
"""HyDE evaluation — local-LLM version, comparable to diagnose_lift.py.

For each test query:
  1. Call a local LLM (via Ollama HTTP API) to generate a hypothetical
     passage that would answer the query.
  2. Encode that hypothetical passage with the base bi-encoder.
  3. Retrieve top-k using the hypothetical-passage embedding.

Produces a per-query R@10 breakdown bucketed by base R@10 (matching
the layout of `diagnose_lift.py`), so HyDE results can be diffed
against BoD results from the same corpus.

Hypothetical passages are cached to <data_dir>/hyde_passages_<llm>.jsonl
so a failed run resumes without re-generating.

Usage:
    python evaluation/eval_hyde.py \\
        --catalog scifact_data/titles.json \\
        --product-ids scifact_data/doc_ids.json \\
        --qrels scifact_data/test_qrels.jsonl --min-relevance 1 \\
        --queries scifact_data/test_queries.jsonl \\
        --base-model all-MiniLM-L6-v2 \\
        --base-vecs scifact_data/base_catalog.vecs.fp16.npy \\
        --llm-model llama3.1:8b-instruct-q4_K_M \\
        --label scifact-hyde
"""

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from sentence_transformers import SentenceTransformer  # noqa: E402

try:
    import requests  # noqa: E402
except ImportError:
    print("ERROR: `requests` not installed. `pip install requests`.", file=sys.stderr)
    sys.exit(1)

OLLAMA_DEFAULT_URL = "http://localhost:11434/api/generate"
HYDE_PROMPT = (
    "Write a short factual passage that would answer the following query. "
    "Be concrete and specific. Provide an answer; do not repeat the query.\n\n"
    "Query: {query}\nPassage:"
)


def generate_one(session, url, model, query, max_tokens=200, temperature=0.0, seed=None):
    """Generate a single hypothetical passage for `query` via Ollama."""
    options = {
        "temperature": temperature,
        "num_predict": max_tokens,
    }
    if seed is not None:
        options["seed"] = seed
    payload = {
        "model": model,
        "prompt": HYDE_PROMPT.format(query=query),
        "stream": False,
        "options": options,
    }
    r = session.post(url, json=payload, timeout=120)
    r.raise_for_status()
    return r.json().get("response", "").strip()


def load_or_generate_passages(args, qids, queries):
    """Generate (or load cached) hypothetical passages. Returns a list of
    lists of passages parallel to `qids` (one list per query, length
    args.n_samples). Writes incremental cache so a crash resumes."""
    cache_safe = args.llm_model.replace(":", "_").replace("/", "_")
    suffix = "" if args.n_samples == 1 else f"_n{args.n_samples}"
    cache_path = os.path.join(
        os.path.dirname(args.queries) or ".",
        f"hyde_passages_{cache_safe}{suffix}.jsonl",
    )
    # For n=1, keep the existing flat format ({query_id, query, passage}).
    # For n>1, use ({query_id, query, passage, sample_idx}) with multiple rows
    # per qid.
    cached_by_qid = {}  # qid -> list of passages (indexed by sample_idx)
    if os.path.exists(cache_path):
        with open(cache_path) as f:
            for line in f:
                d = json.loads(line)
                qid = d["query_id"]
                idx = d.get("sample_idx", 0)
                cached_by_qid.setdefault(qid, [None] * args.n_samples)
                if idx < args.n_samples:
                    cached_by_qid[qid][idx] = d["passage"]
        n_complete = sum(1 for v in cached_by_qid.values() if v and all(p is not None for p in v))
        print(
            f"loaded {n_complete:,} fully-cached qids ({sum(1 for v in cached_by_qid.values() for p in v if p is not None):,} total samples) from {cache_path}",
            flush=True,
        )

    session = requests.Session()
    passages_per_query = []
    t0 = time.time()
    n_done = 0
    with open(cache_path, "a") as cache_handle:
        for qid, q in zip(qids, queries):
            samples = cached_by_qid.get(qid) or [None] * args.n_samples
            for sample_idx in range(args.n_samples):
                if samples[sample_idx] is not None:
                    continue
                # Use a per-sample seed for reproducibility; T>0 for diversity.
                seed = (hash(qid) % 100000) + sample_idx
                try:
                    p = generate_one(
                        session,
                        args.ollama_url,
                        args.llm_model,
                        q,
                        temperature=args.temperature,
                        seed=seed,
                    )
                except Exception as e:
                    print(f"  [{qid}/s{sample_idx}] LLM error: {e} — empty", flush=True)
                    p = ""
                samples[sample_idx] = p
                row = {"query_id": qid, "query": q, "passage": p}
                if args.n_samples > 1:
                    row["sample_idx"] = sample_idx
                cache_handle.write(json.dumps(row) + "\n")
                cache_handle.flush()
            passages_per_query.append(samples)
            n_done += 1
            if n_done % 25 == 0 or n_done == len(qids):
                elapsed = time.time() - t0
                avg_chars = int(np.mean([len(p) for ps in passages_per_query for p in ps if p]))
                rate = n_done / elapsed if elapsed > 0 else 0.0
                print(
                    f"  generated {n_done:,}/{len(qids):,} qids "
                    f"({n_done * args.n_samples:,} total samples; "
                    f"{rate:.2f} qids/s; avg passage chars={avg_chars})",
                    flush=True,
                )
    return passages_per_query, cache_path


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--catalog", required=True, help="titles.json")
    ap.add_argument("--product-ids", required=True)
    ap.add_argument("--qrels", required=True)
    ap.add_argument("--queries", required=True)
    ap.add_argument("--min-relevance", type=int, default=1)
    ap.add_argument("--base-model", required=True)
    ap.add_argument("--base-vecs", default=None, help="cached base catalog .npy")
    ap.add_argument("--llm-model", default="llama3.1:8b-instruct-q4_K_M")
    ap.add_argument("--ollama-url", default=OLLAMA_DEFAULT_URL)
    ap.add_argument("--label", default="corpus")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument(
        "--n-samples",
        type=int,
        default=1,
        help=(
            "number of hypothetical passages to generate per query; their "
            "embeddings are averaged + renormalized for retrieval. Original "
            "HyDE paper used N=8. For N>1, set --temperature > 0 for diversity."
        ),
    )
    ap.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="LLM sampling temperature; 0.0 deterministic, ~0.7-1.0 for multi-sample diversity",
    )
    args = ap.parse_args()

    print("loading data...", flush=True)
    with open(args.catalog) as f:
        titles = json.load(f)
    with open(args.product_ids) as f:
        pids = json.load(f)
    pid_to_idx = {p: i for i, p in enumerate(pids)}

    queries_by_qid = {}
    with open(args.queries) as f:
        for line in f:
            d = json.loads(line)
            queries_by_qid[d["query_id"]] = d["query"]

    pos = defaultdict(set)
    field = None
    with open(args.qrels) as f:
        for line in f:
            r = json.loads(line)
            if field is None:
                field = "product_id" if "product_id" in r else "doc_id"
            if r[field] not in pid_to_idx:
                continue
            if r["relevance"] < args.min_relevance:
                continue
            pos[r["query_id"]].add(pid_to_idx[r[field]])

    qids = sorted(queries_by_qid)
    queries = [queries_by_qid[q] for q in qids]
    n_pos_q = sum(1 for v in pos.values() if v)
    print(
        f"  catalog={len(pids):,}  queries={len(queries):,}  pos-bearing={n_pos_q:,}",
        flush=True,
    )

    device = "mps" if torch.backends.mps.is_available() else "cpu"

    if args.base_vecs and os.path.exists(args.base_vecs):
        print(f"loading cached base vecs {args.base_vecs}...", flush=True)
        base_pv = np.load(args.base_vecs).astype(np.float32)
    else:
        print(f"encoding catalog with {args.base_model} on {device}...", flush=True)
        m = SentenceTransformer(args.base_model, device=device)
        base_pv = m.encode(
            titles, normalize_embeddings=True, batch_size=128, show_progress_bar=True
        ).astype(np.float32)
        del m
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

    print(
        f"\ngenerating HyDE passages via {args.llm_model} "
        f"(n_samples={args.n_samples}, T={args.temperature}) ({args.ollama_url})...",
        flush=True,
    )
    passages_per_query, cache_path = load_or_generate_passages(args, qids, queries)
    print(f"  cached at {cache_path}", flush=True)

    print("\nencoding base queries (raw text)...", flush=True)
    encoder = SentenceTransformer(args.base_model, device=device)
    base_qv = encoder.encode(
        queries, normalize_embeddings=True, batch_size=256, show_progress_bar=True
    ).astype(np.float32)
    # Encode all N passages per query in one flat batch; then average per-query
    # and renormalize. Matches the original HyDE multi-sample procedure.
    flat_passages = [p for ps in passages_per_query for p in ps]
    print(
        f"encoding {len(flat_passages):,} HyDE passages ({args.n_samples} per query)...",
        flush=True,
    )
    flat_qv = encoder.encode(
        flat_passages, normalize_embeddings=True, batch_size=128, show_progress_bar=True
    ).astype(np.float32)
    # Reshape (n_queries * n_samples, dim) -> (n_queries, n_samples, dim) -> mean
    flat_qv = flat_qv.reshape(len(qids), args.n_samples, -1)
    hyde_qv = flat_qv.mean(axis=1)
    # Renormalize after averaging
    norms = np.linalg.norm(hyde_qv, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    hyde_qv = hyde_qv / norms
    del encoder, flat_qv

    print("\nretrieving + scoring (base and HyDE side-by-side)...", flush=True)
    base_top = np.argpartition(-(base_qv @ base_pv.T), args.k, axis=1)[:, : args.k]
    hyde_top = np.argpartition(-(hyde_qv @ base_pv.T), args.k, axis=1)[:, : args.k]

    per_q = []
    for j, qid in enumerate(qids):
        g = pos.get(qid, set())
        if not g:
            continue
        b_hit = len({int(x) for x in base_top[j]} & g)
        h_hit = len({int(x) for x in hyde_top[j]} & g)
        per_q.append((qid, len(g), b_hit, h_hit))

    n = len(per_q)
    base_r = np.mean([h / g for _, g, h, _ in per_q])
    hyde_r = np.mean([h / g for _, g, _, h in per_q])

    # Bucket by base R@10 to mirror diagnose_lift output.
    print(f"\nHyDE vs base — {args.label}  (R@10 on relevance>={args.min_relevance})")
    print("  bucket                              n     base     HyDE        Δ")
    buckets = [
        ("0.0 (base misses entirely)", lambda b, g: b == 0),
        ("0.0-0.5", lambda b, g: 0 < b / g <= 0.5),
        ("0.5-1.0", lambda b, g: 0.5 < b / g < 1.0),
        ("1.0 (base perfect)", lambda b, g: b == g),
    ]
    for label, pred in buckets:
        rows = [(g, b, h) for _, g, b, h in per_q if pred(b, g)]
        if not rows:
            continue
        bn = len(rows)
        br = np.mean([b / g for g, b, _ in rows])
        hr = np.mean([h / g for g, _, h in rows])
        pct = bn / n * 100
        print(f"  {label:<30}  {bn:>5} ({pct:>4.1f}%)  {br:>7.3f}  {hr:>7.3f}  {hr - br:+7.3f}")
    print(f"  overall (n={n:,}): base={base_r:.3f}  HyDE={hyde_r:.3f}  Δ={hyde_r - base_r:+.3f}")

    # Also write a per-query JSONL for the overlap analysis with BoD.
    out_path = os.path.join(
        os.path.dirname(args.queries) or ".",
        f"hyde_per_query_{args.label}.jsonl",
    )
    with open(out_path, "w") as f:
        for qid, g, b, h in per_q:
            f.write(json.dumps({"query_id": qid, "n_gold": g, "base_hit": b, "hyde_hit": h}) + "\n")
    print(f"\nper-query results at {out_path}", flush=True)


if __name__ == "__main__":
    main()
