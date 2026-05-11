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


def generate_one(session, url, model, query, max_tokens=200, temperature=0.0):
    """Generate a single hypothetical passage for `query` via Ollama."""
    payload = {
        "model": model,
        "prompt": HYDE_PROMPT.format(query=query),
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens,
        },
    }
    r = session.post(url, json=payload, timeout=120)
    r.raise_for_status()
    return r.json().get("response", "").strip()


def load_or_generate_passages(args, qids, queries):
    """Generate (or load cached) hypothetical passages. Returns list of strings
    parallel to `qids`. Writes incremental cache so a crash resumes."""
    cache_safe = args.llm_model.replace(":", "_").replace("/", "_")
    cache_path = os.path.join(
        os.path.dirname(args.queries) or ".",
        f"hyde_passages_{cache_safe}.jsonl",
    )
    cached = {}
    if os.path.exists(cache_path):
        with open(cache_path) as f:
            for line in f:
                d = json.loads(line)
                cached[d["query_id"]] = d["passage"]
        print(f"loaded {len(cached):,} cached HyDE passages from {cache_path}", flush=True)

    session = requests.Session()
    passages = []
    t0 = time.time()
    with open(cache_path, "a") as cache_handle:
        for i, (qid, q) in enumerate(zip(qids, queries)):
            if qid in cached:
                passages.append(cached[qid])
                continue
            try:
                p = generate_one(session, args.ollama_url, args.llm_model, q)
            except Exception as e:
                print(f"  [{qid}] LLM error: {e} — empty passage", flush=True)
                p = ""
            passages.append(p)
            cache_handle.write(json.dumps({"query_id": qid, "query": q, "passage": p}) + "\n")
            cache_handle.flush()
            cached[qid] = p
            if (i + 1) % 25 == 0 or i + 1 == len(qids):
                elapsed = time.time() - t0
                rate = (i + 1) / elapsed if elapsed > 0 else 0.0
                print(
                    f"  generated {i + 1:,}/{len(qids):,} ({rate:.1f} qps; "
                    f"avg passage chars={int(np.mean([len(p) for p in passages]))})",
                    flush=True,
                )
    return passages, cache_path


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

    print(f"\ngenerating HyDE passages via {args.llm_model} ({args.ollama_url})...", flush=True)
    passages, cache_path = load_or_generate_passages(args, qids, queries)
    print(f"  cached at {cache_path}", flush=True)

    print("\nencoding base queries (raw text)...", flush=True)
    encoder = SentenceTransformer(args.base_model, device=device)
    base_qv = encoder.encode(
        queries, normalize_embeddings=True, batch_size=256, show_progress_bar=True
    ).astype(np.float32)
    print("encoding HyDE queries (hypothetical passages)...", flush=True)
    hyde_qv = encoder.encode(
        passages, normalize_embeddings=True, batch_size=128, show_progress_bar=True
    ).astype(np.float32)
    del encoder

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
