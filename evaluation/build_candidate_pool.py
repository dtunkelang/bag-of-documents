#!/usr/bin/env python3
"""Build a candidate pool from multiple retrievers for LLM-relevance-judging.

For each test query, retrieves top-K candidates from one or more retrievers,
unions them per query, and writes a JSONL input file ready for
evaluation/llm_relevance_judge.py.

Each retriever is specified as a 4-tuple via --retriever:
    retrievers_spec ::= MODEL_KIND:VEC_PATH:MODEL_ID[:DIM]

Where MODEL_KIND is one of:
    - "st"     sentence-transformers model id (encodes queries on MPS)
    - "openai" OpenAI embedding model name (encodes queries via API; --dim used)

Examples:
  # Pool from MiniLM-base + bge-large + te3-large for NFCorpus
  .venv/bin/python evaluation/build_candidate_pool.py \\
      --data-dir nfcorpus_data --k 50 \\
      --retriever st:nfcorpus_data/base_catalog.vecs.fp16.npy:sentence-transformers/all-MiniLM-L6-v2 \\
      --retriever st:nfcorpus_data/bge_large_catalog.vecs.fp16.npy:BAAI/bge-large-en-v1.5 \\
      --retriever openai:nfcorpus_data/openai_te3large_1024.vecs.fp16.npy:text-embedding-3-large:1024 \\
      --output /tmp/pilot_b/nfcorpus_candidates.jsonl

Output rows: {"qid": ..., "query": ..., "did": ..., "doc_text": ...,
              "retrievers": [list of which retrievers found it in top-K]}
"""

import argparse
import datetime
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
from dotenv import load_dotenv  # noqa: E402

load_dotenv(override=True)


SPEND_LEDGER = Path(__file__).resolve().parent.parent / ".api_spend.jsonl"

OPENAI_PRICES_PER_M_TOKENS = {
    "text-embedding-3-large": 0.13,
    "text-embedding-3-small": 0.02,
}


def record_spend(model: str, tokens: int, cost_usd: float, purpose: str):
    rec = {
        "ts": datetime.datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "provider": "openai",
        "model": model,
        "tokens": int(tokens),
        "cost_usd": round(float(cost_usd), 4),
        "purpose": purpose,
    }
    with open(SPEND_LEDGER, "a") as f:
        f.write(json.dumps(rec) + "\n")


def encode_queries_st(queries: list[str], model_id: str) -> np.ndarray:
    from sentence_transformers import SentenceTransformer

    print(f"  loading ST model {model_id}...", flush=True)
    t0 = time.time()
    model = SentenceTransformer(model_id, device="mps")
    print(f"  loaded in {time.time() - t0:.1f}s", flush=True)
    t0 = time.time()
    vecs = model.encode(
        queries,
        normalize_embeddings=True,
        batch_size=64,
        show_progress_bar=False,
    ).astype(np.float32)
    print(f"  encoded {len(queries)} queries in {time.time() - t0:.1f}s", flush=True)
    return vecs


def encode_queries_openai(
    queries: list[str], model: str, dim: int | None, purpose: str
) -> np.ndarray:
    from openai import OpenAI

    client = OpenAI()
    print(f"  encoding {len(queries)} queries via {model} (dim={dim})...", flush=True)
    t0 = time.time()
    out = []
    tokens = 0
    batch = 512
    for s in range(0, len(queries), batch):
        e = min(s + batch, len(queries))
        kwargs = {"model": model, "input": queries[s:e]}
        if dim is not None:
            kwargs["dimensions"] = dim
        resp = client.embeddings.create(**kwargs)
        vecs = np.array([d.embedding for d in resp.data], dtype=np.float32)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        vecs = vecs / np.maximum(norms, 1e-9)
        out.append(vecs)
        tokens += resp.usage.total_tokens if hasattr(resp, "usage") else 0
    qv = np.vstack(out)
    cost = tokens * OPENAI_PRICES_PER_M_TOKENS.get(model, 0.0) / 1e6
    print(
        f"  done in {time.time() - t0:.1f}s; tokens={tokens:,} cost=${cost:.4f}",
        flush=True,
    )
    record_spend(model, tokens, cost, purpose)
    return qv


def retrieve_topk(qv: np.ndarray, catalog: np.ndarray, k: int) -> np.ndarray:
    """Return (n_q, k) array of top-k catalog indices."""
    n_q = qv.shape[0]
    out = np.zeros((n_q, k), dtype=np.int64)
    chunk = 1024
    cat = np.asarray(catalog).astype(np.float32)
    for s in range(0, n_q, chunk):
        e = min(s + chunk, n_q)
        sims = qv[s:e] @ cat.T
        tk = np.argpartition(-sims, k - 1, axis=1)[:, :k]
        for i in range(e - s):
            tk[i] = tk[i][np.argsort(-sims[i, tk[i]])]
        out[s:e] = tk
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--queries-file", default="test_queries.jsonl")
    ap.add_argument("--titles-file", default="titles.json")
    ap.add_argument("--ids-file", default=None, help="default: doc_ids.json else product_ids.json")
    ap.add_argument(
        "--qrels-file",
        default="test_qrels.jsonl",
        help="used only to filter to queries with >= min-relevance positives",
    )
    ap.add_argument("--min-relevance", type=int, default=2, help="match eval-side default")
    ap.add_argument("--k", type=int, default=50)
    ap.add_argument(
        "--retriever",
        action="append",
        required=True,
        help='one or more "KIND:VEC_PATH:MODEL_ID[:DIM]" specs',
    )
    ap.add_argument("--output", required=True)
    ap.add_argument(
        "--max-doc-chars",
        type=int,
        default=1600,
        help="truncate doc_text to this many characters in the output JSONL",
    )
    ap.add_argument(
        "--purpose-tag",
        default="candidate pool",
        help="ledger purpose tag for OpenAI query encoding records",
    )
    args = ap.parse_args()

    data = Path(args.data_dir)

    # Resolve IDs + titles
    if args.ids_file:
        ids_path = data / args.ids_file
    elif (data / "doc_ids.json").exists():
        ids_path = data / "doc_ids.json"
    else:
        ids_path = data / "product_ids.json"
    with open(ids_path) as f:
        pids = json.load(f)
    with open(data / args.titles_file) as f:
        titles = json.load(f)
    print(f"catalog: {len(pids):,} from {ids_path}", flush=True)

    # Queries
    queries_all = {}
    with open(data / args.queries_file) as f:
        for line in f:
            d = json.loads(line)
            queries_all[d["query_id"]] = d["query"]
    qrels = defaultdict(dict)
    with open(data / args.qrels_file) as f:
        for line in f:
            r = json.loads(line)
            qrels[r["query_id"]][r["product_id"]] = r["relevance"]
    eval_qids = sorted(
        qid
        for qid in queries_all
        if qid in qrels and any(g >= args.min_relevance for g in qrels[qid].values())
    )
    queries = [queries_all[qid] for qid in eval_qids]
    print(f"  {len(eval_qids):,} eval queries (min_relevance={args.min_relevance})", flush=True)

    # Build pool
    pool = defaultdict(set)  # qid -> set of did (pids)
    pool_origins = defaultdict(lambda: defaultdict(list))  # qid -> did -> [retriever_names]

    for spec in args.retriever:
        parts = spec.split(":", 3)
        if len(parts) < 3:
            raise SystemExit(f"bad --retriever spec: {spec}")
        kind = parts[0]
        vec_path = parts[1]
        model_id = parts[2]
        dim = int(parts[3]) if len(parts) == 4 else None

        retriever_name = f"{kind}/{Path(vec_path).stem}"
        print(
            f"\nretriever: {retriever_name}  (kind={kind}, model={model_id}, dim={dim})", flush=True
        )

        catalog = np.load(vec_path, mmap_mode="r")
        if catalog.shape[0] != len(pids):
            raise SystemExit(f"{vec_path}: catalog rows ({catalog.shape[0]}) != ids ({len(pids)})")
        print(f"  catalog: {catalog.shape} dtype={catalog.dtype}", flush=True)

        if kind == "st":
            qv = encode_queries_st(queries, model_id)
        elif kind == "openai":
            qv = encode_queries_openai(
                queries, model_id, dim, f"{args.purpose_tag} query encode {data.name}"
            )
        else:
            raise SystemExit(f"unknown retriever kind: {kind}")

        # Match dim
        if qv.shape[1] != catalog.shape[1]:
            raise SystemExit(f"qv dim {qv.shape[1]} != catalog dim {catalog.shape[1]}")

        topk = retrieve_topk(qv, catalog, args.k)
        for i, qid in enumerate(eval_qids):
            for idx in topk[i]:
                did = pids[int(idx)]
                pool[qid].add(did)
                pool_origins[qid][did].append(retriever_name)

    # Write output JSONL
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    n_pairs = 0
    pid_to_idx = {p: i for i, p in enumerate(pids)}
    with open(args.output, "w") as fout:
        for qid in eval_qids:
            q = queries_all[qid]
            for did in sorted(pool[qid]):
                doc_text = titles[pid_to_idx[did]][: args.max_doc_chars]
                row = {
                    "qid": qid,
                    "query": q,
                    "did": did,
                    "doc_text": doc_text,
                    "retrievers": pool_origins[qid][did],
                }
                fout.write(json.dumps(row) + "\n")
                n_pairs += 1

    avg = n_pairs / max(len(eval_qids), 1)
    print(
        f"\nwrote {n_pairs:,} (query, doc) pairs across {len(eval_qids):,} queries  "
        f"(avg {avg:.1f} candidates/query)  to {args.output}",
        flush=True,
    )


if __name__ == "__main__":
    main()
