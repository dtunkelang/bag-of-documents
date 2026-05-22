#!/usr/bin/env python3
"""Re-evaluate multiple retrievers on a corpus under multiple qrels variants.

Pilot B analysis runner: compares retrievers under {original qrels} vs
{LLM-strict qrels} vs {LLM-liberal qrels} to test whether BoD's
qrels-conditioned result flips when the ground truth is cleaned/densified.

Each retriever is specified the same way as evaluation/build_candidate_pool.py:
    KIND:VEC_PATH:MODEL_ID[:DIM]
Plus an optional friendly --name.

Usage (example):
  .venv/bin/python evaluation/eval_pilot_b_retrievers.py \\
    --data-dir nfcorpus_data \\
    --retriever name=base:st:nfcorpus_data/base_catalog.vecs.fp16.npy:sentence-transformers/all-MiniLM-L6-v2 \\
    --retriever name=bge-large:st:nfcorpus_data/bge_large_catalog.vecs.fp16.npy:BAAI/bge-large-en-v1.5 \\
    --retriever name=te3-large:openai:nfcorpus_data/openai_te3large_1024.vecs.fp16.npy:text-embedding-3-large:1024 \\
    --retriever name=bod:st:nfcorpus_data/base_catalog.vecs.fp16.npy:query_model_nfcorpus_bod \\
    --qrels-variant name=original:nfcorpus_data/test_qrels.jsonl:1:2 \\
    --qrels-variant name=strict-LLM:nfcorpus_data/test_qrels_llm_strict.jsonl:2:2 \\
    --qrels-variant name=liberal-LLM:nfcorpus_data/test_qrels_llm_liberal.jsonl:1:2 \\
    --output /tmp/pilot_b/nfcorpus_results.json --k 10
"""

import argparse
import datetime
import json
import math
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


def record_spend(model, tokens, cost_usd, purpose):
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


def encode_queries_st(queries, model_id):
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_id, device="mps")
    return model.encode(
        queries,
        normalize_embeddings=True,
        batch_size=64,
        show_progress_bar=False,
    ).astype(np.float32)


def encode_queries_openai(queries, model, dim, purpose):
    from openai import OpenAI

    client = OpenAI()
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
    record_spend(model, tokens, cost, purpose)
    return qv


def retrieve_topk(qv, catalog, k):
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


def per_query_metrics(retrieved_pids, qrels_q, k, min_rel, exact_rel):
    pos = {p for p, g in qrels_q.items() if g >= min_rel}
    if not pos:
        return None
    pos_e = {p for p, g in qrels_q.items() if g >= exact_rel}
    top_k = retrieved_pids[:k]
    n_hits = sum(1 for p in top_k if p in pos)
    recall = n_hits / min(len(pos), k)
    # DCG@k
    dcg = sum((qrels_q.get(p, 0) >= min_rel) / math.log2(idx + 2) for idx, p in enumerate(top_k))
    idcg = sum(1.0 / math.log2(i + 2) for i in range(min(len(pos), k)))
    ndcg = dcg / idcg if idcg > 0 else 0.0
    if pos_e:
        e1 = 1.0 if top_k[0] in pos_e else 0.0
        e3 = sum(1 for p in top_k[:3] if p in pos_e) / min(3, len(pos_e))
    else:
        e1 = float("nan")
        e3 = float("nan")
    return recall, ndcg, e1, e3


def parse_retriever(spec):
    """Parse `name=NAME:KIND:VEC_PATH:MODEL_ID[:DIM]` (or omit name=)."""
    name = None
    if spec.startswith("name="):
        head, _, rest = spec[len("name=") :].partition(":")
        name = head
        spec = rest
    parts = spec.split(":", 3)
    if len(parts) < 3:
        raise SystemExit(f"bad retriever spec: {spec}")
    kind = parts[0]
    vec_path = parts[1]
    model_id = parts[2]
    dim = int(parts[3]) if len(parts) == 4 else None
    if name is None:
        name = f"{kind}/{Path(vec_path).stem}"
    return {"name": name, "kind": kind, "vec_path": vec_path, "model_id": model_id, "dim": dim}


def parse_qrels_variant(spec):
    """Parse `name=NAME:PATH:MIN_REL:EXACT_REL`."""
    name = None
    if spec.startswith("name="):
        head, _, rest = spec[len("name=") :].partition(":")
        name = head
        spec = rest
    parts = spec.split(":")
    if len(parts) < 3:
        raise SystemExit(f"bad qrels-variant spec: {spec}")
    path = parts[0]
    min_rel = int(parts[1])
    exact_rel = int(parts[2])
    if name is None:
        name = Path(path).stem
    return {"name": name, "path": path, "min_rel": min_rel, "exact_rel": exact_rel}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--queries-file", default="test_queries.jsonl")
    ap.add_argument("--titles-file", default="titles.json")
    ap.add_argument("--ids-file", default=None)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--retriever", action="append", required=True, help="see module docstring")
    ap.add_argument("--qrels-variant", action="append", required=True, help="see module docstring")
    ap.add_argument("--output", required=True, help="JSON results path")
    args = ap.parse_args()

    data = Path(args.data_dir)
    # Resolve IDs
    if args.ids_file:
        ids_path = data / args.ids_file
    elif (data / "doc_ids.json").exists():
        ids_path = data / "doc_ids.json"
    else:
        ids_path = data / "product_ids.json"
    with open(ids_path) as f:
        pids = json.load(f)

    # Queries (all of them, indexed by qid)
    queries_all = {}
    with open(data / args.queries_file) as f:
        for line in f:
            d = json.loads(line)
            queries_all[d["query_id"]] = d["query"]

    # Parse qrels variants
    variants = [parse_qrels_variant(s) for s in args.qrels_variant]
    variant_qrels = {}
    for v in variants:
        q = defaultdict(dict)
        path = v["path"] if Path(v["path"]).is_absolute() else (data / v["path"])
        if not Path(path).exists():
            # also try data-dir relative
            path = data / Path(v["path"]).name
        with open(path) as f:
            for line in f:
                r = json.loads(line)
                q[r["query_id"]][r["product_id"]] = r["relevance"]
        v["resolved_path"] = str(path)
        variant_qrels[v["name"]] = q
        print(
            f"variant {v['name']:>18s}  {sum(len(x) for x in q.values()):,} rows, "
            f"{len(q):,} queries  ({path})",
            flush=True,
        )

    # Determine eval qid set per variant (qids with >= min_rel positive)
    variant_eval_qids = {}
    for v in variants:
        qrels = variant_qrels[v["name"]]
        eligible = sorted(
            qid
            for qid in queries_all
            if qid in qrels and any(g >= v["min_rel"] for g in qrels[qid].values())
        )
        variant_eval_qids[v["name"]] = eligible
        print(
            f"  variant {v['name']:>18s} eval queries (min_rel={v['min_rel']}): {len(eligible):,}",
            flush=True,
        )

    # Use union of all variant qid sets to run retrieval (run each retriever once)
    all_eval_qids = sorted(set().union(*variant_eval_qids.values()))
    queries = [queries_all[qid] for qid in all_eval_qids]
    print(f"\nrunning retrieval for {len(all_eval_qids):,} unique eval queries\n", flush=True)

    retrievers = [parse_retriever(s) for s in args.retriever]
    retriever_topk = {}  # name -> array of (n_q, k_max) pids
    K_MAX = args.k

    for r in retrievers:
        print(
            f"retriever {r['name']:>18s}  kind={r['kind']} model={r['model_id']} dim={r['dim']}",
            flush=True,
        )
        catalog = np.load(r["vec_path"], mmap_mode="r")
        if catalog.shape[0] != len(pids):
            raise SystemExit(
                f"{r['vec_path']}: catalog rows ({catalog.shape[0]}) != ids ({len(pids)})"
            )
        t0 = time.time()
        if r["kind"] == "st":
            qv = encode_queries_st(queries, r["model_id"])
        elif r["kind"] == "openai":
            qv = encode_queries_openai(
                queries, r["model_id"], r["dim"], f"pilot B re-eval {r['name']} {data.name}"
            )
        else:
            raise SystemExit(f"unknown retriever kind: {r['kind']}")
        if qv.shape[1] != catalog.shape[1]:
            raise SystemExit(f"qv dim {qv.shape[1]} != catalog dim {catalog.shape[1]}")
        top_pos = retrieve_topk(qv, catalog, K_MAX)
        retriever_topk[r["name"]] = top_pos
        print(
            f"  encoded+retrieved in {time.time() - t0:.1f}s  ({len(queries) / max(time.time() - t0, 1e-3):.0f} q/s)",
            flush=True,
        )

    # Per-variant scoring
    results = {"data_dir": str(data), "k": K_MAX, "rows": []}
    print("\n=== Results ===\n", flush=True)
    header = f"{'retriever':>18s}  {'variant':>18s}  {'n':>5s}  {'R@K':>7s}  {'nDCG':>7s}  {'E@1':>7s}  {'E@3':>7s}"
    print(header, flush=True)
    print("-" * len(header), flush=True)

    qid_to_idx = {qid: i for i, qid in enumerate(all_eval_qids)}
    for r in retrievers:
        topk_arr = retriever_topk[r["name"]]
        for v in variants:
            eval_qids = variant_eval_qids[v["name"]]
            qrels = variant_qrels[v["name"]]
            rs, ns, e1s, e3s = [], [], [], []
            for qid in eval_qids:
                idx = qid_to_idx[qid]
                ordering = [pids[int(p)] for p in topk_arr[idx]]
                m = per_query_metrics(ordering, qrels[qid], K_MAX, v["min_rel"], v["exact_rel"])
                if m is None:
                    continue
                rec, nd, e1, e3 = m
                rs.append(rec)
                ns.append(nd)
                if not math.isnan(e1):
                    e1s.append(e1)
                    e3s.append(e3)
            row = {
                "retriever": r["name"],
                "variant": v["name"],
                "n": len(rs),
                "R_at_K": float(np.mean(rs)) if rs else float("nan"),
                "nDCG": float(np.mean(ns)) if ns else float("nan"),
                "E_at_1": float(np.mean(e1s)) if e1s else float("nan"),
                "E_at_3": float(np.mean(e3s)) if e3s else float("nan"),
            }
            results["rows"].append(row)
            print(
                f"{r['name']:>18s}  {v['name']:>18s}  {row['n']:>5d}  "
                f"{row['R_at_K']:>7.4f}  {row['nDCG']:>7.4f}  "
                f"{row['E_at_1']:>7.4f}  {row['E_at_3']:>7.4f}",
                flush=True,
            )

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nresults json: {args.output}", flush=True)


if __name__ == "__main__":
    main()
