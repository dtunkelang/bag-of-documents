#!/usr/bin/env python3
"""ESCI E-only top-K evaluation — does BoD-as-retriever win on narrow-intent
queries when we stop pooling Substitute matches into "relevant"?

For each model, encode all test queries and retrieve top-K from the model's
native product space (the existing MiniLM FAISS index for base/cosine-distilled
models; the cached fp16 vecs for MNRL-trained models). Then for each query:
  - E@1 = 1 if the top-1 product is labeled Exact in qrels, else 0.
  - E@3 = (# Exact in top-3) / min(3, # Exact judgments for this query).

Aggregates: overall E@1 / E@3 across all queries with at least one E judgment,
plus a per-bin breakdown by within-E cohesion (intra-bag pairwise cosine of
the Exact products under base MiniLM, computed on the fly). The bin labels
identify queries where the cluster hypothesis holds tightly vs loosely.

Usage:
    python eval_e_at_k.py
    python eval_e_at_k.py --limit 5000 --models base,query_model_us_full_6m_mnrl
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import json
import os
import statistics
import subprocess
from collections import defaultdict

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np  # noqa: E402

SCRIPT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INDEX_DIR = os.path.join(SCRIPT_DIR, "combined_index_us_minilm")


def encode_subproc(model_path, queries, device="auto"):
    """Subprocess-isolated encoding (avoids FAISS+SentenceTransformer init segfault)."""
    code = f"""
import os, json, sys
os.environ['TOKENIZERS_PARALLELISM']='false'
os.environ['OMP_NUM_THREADS']='1'
os.chdir({SCRIPT_DIR!r})
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
device = {device!r}
if device == 'auto':
    device = 'mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu')
m = SentenceTransformer({model_path!r}, device=device)
qs = json.loads(sys.stdin.read())
v = m.encode(qs, normalize_embeddings=True, batch_size=128, show_progress_bar=False)
sys.stdout.write(json.dumps(np.asarray(v, dtype=np.float32).tolist()))
"""
    out = subprocess.check_output(
        [".venv/bin/python", "-c", code],
        input=json.dumps(queries),
        stderr=subprocess.DEVNULL,
        text=True,
    )
    return np.array(json.loads(out), dtype=np.float32)


def faiss_top_k(q_vecs, faiss_path, k):
    import faiss

    idx = faiss.read_index(faiss_path)
    if hasattr(idx, "hnsw"):
        idx.hnsw.efSearch = 128
    elif hasattr(idx, "nprobe"):
        idx.nprobe = 32
    _, I = idx.search(q_vecs.astype(np.float32), k)
    return I


def cached_top_k(q_vecs, vecs_path, k):
    """Brute-force top-K via numpy matmul against a cached fp16 product matrix."""
    pv = np.load(vecs_path).astype(np.float32)  # 1.2M × 384
    # Process in batches to bound memory.
    n = q_vecs.shape[0]
    out = np.zeros((n, k), dtype=np.int64)
    BATCH = 256
    for start in range(0, n, BATCH):
        end = min(start + BATCH, n)
        sims = q_vecs[start:end] @ pv.T  # (batch, 1.2M)
        out[start:end] = np.argpartition(-sims, k, axis=1)[:, :k]
        # argpartition isn't fully sorted; sort the top-K by sim
        for i in range(end - start):
            row = out[start + i]
            row_sims = sims[i, row]
            order = np.argsort(-row_sims)
            out[start + i] = row[order]
    return out


def load_qrels(path):
    by_query = defaultdict(dict)
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            by_query[r["query_id"]][r["product_id"]] = r["relevance"]
    return dict(by_query)


def load_queries(path):
    out = {}
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            out[d["query_id"]] = d["query"]
    return out


# ESCI-style: 3 = Exact, 2 = Substitute, 1 = Complement, 0 = Irrelevant
EXACT = 3


def metrics_for(retrieved_pids, qrels_q):
    e_pids = {p for p, g in qrels_q.items() if g == EXACT}
    if not e_pids:
        return None
    top1_hit = 1.0 if retrieved_pids and retrieved_pids[0] in e_pids else 0.0
    top3 = retrieved_pids[:3]
    e_in_top3 = sum(1 for p in top3 if p in e_pids)
    e_at_3 = e_in_top3 / min(3, len(e_pids))
    return {"e_at_1": top1_hit, "e_at_3": e_at_3, "n_exact": len(e_pids)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--models",
        default="base,query_model_amazon,query_model_us_full_6m_mnrl,query_model_us_qrels_mnrl_hardneg",
    )
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    models = args.models.split(",")

    print("loading qrels + queries...", flush=True)
    queries_all = load_queries(os.path.join(SCRIPT_DIR, "esci_us_data/test_queries.jsonl"))
    qrels = load_qrels(os.path.join(SCRIPT_DIR, "esci_us_data/test_qrels.jsonl"))

    with open(os.path.join(SCRIPT_DIR, "esci_us_data/product_ids.json")) as f:
        esci_pids = json.load(f)
    with open(os.path.join(SCRIPT_DIR, "esci_us_data/titles.json")) as f:
        esci_titles_arr = json.load(f)
    with open(os.path.join(INDEX_DIR, "titles.json")) as f:
        index_titles = json.load(f)
    title_to_pid = {t: p for p, t in zip(esci_pids, esci_titles_arr)}
    faiss_pos_to_pid = [title_to_pid.get(t) for t in index_titles]

    qids = [
        qid for qid in queries_all if qid in qrels and any(g == EXACT for g in qrels[qid].values())
    ]
    if args.limit:
        import random

        rng = random.Random(args.seed)
        rng.shuffle(qids)
        qids = qids[: args.limit]
    print(f"  {len(qids):,} eval queries (have ≥1 Exact)", flush=True)
    queries = [queries_all[qid] for qid in qids]

    K = 3

    # Build retrieval setup per model: source = (kind, path)
    # kind: "faiss" → search index.faiss directly; "cached" → brute-force cached fp16 vecs.
    sources = {
        "base": ("faiss", os.path.join(INDEX_DIR, "index.faiss"), "all-MiniLM-L6-v2"),
        "query_model_amazon": (
            "faiss",
            os.path.join(INDEX_DIR, "index.faiss"),
            os.path.join(SCRIPT_DIR, "query_model_amazon"),
        ),
        "query_model_us_full_6m_mnrl": (
            "cached",
            os.path.join(INDEX_DIR, "rerank_A.vecs.fp16.npy"),
            os.path.join(SCRIPT_DIR, "query_model_us_full_6m_mnrl"),
        ),
        "query_model_us_qrels_mnrl_hardneg": (
            "cached",
            os.path.join(INDEX_DIR, "rerank_B.vecs.fp16.npy"),
            os.path.join(SCRIPT_DIR, "query_model_us_qrels_mnrl_hardneg"),
        ),
    }

    results = {}
    for name in models:
        if name not in sources:
            print(f"SKIP {name} (no source defined)")
            continue
        kind, src, model_path = sources[name]
        print(f"\n=== {name} ===", flush=True)
        print(f"  encode with {model_path}", flush=True)
        q_vecs = encode_subproc(model_path, queries)
        print(f"  retrieve top-{K} via {kind}: {src}", flush=True)
        if kind == "faiss":
            I = faiss_top_k(q_vecs, src, K)
            # FAISS positions → pids (cached_vecs path for INDEX_DIR is parallel to titles.json)
            retrieved_pids_per_q = [[faiss_pos_to_pid[int(p)] for p in row if p >= 0] for row in I]
        else:
            I = cached_top_k(q_vecs, src, K)
            # Cached vec rows are parallel to combined_index_us_minilm/titles.json
            retrieved_pids_per_q = [[faiss_pos_to_pid[int(p)] for p in row if p >= 0] for row in I]

        e1, e3, n_with_exact = [], [], 0
        for qi, qid in enumerate(qids):
            m = metrics_for(retrieved_pids_per_q[qi], qrels[qid])
            if m:
                e1.append(m["e_at_1"])
                e3.append(m["e_at_3"])
                n_with_exact += 1
        results[name] = {
            "n": n_with_exact,
            "E@1": statistics.mean(e1) if e1 else 0,
            "E@3": statistics.mean(e3) if e3 else 0,
        }
        print(
            f"  E@1={results[name]['E@1']:.4f}  E@3={results[name]['E@3']:.4f}  (n={n_with_exact})"
        )

    print(f"\n\n{'=' * 60}")
    print("=== Summary ===")
    print(f"{'=' * 60}")
    print(f"{'model':<40} {'n':>6} {'E@1':>8} {'E@3':>8}")
    print("-" * 65)
    for name in models:
        if name not in results:
            continue
        r = results[name]
        print(f"{name:<40} {r['n']:>6} {r['E@1']:>8.2%} {r['E@3']:>8.2%}")


if __name__ == "__main__":
    main()
