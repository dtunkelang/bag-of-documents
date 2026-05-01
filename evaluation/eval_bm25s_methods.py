#!/usr/bin/env python3
"""bm25s scoring method variants at (k1=0.3, b=0.6).

bm25s supports multiple BM25 variants via `method=`:
  - "robertson" (the textbook BM25, current default in bm25s 0.3.x)
  - "lucene"  (Lucene's BM25 with epsilon protection on negative IDF)
  - "atire"   (ATIRE BM25)
  - "bm25l"   (BM25L, which adds a tunable lift to short-doc TFs)
  - "bm25+"   (BM25+, which adds delta to score the way BM25L does in IDF)

Probe whether any non-default scoring variant beats robertson + (0.3, 0.6).
Cheap; ~5 min total.

Usage:
    python evaluation/eval_bm25s_methods.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json  # noqa: E402
import math  # noqa: E402
import os  # noqa: E402
import time  # noqa: E402
from collections import defaultdict  # noqa: E402

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import bm25s  # noqa: E402
import numpy as np  # noqa: E402
import Stemmer  # noqa: E402

SCRIPT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INDEX_DIR = os.path.join(SCRIPT_DIR, "combined_index_us_minilm")

K_RETRIEVE = 50
K_EVAL = 10

METHODS = ["robertson", "lucene", "atire", "bm25l", "bm25+"]


def encode_subproc(model_path, queries):
    import subprocess

    code = f"""
import os, json, sys
os.environ['TOKENIZERS_PARALLELISM']='false'
os.environ['OMP_NUM_THREADS']='1'
os.chdir({SCRIPT_DIR!r})
import numpy as np, torch
from sentence_transformers import SentenceTransformer
m = SentenceTransformer({model_path!r})
device = 'mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu')
m = m.to(device)
queries = json.loads(sys.stdin.read())
v = m.encode(queries, batch_size=128, show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=True)
np.save('/tmp/_qenc.npy', v.astype(np.float32))
print('OK')
"""
    p = subprocess.run(
        [".venv/bin/python", "-c", code],
        input=json.dumps(queries),
        capture_output=True,
        text=True,
        cwd=SCRIPT_DIR,
        timeout=600,
    )
    if "OK" not in p.stdout:
        raise RuntimeError(f"encode failed: {p.stderr}")
    return np.load("/tmp/_qenc.npy")


def per_query_metrics(retrieved_pids, qrels_q):
    if not retrieved_pids:
        return None
    pos_e = {pid for pid, g in qrels_q.items() if g >= 3}
    pos_es = {pid for pid, g in qrels_q.items() if g >= 2}
    if not pos_es:
        return None
    top_k = retrieved_pids[:K_EVAL]
    recall = sum(1 for p in top_k if p in pos_es) / len(pos_es)
    gains = [1.0 if p in pos_e else (0.1 if p in pos_es else 0.0) for p in top_k]
    dcg = sum(g / math.log2(i + 2) for i, g in enumerate(gains))
    ideal = sorted((1.0 if p in pos_e else 0.1 for p in pos_es), reverse=True)[:K_EVAL]
    idcg = sum(g / math.log2(i + 2) for i, g in enumerate(ideal))
    ndcg = dcg / idcg if idcg > 0 else 0.0
    e1 = (1.0 if any(p in pos_e for p in top_k[:1]) else 0.0) if pos_e else None
    return recall, ndcg, e1


def main():
    qrels = defaultdict(dict)
    with open(os.path.join(SCRIPT_DIR, "esci_us_data/test_qrels.jsonl")) as f:
        for line in f:
            r = json.loads(line)
            qrels[r["query_id"]][r["product_id"]] = r["relevance"]
    queries_all = {}
    with open(os.path.join(SCRIPT_DIR, "esci_us_data/test_queries.jsonl")) as f:
        for line in f:
            d = json.loads(line)
            queries_all[d["query_id"]] = d["query"]
    with open(os.path.join(SCRIPT_DIR, "esci_us_data/product_ids.json")) as f:
        esci_pids = json.load(f)
    with open(os.path.join(SCRIPT_DIR, "esci_us_data/titles.json")) as f:
        esci_titles_arr = json.load(f)
    with open(os.path.join(INDEX_DIR, "titles.json")) as f:
        index_titles = json.load(f)
    title_to_pid = {t: p for p, t in zip(esci_pids, esci_titles_arr)}
    faiss_pos_to_pid = [title_to_pid.get(t) for t in index_titles]

    qids = [qid for qid in queries_all if qid in qrels and any(g >= 2 for g in qrels[qid].values())]
    queries = [queries_all[qid] for qid in qids]
    print(f"  {len(qids):,} eval queries, {len(index_titles):,} titles", flush=True)

    print("encoding queries...", flush=True)
    qv_a = encode_subproc(os.path.join(SCRIPT_DIR, "query_model_us_full_6m_mnrl"), queries)
    qv_b = encode_subproc(os.path.join(SCRIPT_DIR, "query_model_us_qrels_mnrl_hardneg"), queries)
    qv_g = encode_subproc(os.path.join(SCRIPT_DIR, "query_model_us_esci_supervised"), queries)

    print("loading product vecs...", flush=True)
    pv_a = np.load(os.path.join(INDEX_DIR, "rerank_A.vecs.fp16.npy")).astype(np.float32)
    pv_b = np.load(os.path.join(INDEX_DIR, "rerank_B.vecs.fp16.npy")).astype(np.float32)
    pv_g = np.load(os.path.join(INDEX_DIR, "rerank_G.vecs.fp16.npy")).astype(np.float32)

    stemmer = Stemmer.Stemmer("english")
    print("tokenizing titles + queries...", flush=True)
    title_tokens = bm25s.tokenize(
        index_titles, stopwords="en", stemmer=stemmer, show_progress=False
    )
    query_tokens = bm25s.tokenize(queries, stopwords="en", stemmer=stemmer, show_progress=False)

    print(
        f"\n{'method':<14} | {'CC3-50 R@10':>11} {'nDCG@10':>9} {'E@1':>7} {'time':>5}",
        flush=True,
    )
    print("-" * 60)
    for method in METHODS:
        t0 = time.time()
        try:
            idx = bm25s.BM25(k1=0.3, b=0.6, method=method)
            idx.index(title_tokens, show_progress=False)
            results, _ = idx.retrieve(query_tokens, k=K_RETRIEVE, show_progress=False)
            I = np.asarray(results, dtype=np.int64)
        except Exception as e:
            print(f"{method:<14} | FAILED: {e}", flush=True)
            continue

        rs, ns, e1s = [], [], []
        for qi, qid in enumerate(qids):
            positions = [int(p) for p in I[qi] if p >= 0]
            if not positions:
                continue
            sims = (
                pv_a[positions] @ qv_a[qi] + pv_b[positions] @ qv_b[qi] + pv_g[positions] @ qv_g[qi]
            ) / 3
            order = np.argsort(-sims)[:K_EVAL]
            ordering = [faiss_pos_to_pid[positions[int(j)]] for j in order]
            m = per_query_metrics(ordering, qrels[qid])
            if m is None:
                continue
            r, n, e1 = m
            rs.append(r)
            ns.append(n)
            if e1 is not None:
                e1s.append(e1)
        print(
            f"{method:<14} | {np.mean(rs):>10.2%} {np.mean(ns):>9.4f} {np.mean(e1s):>6.2%}  "
            f"({time.time() - t0:.0f}s)",
            flush=True,
        )
        del idx, I
        import gc

        gc.collect()


if __name__ == "__main__":
    main()
