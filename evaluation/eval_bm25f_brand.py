#!/usr/bin/env python3
"""BM25F-style probe: split each title into [first-token brand] + [rest]
and score the two streams separately. The first token of an Amazon
product title is the brand for ~70-80% of titles ('Logitech K380...',
'Samsung 860 EVO...'). For brand-driven queries this should give the
brand match more weight than its raw frequency.

We don't actually use BM25F (which requires per-field length normalization
inside one BM25 score). Instead we build two bm25s indices — one over
just the first token, one over the rest of the title — and fuse their
scores with a sweep over brand_weight in {0.0, 0.25, 0.4, 0.5, 0.6, 0.75, 1.0}.

Cheap; ~10 min total.

Usage:
    python evaluation/eval_bm25f_brand.py
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
    e3 = (1.0 if any(p in pos_e for p in top_k[:3]) else 0.0) if pos_e else None
    return recall, ndcg, e1, e3


def split_title(t):
    """First whitespace-token = brand-stream. Rest = description-stream.
    Both can be the same as the original title for short titles."""
    parts = t.strip().split(maxsplit=1)
    brand = parts[0] if parts else ""
    rest = parts[1] if len(parts) > 1 else ""
    return brand, rest


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
    print(f"  {len(qids):,} eval queries", flush=True)

    print("encoding queries...", flush=True)
    qv_a = encode_subproc(os.path.join(SCRIPT_DIR, "query_model_us_full_6m_mnrl"), queries)
    qv_b = encode_subproc(os.path.join(SCRIPT_DIR, "query_model_us_qrels_mnrl_hardneg"), queries)
    qv_g = encode_subproc(os.path.join(SCRIPT_DIR, "query_model_us_esci_supervised"), queries)

    print("loading product vecs...", flush=True)
    pv_a = np.load(os.path.join(INDEX_DIR, "rerank_A.vecs.fp16.npy")).astype(np.float32)
    pv_b = np.load(os.path.join(INDEX_DIR, "rerank_B.vecs.fp16.npy")).astype(np.float32)
    pv_g = np.load(os.path.join(INDEX_DIR, "rerank_G.vecs.fp16.npy")).astype(np.float32)

    print("splitting titles into [brand] + [rest]...", flush=True)
    brands = []
    rests = []
    for t in index_titles:
        b, r = split_title(t)
        brands.append(b)
        rests.append(r if r else b)  # short titles keep brand in rest too

    stemmer = Stemmer.Stemmer("english")
    print("tokenizing query stream...", flush=True)
    qt = bm25s.tokenize(queries, stopwords="en", stemmer=stemmer, show_progress=False)

    print("building brand-only bm25s index (k1=0.3, b=0.6)...", flush=True)
    t0 = time.time()
    brand_tokens = bm25s.tokenize(brands, stopwords="en", stemmer=stemmer, show_progress=False)
    idx_brand = bm25s.BM25(k1=0.3, b=0.6)
    idx_brand.index(brand_tokens, show_progress=False)
    print(f"  done {time.time() - t0:.0f}s", flush=True)

    print("building rest-only bm25s index (k1=0.3, b=0.6)...", flush=True)
    t0 = time.time()
    rest_tokens = bm25s.tokenize(rests, stopwords="en", stemmer=stemmer, show_progress=False)
    idx_rest = bm25s.BM25(k1=0.3, b=0.6)
    idx_rest.index(rest_tokens, show_progress=False)
    print(f"  done {time.time() - t0:.0f}s", flush=True)

    # Score both streams for all queries.
    print("scoring brand stream...", flush=True)
    t0 = time.time()
    R_brand, S_brand = idx_brand.retrieve(qt, k=200, show_progress=False)
    R_brand = np.asarray(R_brand, dtype=np.int64)
    S_brand = np.asarray(S_brand, dtype=np.float32)
    print(f"  done {time.time() - t0:.0f}s", flush=True)

    print("scoring rest stream...", flush=True)
    t0 = time.time()
    R_rest, S_rest = idx_rest.retrieve(qt, k=200, show_progress=False)
    R_rest = np.asarray(R_rest, dtype=np.int64)
    S_rest = np.asarray(S_rest, dtype=np.float32)
    print(f"  done {time.time() - t0:.0f}s", flush=True)

    # For each query, fuse the two streams' top-N candidate scores.
    # Build a {position: brand_score} and {position: rest_score} map per query
    # so we can sum them, then take top-K_RETRIEVE for downstream rerank.
    weights = [0.0, 0.25, 0.4, 0.5, 0.6, 0.75, 1.0]
    print(
        f"\n{'brand_w':>8} | {'CC3-50 R@10':>11} {'nDCG@10':>9} {'E@1':>7} {'E@3':>7}",
        flush=True,
    )
    print("-" * 64)
    for w in weights:
        rs, ns, e1s, e3s = [], [], [], []
        for qi, qid in enumerate(qids):
            scoremap = {}
            for j in range(R_brand.shape[1]):
                pos = int(R_brand[qi, j])
                if pos < 0:
                    continue
                scoremap[pos] = scoremap.get(pos, 0.0) + w * float(S_brand[qi, j])
            for j in range(R_rest.shape[1]):
                pos = int(R_rest[qi, j])
                if pos < 0:
                    continue
                scoremap[pos] = scoremap.get(pos, 0.0) + (1 - w) * float(S_rest[qi, j])
            if not scoremap:
                continue
            top = sorted(scoremap.items(), key=lambda kv: -kv[1])[:K_RETRIEVE]
            positions = [p for p, _ in top]
            sims = (
                pv_a[positions] @ qv_a[qi] + pv_b[positions] @ qv_b[qi] + pv_g[positions] @ qv_g[qi]
            ) / 3
            order = np.argsort(-sims)[:K_EVAL]
            ordering = [faiss_pos_to_pid[positions[int(jj)]] for jj in order]
            m = per_query_metrics(ordering, qrels[qid])
            if m is None:
                continue
            r, n, e1, e3 = m
            rs.append(r)
            ns.append(n)
            if e1 is not None:
                e1s.append(e1)
                e3s.append(e3)
        print(
            f"{w:>8.2f} | {np.mean(rs):>10.2%} {np.mean(ns):>9.4f} {np.mean(e1s):>6.2%} {np.mean(e3s):>6.2%}",
            flush=True,
        )


if __name__ == "__main__":
    main()
