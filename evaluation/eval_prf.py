#!/usr/bin/env python3
"""Pseudo-relevance feedback (PRF) probe.

For each query, take the top-3 BM25 results, extract their highest-IDF
tokens not already in the query, append to the query, re-retrieve via
bm25s, and re-rank with CC3-50.

Hypothesis: under-specified queries (e.g. '#2 pencils', 'fence end
spacer') would benefit from candidate-pool expansion via terms found in
the top-3 retrieval results.

Output: PRF-expanded query strings + per-query CC3-50 metrics.

Usage:
    python evaluation/eval_prf.py
    python evaluation/eval_prf.py --feedback-docs 5 --new-tokens 2
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import os  # noqa: E402
import re  # noqa: E402
import time  # noqa: E402
from collections import Counter, defaultdict  # noqa: E402

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import bm25s  # noqa: E402
import numpy as np  # noqa: E402
import Stemmer  # noqa: E402

SCRIPT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INDEX_DIR = os.path.join(SCRIPT_DIR, "combined_index_us_minilm")
K_EVAL = 10
K_RET = 50
TOK_RE = re.compile(r"[a-z0-9]+")


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
    if pos_e:
        e1 = sum(1 for p in top_k[:1] if p in pos_e) / min(1, len(pos_e))
        e3 = sum(1 for p in top_k[:3] if p in pos_e) / min(3, len(pos_e))
    else:
        e1 = e3 = float("nan")
    return recall, ndcg, e1, e3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feedback-docs", type=int, default=3, help="top-N BM25 docs for feedback")
    ap.add_argument("--new-tokens", type=int, default=2, help="N highest-IDF tokens to add")
    args = ap.parse_args()

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

    # Build IDF table from the catalog.
    print("computing token DFs from catalog...", flush=True)
    t0 = time.time()
    df = Counter()
    n_docs = len(index_titles)
    for t in index_titles:
        for tok in set(TOK_RE.findall(t.lower())):
            df[tok] += 1
    idf = {tok: math.log((n_docs + 1) / (d + 1)) for tok, d in df.items()}
    print(f"  {len(idf):,} unique tokens, {time.time() - t0:.0f}s", flush=True)

    # Load BM25 top-3 from cache (top-200 array; slice).
    I_orig = np.load(os.path.join(INDEX_DIR, "bm25s_top200.npy"))[:, : args.feedback_docs]

    # PRF expansion.
    print(f"expanding {len(queries):,} queries with PRF...", flush=True)
    t0 = time.time()
    expanded_queries = []
    n_expanded = 0
    for qi, q in enumerate(queries):
        q_tokens = set(TOK_RE.findall(q.lower()))
        token_scores = Counter()
        for j in range(args.feedback_docs):
            pos = int(I_orig[qi, j])
            if pos < 0:
                continue
            for tok in TOK_RE.findall(index_titles[pos].lower()):
                if tok in q_tokens or tok not in idf:
                    continue
                token_scores[tok] += idf[tok]
        # Top-N candidates
        top = [tok for tok, _ in token_scores.most_common(args.new_tokens)]
        if top:
            expanded_queries.append(q + " " + " ".join(top))
            n_expanded += 1
        else:
            expanded_queries.append(q)
    print(f"  {n_expanded:,} expanded ({n_expanded / len(queries):.1%}), {time.time() - t0:.0f}s")

    print("\nsample PRF expansions:")
    samples = 0
    for qi in range(len(queries)):
        if expanded_queries[qi] != queries[qi] and samples < 12:
            print(f"  '{queries[qi]}' → '{expanded_queries[qi]}'")
            samples += 1

    # Save expansions.
    prf_dir = os.path.join(INDEX_DIR, "prf")
    os.makedirs(prf_dir, exist_ok=True)
    with open(os.path.join(prf_dir, "expanded_queries.json"), "w") as f:
        json.dump(expanded_queries, f)

    # Re-retrieve with PRF queries via bm25s.
    print("\nre-retrieving via bm25s...", flush=True)
    t0 = time.time()
    bm25s_idx = bm25s.BM25.load(os.path.join(INDEX_DIR, "bm25s_index"), mmap=False)
    stemmer = Stemmer.Stemmer("english")
    qt = bm25s.tokenize(expanded_queries, stopwords="en", stemmer=stemmer, show_progress=False)
    results, _ = bm25s_idx.retrieve(qt, k=100, show_progress=False)
    I_prf = np.asarray(results, dtype=np.int64)
    np.save(os.path.join(prf_dir, "I_prf_top100.npy"), I_prf)
    print(f"  done {time.time() - t0:.0f}s", flush=True)

    # CC3-50 eval (no CE for speed).
    print(
        "\nencoding original queries (NOT expanded — bi-encoders process the user's intent, not BM25 expansion)...",
        flush=True,
    )
    qv_a = encode_subproc(os.path.join(SCRIPT_DIR, "query_model_us_full_6m_mnrl"), queries)
    qv_b = encode_subproc(os.path.join(SCRIPT_DIR, "query_model_us_qrels_mnrl_hardneg"), queries)
    qv_g = encode_subproc(os.path.join(SCRIPT_DIR, "query_model_us_esci_supervised"), queries)
    pv_a = np.load(os.path.join(INDEX_DIR, "rerank_A.vecs.fp16.npy")).astype(np.float32)
    pv_b = np.load(os.path.join(INDEX_DIR, "rerank_B.vecs.fp16.npy")).astype(np.float32)
    pv_g = np.load(os.path.join(INDEX_DIR, "rerank_G.vecs.fp16.npy")).astype(np.float32)

    print("computing CC3-50 with PRF candidates...", flush=True)
    rs, ns, e1s, e3s = [], [], [], []
    rs_per_query = np.full(len(qids), np.nan, dtype=np.float32)
    e1s_per_query = np.full(len(qids), np.nan, dtype=np.float32)
    for qi, qid in enumerate(qids):
        positions = [int(p) for p in I_prf[qi, :K_RET] if p >= 0]
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
        r, n, e1, e3 = m
        rs.append(r)
        ns.append(n)
        rs_per_query[qi] = r
        if e1 is not None and not math.isnan(e1):
            e1s.append(e1)
            e3s.append(e3)
            e1s_per_query[qi] = e1
    np.save(os.path.join(prf_dir, "prf_per_query_R10.npy"), rs_per_query)
    np.save(os.path.join(prf_dir, "prf_per_query_E1.npy"), e1s_per_query)

    print(
        f"\nCC3-50 with PRF: R@10 {np.mean(rs):.2%} nDCG {np.mean(ns):.4f} "
        f"E@1 {np.mean(e1s):.2%} E@3 {np.mean(e3s):.2%}"
    )

    # Baseline reference.
    print("\nbaseline (cached): CC3-50 R@10 21.60%, E@1 42.10% (from earlier eval)", flush=True)


if __name__ == "__main__":
    main()
