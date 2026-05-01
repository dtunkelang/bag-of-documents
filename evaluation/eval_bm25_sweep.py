#!/usr/bin/env python3
"""BM25 (k1, b) sweep against the K and CC3-50 ESCI eval setups.

The shipped retriever uses tantivy's default BM25 parameters (k1=1.2, b=0.75
per Lucene defaults), which were never tuned for this corpus. ESCI titles are
short, keyword-heavy, brand-anchored — a regime where the defaults are often
a hair off optimal.

This script swaps in `bm25s` (a fast numpy-backed BM25 with configurable k1/b)
to sweep candidate combos and measure the downstream effect on K (BM25 + 2-way
ensemble rerank) and CC3-50 (BM25 top-50 + 3-way ensemble rerank, current SOTA).

Output: per-combo R@10 / nDCG@10 / E@1 / E@3 for K and CC3-50, with deltas
relative to the tantivy baseline (K=21.11%, CC3-50=21.32%).

Usage:
    python evaluation/eval_bm25_sweep.py
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

# Boundary probe: the second sweep found a local-max region around
# (k1=0.5, b=0.5-0.6). Test slightly smaller k1 and larger b to see if
# the optimum lies further out, or if (0.4-0.6, 0.5-0.6) is the true plateau.
SWEEP = [
    (1.2, 0.75),  # baseline / sanity check
    (0.5, 0.60),  # current best (verify)
    (0.3, 0.50),
    (0.3, 0.60),
    (0.2, 0.50),
    (0.4, 0.60),
    (0.4, 0.70),
    (0.5, 0.70),
    (0.5, 0.80),
    (0.6, 0.70),
]

K_RETRIEVE_K = 100  # for setup K
K_RETRIEVE_CC = 50  # for CC3-50
K_EVAL = 10


def encode_subproc(model_path, queries):
    """Encode queries with a sentence-transformers model in a subprocess so
    MPS state doesn't leak across encoders."""
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
        print("ENCODE STDOUT:", p.stdout, file=sys.stderr)
        print("ENCODE STDERR:", p.stderr, file=sys.stderr)
        raise RuntimeError(f"encode failed for {model_path}")
    return np.load("/tmp/_qenc.npy")


def per_query_metrics(retrieved_pids, qrels_q, k_eval=10):
    """R@10 (E+S relevant), nDCG@10 (E=1.0/S=0.1), E@1, E@3."""
    if not retrieved_pids:
        return {}
    pos_e = {pid for pid, g in qrels_q.items() if g >= 3}
    pos_es = {pid for pid, g in qrels_q.items() if g >= 2}
    rel_es_total = len(pos_es)
    if rel_es_total == 0:
        return {}
    top_k = retrieved_pids[:k_eval]
    hits_es = sum(1 for p in top_k if p in pos_es)
    recall = hits_es / rel_es_total
    # nDCG with E=1.0, S=0.1
    gains = []
    for p in top_k:
        if p in pos_e:
            gains.append(1.0)
        elif p in pos_es:
            gains.append(0.1)
        else:
            gains.append(0.0)
    dcg = sum(g / math.log2(i + 2) for i, g in enumerate(gains))
    ideal = sorted(
        (1.0 if p in pos_e else 0.1 for p in pos_es),
        reverse=True,
    )[:k_eval]
    idcg = sum(g / math.log2(i + 2) for i, g in enumerate(ideal))
    ndcg = dcg / idcg if idcg > 0 else 0.0
    out = {"recall": recall, "ndcg": ndcg}
    if pos_e:
        out["e_at_1"] = 1.0 if any(p in pos_e for p in top_k[:1]) else 0.0
        out["e_at_3"] = 1.0 if any(p in pos_e for p in top_k[:3]) else 0.0
    return out


def aggregate(orderings, qids, qrels):
    metrics = defaultdict(list)
    e_metrics = defaultdict(list)
    for qi, qid in enumerate(qids):
        m = per_query_metrics(orderings[qi], qrels[qid])
        if "recall" not in m:
            continue
        metrics["recall"].append(m["recall"])
        metrics["ndcg"].append(m["ndcg"])
        if "e_at_1" in m:
            e_metrics["e_at_1"].append(m["e_at_1"])
            e_metrics["e_at_3"].append(m["e_at_3"])
    return {
        "R@10": np.mean(metrics["recall"]) if metrics["recall"] else 0.0,
        "nDCG@10": np.mean(metrics["ndcg"]) if metrics["ndcg"] else 0.0,
        "E@1": np.mean(e_metrics["e_at_1"]) if e_metrics["e_at_1"] else 0.0,
        "E@3": np.mean(e_metrics["e_at_3"]) if e_metrics["e_at_3"] else 0.0,
    }


def main():
    print("loading queries + qrels + product map...", flush=True)
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
    print(f"  {len(qids):,} eval queries, {len(index_titles):,} titles in index", flush=True)

    # --- Tokenize titles (one-time) ---
    print("tokenizing titles with bm25s+stemmer...", flush=True)
    stemmer = Stemmer.Stemmer("english")
    t0 = time.time()
    title_tokens = bm25s.tokenize(
        index_titles, stopwords="en", stemmer=stemmer, show_progress=False
    )
    print(f"  done in {time.time() - t0:.0f}s", flush=True)

    # --- Tokenize queries (one-time) ---
    print("tokenizing queries...", flush=True)
    t0 = time.time()
    query_tokens = bm25s.tokenize(queries, stopwords="en", stemmer=stemmer, show_progress=False)
    print(f"  done in {time.time() - t0:.0f}s", flush=True)

    # --- Encode queries with rerank_a, rerank_b, rerank_g (cached vecs) ---
    print("encoding queries with rerank_a, rerank_b, rerank_g...", flush=True)
    t0 = time.time()
    qv_a = encode_subproc(os.path.join(SCRIPT_DIR, "query_model_us_full_6m_mnrl"), queries)
    qv_b = encode_subproc(os.path.join(SCRIPT_DIR, "query_model_us_qrels_mnrl_hardneg"), queries)
    qv_g = encode_subproc(os.path.join(SCRIPT_DIR, "query_model_us_esci_supervised"), queries)
    print(f"  done in {time.time() - t0:.0f}s", flush=True)

    print("loading cached product vecs...", flush=True)
    pv_a = np.load(os.path.join(INDEX_DIR, "rerank_A.vecs.fp16.npy")).astype(np.float32)
    pv_b = np.load(os.path.join(INDEX_DIR, "rerank_B.vecs.fp16.npy")).astype(np.float32)
    pv_g = np.load(os.path.join(INDEX_DIR, "rerank_G.vecs.fp16.npy")).astype(np.float32)

    # --- Sweep ---
    rows = []
    for k1, b in SWEEP:
        print(f"\n=== k1={k1}, b={b} ===", flush=True)
        t0 = time.time()
        idx = bm25s.BM25(k1=k1, b=b)
        idx.index(title_tokens, show_progress=False)
        print(f"  indexed in {time.time() - t0:.0f}s", flush=True)

        t0 = time.time()
        # bm25s returns (results, scores); results is array of doc indices.
        results, _ = idx.retrieve(
            query_tokens, k=max(K_RETRIEVE_K, K_RETRIEVE_CC), show_progress=False
        )
        print(f"  retrieved in {time.time() - t0:.0f}s", flush=True)
        # results: (n_queries, k), each entry is title-row index
        I_bm25 = np.asarray(results, dtype=np.int64)

        # H: BM25 alone, top-10
        h_orderings = []
        for qi in range(len(queries)):
            row = I_bm25[qi, :K_EVAL]
            h_orderings.append([faiss_pos_to_pid[int(p)] for p in row if p >= 0])
        h_metrics = aggregate(h_orderings, qids, qrels)

        # K: BM25 top-100, 2-way (A+B) sumsim rerank
        k_orderings = []
        for qi in range(len(queries)):
            positions = [int(p) for p in I_bm25[qi, :K_RETRIEVE_K] if p >= 0]
            if not positions:
                k_orderings.append([])
                continue
            sims = (pv_a[positions] @ qv_a[qi] + pv_b[positions] @ qv_b[qi]) / 2
            order = np.argsort(-sims)[:K_EVAL]
            k_orderings.append([faiss_pos_to_pid[positions[int(j)]] for j in order])
        k_metrics = aggregate(k_orderings, qids, qrels)

        # CC3-50: BM25 top-50, 3-way (A+B+G) sumsim rerank
        cc_orderings = []
        for qi in range(len(queries)):
            positions = [int(p) for p in I_bm25[qi, :K_RETRIEVE_CC] if p >= 0]
            if not positions:
                cc_orderings.append([])
                continue
            sims = (
                pv_a[positions] @ qv_a[qi] + pv_b[positions] @ qv_b[qi] + pv_g[positions] @ qv_g[qi]
            ) / 3
            order = np.argsort(-sims)[:K_EVAL]
            cc_orderings.append([faiss_pos_to_pid[positions[int(j)]] for j in order])
        cc_metrics = aggregate(cc_orderings, qids, qrels)

        rows.append((k1, b, h_metrics, k_metrics, cc_metrics))

        # Free index memory before the next combo
        del idx, I_bm25, results
        import gc

        gc.collect()

    # --- Report ---
    # Tantivy baseline reference (cached)
    tantivy_baseline = {
        "H": {"R@10": 0.1950, "nDCG@10": 0.3322, "E@1": 0.3879, "E@3": 0.3572},
        "K": {"R@10": 0.2111, "nDCG@10": 0.3566, "E@1": 0.4087, "E@3": 0.3804},
        "CC3-50": {"R@10": 0.2132, "nDCG@10": 0.3613, "E@1": 0.4164, "E@3": 0.3880},
    }

    print("\n" + "=" * 130)
    print(f"=== BM25 (k1, b) sweep on {len(qids):,} ESCI test queries ===")
    print("=" * 130)
    print(
        f"{'combo':<14} | {'H R@10':>9} {'H E@1':>8} | {'K R@10':>9} {'K-K0':>7} {'K E@1':>8} | {'CC R@10':>9} {'CC-CC0':>7} {'CC E@1':>8}"
    )
    print("-" * 130)
    print(
        f"{'tantivy':<14} | "
        f"{tantivy_baseline['H']['R@10']:>8.2%}  {tantivy_baseline['H']['E@1']:>7.2%} | "
        f"{tantivy_baseline['K']['R@10']:>8.2%}  {'   ':>5}     "
        f"{tantivy_baseline['K']['E@1']:>7.2%} | "
        f"{tantivy_baseline['CC3-50']['R@10']:>8.2%}  {'   ':>5}     "
        f"{tantivy_baseline['CC3-50']['E@1']:>7.2%}"
    )

    for k1, b, hm, km, ccm in rows:
        d_k = (km["R@10"] - tantivy_baseline["K"]["R@10"]) * 100
        d_cc = (ccm["R@10"] - tantivy_baseline["CC3-50"]["R@10"]) * 100
        print(
            f"k1={k1}, b={b:<5} | "
            f"{hm['R@10']:>8.2%}  {hm['E@1']:>7.2%} | "
            f"{km['R@10']:>8.2%}  {d_k:>+6.2f}    {km['E@1']:>7.2%} | "
            f"{ccm['R@10']:>8.2%}  {d_cc:>+6.2f}    {ccm['E@1']:>7.2%}"
        )

    print("\nFull metrics per combo:")
    for k1, b, hm, km, ccm in rows:
        print(
            f"  k1={k1}, b={b}: "
            f"H R@10={hm['R@10']:.2%} nDCG={hm['nDCG@10']:.4f} | "
            f"K R@10={km['R@10']:.2%} nDCG={km['nDCG@10']:.4f} | "
            f"CC R@10={ccm['R@10']:.2%} nDCG={ccm['nDCG@10']:.4f} E@1={ccm['E@1']:.2%}"
        )


if __name__ == "__main__":
    main()
