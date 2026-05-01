#!/usr/bin/env python3
"""Two retriever-side follow-ups to the (k1=0.3, b=0.6) bm25s win.

Probe 1: Per-query-length k1 sweep (b fixed at 0.6).
   Hypothesis: short queries (1-2 tokens) and long queries (4+ tokens)
   want different k1 because short queries have ~no within-query TF
   variation while long queries have many. (k1=0.3, b=0.6) was averaged
   over both regimes; per-bin optima may differ.

Probe 2: Doc-side tokenization variants (k1=0.3, b=0.6 fixed).
   Hypothesis: Snowball + en stopwords might over-stem model numbers
   ("K380") or drop signal words. Tests four (stemmer × stopwords)
   combos to see if a different tokenization releases more lift.

Reuses cached query encodings and product vecs to keep total runtime
under an hour.

Usage:
    python evaluation/eval_bm25_followups.py
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

K_RETRIEVE_CC = 50
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
        print("ENCODE STDOUT:", p.stdout, file=sys.stderr)
        print("ENCODE STDERR:", p.stderr, file=sys.stderr)
        raise RuntimeError(f"encode failed for {model_path}")
    return np.load("/tmp/_qenc.npy")


def per_query_metrics(retrieved_pids, qrels_q):
    if not retrieved_pids:
        return None
    pos_e = {pid for pid, g in qrels_q.items() if g >= 3}
    pos_es = {pid for pid, g in qrels_q.items() if g >= 2}
    if not pos_es:
        return None
    top_k = retrieved_pids[:K_EVAL]
    hits_es = sum(1 for p in top_k if p in pos_es)
    recall = hits_es / len(pos_es)
    gains = [1.0 if p in pos_e else (0.1 if p in pos_es else 0.0) for p in top_k]
    dcg = sum(g / math.log2(i + 2) for i, g in enumerate(gains))
    ideal = sorted((1.0 if p in pos_e else 0.1 for p in pos_es), reverse=True)[:K_EVAL]
    idcg = sum(g / math.log2(i + 2) for i, g in enumerate(ideal))
    ndcg = dcg / idcg if idcg > 0 else 0.0
    e1 = 1.0 if any(p in pos_e for p in top_k[:1]) and pos_e else (None if not pos_e else 0.0)
    return recall, ndcg, e1


def cc3_50_metrics_per_query(
    I_bm25_200, qids, qrels, qv_a, qv_b, qv_g, pv_a, pv_b, pv_g, faiss_pos_to_pid
):
    """For each query, compute CC3-50 (R@10, nDCG@10, E@1) using top-50 candidates."""
    out = []
    for qi, qid in enumerate(qids):
        positions = [int(p) for p in I_bm25_200[qi, :K_RETRIEVE_CC] if p >= 0]
        if not positions:
            out.append(None)
            continue
        sims = (
            pv_a[positions] @ qv_a[qi] + pv_b[positions] @ qv_b[qi] + pv_g[positions] @ qv_g[qi]
        ) / 3
        order = np.argsort(-sims)[:K_EVAL]
        ordering = [faiss_pos_to_pid[positions[int(j)]] for j in order]
        out.append(per_query_metrics(ordering, qrels[qid]))
    return out


def aggregate(metrics_per_query, indices=None):
    """Aggregate over either all queries or a subset (indices)."""
    if indices is None:
        indices = range(len(metrics_per_query))
    rs, ns, e1s = [], [], []
    for i in indices:
        m = metrics_per_query[i]
        if m is None:
            continue
        r, n, e1 = m
        rs.append(r)
        ns.append(n)
        if e1 is not None:
            e1s.append(e1)
    return {
        "n": len(rs),
        "R@10": np.mean(rs) if rs else 0.0,
        "nDCG@10": np.mean(ns) if ns else 0.0,
        "E@1": np.mean(e1s) if e1s else 0.0,
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
    print(f"  {len(qids):,} eval queries, {len(index_titles):,} titles", flush=True)

    # Encode queries with reranker models (subprocess to keep MPS clean).
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

    # =================================================================
    # PROBE 1: per-query-length k1 sweep (b=0.6 fixed)
    # =================================================================
    print("\n" + "=" * 100)
    print("PROBE 1: per-query-length k1 sweep (b=0.6 fixed)")
    print("=" * 100)

    # Build query-length bins from raw token count.
    query_lens = [len(q.split()) for q in queries]
    bins = {"1": [], "2": [], "3": [], "4+": []}
    for qi, ql in enumerate(query_lens):
        if ql == 1:
            bins["1"].append(qi)
        elif ql == 2:
            bins["2"].append(qi)
        elif ql == 3:
            bins["3"].append(qi)
        else:
            bins["4+"].append(qi)
    print(
        f"\nquery-length bins: 1tok={len(bins['1']):,}, 2tok={len(bins['2']):,}, "
        f"3tok={len(bins['3']):,}, 4+tok={len(bins['4+']):,}",
        flush=True,
    )

    # Tokenize once; only k1 changes between sweep entries.
    stemmer = Stemmer.Stemmer("english")
    print("tokenizing titles + queries (snowball+stop)...", flush=True)
    title_tokens = bm25s.tokenize(
        index_titles, stopwords="en", stemmer=stemmer, show_progress=False
    )
    query_tokens = bm25s.tokenize(queries, stopwords="en", stemmer=stemmer, show_progress=False)

    k1_sweep = [0.1, 0.2, 0.3, 0.5, 1.0, 1.6]
    print(
        f"\n{'k1':>5} | " + " | ".join(f"{b:>16}" for b in ["all", "1tok", "2tok", "3tok", "4+tok"])
    )
    print("-" * 96)
    by_k1_metrics = {}
    for k1 in k1_sweep:
        t0 = time.time()
        idx = bm25s.BM25(k1=k1, b=0.6)
        idx.index(title_tokens, show_progress=False)
        results, _ = idx.retrieve(query_tokens, k=K_RETRIEVE_CC, show_progress=False)
        I = np.asarray(results, dtype=np.int64)
        per_q = cc3_50_metrics_per_query(
            I, qids, qrels, qv_a, qv_b, qv_g, pv_a, pv_b, pv_g, faiss_pos_to_pid
        )
        by_k1_metrics[k1] = per_q

        # Aggregate per bin
        agg_all = aggregate(per_q)
        agg1 = aggregate(per_q, bins["1"])
        agg2 = aggregate(per_q, bins["2"])
        agg3 = aggregate(per_q, bins["3"])
        agg4 = aggregate(per_q, bins["4+"])

        def cell(a):
            return f"{a['R@10']:>6.2%}/{a['E@1']:>6.2%}"

        print(
            f"{k1:>5} | {cell(agg_all):>16} | {cell(agg1):>16} | {cell(agg2):>16} | "
            f"{cell(agg3):>16} | {cell(agg4):>16}  ({time.time() - t0:.0f}s)"
        )
        del idx, I, results
        import gc

        gc.collect()

    # Find per-bin optimum
    print("\nPer-bin optimum (R@10):")
    for binname in ["1", "2", "3", "4+"]:
        best_k1 = None
        best_r = -1
        for k1 in k1_sweep:
            agg = aggregate(by_k1_metrics[k1], bins[binname])
            if agg["R@10"] > best_r:
                best_r = agg["R@10"]
                best_k1 = k1
        print(f"  {binname:>5}-tok: best k1={best_k1} (R@10={best_r:.2%})")

    # If per-bin optima differ, score the routed setup
    routed = []
    for qi, ql in enumerate(query_lens):
        binname = "1" if ql == 1 else "2" if ql == 2 else "3" if ql == 3 else "4+"
        # Use per-bin best k1 (chosen by R@10)
        best_k1 = None
        best_r = -1
        for k1 in k1_sweep:
            agg = aggregate(by_k1_metrics[k1], bins[binname])
            if agg["R@10"] > best_r:
                best_r = agg["R@10"]
                best_k1 = k1
        routed.append(by_k1_metrics[best_k1][qi])

    routed_agg = aggregate(routed)
    print(
        f"\nRouted (per-bin best k1): R@10={routed_agg['R@10']:.4f} "
        f"nDCG@10={routed_agg['nDCG@10']:.4f} E@1={routed_agg['E@1']:.4f}",
        flush=True,
    )
    print(f"vs uniform k1=0.3: R@10={aggregate(by_k1_metrics[0.3])['R@10']:.4f}")

    # =================================================================
    # PROBE 2: doc-side tokenization variants (k1=0.3, b=0.6 fixed)
    # =================================================================
    print("\n" + "=" * 100)
    print("PROBE 2: tokenization variants (k1=0.3, b=0.6 fixed)")
    print("=" * 100)

    variants = [
        ("snowball+stopw (current)", "en", stemmer),
        ("snowball, no stopw", None, stemmer),
        ("no stem, stopw", "en", None),
        ("no stem, no stopw", None, None),
    ]
    print(f"\n{'variant':<28} | {'R@10':>8} {'nDCG@10':>9} {'E@1':>8} {'time':>5}")
    print("-" * 70)
    for name, sw, st in variants:
        t0 = time.time()
        tt = bm25s.tokenize(index_titles, stopwords=sw, stemmer=st, show_progress=False)
        qt = bm25s.tokenize(queries, stopwords=sw, stemmer=st, show_progress=False)
        idx = bm25s.BM25(k1=0.3, b=0.6)
        idx.index(tt, show_progress=False)
        results, _ = idx.retrieve(qt, k=K_RETRIEVE_CC, show_progress=False)
        I = np.asarray(results, dtype=np.int64)
        per_q = cc3_50_metrics_per_query(
            I, qids, qrels, qv_a, qv_b, qv_g, pv_a, pv_b, pv_g, faiss_pos_to_pid
        )
        agg = aggregate(per_q)
        print(
            f"{name:<28} | {agg['R@10']:>7.2%} {agg['nDCG@10']:>9.4f} {agg['E@1']:>7.2%} "
            f"  ({time.time() - t0:.0f}s)"
        )
        del idx, I, results, tt, qt
        import gc

        gc.collect()

    print("\nDone.")


if __name__ == "__main__":
    main()
