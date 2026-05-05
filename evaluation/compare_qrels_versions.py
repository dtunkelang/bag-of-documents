#!/usr/bin/env python3
"""Recompute aggregate metrics for cached per-query orderings against both
the original and cleaned qrels. Reports deltas (cleaned minus original).

Sources of cached orderings (per-query top-10 PIDs):
  /tmp/faisshn_skip30_probe_per_query.json (per_query R@10, E@1 pre-aggregated)
  /tmp/router_probe.json                   (per-query A_R/B_R/G_R/K_R + E1)

Since the per_query data already aggregates only R@10 and E@1 (not nDCG),
we compute R@10 and E@1 deltas only.

For setups where the saved data only has metric values (not the actual
top-10 PIDs), we cannot recompute — so we use this as a sanity check on
the architecture deltas, not absolute numbers.

A more accurate approach would require re-running the architectures with
both qrels. This script gives a fast direct comparison from cached data.
"""

import json
import os
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

SCRIPT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_qrels(path):
    qrels = defaultdict(dict)
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            qrels[r["query_id"]][r["product_id"]] = r["relevance"]
    return dict(qrels)


def metrics_for(retrieved_pids, qrels_q, k=10):
    e_pids = {p for p, g in qrels_q.items() if g == 3}
    es_pids = {p for p, g in qrels_q.items() if g >= 2}
    out = {}
    if es_pids:
        top_k = retrieved_pids[:k]
        out["recall"] = sum(1 for p in top_k if p in es_pids) / len(es_pids)
    if e_pids:
        out["e_at_1"] = 1.0 if retrieved_pids and retrieved_pids[0] in e_pids else 0.0
    return out


def main():
    qrels_orig = load_qrels(os.path.join(SCRIPT_DIR, "esci_us_data/test_qrels.jsonl"))
    qrels_clean = load_qrels(os.path.join(SCRIPT_DIR, "esci_us_data/test_qrels_cleaned.jsonl"))
    print(
        f"loaded qrels: orig={sum(len(v) for v in qrels_orig.values())} "
        f"rows, clean={sum(len(v) for v in qrels_clean.values())} rows",
        flush=True,
    )

    # Also count differences
    n_diff = 0
    for qid, qrs in qrels_orig.items():
        for pid, g in qrs.items():
            if qrels_clean.get(qid, {}).get(pid, g) != g:
                n_diff += 1
    print(f"  differences: {n_diff} qrels rows changed", flush=True)

    # Compute eligible queries under each qrels: must have >=1 E or S product
    qids_orig = {qid for qid, qrs in qrels_orig.items() if any(g >= 2 for g in qrs.values())}
    qids_clean = {qid for qid, qrs in qrels_clean.items() if any(g >= 2 for g in qrs.values())}
    print(f"  eligible queries: orig={len(qids_orig)} clean={len(qids_clean)}", flush=True)

    # No saved top-10 PIDs in /tmp/router_probe.json (it stored R/E1 per-query
    # already aggregated against the *original* qrels). Same for faisshn_probe.
    # So we cannot recompute from those caches.
    #
    # We will instead run a focused comparison: brute-force retrieval
    # with rerank_A (production retriever) and write top-10 PIDs, then
    # evaluate against both qrels.
    print("\n>>> need to recompute orderings; running rerank_A retrieval pass...", flush=True)
    print("    (loads precomputed catalog vectors; ~2-5 min on MPS)", flush=True)

    import subprocess

    import numpy as np

    INDEX_DIR = os.path.join(SCRIPT_DIR, "combined_index_us_minilm")
    queries_all = {}
    with open(os.path.join(SCRIPT_DIR, "esci_us_data/test_queries.jsonl")) as f:
        for line in f:
            d = json.loads(line)
            queries_all[d["query_id"]] = d["query"]

    # Preserve test_queries.jsonl iteration order (matches BM25 cache alignment)
    qids = [
        qid
        for qid in queries_all
        if qid in qrels_orig and any(g >= 2 for g in qrels_orig[qid].values())
    ]
    queries = [queries_all[qid] for qid in qids]
    print(f"    encoding {len(queries)} queries with rerank_A...", flush=True)

    code = f"""
import os, json, sys
os.environ['TOKENIZERS_PARALLELISM']='false'
os.environ['OMP_NUM_THREADS']='1'
os.chdir({SCRIPT_DIR!r})
import numpy as np, torch
from sentence_transformers import SentenceTransformer
device = 'mps' if torch.backends.mps.is_available() else 'cpu'
m = SentenceTransformer('query_model_us_full_6m_mnrl', device=device)
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
    qv_a = np.array(json.loads(out), dtype=np.float32)
    print(f"    encoded shape={qv_a.shape}", flush=True)

    pv_a = np.load(os.path.join(INDEX_DIR, "rerank_A.vecs.fp16.npy")).astype(np.float32)
    print(f"    catalog shape={pv_a.shape}", flush=True)

    # BM25 top-50 (we'll also do BM25+rerank_A)
    bm25_path = os.path.join(INDEX_DIR, "bm25s_top200.npy")
    if not os.path.exists(bm25_path):
        bm25_path = os.path.join(INDEX_DIR, "bm25_top200.npy")
    bm25_top50 = np.load(bm25_path)[:, :50]

    with open(os.path.join(SCRIPT_DIR, "esci_us_data/product_ids.json")) as f:
        esci_pids = json.load(f)
    with open(os.path.join(SCRIPT_DIR, "esci_us_data/titles.json")) as f:
        esci_titles_arr = json.load(f)
    with open(os.path.join(INDEX_DIR, "titles.json")) as f:
        index_titles = json.load(f)
    title_to_pid = {t: p for p, t in zip(esci_pids, esci_titles_arr)}
    faiss_pos_to_pid = [title_to_pid.get(t) for t in index_titles]

    # Retrieval top-10 (rerank_A as retriever)
    print("    computing rerank_A retrieval top-10...", flush=True)
    K = 10
    BATCH = 64
    n = qv_a.shape[0]
    a_retr_top10 = np.zeros((n, K), dtype=np.int64)
    for start in range(0, n, BATCH):
        end = min(start + BATCH, n)
        sims = qv_a[start:end] @ pv_a.T
        top = np.argpartition(-sims, K, axis=1)[:, :K]
        for i in range(end - start):
            row = sims[i, top[i]]
            order = np.argsort(-row)
            a_retr_top10[start + i] = top[i, order]

    # BM25+rerank_A top-10 (rerank_A as single-stream reranker over BM25 top-50)
    print("    computing BM25+rerank_A rerank top-10...", flush=True)
    a_rerank_top10 = np.zeros((n, K), dtype=np.int64)
    for i in range(n):
        cand = bm25_top50[i]
        cand = cand[cand >= 0]
        if len(cand) == 0:
            a_rerank_top10[i, :] = -1
            continue
        sims = qv_a[i] @ pv_a[cand].T
        order = np.argsort(-sims)[:K]
        topk = cand[order]
        if len(topk) < K:
            topk = np.concatenate([topk, np.full(K - len(topk), -1, dtype=np.int64)])
        a_rerank_top10[i] = topk

    # 3-way ensemble (uniform_K, ≈ CC3-50): mean of A, B, G stream sims over BM25 top-50
    print("    encoding queries with rerank_B and rerank_G...", flush=True)
    code_b = code.replace("query_model_us_full_6m_mnrl", "query_model_us_qrels_mnrl_hardneg")
    code_g = code.replace("query_model_us_full_6m_mnrl", "query_model_us_esci_supervised")
    out_b = subprocess.check_output(
        [".venv/bin/python", "-c", code_b],
        input=json.dumps(queries),
        stderr=subprocess.DEVNULL,
        text=True,
    )
    qv_b = np.array(json.loads(out_b), dtype=np.float32)
    out_g = subprocess.check_output(
        [".venv/bin/python", "-c", code_g],
        input=json.dumps(queries),
        stderr=subprocess.DEVNULL,
        text=True,
    )
    qv_g = np.array(json.loads(out_g), dtype=np.float32)
    pv_b = np.load(os.path.join(INDEX_DIR, "rerank_B.vecs.fp16.npy")).astype(np.float32)
    pv_g = np.load(os.path.join(INDEX_DIR, "rerank_G.vecs.fp16.npy")).astype(np.float32)

    print("    computing 3-way ensemble (CC3-50) top-10...", flush=True)
    cc3_top10 = np.zeros((n, K), dtype=np.int64)
    for i in range(n):
        cand = bm25_top50[i]
        cand = cand[cand >= 0]
        if len(cand) == 0:
            cc3_top10[i, :] = -1
            continue
        s = (qv_a[i] @ pv_a[cand].T + qv_b[i] @ pv_b[cand].T + qv_g[i] @ pv_g[cand].T) / 3
        order = np.argsort(-s)[:K]
        topk = cand[order]
        if len(topk) < K:
            topk = np.concatenate([topk, np.full(K - len(topk), -1, dtype=np.int64)])
        cc3_top10[i] = topk

    def to_pids(ord_array):
        return [
            [faiss_pos_to_pid[p] if 0 <= p < len(faiss_pos_to_pid) else None for p in row]
            for row in ord_array
        ]

    setups = {
        "rerank_A_retriever": to_pids(a_retr_top10),
        "BM25_top50+rerank_A": to_pids(a_rerank_top10),
        "CC3-50_3way_ensemble": to_pids(cc3_top10),
    }

    # Evaluate each on orig and cleaned qrels
    print("\n=== METRICS ON ORIGINAL VS CLEANED QRELS ===", flush=True)
    name_pad = max(len(n) for n in setups)
    print(f"{'setup':<{name_pad}}  R@10 orig  R@10 clean  delta    E@1 orig  E@1 clean  delta")
    for name, orderings in setups.items():
        recalls_o = []
        recalls_c = []
        e1_o = []
        e1_c = []
        for qi, qid in enumerate(qids):
            mo = metrics_for(orderings[qi], qrels_orig[qid])
            mc = metrics_for(orderings[qi], qrels_clean[qid])
            if "recall" in mo:
                recalls_o.append(mo["recall"])
            if "recall" in mc:
                recalls_c.append(mc["recall"])
            if "e_at_1" in mo:
                e1_o.append(mo["e_at_1"])
            if "e_at_1" in mc:
                e1_c.append(mc["e_at_1"])

        ro = sum(recalls_o) / len(recalls_o) if recalls_o else 0
        rc = sum(recalls_c) / len(recalls_c) if recalls_c else 0
        eo = sum(e1_o) / len(e1_o) if e1_o else 0
        ec = sum(e1_c) / len(e1_c) if e1_c else 0
        print(
            f"{name:<{name_pad}}  {ro * 100:8.2f}  {rc * 100:9.2f}  {(rc - ro) * 100:+6.2f}  "
            f"{eo * 100:8.2f}  {ec * 100:8.2f}  {(ec - eo) * 100:+6.2f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
