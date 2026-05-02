#!/usr/bin/env python3
"""Evaluate the CE-distilled bi-encoder (rerank_K) on ESCI test set.

Reuses cached top-100 CE artifacts (ce_top100_*.npy) so we only need to
compute the new encoder's per-(q, candidate) similarities.

Setups compared:
  K alone                  - distilled bi-encoder over BM25 top-100
  3-way + K (replaces CE)  - A+B+G+K equal-average over BM25 top-100
  CC3-100 (3-way only)     - reference fast-tier R@10
  CC4-100 (3-way + CE)     - reference quality-tier R@10
  4-way (A+B+G+K) min-max  - distilled student fused with 3-way like CE was
  3-way + 0.25*K mixer     - K used as a CE replacement at the same w_ce

If "3-way + 0.25*K" matches CC4-100 R@10 within ~0.1pp, distillation is a
deployable win: fast-tier latency at quality-tier accuracy.

Usage:
    python evaluation/eval_ce_distilled.py
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

import numpy as np  # noqa: E402

SCRIPT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INDEX_DIR = os.path.join(SCRIPT_DIR, "combined_index_us_minilm")
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
    if pos_e:
        e1 = sum(1 for p in top_k[:1] if p in pos_e) / min(1, len(pos_e))
        e3 = sum(1 for p in top_k[:3] if p in pos_e) / min(3, len(pos_e))
    else:
        e1 = e3 = None
    return recall, ndcg, e1, e3


def normalize_per_query(scores, valid_mask):
    out = scores.copy()
    for qi in range(out.shape[0]):
        v = out[qi, valid_mask[qi]]
        if v.size == 0:
            continue
        lo, hi = float(v.min()), float(v.max())
        out[qi, valid_mask[qi]] = (v - lo) / max(hi - lo, 1e-8)
    return out


def aggregate(scores, candidate_pos, valid, qids, qrels, faiss_pos_to_pid):
    rs, ns, e1s, e3s = [], [], [], []
    for qi, qid in enumerate(qids):
        if not valid[qi].any():
            continue
        s = scores[qi].copy()
        s[~valid[qi]] = -np.inf
        order = np.argsort(-s)[:K_EVAL]
        ordering = [faiss_pos_to_pid[int(candidate_pos[qi, int(j)])] for j in order]
        m = per_query_metrics(ordering, qrels[qid])
        if m is None:
            continue
        r, n, e1, e3 = m
        rs.append(r)
        ns.append(n)
        if e1 is not None:
            e1s.append(e1)
            e3s.append(e3)
    return (
        np.mean(rs) if rs else 0.0,
        np.mean(ns) if ns else 0.0,
        np.mean(e1s) if e1s else 0.0,
        np.mean(e3s) if e3s else 0.0,
    )


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

    print("loading cached top-100 artifacts...", flush=True)
    candidate_pos = np.load(os.path.join(INDEX_DIR, "ce_top100_candidates.npy"))
    sumsim = np.load(os.path.join(INDEX_DIR, "ce_top100_sumsim.npy"))
    ce_scores = np.load(os.path.join(INDEX_DIR, "ce_top100_scores.npy"))
    sims_a = np.load(os.path.join(INDEX_DIR, "ce_top100_sims_a.npy"))
    sims_b = np.load(os.path.join(INDEX_DIR, "ce_top100_sims_b.npy"))
    sims_g = np.load(os.path.join(INDEX_DIR, "ce_top100_sims_g.npy"))
    valid = candidate_pos >= 0

    # Encode queries with the distilled model + load product vecs.
    print("encoding test queries with rerank_K (distilled)...", flush=True)
    t0 = time.time()
    qv_k = encode_subproc(os.path.join(SCRIPT_DIR, "query_model_us_ce_distilled"), queries)
    print(f"  done in {time.time() - t0:.0f}s", flush=True)

    pv_k_path = os.path.join(INDEX_DIR, "rerank_K.vecs.fp16.npy")
    pv_k = np.load(pv_k_path).astype(np.float32)
    print(f"  loaded pv_k {pv_k.shape}", flush=True)

    # Compute per-encoder K cosines on the same candidate set as cached.
    print("computing rerank_K cosines on top-100 candidates...", flush=True)
    sims_k = np.zeros_like(sumsim)
    for qi in range(len(qids)):
        idx = candidate_pos[qi]
        good = idx >= 0
        if not good.any():
            continue
        sims_k[qi, good] = pv_k[idx[good]] @ qv_k[qi]

    # Normalize for fusion setups.
    nm_sum3 = normalize_per_query(sumsim, valid)  # 3-way A+B+G mean
    nm_ce = normalize_per_query(ce_scores, valid)
    nm_k = normalize_per_query(sims_k, valid)
    nm_a = normalize_per_query(sims_a, valid)
    nm_b = normalize_per_query(sims_b, valid)
    nm_g = normalize_per_query(sims_g, valid)

    # Build the eval table.
    setups = []
    setups.append(("CC3-100 (3-way A+B+G only)", sumsim))  # fast-tier reference
    setups.append(("CC4-100 (3-way + CE @ w=0.25)", 0.75 * nm_sum3 + 0.25 * nm_ce))  # quality ref
    setups.append(("K alone (distilled student)", sims_k))
    setups.append(("3-way + K @ w=0.25 minmax", 0.75 * nm_sum3 + 0.25 * nm_k))
    setups.append(("3-way + K @ w=0.50 minmax", 0.50 * nm_sum3 + 0.50 * nm_k))
    setups.append(("4-way A+B+G+K equal mean (raw)", (sims_a + sims_b + sims_g + sims_k) / 4))
    setups.append(("4-way A+B+G+K equal mean (minmax)", (nm_a + nm_b + nm_g + nm_k) / 4))
    setups.append(("5-way A+B+G+K+CE (minmax)", (nm_a + nm_b + nm_g + nm_k + nm_ce) / 5))
    setups.append(
        (
            "3-way + K @ w=0.25 + CE @ w=0.25 (so 0.5/0.25/0.25)",
            0.5 * nm_sum3 + 0.25 * nm_k + 0.25 * nm_ce,
        )
    )

    print(f"\n{'setup':<55} | {'R@10':>7} {'nDCG@10':>9} {'E@1':>7} {'E@3':>7}", flush=True)
    print("-" * 100)
    for label, scores in setups:
        r, n, e1, e3 = aggregate(scores, candidate_pos, valid, qids, qrels, faiss_pos_to_pid)
        print(f"{label:<55} | {r:>6.2%} {n:>9.4f} {e1:>6.2%} {e3:>6.2%}", flush=True)


if __name__ == "__main__":
    main()
