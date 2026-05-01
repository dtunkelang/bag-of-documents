#!/usr/bin/env python3
"""Probe 1: CE rerank on BM25 top-100, bypassing the bi-encoder first stage.

Currently CC4-50 = bm25s top-50 -> 3-way bi-encoder filter -> CE fusion.
The CE only sees what the bi-encoder ensemble surfaced. Question: does
CE on the unfiltered BM25 top-100 (more candidates, no bi-encoder gate)
catch additional relevant products?

Computes:
  - per-encoder cosines on BM25 top-100 (rerank_a, rerank_b, rerank_g)
  - CE scores on BM25 top-100
Saves all to combined_index_us_minilm/ for reuse.

Evaluates these setups (by R@10, nDCG@10, E@1, E@3 on 22,458 ESCI queries):
  - CC3-K (3-way sumsim alone) for K in {30, 50, 75, 100}
  - CE alone (top-K by CE only) for K in {30, 50, 75, 100}
  - CE + sumsim fusion (w_ce=0.25 minmax) for K in {30, 50, 75, 100}
  - CE + sumsim fusion (w_ce=0.50 minmax) for K in {30, 50, 75, 100}

Usage:
    python evaluation/eval_ce_top100.py
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
import torch  # noqa: E402

SCRIPT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INDEX_DIR = os.path.join(SCRIPT_DIR, "combined_index_us_minilm")

K_EVAL = 10
TOP_K_POOL = 100  # CE-score the BM25 top-100


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


def aggregate_metrics(scores_per_q, candidate_pos, valid, qids, qrels, faiss_pos_to_pid, K=None):
    """Take top-10 by `scores_per_q`, restricted to valid candidates and
    (optionally) the first K positions in candidate_pos."""
    rs, ns, e1s, e3s = [], [], [], []
    for qi, qid in enumerate(qids):
        if not valid[qi].any():
            continue
        s = scores_per_q[qi].copy()
        s[~valid[qi]] = -np.inf
        if K is not None and s.shape[0] > K:
            s[K:] = -np.inf
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


def normalize_per_query(scores, valid_mask):
    """Per-query min-max over the valid candidate set."""
    out = scores.copy()
    for qi in range(out.shape[0]):
        v = out[qi, valid_mask[qi]]
        if v.size == 0:
            continue
        lo, hi = float(v.min()), float(v.max())
        out[qi, valid_mask[qi]] = (v - lo) / max(hi - lo, 1e-8)
    return out


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

    print("loading product vecs + bm25s top-100 candidates...", flush=True)
    pv_a = np.load(os.path.join(INDEX_DIR, "rerank_A.vecs.fp16.npy")).astype(np.float32)
    pv_b = np.load(os.path.join(INDEX_DIR, "rerank_B.vecs.fp16.npy")).astype(np.float32)
    pv_g = np.load(os.path.join(INDEX_DIR, "rerank_G.vecs.fp16.npy")).astype(np.float32)
    I_bm25_200 = np.load(os.path.join(INDEX_DIR, "bm25s_top200.npy"))
    candidate_pos = I_bm25_200[:, :TOP_K_POOL].astype(np.int64).copy()
    valid = candidate_pos >= 0
    print(f"  candidate_pos shape={candidate_pos.shape}", flush=True)

    print("encoding queries with rerank_a, rerank_b, rerank_g...", flush=True)
    t0 = time.time()
    qv_a = encode_subproc(os.path.join(SCRIPT_DIR, "query_model_us_full_6m_mnrl"), queries)
    qv_b = encode_subproc(os.path.join(SCRIPT_DIR, "query_model_us_qrels_mnrl_hardneg"), queries)
    qv_g = encode_subproc(os.path.join(SCRIPT_DIR, "query_model_us_esci_supervised"), queries)
    print(f"  done in {time.time() - t0:.0f}s", flush=True)

    # Compute per-encoder cosines on the top-100 candidates.
    print("computing per-encoder cosines on top-100 candidates...", flush=True)
    t0 = time.time()
    sims_a = np.zeros((len(qids), TOP_K_POOL), dtype=np.float32)
    sims_b = np.zeros((len(qids), TOP_K_POOL), dtype=np.float32)
    sims_g = np.zeros((len(qids), TOP_K_POOL), dtype=np.float32)
    for qi in range(len(qids)):
        positions = candidate_pos[qi]
        good = positions >= 0
        if not good.any():
            continue
        idx = positions[good]
        sims_a[qi, good] = pv_a[idx] @ qv_a[qi]
        sims_b[qi, good] = pv_b[idx] @ qv_b[qi]
        sims_g[qi, good] = pv_g[idx] @ qv_g[qi]
    sumsim = (sims_a + sims_b + sims_g) / 3
    print(f"  done in {time.time() - t0:.0f}s", flush=True)

    # CE-score each (query, title) pair on top-100.
    from sentence_transformers import CrossEncoder

    device = (
        "mps"
        if torch.backends.mps.is_available()
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    ce_model_name = "LiYuan/Amazon-Cup-Cross-Encoder-Regression"
    print(f"\nloading CE model {ce_model_name} on {device}...", flush=True)
    ce = CrossEncoder(ce_model_name, device=device)

    print(f"scoring {len(qids):,} queries x {TOP_K_POOL} candidates...", flush=True)
    ce_scores = np.zeros((len(qids), TOP_K_POOL), dtype=np.float32)
    pairs_buf = []
    locs_buf = []
    n_pairs_total = int(valid.sum())
    n_done = 0
    t0 = time.time()
    for qi, _qid in enumerate(qids):
        q = queries[qi]
        for j in range(TOP_K_POOL):
            pos = int(candidate_pos[qi, j])
            if pos < 0:
                continue
            pairs_buf.append((q, index_titles[pos]))
            locs_buf.append((qi, j))
        if len(pairs_buf) >= 4096 or qi == len(qids) - 1:
            scores = ce.predict(pairs_buf, batch_size=64, show_progress_bar=False)
            for (qi2, j2), sc in zip(locs_buf, scores):
                ce_scores[qi2, j2] = float(sc)
            n_done += len(pairs_buf)
            elapsed = time.time() - t0
            rate = n_done / max(elapsed, 1e-3)
            eta = (n_pairs_total - n_done) / max(rate, 1e-3)
            print(
                f"  {n_done:,}/{n_pairs_total:,} pairs ({n_done / n_pairs_total:.1%}) "
                f"@ {rate:.0f} pairs/s, eta {eta / 60:.1f} min",
                flush=True,
            )
            pairs_buf.clear()
            locs_buf.clear()

    print("\nsaving artifacts...", flush=True)
    np.save(os.path.join(INDEX_DIR, "ce_top100_candidates.npy"), candidate_pos)
    np.save(os.path.join(INDEX_DIR, "ce_top100_sumsim.npy"), sumsim)
    np.save(os.path.join(INDEX_DIR, "ce_top100_sims_a.npy"), sims_a)
    np.save(os.path.join(INDEX_DIR, "ce_top100_sims_b.npy"), sims_b)
    np.save(os.path.join(INDEX_DIR, "ce_top100_sims_g.npy"), sims_g)
    np.save(os.path.join(INDEX_DIR, "ce_top100_scores.npy"), ce_scores)

    # Evaluate.
    print(f"\n{'setup':<40} | {'R@10':>7} {'nDCG@10':>9} {'E@1':>7} {'E@3':>7}", flush=True)
    print("-" * 84)
    nm_sum = normalize_per_query(sumsim, valid)
    nm_ce = normalize_per_query(ce_scores, valid)

    for K in [30, 50, 75, 100]:
        valid_k = valid.copy()
        valid_k[:, K:] = False

        def eval_with(score, vk=valid_k):
            score = score.copy()
            score[~vk] = -np.inf
            return aggregate_metrics(score, candidate_pos, vk, qids, qrels, faiss_pos_to_pid)

        r, n, e1, e3 = eval_with(sumsim)
        print(
            f"{'CC3-' + str(K) + ' (sumsim only)':<40} | {r:>6.2%} {n:>9.4f} {e1:>6.2%} {e3:>6.2%}",
            flush=True,
        )
        r, n, e1, e3 = eval_with(ce_scores)
        print(
            f"{'CE alone over BM25 top-' + str(K):<40} | {r:>6.2%} {n:>9.4f} {e1:>6.2%} {e3:>6.2%}",
            flush=True,
        )
        for w in (0.25, 0.5):
            fused = (1 - w) * nm_sum + w * nm_ce
            r, n, e1, e3 = eval_with(fused)
            label = f"CC4-{K} (w_ce={w}) fused"
            print(
                f"{label:<40} | {r:>6.2%} {n:>9.4f} {e1:>6.2%} {e3:>6.2%}",
                flush=True,
            )


if __name__ == "__main__":
    main()
