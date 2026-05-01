#!/usr/bin/env python3
"""Cross-encoder rerank as a 4th stage on CC3-50's top-K candidates.

Uses LiYuan/Amazon-Cup-Cross-Encoder-Regression — the same RoBERTa CE we
used for bag construction. Trained on ESCI labels with full attention,
so it has supervision the bi-encoder rerankers don't.

For each test query, take CC3-50's top-K candidates (after the bi-encoder
ensemble). Score (query, title) pairs with the CE. Re-sort.

Saves the per-(qi, candidate) CE scores to combined_index_us_minilm/
ce_scores_top50.npy (shape 22458x50, fp32) so downstream fusion sweeps
can reuse them.

Usage:
    python evaluation/eval_ce_rerank.py
    python evaluation/eval_ce_rerank.py --top-k 20 --batch-size 64
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse  # noqa: E402
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ce-model", default="LiYuan/Amazon-Cup-Cross-Encoder-Regression")
    ap.add_argument("--top-k", type=int, default=50, help="rerank pool size")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--save-scores", default=None, help="path to save fp32 ce_scores npy")
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
    print(f"  {len(qids):,} eval queries, top-{args.top_k} per query", flush=True)

    # Step 1: build CC3-{top_k} candidate pool from cached vecs.
    print("\nencoding queries with rerank_a/b/g...", flush=True)
    qv_a = encode_subproc(os.path.join(SCRIPT_DIR, "query_model_us_full_6m_mnrl"), queries)
    qv_b = encode_subproc(os.path.join(SCRIPT_DIR, "query_model_us_qrels_mnrl_hardneg"), queries)
    qv_g = encode_subproc(os.path.join(SCRIPT_DIR, "query_model_us_esci_supervised"), queries)

    print("loading product vecs + bm25s top-200...", flush=True)
    pv_a = np.load(os.path.join(INDEX_DIR, "rerank_A.vecs.fp16.npy")).astype(np.float32)
    pv_b = np.load(os.path.join(INDEX_DIR, "rerank_B.vecs.fp16.npy")).astype(np.float32)
    pv_g = np.load(os.path.join(INDEX_DIR, "rerank_G.vecs.fp16.npy")).astype(np.float32)
    I_bm25_200 = np.load(os.path.join(INDEX_DIR, "bm25s_top200.npy"))

    print(f"computing CC3-{args.top_k} candidate pools (sumsim sort)...", flush=True)
    K_RET = args.top_k
    candidate_pos = np.full((len(qids), K_RET), -1, dtype=np.int64)
    sumsim_scores = np.zeros((len(qids), K_RET), dtype=np.float32)
    for qi in range(len(qids)):
        positions = [int(p) for p in I_bm25_200[qi, :K_RET] if p >= 0]
        if not positions:
            continue
        sims = (
            pv_a[positions] @ qv_a[qi] + pv_b[positions] @ qv_b[qi] + pv_g[positions] @ qv_g[qi]
        ) / 3
        order = np.argsort(-sims)
        for j, idx in enumerate(order):
            if j >= K_RET:
                break
            candidate_pos[qi, j] = positions[int(idx)]
            sumsim_scores[qi, j] = sims[int(idx)]

    # Step 2: CE-score each candidate.
    from sentence_transformers import CrossEncoder

    device = (
        "mps"
        if torch.backends.mps.is_available()
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"\nloading CE model {args.ce_model} on {device}...", flush=True)
    ce = CrossEncoder(args.ce_model, device=device)
    print(
        f"  loaded ({sum(p.numel() for p in ce.model.parameters()) / 1e6:.0f}M params)", flush=True
    )

    print(f"scoring {len(qids):,} queries x {K_RET} candidates...", flush=True)
    ce_scores = np.zeros((len(qids), K_RET), dtype=np.float32)
    pairs_buf = []
    locs_buf = []
    n_pairs_total = len(qids) * K_RET
    n_done = 0
    t0 = time.time()
    for qi, _qid in enumerate(qids):
        q = queries[qi]
        for j in range(K_RET):
            pos = int(candidate_pos[qi, j])
            if pos < 0:
                continue
            pairs_buf.append((q, index_titles[pos]))
            locs_buf.append((qi, j))
        if len(pairs_buf) >= 4096 or qi == len(qids) - 1:
            scores = ce.predict(pairs_buf, batch_size=args.batch_size, show_progress_bar=False)
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

    if args.save_scores:
        np.save(args.save_scores, ce_scores)
        print(f"\nsaved CE scores to {args.save_scores}: shape={ce_scores.shape}", flush=True)
    # Also save candidate positions and sumsim scores so fusion sweeps can reuse.
    np.save(os.path.join(INDEX_DIR, "ce_candidates.npy"), candidate_pos)
    np.save(os.path.join(INDEX_DIR, "ce_sumsim.npy"), sumsim_scores)
    np.save(os.path.join(INDEX_DIR, "ce_scores.npy"), ce_scores)
    print(f"saved {INDEX_DIR}/ce_candidates.npy, ce_sumsim.npy, ce_scores.npy", flush=True)

    # Step 3: evaluate CE-only re-rank vs CC3 baseline.
    print(f"\n{'setup':<32} | {'R@10':>7} {'nDCG@10':>9} {'E@1':>7} {'E@3':>7}", flush=True)
    print("-" * 76)

    for label, score_matrix in [
        (f"CC3-{K_RET} (sumsim baseline)", sumsim_scores),
        (f"CE alone over CC3-{K_RET} top", ce_scores),
    ]:
        rs, ns, e1s, e3s = [], [], [], []
        for qi, qid in enumerate(qids):
            valid = candidate_pos[qi] >= 0
            if not valid.any():
                continue
            scores = score_matrix[qi].copy()
            scores[~valid] = -np.inf
            order = np.argsort(-scores)[:K_EVAL]
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
        print(
            f"{label:<32} | {np.mean(rs):>6.2%} {np.mean(ns):>9.4f} "
            f"{np.mean(e1s):>6.2%} {np.mean(e3s):>6.2%}",
            flush=True,
        )


if __name__ == "__main__":
    main()
