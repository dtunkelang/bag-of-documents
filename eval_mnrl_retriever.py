#!/usr/bin/env python3
"""Evaluate the 6M-MNRL model as primary retriever, with optional hardneg rerank.

Setups computed in one pass on the 22,458-query ESCI test set:
  A. base alone                          (canonical baseline)
  B. 6M-MNRL retriever alone             (no rerank)
  C. base + 6M+hardneg ensemble rerank   (current deployable, +2.75pp R@10)
  D. 6M-MNRL retriever + hardneg rerank  (the candidate new architecture)
  E. 6M-MNRL retriever + 6M+hardneg ensemble rerank (sanity)

Metrics: R@10 (E+S relevant), nDCG@10 (E=1.0/S=0.1/C=0.01/I=0), E@1, E@3.

Retrieval is brute-force top-100 against cached fp16 product matrices
(combined_index_us_minilm/rerank_{A,B}.vecs.fp16.npy) — ~100s per model.
"""

import json
import math
import os
import statistics
import subprocess
import time
from collections import defaultdict

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np  # noqa: E402

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INDEX_DIR = os.path.join(SCRIPT_DIR, "combined_index_us_minilm")


def encode_subproc(model_path, queries):
    code = f"""
import os, json, sys
os.environ['TOKENIZERS_PARALLELISM']='false'
os.environ['OMP_NUM_THREADS']='1'
os.chdir({SCRIPT_DIR!r})
import numpy as np, torch
from sentence_transformers import SentenceTransformer
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


def brute_top_k(q_vecs, p_vecs, k):
    """Top-K cosine via batched matmul. p_vecs assumed L2-normalized."""
    n = q_vecs.shape[0]
    out_idx = np.zeros((n, k), dtype=np.int64)
    BATCH = 128
    for start in range(0, n, BATCH):
        end = min(start + BATCH, n)
        sims = q_vecs[start:end] @ p_vecs.T  # (B, P)
        top = np.argpartition(-sims, k, axis=1)[:, :k]
        for i in range(end - start):
            row_sims = sims[i, top[i]]
            order = np.argsort(-row_sims)
            out_idx[start + i] = top[i, order]
        if (start // BATCH) % 20 == 0:
            print(f"    brute_top_k progress {end}/{n}", flush=True)
    return out_idx


GAIN = {3: 1.0, 2: 0.1, 1: 0.01, 0: 0.0}


def metrics_for(retrieved_pids, qrels_q, k_eval=10):
    e_pids = {p for p, g in qrels_q.items() if g == 3}
    es_pids = {p for p, g in qrels_q.items() if g >= 2}
    out = {}
    if es_pids:
        top_k = retrieved_pids[:k_eval]
        out["recall"] = sum(1 for p in top_k if p in es_pids) / len(es_pids)
        gains = [GAIN.get(qrels_q.get(p, 0), 0.0) for p in top_k]
        dcg = sum(g / math.log2(i + 2) for i, g in enumerate(gains))
        ideal = sorted(qrels_q.values(), reverse=True)[:k_eval]
        idcg = sum(GAIN.get(g, 0) / math.log2(i + 2) for i, g in enumerate(ideal))
        out["ndcg"] = dcg / idcg if idcg > 0 else 0.0
    if e_pids:
        out["e_at_1"] = 1.0 if retrieved_pids and retrieved_pids[0] in e_pids else 0.0
        top3 = retrieved_pids[:3]
        out["e_at_3"] = sum(1 for p in top3 if p in e_pids) / min(3, len(e_pids))
    return out


def aggregate(orderings_by_setup, qids, qrels_by_qid):
    summary = {}
    for setup_name, orderings in orderings_by_setup.items():
        recalls, ndcgs, e1s, e3s = [], [], [], []
        for qi, qid in enumerate(qids):
            m = metrics_for(orderings[qi], qrels_by_qid[qid])
            if "recall" in m:
                recalls.append(m["recall"])
                ndcgs.append(m["ndcg"])
            if "e_at_1" in m:
                e1s.append(m["e_at_1"])
                e3s.append(m["e_at_3"])
        summary[setup_name] = {
            "n_es": len(recalls),
            "n_e": len(e1s),
            "R@10": statistics.mean(recalls) if recalls else 0,
            "nDCG@10": statistics.mean(ndcgs) if ndcgs else 0,
            "E@1": statistics.mean(e1s) if e1s else 0,
            "E@3": statistics.mean(e3s) if e3s else 0,
        }
    return summary


def main():
    print("loading qrels + queries + product map...", flush=True)
    qrels = defaultdict(dict)
    with open(os.path.join(SCRIPT_DIR, "esci_us_data/test_qrels.jsonl")) as f:
        for line in f:
            r = json.loads(line)
            qrels[r["query_id"]][r["product_id"]] = r["relevance"]
    qrels = dict(qrels)
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

    K_RETRIEVE = 100
    K_EVAL = 10

    # Encode queries with 3 models
    print("\nencoding queries...", flush=True)
    t0 = time.time()
    qv_base = encode_subproc("all-MiniLM-L6-v2", queries)
    print(f"  base: {time.time() - t0:.0f}s", flush=True)
    t0 = time.time()
    qv_a = encode_subproc(os.path.join(SCRIPT_DIR, "query_model_us_full_6m_mnrl"), queries)
    print(f"  6M-MNRL: {time.time() - t0:.0f}s", flush=True)
    t0 = time.time()
    qv_b = encode_subproc(os.path.join(SCRIPT_DIR, "query_model_us_qrels_mnrl_hardneg"), queries)
    print(f"  hardneg: {time.time() - t0:.0f}s", flush=True)

    # Optional third reranker: query_model_amazon (cosine-distilled).
    rerank_c_vecs_path = os.path.join(INDEX_DIR, "rerank_C.vecs.fp16.npy")
    pv_c = None
    qv_c = None
    if os.path.exists(rerank_c_vecs_path):
        t0 = time.time()
        qv_c = encode_subproc(os.path.join(SCRIPT_DIR, "query_model_amazon"), queries)
        print(f"  amazon (rerank_C): {time.time() - t0:.0f}s", flush=True)

    # Load cached product matrices
    print("\nloading cached product vecs...", flush=True)
    pv_a = np.load(os.path.join(INDEX_DIR, "rerank_A.vecs.fp16.npy")).astype(np.float32)
    pv_b = np.load(os.path.join(INDEX_DIR, "rerank_B.vecs.fp16.npy")).astype(np.float32)
    print(f"  pv_a: {pv_a.shape}  pv_b: {pv_b.shape}", flush=True)
    if qv_c is not None:
        pv_c = np.load(rerank_c_vecs_path).astype(np.float32)
        print(f"  pv_c: {pv_c.shape}", flush=True)

    # === Retrieve top-K_RETRIEVE per setup ===
    print("\nbase top-100 from MiniLM FAISS...", flush=True)
    t0 = time.time()
    I_base = faiss_top_k(qv_base, os.path.join(INDEX_DIR, "index.faiss"), K_RETRIEVE)
    print(f"  done {time.time() - t0:.0f}s", flush=True)

    print("6M-MNRL top-100 from cached_A...", flush=True)
    t0 = time.time()
    I_mnrl = brute_top_k(qv_a, pv_a, K_RETRIEVE)
    print(f"  done {time.time() - t0:.0f}s", flush=True)

    # Convert FAISS positions → pids per query, per setup
    def to_pids(I_arr):
        return [[faiss_pos_to_pid[int(p)] for p in row if p >= 0] for row in I_arr]

    base_pids = to_pids(I_base)
    mnrl_pids = to_pids(I_mnrl)

    # === Build orderings per setup ===
    orderings = {}
    # A: base alone, top-10 of its top-100
    orderings["A: base"] = [row[:K_EVAL] for row in base_pids]
    # B: 6M-MNRL alone, top-10 of its top-100
    orderings["B: 6M-MNRL retriever"] = [row[:K_EVAL] for row in mnrl_pids]

    # C: base + 6M+hardneg ensemble rerank
    print("\nbuilding C: base + ensemble rerank...", flush=True)
    c_orderings = []
    for qi in range(len(queries)):
        positions = [int(p) for p in I_base[qi] if p >= 0]
        if not positions:
            c_orderings.append([])
            continue
        cand_a = pv_a[positions]
        cand_b = pv_b[positions]
        avg = (cand_a @ qv_a[qi] + cand_b @ qv_b[qi]) / 2
        order = np.argsort(-avg)[:K_EVAL]
        c_orderings.append([base_pids[qi][int(j)] for j in order])
    orderings["C: base + ensemble rerank"] = c_orderings

    # D: 6M-MNRL retriever + hardneg rerank
    print("building D: 6M-MNRL + hardneg rerank...", flush=True)
    d_orderings = []
    for qi in range(len(queries)):
        positions = [int(p) for p in I_mnrl[qi] if p >= 0]
        if not positions:
            d_orderings.append([])
            continue
        cand_b = pv_b[positions]
        sims = cand_b @ qv_b[qi]
        order = np.argsort(-sims)[:K_EVAL]
        d_orderings.append([mnrl_pids[qi][int(j)] for j in order])
    orderings["D: 6M-MNRL + hardneg rerank"] = d_orderings

    # E: 6M-MNRL retriever + ensemble rerank
    print("building E: 6M-MNRL + ensemble rerank...", flush=True)
    e_orderings = []
    for qi in range(len(queries)):
        positions = [int(p) for p in I_mnrl[qi] if p >= 0]
        if not positions:
            e_orderings.append([])
            continue
        cand_a = pv_a[positions]
        cand_b = pv_b[positions]
        avg = (cand_a @ qv_a[qi] + cand_b @ qv_b[qi]) / 2
        order = np.argsort(-avg)[:K_EVAL]
        e_orderings.append([mnrl_pids[qi][int(j)] for j in order])
    orderings["E: 6M-MNRL + ensemble rerank"] = e_orderings

    # RRF fusion of base + 6M-MNRL retrieval. Standard RRF constant c=60.
    # F: RRF retrieval only (no rerank), top-10 of fused list.
    # G: RRF top-100 candidates → ensemble rerank (the apples-to-apples test
    #    of whether base retrieves anything additive over 6M-MNRL).
    RRF_C = 60
    print("building F/G: RRF(base, 6M-MNRL) ...", flush=True)
    f_orderings = []
    g_orderings = []
    for qi in range(len(queries)):
        rrf = {}
        for rank, p in enumerate(int(x) for x in I_base[qi] if x >= 0):
            rrf[p] = rrf.get(p, 0.0) + 1.0 / (rank + 1 + RRF_C)
        for rank, p in enumerate(int(x) for x in I_mnrl[qi] if x >= 0):
            rrf[p] = rrf.get(p, 0.0) + 1.0 / (rank + 1 + RRF_C)
        if not rrf:
            f_orderings.append([])
            g_orderings.append([])
            continue
        fused = sorted(rrf.items(), key=lambda kv: -kv[1])[:K_RETRIEVE]
        positions = [p for p, _ in fused]
        f_orderings.append([faiss_pos_to_pid[p] for p in positions[:K_EVAL]])

        cand_a = pv_a[positions]
        cand_b = pv_b[positions]
        avg = (cand_a @ qv_a[qi] + cand_b @ qv_b[qi]) / 2
        order = np.argsort(-avg)[:K_EVAL]
        g_orderings.append([faiss_pos_to_pid[positions[int(j)]] for j in order])
    orderings["F: RRF(base, 6M-MNRL) retrieval"] = f_orderings
    orderings["G: RRF(base, 6M-MNRL) + ensemble rerank"] = g_orderings

    # BM25 retrieval, if precomputed top-100 positions are available. Tests
    # whether a *non-dense* retriever supplies candidates that base+BoD miss
    # — the brand+entity hard regime where dense retrievers collapse.
    bm25_path = os.path.join(INDEX_DIR, "bm25_top100.npy")
    if os.path.exists(bm25_path):
        print("\nbuilding H/I/J: BM25 fusion setups...", flush=True)
        I_bm25 = np.load(bm25_path)
        bm25_pids = to_pids(I_bm25)
        # H: BM25 retrieval alone, top-10
        orderings["H: BM25 retrieval"] = [row[:K_EVAL] for row in bm25_pids]

        i_orderings = []
        j_orderings = []
        for qi in range(len(queries)):
            rrf2 = {}
            for rank, p in enumerate(int(x) for x in I_bm25[qi] if x >= 0):
                rrf2[p] = rrf2.get(p, 0.0) + 1.0 / (rank + 1 + RRF_C)
            for rank, p in enumerate(int(x) for x in I_mnrl[qi] if x >= 0):
                rrf2[p] = rrf2.get(p, 0.0) + 1.0 / (rank + 1 + RRF_C)
            rrf3 = dict(rrf2)
            for rank, p in enumerate(int(x) for x in I_base[qi] if x >= 0):
                rrf3[p] = rrf3.get(p, 0.0) + 1.0 / (rank + 1 + RRF_C)

            for source, dest in ((rrf2, i_orderings), (rrf3, j_orderings)):
                if not source:
                    dest.append([])
                    continue
                fused = sorted(source.items(), key=lambda kv: -kv[1])[:K_RETRIEVE]
                positions = [p for p, _ in fused]
                cand_a = pv_a[positions]
                cand_b = pv_b[positions]
                avg = (cand_a @ qv_a[qi] + cand_b @ qv_b[qi]) / 2
                order = np.argsort(-avg)[:K_EVAL]
                dest.append([faiss_pos_to_pid[positions[int(j)]] for j in order])
        orderings["I: RRF(BM25, 6M-MNRL) + ensemble rerank"] = i_orderings
        orderings["J: RRF(BM25, base, 6M-MNRL) + ensemble rerank"] = j_orderings

        # K: BM25-only candidate pool → ensemble rerank. Tests whether 6M-MNRL
        # retrieval contributes additive candidates over BM25 in the SOTA mix,
        # or whether all the lift in I comes from the rerank stack.
        print("building K: BM25 + ensemble rerank...", flush=True)
        k_orderings = []
        for qi in range(len(queries)):
            positions = [int(p) for p in I_bm25[qi] if p >= 0]
            if not positions:
                k_orderings.append([])
                continue
            cand_a = pv_a[positions]
            cand_b = pv_b[positions]
            avg = (cand_a @ qv_a[qi] + cand_b @ qv_b[qi]) / 2
            order = np.argsort(-avg)[:K_EVAL]
            k_orderings.append([faiss_pos_to_pid[positions[int(j)]] for j in order])
        orderings["K: BM25 + ensemble rerank"] = k_orderings

        # K-variants: explore the local maximum around BM25 + ensemble rerank.
        # L1, L2: each reranker alone — does the ensemble actually help?
        # M:     sumrank fusion instead of sumsim — does the fusion choice matter?
        print("building L/M variants on BM25 candidates...", flush=True)
        l1_orderings, l2_orderings, m_orderings = [], [], []
        for qi in range(len(queries)):
            positions = [int(p) for p in I_bm25[qi] if p >= 0]
            if not positions:
                l1_orderings.append([])
                l2_orderings.append([])
                m_orderings.append([])
                continue
            cand_a = pv_a[positions]
            cand_b = pv_b[positions]
            sims_a = cand_a @ qv_a[qi]
            sims_b = cand_b @ qv_b[qi]
            order_a = np.argsort(-sims_a)[:K_EVAL]
            order_b = np.argsort(-sims_b)[:K_EVAL]
            l1_orderings.append([faiss_pos_to_pid[positions[int(j)]] for j in order_a])
            l2_orderings.append([faiss_pos_to_pid[positions[int(j)]] for j in order_b])
            ranks_a = np.argsort(np.argsort(-sims_a))
            ranks_b = np.argsort(np.argsort(-sims_b))
            order_m = np.argsort(ranks_a + ranks_b)[:K_EVAL]
            m_orderings.append([faiss_pos_to_pid[positions[int(j)]] for j in order_m])
        orderings["L1: BM25 + 6M-MNRL only"] = l1_orderings
        orderings["L2: BM25 + hardneg only"] = l2_orderings
        orderings["M: BM25 + sumrank ensemble"] = m_orderings

        # N50, N200: K_retrieve sweep at 50 and 200 (vs K's 100). Requires
        # bm25_top200.npy with K_retrieve >= 200.
        bm25_top200_path = os.path.join(INDEX_DIR, "bm25_top200.npy")
        if os.path.exists(bm25_top200_path):
            print("building N: K_retrieve sweep on BM25 + ensemble rerank...", flush=True)
            I_bm25_200 = np.load(bm25_top200_path)
            for k_ret, label in (
                (40, "N40"),
                (50, "N50"),
                (60, "N60"),
                (75, "N75"),
                (200, "N200"),
            ):
                ords = []
                for qi in range(len(queries)):
                    positions = [int(p) for p in I_bm25_200[qi, :k_ret] if p >= 0]
                    if not positions:
                        ords.append([])
                        continue
                    cand_a = pv_a[positions]
                    cand_b = pv_b[positions]
                    avg = (cand_a @ qv_a[qi] + cand_b @ qv_b[qi]) / 2
                    order = np.argsort(-avg)[:K_EVAL]
                    ords.append([faiss_pos_to_pid[positions[int(j)]] for j in order])
                orderings[f"{label}: BM25 top-{k_ret} + ensemble rerank"] = ords

            # Q: min-max normalize each reranker's similarities to [0,1] before
            # averaging. Tests whether one reranker's wider sim range was
            # silently dominating the sumsim ensemble.
            print("building Q: min-max normalized ensemble rerank...", flush=True)
            q_orderings = []
            for qi in range(len(queries)):
                positions = [int(p) for p in I_bm25_200[qi, :100] if p >= 0]
                if not positions:
                    q_orderings.append([])
                    continue
                cand_a = pv_a[positions]
                cand_b = pv_b[positions]
                sims_a = cand_a @ qv_a[qi]
                sims_b = cand_b @ qv_b[qi]

                def _minmax(x):
                    lo, hi = float(x.min()), float(x.max())
                    return (x - lo) / (hi - lo) if hi > lo else np.zeros_like(x)

                avg = (_minmax(sims_a) + _minmax(sims_b)) / 2
                order = np.argsort(-avg)[:K_EVAL]
                q_orderings.append([faiss_pos_to_pid[positions[int(j)]] for j in order])
            orderings["Q: BM25 top-100 + min-max ensemble"] = q_orderings

            # R: 3-way ensemble rerank with rerank_a (6M-MNRL), rerank_b
            # (hardneg), rerank_c (cosine-distilled query_model_amazon).
            # Tests whether a third encoder with a *different* training signal
            # (cosine loss, smaller construction set) adds orthogonal lift to
            # the SOTA. Run at K_retrieve=40 and 50 to bracket the N-sweep peak.
            if pv_c is not None and qv_c is not None:
                print("building R: BM25 + 3-way ensemble rerank...", flush=True)
                for k_ret, label in ((40, "R40"), (50, "R50"), (100, "R100")):
                    ords = []
                    for qi in range(len(queries)):
                        positions = [int(p) for p in I_bm25_200[qi, :k_ret] if p >= 0]
                        if not positions:
                            ords.append([])
                            continue
                        cand_a = pv_a[positions]
                        cand_b = pv_b[positions]
                        cand_c = pv_c[positions]
                        avg = (cand_a @ qv_a[qi] + cand_b @ qv_b[qi] + cand_c @ qv_c[qi]) / 3
                        order = np.argsort(-avg)[:K_EVAL]
                        ords.append([faiss_pos_to_pid[positions[int(j)]] for j in order])
                    orderings[f"{label}: BM25 top-{k_ret} + 3-way rerank"] = ords
        else:
            print(f"\nskipping N: {bm25_top200_path} not found", flush=True)

        # P: BM25 with default tokenizer (no stem) → ensemble rerank.
        bm25_default_path = os.path.join(INDEX_DIR, "bm25_default_top200.npy")
        if os.path.exists(bm25_default_path):
            print("building P: default-tokenizer BM25 + ensemble rerank...", flush=True)
            I_bm25_def = np.load(bm25_default_path)
            p_orderings = []
            for qi in range(len(queries)):
                positions = [int(p) for p in I_bm25_def[qi, :100] if p >= 0]
                if not positions:
                    p_orderings.append([])
                    continue
                cand_a = pv_a[positions]
                cand_b = pv_b[positions]
                avg = (cand_a @ qv_a[qi] + cand_b @ qv_b[qi]) / 2
                order = np.argsort(-avg)[:K_EVAL]
                p_orderings.append([faiss_pos_to_pid[positions[int(j)]] for j in order])
            orderings["P: BM25 (default tok) + ensemble rerank"] = p_orderings
        else:
            print(f"\nskipping P: {bm25_default_path} not found", flush=True)
    else:
        print(f"\nskipping H/I/J: {bm25_path} not found", flush=True)

    summary = aggregate(orderings, qids, qrels)

    print(f"\n\n{'=' * 80}")
    print("=== Summary (22,458 ESCI test queries) ===")
    print(f"{'=' * 80}")
    print(f"{'setup':<38} {'R@10':>8} {'nDCG@10':>9} {'E@1':>8} {'E@3':>8}")
    print("-" * 80)
    for name, r in summary.items():
        print(
            f"{name:<38} {r['R@10']:>8.2%} {r['nDCG@10']:>9.4f} {r['E@1']:>8.2%} {r['E@3']:>8.2%}"
        )


if __name__ == "__main__":
    main()
