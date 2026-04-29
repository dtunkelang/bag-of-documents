#!/usr/bin/env python3
"""Per-query oracle routing: upper bound on what a query-router could ever
achieve by picking between A (base MiniLM retrieval) and K (BM25 + ensemble
rerank) per query.

Reuses the same encoders and BM25 top-100 cache as eval_per_query_bins.py.
For each query, computes both A's and K's R@10/E@1, then aggregates under
three policies:
  - A only:                  base MiniLM retrieval
  - K only:                  BM25 + ensemble rerank (current shipped SOTA)
  - oracle pick (per query): max(A R@10, K R@10) — upper bound for any router

If oracle is within ~0.2pp of K, routing has no headroom and the work to
build a router is wasted.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
import math
import os
import statistics
import subprocess
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
    _, I = idx.search(q_vecs.astype(np.float32), k)
    return I


GAIN = {3: 1.0, 2: 0.1, 1: 0.01, 0: 0.0}


def per_query_metrics(retrieved_pids, qrels_q, k_eval=10):
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
    else:
        out["recall"] = 0.0
        out["ndcg"] = 0.0
    if e_pids:
        out["e_at_1"] = 1.0 if retrieved_pids and retrieved_pids[0] in e_pids else 0.0
        top3 = retrieved_pids[:3]
        out["e_at_3"] = sum(1 for p in top3 if p in e_pids) / min(3, len(e_pids))
    return out


def main():
    print("loading qrels + queries + product map...", flush=True)
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

    K_RETRIEVE = 100
    K_EVAL = 10

    print("encoding queries with base, rerank_a, rerank_b...", flush=True)
    qv_base = encode_subproc("all-MiniLM-L6-v2", queries)
    qv_a = encode_subproc(os.path.join(SCRIPT_DIR, "query_model_us_full_6m_mnrl"), queries)
    qv_b = encode_subproc(os.path.join(SCRIPT_DIR, "query_model_us_qrels_mnrl_hardneg"), queries)

    print("loading cached product vecs + BM25 top-100...", flush=True)
    pv_a = np.load(os.path.join(INDEX_DIR, "rerank_A.vecs.fp16.npy")).astype(np.float32)
    pv_b = np.load(os.path.join(INDEX_DIR, "rerank_B.vecs.fp16.npy")).astype(np.float32)
    I_bm25 = np.load(os.path.join(INDEX_DIR, "bm25_top100.npy"))

    print("base FAISS top-100...", flush=True)
    import faiss

    base_idx = faiss.read_index(os.path.join(INDEX_DIR, "index.faiss"))
    if hasattr(base_idx, "hnsw"):
        base_idx.hnsw.efSearch = 128
    D_base, I_base = base_idx.search(qv_base.astype(np.float32), K_RETRIEVE)
    metric = base_idx.metric_type
    base_top1_sim = []
    for d in D_base[:, 0]:
        sim = float(d) if metric == faiss.METRIC_INNER_PRODUCT else float(1 - d / 2)
        base_top1_sim.append(sim)
    base_pids = [[faiss_pos_to_pid[int(p)] for p in row if p >= 0] for row in I_base]

    a_order = [row[:K_EVAL] for row in base_pids]
    k_order = []
    for qi in range(len(queries)):
        positions = [int(p) for p in I_bm25[qi] if p >= 0]
        if not positions:
            k_order.append([])
            continue
        sims = (pv_a[positions] @ qv_a[qi] + pv_b[positions] @ qv_b[qi]) / 2
        order = np.argsort(-sims)[:K_EVAL]
        k_order.append([faiss_pos_to_pid[positions[int(j)]] for j in order])

    # Per-query: which lane wins?
    print("computing per-query lane comparison...", flush=True)
    a_recall, k_recall = [], []
    a_e1, k_e1 = [], []
    oracle_recall, oracle_e1 = [], []
    n_a_wins = 0
    n_k_wins = 0
    n_ties = 0
    for qi, qid in enumerate(qids):
        ma = per_query_metrics(a_order[qi], qrels[qid])
        mk = per_query_metrics(k_order[qi], qrels[qid])
        a_recall.append(ma["recall"])
        k_recall.append(mk["recall"])
        oracle_recall.append(max(ma["recall"], mk["recall"]))
        if "e_at_1" in ma:
            a_e1.append(ma["e_at_1"])
            k_e1.append(mk.get("e_at_1", 0.0))
            oracle_e1.append(max(ma["e_at_1"], mk.get("e_at_1", 0.0)))
        if mk["recall"] > ma["recall"]:
            n_k_wins += 1
        elif ma["recall"] > mk["recall"]:
            n_a_wins += 1
        else:
            n_ties += 1

    n = len(a_recall)
    print("\n" + "=" * 80)
    print(f"=== Oracle routing analysis on {n:,} queries (E+S relevant) ===")
    print("=" * 80)
    print(f"{'policy':<28} {'R@10':>10} {'E@1':>10}")
    print("-" * 80)
    print(
        f"{'A only (base)':<28} {statistics.mean(a_recall):>10.4f} {statistics.mean(a_e1):>10.4f}"
    )
    print(
        f"{'K only (BM25+rerank)':<28} {statistics.mean(k_recall):>10.4f} {statistics.mean(k_e1):>10.4f}"
    )
    print(
        f"{'oracle max(A, K) per query':<28} {statistics.mean(oracle_recall):>10.4f} {statistics.mean(oracle_e1):>10.4f}"
    )
    print()
    print(
        f"per-query lane winners (R@10):  K wins {n_k_wins:,} ({n_k_wins / n:.1%}),  "
        f"A wins {n_a_wins:,} ({n_a_wins / n:.1%}),  ties {n_ties:,} ({n_ties / n:.1%})"
    )
    headroom = (statistics.mean(oracle_recall) - statistics.mean(k_recall)) * 100
    print(
        f"\nHEADROOM: oracle - K = {headroom:+.2f}pp R@10. "
        f"This is the upper bound for any per-query router."
    )

    # Threshold sweep on base-FAISS top-1 cosine similarity as routing signal:
    # if base is confident (high top-1 sim), use A; otherwise use K.
    print("\n" + "=" * 80)
    print("=== Router: base-FAISS top-1 cosine similarity threshold sweep ===")
    print("=" * 80)
    print(f"{'threshold':<12} {'%->A':>8} {'R@10':>10} {'E@1':>10} {'gap to K':>12}")
    print("-" * 80)
    base_top1_sim_arr = np.array(base_top1_sim)
    for thr in [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]:
        chosen_recall = []
        chosen_e1 = []
        n_route_a = 0
        for qi, qid in enumerate(qids):
            ma = per_query_metrics(a_order[qi], qrels[qid])
            mk = per_query_metrics(k_order[qi], qrels[qid])
            if base_top1_sim_arr[qi] >= thr:
                chosen_recall.append(ma["recall"])
                n_route_a += 1
                if "e_at_1" in ma:
                    chosen_e1.append(ma["e_at_1"])
            else:
                chosen_recall.append(mk["recall"])
                if "e_at_1" in mk:
                    chosen_e1.append(mk["e_at_1"])
        rec = statistics.mean(chosen_recall)
        e1 = statistics.mean(chosen_e1) if chosen_e1 else 0.0
        gap = (rec - statistics.mean(k_recall)) * 100
        print(f"{thr:<12.2f} {n_route_a / n:>7.1%} {rec:>10.4f} {e1:>10.4f} {gap:>+11.2f}pp")

    # Where do A-wins live? Bin by base R@10 and count A-wins, K-wins, ties.
    print("\n" + "=" * 80)
    print("=== A-wins by base-R@10 bin (where is the routing headroom?) ===")
    print("=" * 80)
    bin_counts = {
        "0 (zero recall)": [0, 0, 0],
        "(0, 0.25]": [0, 0, 0],
        "(0.25, 0.50]": [0, 0, 0],
        "(0.50, 1.00]": [0, 0, 0],
    }  # [A_wins, K_wins, ties]
    for ar, kr in zip(a_recall, k_recall):
        if ar == 0:
            bk = "0 (zero recall)"
        elif ar <= 0.25:
            bk = "(0, 0.25]"
        elif ar <= 0.50:
            bk = "(0.25, 0.50]"
        else:
            bk = "(0.50, 1.00]"
        if ar > kr:
            bin_counts[bk][0] += 1
        elif kr > ar:
            bin_counts[bk][1] += 1
        else:
            bin_counts[bk][2] += 1
    print(f"{'bin':<22} {'A wins':>8} {'K wins':>8} {'ties':>8} {'A-win %':>10}")
    print("-" * 80)
    for bk, (aw, kw, t) in bin_counts.items():
        total = aw + kw + t
        pct = aw / total if total else 0.0
        print(f"{bk:<22} {aw:>8} {kw:>8} {t:>8} {pct:>9.1%}")

    # Heuristic features: query length, has-digit. Test simple rules.
    print("\n" + "=" * 80)
    print("=== Heuristic router: query length + has-digit ===")
    print("=" * 80)
    has_digit = [any(c.isdigit() for c in q) for q in queries]
    n_tokens = [len(q.split()) for q in queries]
    rules = [
        ("base only (A always)", lambda i: True),
        ("K only", lambda i: False),
        ("A if 1 token", lambda i: n_tokens[i] == 1),
        ("A if no digit", lambda i: not has_digit[i]),
        ("A if 1-2 tokens, no digit", lambda i: n_tokens[i] <= 2 and not has_digit[i]),
        ("A if 1-3 tokens, no digit", lambda i: n_tokens[i] <= 3 and not has_digit[i]),
    ]
    print(f"{'rule':<32} {'%->A':>8} {'R@10':>10} {'E@1':>10} {'gap to K':>12}")
    print("-" * 80)
    for name, fn in rules:
        chosen_recall = []
        chosen_e1 = []
        n_route_a = 0
        for qi, qid in enumerate(qids):
            ma = per_query_metrics(a_order[qi], qrels[qid])
            mk = per_query_metrics(k_order[qi], qrels[qid])
            if fn(qi):
                chosen_recall.append(ma["recall"])
                n_route_a += 1
                if "e_at_1" in ma:
                    chosen_e1.append(ma["e_at_1"])
            else:
                chosen_recall.append(mk["recall"])
                if "e_at_1" in mk:
                    chosen_e1.append(mk["e_at_1"])
        rec = statistics.mean(chosen_recall)
        e1 = statistics.mean(chosen_e1) if chosen_e1 else 0.0
        gap = (rec - statistics.mean(k_recall)) * 100
        print(f"{name:<32} {n_route_a / n:>7.1%} {rec:>10.4f} {e1:>10.4f} {gap:>+11.2f}pp")


if __name__ == "__main__":
    main()
