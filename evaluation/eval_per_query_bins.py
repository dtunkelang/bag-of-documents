#!/usr/bin/env python3
"""Per-query-bin breakdown of setup K's lift over base on the ESCI test set.

Bins the 22,458 queries by base MiniLM's R@10 to surface where the lexical-
plus-rerank lift in K=21.11% (vs base A=15.60%) is concentrated. The aggregate
+5.51pp R@10 is almost certainly bimodal — concentrated on hard-regime
entity-anchored queries (samsung 860 evo, lego game of thrones, etc.) where
dense retrieval collapses to zero recall but BM25 surfaces the right product.

Output: a per-bin × per-setup table of R@10 / nDCG@10 / E@1 / E@3, plus the
absolute count of queries falling into each bin.
"""

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
    if e_pids:
        out["e_at_1"] = 1.0 if retrieved_pids and retrieved_pids[0] in e_pids else 0.0
        top3 = retrieved_pids[:3]
        out["e_at_3"] = sum(1 for p in top3 if p in e_pids) / min(3, len(e_pids))
    return out


def bin_label(base_recall):
    """Bin by base MiniLM R@10. Hard regime is base recall = 0; the rest are
    quartile-ish thresholds."""
    if base_recall == 0:
        return "0 (zero recall)"
    if base_recall <= 0.25:
        return "(0, 0.25]"
    if base_recall <= 0.50:
        return "(0.25, 0.50]"
    return "(0.50, 1.00]"


BIN_ORDER = ["0 (zero recall)", "(0, 0.25]", "(0.25, 0.50]", "(0.50, 1.00]"]


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

    # Encode queries with the 3 models we need (skip 6M-MNRL retrieval —
    # we don't need it for setups A or K).
    print("encoding queries with base, rerank_a, rerank_b...", flush=True)
    qv_base = encode_subproc("all-MiniLM-L6-v2", queries)
    qv_a = encode_subproc(os.path.join(SCRIPT_DIR, "query_model_us_full_6m_mnrl"), queries)
    qv_b = encode_subproc(os.path.join(SCRIPT_DIR, "query_model_us_qrels_mnrl_hardneg"), queries)

    # Cached product vecs
    print("loading cached product vecs...", flush=True)
    pv_a = np.load(os.path.join(INDEX_DIR, "rerank_A.vecs.fp16.npy")).astype(np.float32)
    pv_b = np.load(os.path.join(INDEX_DIR, "rerank_B.vecs.fp16.npy")).astype(np.float32)

    # Setup A: base FAISS top-10
    print("retrieving base FAISS top-100...", flush=True)
    I_base = faiss_top_k(qv_base, os.path.join(INDEX_DIR, "index.faiss"), K_RETRIEVE)
    base_pids = [[faiss_pos_to_pid[int(p)] for p in row if p >= 0] for row in I_base]

    # Setup K: BM25 top-100 -> ensemble rerank
    print("loading BM25 top-100...", flush=True)
    I_bm25 = np.load(os.path.join(INDEX_DIR, "bm25_top100.npy"))

    # Per-query orderings
    a_order = [row[:K_EVAL] for row in base_pids]
    h_order = [
        [faiss_pos_to_pid[int(p)] for p in row[:K_EVAL] if p >= 0] for row in I_bm25
    ]  # BM25 top-10, no rerank
    k_order = []
    for qi in range(len(queries)):
        positions = [int(p) for p in I_bm25[qi] if p >= 0]
        if not positions:
            k_order.append([])
            continue
        sims = (pv_a[positions] @ qv_a[qi] + pv_b[positions] @ qv_b[qi]) / 2
        order = np.argsort(-sims)[:K_EVAL]
        k_order.append([faiss_pos_to_pid[positions[int(j)]] for j in order])

    # Per-query metrics
    print("computing per-query metrics + binning...", flush=True)
    bins_es = defaultdict(lambda: defaultdict(list))  # bin -> setup -> list of recalls
    bins_n_es = defaultdict(int)
    bins_n_e = defaultdict(int)

    for qi, qid in enumerate(qids):
        a_m = per_query_metrics(a_order[qi], qrels[qid])
        if "recall" not in a_m:
            continue  # skip queries with no E+S relevant (shouldn't happen given qids filter)
        bin_key = bin_label(a_m["recall"])
        bins_n_es[bin_key] += 1
        bins_es[bin_key]["A"].append(a_m["recall"])
        bins_es[bin_key]["A_ndcg"].append(a_m["ndcg"])
        h_m = per_query_metrics(h_order[qi], qrels[qid])
        bins_es[bin_key]["H"].append(h_m.get("recall", 0))
        bins_es[bin_key]["H_ndcg"].append(h_m.get("ndcg", 0))
        k_m = per_query_metrics(k_order[qi], qrels[qid])
        bins_es[bin_key]["K"].append(k_m.get("recall", 0))
        bins_es[bin_key]["K_ndcg"].append(k_m.get("ndcg", 0))
        if "e_at_1" in a_m:
            bins_n_e[bin_key] += 1
            bins_es[bin_key]["A_e1"].append(a_m["e_at_1"])
            bins_es[bin_key]["A_e3"].append(a_m["e_at_3"])
            bins_es[bin_key]["H_e1"].append(h_m.get("e_at_1", 0))
            bins_es[bin_key]["H_e3"].append(h_m.get("e_at_3", 0))
            bins_es[bin_key]["K_e1"].append(k_m.get("e_at_1", 0))
            bins_es[bin_key]["K_e3"].append(k_m.get("e_at_3", 0))

    # Output
    total_es = sum(bins_n_es.values())
    print("\n" + "=" * 100)
    print(f"=== Per-bin breakdown (binned by base MiniLM R@10), {total_es:,} queries with E+S ===")
    print("=" * 100)
    print(
        f"{'bin':<22} {'n':>6} {'A R@10':>9} {'H R@10':>9} {'K R@10':>9} "
        f"{'K-A':>8} {'A E@1':>9} {'K E@1':>9} {'K-A E@1':>10}"
    )
    print("-" * 100)

    def mean(xs):
        return statistics.mean(xs) if xs else 0.0

    for bin_key in BIN_ORDER:
        if bin_key not in bins_n_es:
            continue
        rec_a = mean(bins_es[bin_key]["A"])
        rec_h = mean(bins_es[bin_key]["H"])
        rec_k = mean(bins_es[bin_key]["K"])
        e1_a = mean(bins_es[bin_key]["A_e1"])
        e1_k = mean(bins_es[bin_key]["K_e1"])
        n = bins_n_es[bin_key]
        print(
            f"{bin_key:<22} {n:>6} "
            f"{rec_a:>8.2%} {rec_h:>8.2%} {rec_k:>8.2%} "
            f"{(rec_k - rec_a) * 100:>+7.2f} "
            f"{e1_a:>8.2%} {e1_k:>8.2%} {(e1_k - e1_a) * 100:>+9.2f}"
        )

    print("\n" + "=" * 100)
    print("Aggregate (sanity-check vs eval_mnrl_retriever.py):")
    print("=" * 100)
    all_recalls_a = [r for b in bins_es.values() for r in b["A"]]
    all_recalls_h = [r for b in bins_es.values() for r in b["H"]]
    all_recalls_k = [r for b in bins_es.values() for r in b["K"]]
    all_e1_a = [r for b in bins_es.values() for r in b["A_e1"]]
    all_e1_k = [r for b in bins_es.values() for r in b["K_e1"]]
    print(f"  A R@10 = {mean(all_recalls_a):.4f}  (expected 0.1560)")
    print(f"  H R@10 = {mean(all_recalls_h):.4f}  (expected 0.1950)")
    print(f"  K R@10 = {mean(all_recalls_k):.4f}  (expected 0.2111)")
    print(f"  A E@1  = {mean(all_e1_a):.4f}  (expected 0.3150)")
    print(f"  K E@1  = {mean(all_e1_k):.4f}  (expected 0.4087)")


if __name__ == "__main__":
    main()
