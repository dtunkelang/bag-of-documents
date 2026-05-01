#!/usr/bin/env python3
"""Probes 2 + 3:

Probe 2 — query-conditional w_ce routing. Bin queries by simple features
(token count, has_digit, has_caps, predicted specificity via kNN to bag
centroids). For each bin, find the optimal w_ce via oracle search. Then
test whether per-bin routing beats the uniform w_ce=0.25 and how close
to the oracle a learned router could get.

Probe 3 — weighted A/B/G fusion with CE in the loop. The previous W40/
W60/W70 weighted-fusion sweep happened *without* CE. With CE at
w_ce=0.25, the optimal A/B/G weighting may shift. Sweep (w_a, w_b, w_g)
on a 0.1-grid simplex and find the optimum.

Both probes reuse the cached top-50 artifacts from eval_ce_rerank.py:
  - ce_candidates.npy   (22458, 50)  - candidate FAISS positions
  - ce_sumsim.npy       (22458, 50)  - 3-way bi-encoder mean cosine
  - ce_scores.npy       (22458, 50)  - CE scores
Plus needs per-encoder cosines, which we compute fresh.

Usage:
    python evaluation/eval_routing_and_weights.py
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


def per_query_recall_array(scores, candidate_pos, valid, qids, qrels, faiss_pos_to_pid):
    """Return per-query (R@10, E@1) under top-10 sort by `scores`."""
    rs = np.full(len(qids), np.nan, dtype=np.float32)
    e1s = np.full(len(qids), np.nan, dtype=np.float32)
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
        rs[qi] = m[0]
        if m[2] is not None:
            e1s[qi] = m[2]
    return rs, e1s


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

    candidate_pos = np.load(os.path.join(INDEX_DIR, "ce_candidates.npy"))
    sumsim = np.load(os.path.join(INDEX_DIR, "ce_sumsim.npy"))
    ce_scores = np.load(os.path.join(INDEX_DIR, "ce_scores.npy"))
    valid = candidate_pos >= 0

    nm_sum = normalize_per_query(sumsim, valid)
    nm_ce = normalize_per_query(ce_scores, valid)

    # =================================================================
    # PROBE 2: query-conditional w_ce routing
    # =================================================================
    print("\n" + "=" * 100)
    print("PROBE 2: query-conditional w_ce routing")
    print("=" * 100)

    w_grid = [0.0, 0.1, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.75]

    # Per-query R@10 at every w_ce in grid.
    print("computing per-query R@10 / E@1 at each w in grid...", flush=True)
    per_w_recall = {}
    per_w_e1 = {}
    for w in w_grid:
        fused = (1 - w) * nm_sum + w * nm_ce
        rs, e1s = per_query_recall_array(fused, candidate_pos, valid, qids, qrels, faiss_pos_to_pid)
        per_w_recall[w] = rs
        per_w_e1[w] = e1s
    # Aggregate baseline
    print(f"\n{'w_ce':>5} | {'R@10':>7} {'E@1':>7}")
    print("-" * 25)
    for w in w_grid:
        rs = per_w_recall[w]
        e1s = per_w_e1[w]
        rs_clean = rs[~np.isnan(rs)]
        e1s_clean = e1s[~np.isnan(e1s)]
        print(f"{w:>5.2f} | {np.mean(rs_clean):>6.2%} {np.mean(e1s_clean):>6.2%}")

    # Oracle headroom: per-query best w_ce.
    print("\noracle headroom (best w_ce per query):")
    R = np.stack([per_w_recall[w] for w in w_grid], axis=1)  # (n_q, n_w)
    valid_q = ~np.isnan(R[:, 0])
    R_valid = R[valid_q]
    oracle_recall = np.nanmax(R_valid, axis=1).mean()
    print(f"  oracle R@10 = {oracle_recall:.2%}")

    # Define query feature bins.
    def query_features(q):
        toks = q.split()
        return {
            "n_tokens": len(toks),
            "has_digit": any(any(c.isdigit() for c in t) for t in toks),
            "any_upper": any(c.isupper() for c in q),
            "all_lower": q.lower() == q,
        }

    feats = [query_features(q) for q in queries]

    bins = {
        "1tok": [qi for qi, f in enumerate(feats) if f["n_tokens"] == 1],
        "2tok": [qi for qi, f in enumerate(feats) if f["n_tokens"] == 2],
        "3tok": [qi for qi, f in enumerate(feats) if f["n_tokens"] == 3],
        "4+tok": [qi for qi, f in enumerate(feats) if f["n_tokens"] >= 4],
        "has_digit": [qi for qi, f in enumerate(feats) if f["has_digit"]],
        "no_digit": [qi for qi, f in enumerate(feats) if not f["has_digit"]],
        "all_lower": [qi for qi, f in enumerate(feats) if f["all_lower"]],
        "has_upper": [qi for qi, f in enumerate(feats) if not f["all_lower"]],
    }

    print(f"\n{'bin':<14} {'n':>7} | best_w {'R@10':>7} {'unif':>7} {'lift':>5}")
    print("-" * 55)
    best_w_per_bin = {}
    for binname, bin_qis in bins.items():
        if not bin_qis:
            continue
        bin_qis_arr = np.asarray(bin_qis)
        bin_mask = np.isin(np.arange(len(qids)), bin_qis_arr)
        bin_recalls_per_w = {}
        for w in w_grid:
            v = per_w_recall[w][bin_qis_arr]
            v = v[~np.isnan(v)]
            bin_recalls_per_w[w] = float(v.mean()) if v.size else 0.0
        best_w = max(bin_recalls_per_w, key=lambda w: bin_recalls_per_w[w])
        best_r = bin_recalls_per_w[best_w]
        unif_r = bin_recalls_per_w[0.25]
        n = int(bin_mask.sum())
        print(
            f"{binname:<14} {n:>7,} | {best_w:>5.2f}  {best_r:>6.2%} {unif_r:>6.2%} {(best_r - unif_r) * 100:>+4.2f}"
        )
        best_w_per_bin[binname] = best_w

    # Build a routed setup using a precedence: longest-token-bin best_w wins.
    # Simpler: route by token count alone (most common axis).
    print("\nrouting by n_tokens:")
    tokcount_best_w = {}
    for k, label in zip([1, 2, 3, 4], ["1tok", "2tok", "3tok", "4+tok"]):
        tokcount_best_w[k] = best_w_per_bin.get(label, 0.25)
    routed_recall = np.full(len(qids), np.nan, dtype=np.float32)
    for qi in range(len(qids)):
        nt = min(feats[qi]["n_tokens"], 4)
        w = tokcount_best_w[nt]
        routed_recall[qi] = per_w_recall[w][qi]
    rr = routed_recall[~np.isnan(routed_recall)].mean()
    print(f"  routed R@10 = {rr:.2%}  (vs uniform w=0.25 = {np.nanmean(per_w_recall[0.25]):.2%})")
    print(f"  oracle gap = {(oracle_recall - rr) * 100:.2f}pp")

    # =================================================================
    # PROBE 3: weighted A/B/G fusion with CE in the loop
    # =================================================================
    print("\n" + "=" * 100)
    print("PROBE 3: weighted A/B/G fusion with CE @ w_ce=0.25")
    print("=" * 100)

    # Compute per-encoder cosines on the cached candidate_pos.
    print("encoding queries with rerank_a, b, g...", flush=True)
    t0 = time.time()
    qv_a = encode_subproc(os.path.join(SCRIPT_DIR, "query_model_us_full_6m_mnrl"), queries)
    qv_b = encode_subproc(os.path.join(SCRIPT_DIR, "query_model_us_qrels_mnrl_hardneg"), queries)
    qv_g = encode_subproc(os.path.join(SCRIPT_DIR, "query_model_us_esci_supervised"), queries)
    pv_a = np.load(os.path.join(INDEX_DIR, "rerank_A.vecs.fp16.npy")).astype(np.float32)
    pv_b = np.load(os.path.join(INDEX_DIR, "rerank_B.vecs.fp16.npy")).astype(np.float32)
    pv_g = np.load(os.path.join(INDEX_DIR, "rerank_G.vecs.fp16.npy")).astype(np.float32)
    print(f"  encoded in {time.time() - t0:.0f}s; computing per-encoder cosines...", flush=True)
    t0 = time.time()
    n_q = len(qids)
    K_RET = candidate_pos.shape[1]
    sims_a = np.zeros((n_q, K_RET), dtype=np.float32)
    sims_b = np.zeros((n_q, K_RET), dtype=np.float32)
    sims_g = np.zeros((n_q, K_RET), dtype=np.float32)
    for qi in range(n_q):
        positions = candidate_pos[qi]
        good = positions >= 0
        if not good.any():
            continue
        idx = positions[good]
        sims_a[qi, good] = pv_a[idx] @ qv_a[qi]
        sims_b[qi, good] = pv_b[idx] @ qv_b[qi]
        sims_g[qi, good] = pv_g[idx] @ qv_g[qi]
    print(f"  done in {time.time() - t0:.0f}s", flush=True)

    # Sweep (w_a, w_b, w_g) on a 0.1 grid (constrained to sum to 1).
    # Skip the trivial all-zero combos.
    print(f"\n{'(w_a, w_b, w_g)':<22} | {'R@10':>7} {'E@1':>7}", flush=True)
    print("-" * 42)

    grid = []
    for wa10 in range(0, 11):
        for wb10 in range(0, 11 - wa10):
            wg10 = 10 - wa10 - wb10
            grid.append((wa10 / 10, wb10 / 10, wg10 / 10))

    results = []
    for wa, wb, wg in grid:
        weighted_sumsim = wa * sims_a + wb * sims_b + wg * sims_g
        nm_ws = normalize_per_query(weighted_sumsim, valid)
        fused = 0.75 * nm_ws + 0.25 * nm_ce
        rs, e1s = per_query_recall_array(fused, candidate_pos, valid, qids, qrels, faiss_pos_to_pid)
        rs_clean = rs[~np.isnan(rs)]
        e1s_clean = e1s[~np.isnan(e1s)]
        results.append((wa, wb, wg, float(rs_clean.mean()), float(e1s_clean.mean())))

    # Print top 10 by R@10
    results.sort(key=lambda x: -x[3])
    print("top-10 by R@10:")
    for wa, wb, wg, r, e1 in results[:10]:
        print(f"  ({wa:.1f}, {wb:.1f}, {wg:.1f})        | {r:>6.2%} {e1:>6.2%}")
    print("\nbottom-3 by R@10:")
    for wa, wb, wg, r, e1 in results[-3:]:
        print(f"  ({wa:.1f}, {wb:.1f}, {wg:.1f})        | {r:>6.2%} {e1:>6.2%}")
    # Equal weight reference
    eq = next((w for w in results if abs(w[0] - 1 / 3) < 0.05 and abs(w[1] - 1 / 3) < 0.05), None)
    if eq:
        print(f"\nequal-weight reference (~1/3, 1/3, 1/3): R@10={eq[3]:.2%} E@1={eq[4]:.2%}")
    eq_grid = [w for w in results if w[0] == 0.3 and w[1] == 0.3]
    if eq_grid:
        eq = eq_grid[0]
        print(f"closest grid point (0.3, 0.3, 0.4): R@10={eq[3]:.2%} E@1={eq[4]:.2%}")


if __name__ == "__main__":
    main()
