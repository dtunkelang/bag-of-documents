#!/usr/bin/env python3
"""Bootstrap CIs on the spell-correction lift (spell vs baseline) on the
changed-only subset and on the full test set.

Reuses the spell probe's outputs:
  - combined_index_us_minilm/spell/corrected_queries.json
  - combined_index_us_minilm/spell/changed_mask.npy

Computes per-query CC3-50 metrics for both baseline (cached) and corrected
(fresh sumsim from the corrected query encodings on the new BM25 top-100),
then bootstraps paired deltas.

Usage:
    python evaluation/bootstrap_spell_lift.py
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
K_EVAL = 10
K_RET = 50  # CC3-50


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
        e1 = e3 = float("nan")
    return recall, ndcg, e1, e3


def per_query_cc3_50_metrics(I, qv_a, qv_b, qv_g, pv_a, pv_b, pv_g, qids, qrels, faiss_pos_to_pid):
    n = len(qids)
    rs = np.full(n, np.nan, dtype=np.float32)
    e1s = np.full(n, np.nan, dtype=np.float32)
    for qi, qid in enumerate(qids):
        positions = [int(p) for p in I[qi, :K_RET] if p >= 0]
        if not positions:
            continue
        sims = (
            pv_a[positions] @ qv_a[qi] + pv_b[positions] @ qv_b[qi] + pv_g[positions] @ qv_g[qi]
        ) / 3
        order = np.argsort(-sims)[:K_EVAL]
        ordering = [faiss_pos_to_pid[positions[int(j)]] for j in order]
        m = per_query_metrics(ordering, qrels[qid])
        if m is None:
            continue
        rs[qi] = m[0]
        e1s[qi] = m[2] if m[2] is not None else float("nan")
    return rs, e1s


def main():
    rng = np.random.default_rng(42)

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

    spell_dir = os.path.join(INDEX_DIR, "spell")
    with open(os.path.join(spell_dir, "corrected_queries.json")) as f:
        corrected_queries = json.load(f)
    changed_mask = np.load(os.path.join(spell_dir, "changed_mask.npy"))
    n_changed = int(changed_mask.sum())
    print(
        f"  {len(queries):,} eval queries, {n_changed:,} corrected ({n_changed / len(queries):.1%})",
        flush=True,
    )

    # Baseline metrics (cached top-100, restrict to top-50).
    candidate_pos = np.load(os.path.join(INDEX_DIR, "ce_top100_candidates.npy"))
    sumsim = np.load(os.path.join(INDEX_DIR, "ce_top100_sumsim.npy"))
    valid = candidate_pos >= 0
    rs_baseline = np.full(len(qids), np.nan, dtype=np.float32)
    e1s_baseline = np.full(len(qids), np.nan, dtype=np.float32)
    for qi, qid in enumerate(qids):
        s = sumsim[qi].copy()
        s[~valid[qi]] = -np.inf
        s[K_RET:] = -np.inf
        order = np.argsort(-s)[:K_EVAL]
        ordering = [faiss_pos_to_pid[int(candidate_pos[qi, int(j)])] for j in order]
        m = per_query_metrics(ordering, qrels[qid])
        if m is None:
            continue
        rs_baseline[qi] = m[0]
        e1s_baseline[qi] = m[2] if m[2] is not None else float("nan")

    # Corrected metrics: bm25s on corrected queries + bi-encoder rerank.
    print("loading bm25s + corrected query tokenization...", flush=True)
    bm25s_idx = bm25s.BM25.load(os.path.join(INDEX_DIR, "bm25s_index"), mmap=False)
    stemmer = Stemmer.Stemmer("english")
    qt = bm25s.tokenize(corrected_queries, stopwords="en", stemmer=stemmer, show_progress=False)
    print("retrieving corrected top-100...", flush=True)
    t0 = time.time()
    results, _ = bm25s_idx.retrieve(qt, k=100, show_progress=False)
    I_corr = np.asarray(results, dtype=np.int64)
    print(f"  done {time.time() - t0:.0f}s", flush=True)

    print("encoding corrected queries with rerank_a/b/g...", flush=True)
    t0 = time.time()
    qv_a = encode_subproc(
        os.path.join(SCRIPT_DIR, "query_model_us_full_6m_mnrl"), corrected_queries
    )
    qv_b = encode_subproc(
        os.path.join(SCRIPT_DIR, "query_model_us_qrels_mnrl_hardneg"), corrected_queries
    )
    qv_g = encode_subproc(
        os.path.join(SCRIPT_DIR, "query_model_us_esci_supervised"), corrected_queries
    )
    pv_a = np.load(os.path.join(INDEX_DIR, "rerank_A.vecs.fp16.npy")).astype(np.float32)
    pv_b = np.load(os.path.join(INDEX_DIR, "rerank_B.vecs.fp16.npy")).astype(np.float32)
    pv_g = np.load(os.path.join(INDEX_DIR, "rerank_G.vecs.fp16.npy")).astype(np.float32)
    print(f"  done {time.time() - t0:.0f}s", flush=True)

    print("computing per-query CC3-50 corrected...", flush=True)
    rs_corr, e1s_corr = per_query_cc3_50_metrics(
        I_corr, qv_a, qv_b, qv_g, pv_a, pv_b, pv_g, qids, qrels, faiss_pos_to_pid
    )

    # Bootstrap.
    n_bootstrap = 1000
    n = len(qids)
    print(f"\nbootstrapping {n_bootstrap} resamples...\n")

    def bootstrap_delta(diffs, n, rng, nb):
        diffs = diffs[~np.isnan(diffs)]
        if diffs.size == 0:
            return None
        bs = []
        for _ in range(nb):
            idx = rng.integers(0, diffs.size, size=diffs.size)
            bs.append(diffs[idx].mean())
        bs = np.asarray(bs)
        return bs.mean(), np.percentile(bs, 2.5), np.percentile(bs, 97.5)

    diffs_r_all = rs_corr - rs_baseline
    diffs_e1_all = e1s_corr - e1s_baseline
    print("Full test set (22,458):")
    for label, diffs in (("R@10", diffs_r_all), ("E@1", diffs_e1_all)):
        bs = bootstrap_delta(diffs, n, rng, n_bootstrap)
        if bs is None:
            continue
        mu, lo, hi = bs
        sig = "**" if (lo > 0 and hi > 0) or (lo < 0 and hi < 0) else "  "
        print(f"  Δ {label}: {sig}{mu:+.2%} [{lo:+.2%}, {hi:+.2%}]")

    if n_changed > 0:
        idx_changed = np.where(changed_mask)[0]
        diffs_r_chg = (rs_corr - rs_baseline)[idx_changed]
        diffs_e1_chg = (e1s_corr - e1s_baseline)[idx_changed]
        print(f"\nChanged-only subset ({n_changed:,}):")
        for label, diffs in (("R@10", diffs_r_chg), ("E@1", diffs_e1_chg)):
            bs = bootstrap_delta(diffs, idx_changed.size, rng, n_bootstrap)
            if bs is None:
                continue
            mu, lo, hi = bs
            sig = "**" if (lo > 0 and hi > 0) or (lo < 0 and hi < 0) else "  "
            print(f"  Δ {label}: {sig}{mu:+.2%} [{lo:+.2%}, {hi:+.2%}]")

    # Save per-query metrics for downstream composite eval.
    np.save(os.path.join(spell_dir, "spell_per_query_R10.npy"), rs_corr)
    np.save(os.path.join(spell_dir, "spell_per_query_E1.npy"), e1s_corr)
    np.save(os.path.join(spell_dir, "I_corrected_top100.npy"), I_corr)
    print(f"\nsaved per-query metrics + I_corrected to {spell_dir}/")


if __name__ == "__main__":
    main()
