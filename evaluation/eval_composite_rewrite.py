#!/usr/bin/env python3
"""Composite query rewriting: combine spell + LLM corrections, then
re-evaluate. (PRF and phrase BM25 are candidate-pool transformations,
not query-text transformations, so they're orthogonal — handled
separately by overlapping I_*_top100.npy candidate pools.)

Composite waterfall:
  if LLM rewrote → use LLM
  else if spell corrected → use spell
  else → original

For BM25 candidate pool: option (a) bm25s on the composite text;
option (b) merge candidates from {original, spell, prf, phrase, llm}
top-K via RRF for a candidate-pool-union approach.

We do (a) here as the cleanest test. Saves composite metrics.

Usage:
    python evaluation/eval_composite_rewrite.py
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
K_RET = 50


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

    spell_path = os.path.join(INDEX_DIR, "spell", "corrected_queries.json")
    llm_path = os.path.join(INDEX_DIR, "llm", "rewritten_queries.json")
    spell_changed = np.load(os.path.join(INDEX_DIR, "spell", "changed_mask.npy"))
    llm_changed = np.load(os.path.join(INDEX_DIR, "llm", "changed_mask.npy"))
    with open(spell_path) as f:
        spell_q = json.load(f)
    with open(llm_path) as f:
        llm_q = json.load(f)

    # Composite: LLM > spell > original.
    composite = []
    n_llm = n_spell = n_orig = 0
    for qi in range(len(queries)):
        if llm_changed[qi]:
            composite.append(llm_q[qi])
            n_llm += 1
        elif spell_changed[qi]:
            composite.append(spell_q[qi])
            n_spell += 1
        else:
            composite.append(queries[qi])
            n_orig += 1
    print(f"  composite: {n_llm:,} LLM, {n_spell:,} spell, {n_orig:,} original")

    # Re-retrieve.
    print("retrieving composite top-100 via bm25s...", flush=True)
    bm25s_idx = bm25s.BM25.load(os.path.join(INDEX_DIR, "bm25s_index"), mmap=False)
    stemmer = Stemmer.Stemmer("english")
    qt = bm25s.tokenize(composite, stopwords="en", stemmer=stemmer, show_progress=False)
    results, _ = bm25s_idx.retrieve(qt, k=100, show_progress=False)
    I_comp = np.asarray(results, dtype=np.int64)

    composite_dir = os.path.join(INDEX_DIR, "composite")
    os.makedirs(composite_dir, exist_ok=True)
    with open(os.path.join(composite_dir, "composite_queries.json"), "w") as f:
        json.dump(composite, f)
    np.save(os.path.join(composite_dir, "I_composite_top100.npy"), I_comp)

    # Encode the original queries — bi-encoders should see user intent, not LLM rewrite.
    # However for LLM-changed queries we use the rewrite since the original is conversational.
    # Compromise: use composite text for the bi-encoder too (matches what BM25 saw).
    print("encoding composite queries with rerank_a/b/g...", flush=True)
    t0 = time.time()
    qv_a = encode_subproc(os.path.join(SCRIPT_DIR, "query_model_us_full_6m_mnrl"), composite)
    qv_b = encode_subproc(os.path.join(SCRIPT_DIR, "query_model_us_qrels_mnrl_hardneg"), composite)
    qv_g = encode_subproc(os.path.join(SCRIPT_DIR, "query_model_us_esci_supervised"), composite)
    pv_a = np.load(os.path.join(INDEX_DIR, "rerank_A.vecs.fp16.npy")).astype(np.float32)
    pv_b = np.load(os.path.join(INDEX_DIR, "rerank_B.vecs.fp16.npy")).astype(np.float32)
    pv_g = np.load(os.path.join(INDEX_DIR, "rerank_G.vecs.fp16.npy")).astype(np.float32)
    print(f"  done {time.time() - t0:.0f}s", flush=True)

    print("computing CC3-50 with composite candidates...", flush=True)
    rs, ns, e1s, e3s = [], [], [], []
    for qi, qid in enumerate(qids):
        positions = [int(p) for p in I_comp[qi, :K_RET] if p >= 0]
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
        r, n, e1, e3 = m
        rs.append(r)
        ns.append(n)
        if e1 is not None and not math.isnan(e1):
            e1s.append(e1)
            e3s.append(e3)
    print(
        f"\nCC3-50 with composite query rewriting: R@10 {np.mean(rs):.2%} "
        f"nDCG {np.mean(ns):.4f} E@1 {np.mean(e1s):.2%} E@3 {np.mean(e3s):.2%}"
    )
    print("baseline (original queries): CC3-50 R@10 21.60%, E@1 42.10%")


if __name__ == "__main__":
    main()
