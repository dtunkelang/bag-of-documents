#!/usr/bin/env python3
"""Phrase-aware BM25 probe: extend each title's tokens with adjacent
stemmed bigrams. Build a bm25s index over the combined unigram+bigram
vocabulary and re-retrieve.

Hypothesis: bigrams help on multi-word product entities (e.g.
'samsung 860', 'play doh', 'game of thrones') by giving exact-pair
matches a separate BM25 contribution beyond their individual unigrams.

Reuses (k1=0.3, b=0.6) — the optimum for the unigram-only setup.

Output: phrase-bm25 top-100 candidates + CC3-50 metrics.

Usage:
    python evaluation/eval_phrase_bm25.py
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


def expand_with_bigrams(token_ids_per_doc):
    """Take bm25s.Tokenized.ids (list of [int]) and append bigram pseudo-IDs.

    Bigrams are encoded as (a + 1) * (max_unigram_id + 1) + (b + 1) so they
    don't collide with unigram IDs. Returns: ids_per_doc, max_id.
    """
    out_ids = []
    for ids in token_ids_per_doc:
        bigrams = []
        for i in range(len(ids) - 1):
            bigrams.append(ids[i], ids[i + 1])
        out_ids.append(list(ids) + bigrams)
    return out_ids


def encode_bigrams(token_ids_per_doc, max_unigram_id):
    """Append bigram IDs (offset to avoid collision with unigrams)."""
    base = max_unigram_id + 1
    out = []
    for ids in token_ids_per_doc:
        bigrams = [base + ids[i] * base + ids[i + 1] for i in range(len(ids) - 1)]
        out.append(list(ids) + bigrams)
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

    # Tokenize titles + queries with the same recipe as the deployed bm25s index.
    print("tokenizing titles + queries (Snowball english + en stopwords)...", flush=True)
    stemmer = Stemmer.Stemmer("english")
    title_tok = bm25s.tokenize(index_titles, stopwords="en", stemmer=stemmer, show_progress=False)
    query_tok = bm25s.tokenize(queries, stopwords="en", stemmer=stemmer, show_progress=False)

    # Find max token id across vocab.
    max_id = max(title_tok.vocab.values()) if title_tok.vocab else 0
    print(f"  vocab size: {max_id + 1:,}", flush=True)

    # Append bigrams to each doc / query token stream.
    print("appending adjacent-token bigrams...", flush=True)
    t0 = time.time()
    title_combined_ids = encode_bigrams(title_tok.ids, max_id)
    query_combined_ids = encode_bigrams(query_tok.ids, max_id)

    # Now build a bm25s index manually from the combined ids by abusing the
    # tokenizer output: bm25s expects a Tokenized object. We construct a
    # synthetic vocab mapping that maps every observed combined id to itself.
    used_ids = set()
    for ids in title_combined_ids:
        used_ids.update(ids)
    for ids in query_combined_ids:
        used_ids.update(ids)
    vocab = {f"__id_{i}__": i for i in used_ids}
    print(
        f"  combined unigram+bigram vocab: {len(vocab):,}, build {time.time() - t0:.0f}s",
        flush=True,
    )

    # Construct Tokenized-like objects.
    from bm25s.tokenization import Tokenized

    title_combined = Tokenized(ids=title_combined_ids, vocab=vocab)
    query_combined = Tokenized(ids=query_combined_ids, vocab=vocab)

    print("indexing bm25s with k1=0.3, b=0.6 over phrase vocab...", flush=True)
    t0 = time.time()
    idx = bm25s.BM25(k1=0.3, b=0.6)
    idx.index(title_combined, show_progress=False)
    print(f"  done {time.time() - t0:.0f}s", flush=True)

    print("retrieving phrase top-100...", flush=True)
    t0 = time.time()
    results, _ = idx.retrieve(query_combined, k=100, show_progress=False)
    I_phrase = np.asarray(results, dtype=np.int64)
    print(f"  done {time.time() - t0:.0f}s", flush=True)

    phrase_dir = os.path.join(INDEX_DIR, "phrase")
    os.makedirs(phrase_dir, exist_ok=True)
    np.save(os.path.join(phrase_dir, "I_phrase_top100.npy"), I_phrase)

    # CC3-50 eval.
    print("\nencoding queries with rerank_a/b/g (originals — no expansion)...", flush=True)
    qv_a = encode_subproc(os.path.join(SCRIPT_DIR, "query_model_us_full_6m_mnrl"), queries)
    qv_b = encode_subproc(os.path.join(SCRIPT_DIR, "query_model_us_qrels_mnrl_hardneg"), queries)
    qv_g = encode_subproc(os.path.join(SCRIPT_DIR, "query_model_us_esci_supervised"), queries)
    pv_a = np.load(os.path.join(INDEX_DIR, "rerank_A.vecs.fp16.npy")).astype(np.float32)
    pv_b = np.load(os.path.join(INDEX_DIR, "rerank_B.vecs.fp16.npy")).astype(np.float32)
    pv_g = np.load(os.path.join(INDEX_DIR, "rerank_G.vecs.fp16.npy")).astype(np.float32)

    print("computing CC3-50 with phrase candidates...", flush=True)
    rs, ns, e1s, e3s = [], [], [], []
    rs_pq = np.full(len(qids), np.nan, dtype=np.float32)
    e1s_pq = np.full(len(qids), np.nan, dtype=np.float32)
    for qi, qid in enumerate(qids):
        positions = [int(p) for p in I_phrase[qi, :K_RET] if p >= 0]
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
        rs_pq[qi] = r
        if e1 is not None and not math.isnan(e1):
            e1s.append(e1)
            e3s.append(e3)
            e1s_pq[qi] = e1
    np.save(os.path.join(phrase_dir, "phrase_per_query_R10.npy"), rs_pq)
    np.save(os.path.join(phrase_dir, "phrase_per_query_E1.npy"), e1s_pq)
    print(
        f"\nCC3-50 with phrase BM25: R@10 {np.mean(rs):.2%} nDCG {np.mean(ns):.4f} "
        f"E@1 {np.mean(e1s):.2%} E@3 {np.mean(e3s):.2%}"
    )
    print("baseline (unigram bm25s): CC3-50 R@10 21.60%, E@1 42.10%")


if __name__ == "__main__":
    main()
