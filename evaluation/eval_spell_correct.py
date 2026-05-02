#!/usr/bin/env python3
"""Catalog-vocabulary spell-correction probe.

Builds a SpellChecker dictionary from product-title vocabulary (with
frequencies). Corrects test queries that have out-of-vocabulary tokens.
Measures the lift from correction at each pipeline stage:

  1. BM25 candidate-pool coverage: for changed queries, do relevant
     qrels move into the corrected top-100 that weren't in the original?
     This is the *ceiling* of what spell correction can buy.
  2. CC3-50 (fast SOTA, no CE): bi-encoder rerank on corrected queries.
     Fast: only needs to re-tokenize + bm25s retrieve + bi-encoder pass.
  3. CC4-100 (quality SOTA, with CE): full rerank on changed queries.
     CE re-scoring of new candidates is the slow step (~3h if all 22K
     queries changed; in practice only a fraction will).

Saves the corrected query strings + which-queries-changed mask for
downstream reuse.

Usage:
    python evaluation/eval_spell_correct.py
    python evaluation/eval_spell_correct.py --skip-ce  # skip step 3
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import os  # noqa: E402
import re  # noqa: E402
import time  # noqa: E402
from collections import Counter, defaultdict  # noqa: E402

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import bm25s  # noqa: E402
import numpy as np  # noqa: E402
import Stemmer  # noqa: E402
import torch  # noqa: E402
from spellchecker import SpellChecker  # noqa: E402

SCRIPT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INDEX_DIR = os.path.join(SCRIPT_DIR, "combined_index_us_minilm")

K_EVAL = 10
TOK_RE = re.compile(r"[a-z0-9]+")


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


def build_catalog_vocab(titles):
    """Lowercase + tokenize all titles; return Counter of token frequencies."""
    counter = Counter()
    for t in titles:
        for tok in TOK_RE.findall(t.lower()):
            counter[tok] += 1
    return counter


def correct_query(q, spell, vocab_set, min_freq=2):
    """Correct out-of-vocab tokens. Returns (corrected_query, did_change).

    Skip correction if:
      - Token is in catalog vocab (above frequency threshold)
      - Token is shorter than 3 chars (too short, ambiguous)
      - Token is purely numeric (likely a model number)
      - No correction candidate exists
    """
    tokens = TOK_RE.findall(q.lower())
    out_tokens = []
    changed = False
    for tok in tokens:
        if (
            tok in vocab_set
            or len(tok) <= 2
            or tok.isdigit()
            or any(c.isdigit() for c in tok)  # alphanumeric like "k380"
        ):
            out_tokens.append(tok)
            continue
        # Try correction
        candidates = spell.candidates(tok) or set()
        if not candidates:
            out_tokens.append(tok)
            continue
        # Pick highest-frequency candidate that is in catalog vocab
        best = None
        best_freq = -1
        for c in candidates:
            if c not in vocab_set:
                continue
            f = spell.word_frequency[c]
            if f > best_freq:
                best_freq = f
                best = c
        if best is None or best == tok:
            out_tokens.append(tok)
            continue
        out_tokens.append(best)
        changed = True
    return " ".join(out_tokens), changed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-ce", action="store_true", help="Skip CE rescore step")
    ap.add_argument("--ce-batch", type=int, default=64)
    args = ap.parse_args()

    # Load test data + qrels.
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

    # Build catalog vocab.
    print("building catalog vocabulary from product titles...", flush=True)
    t0 = time.time()
    vocab_counter = build_catalog_vocab(index_titles)
    print(
        f"  {len(vocab_counter):,} unique tokens, top-5: {vocab_counter.most_common(5)} "
        f"({time.time() - t0:.0f}s)",
        flush=True,
    )

    # Filter to tokens with frequency >= 2 (drop singletons; reduces noise).
    MIN_FREQ = 2
    filtered_vocab = {tok: freq for tok, freq in vocab_counter.items() if freq >= MIN_FREQ}
    print(f"  {len(filtered_vocab):,} tokens with freq >= {MIN_FREQ}", flush=True)
    vocab_set = set(filtered_vocab.keys())

    # Initialize spellchecker with catalog vocabulary (no English defaults;
    # we want only catalog tokens as candidates so we don't "correct" model
    # numbers like 'k380' to a generic English word).
    print("initializing SpellChecker with catalog vocab...", flush=True)
    spell = SpellChecker(language=None, distance=2)
    for tok, freq in filtered_vocab.items():
        spell.word_frequency.add(tok, freq)
    print(f"  {len(spell.word_frequency.dictionary):,} dictionary entries", flush=True)

    # Correct each query.
    print("correcting queries...", flush=True)
    t0 = time.time()
    corrected_queries = []
    changed_mask = np.zeros(len(queries), dtype=bool)
    n_oov_total = 0
    for qi, q in enumerate(queries):
        corrected, changed = correct_query(q, spell, vocab_set)
        corrected_queries.append(corrected)
        changed_mask[qi] = changed
        if changed:
            n_oov_total += 1
        if (qi + 1) % 5000 == 0:
            print(
                f"  {qi + 1:,}/{len(queries):,} ({n_oov_total:,} changed) ({time.time() - t0:.0f}s)",
                flush=True,
            )
    print(
        f"  done in {time.time() - t0:.0f}s; {n_oov_total:,} of {len(queries):,} queries changed "
        f"({n_oov_total / len(queries):.1%})",
        flush=True,
    )

    # Sample of corrections.
    print("\nsample corrections:")
    samples = 0
    for qi in range(len(queries)):
        if changed_mask[qi] and samples < 20:
            print(f"  '{queries[qi]}' → '{corrected_queries[qi]}'")
            samples += 1

    # Save artifacts.
    spell_dir = os.path.join(INDEX_DIR, "spell")
    os.makedirs(spell_dir, exist_ok=True)
    with open(os.path.join(spell_dir, "corrected_queries.json"), "w") as f:
        json.dump(corrected_queries, f)
    np.save(os.path.join(spell_dir, "changed_mask.npy"), changed_mask)
    print(f"\nsaved corrections to {spell_dir}/", flush=True)

    # ====================================================================
    # Step 1: BM25 candidate-pool coverage analysis.
    # For queries that changed, did relevant qrels move into the new top-100?
    # ====================================================================
    print("\nloading bm25s + tokenizing corrected queries...", flush=True)
    bm25s_idx = bm25s.BM25.load(os.path.join(INDEX_DIR, "bm25s_index"), mmap=False)
    stemmer = Stemmer.Stemmer("english")
    qt_corrected = bm25s.tokenize(
        corrected_queries, stopwords="en", stemmer=stemmer, show_progress=False
    )
    print("retrieving corrected top-100...", flush=True)
    t0 = time.time()
    results, _ = bm25s_idx.retrieve(qt_corrected, k=100, show_progress=False)
    I_corrected_100 = np.asarray(results, dtype=np.int64)
    print(f"  done in {time.time() - t0:.0f}s", flush=True)

    # Original BM25 top-100 (from cached artifact).
    I_orig_100 = np.load(os.path.join(INDEX_DIR, "bm25s_top200.npy"))[:, :100]

    # Pool-coverage delta on changed queries only.
    print("\npool-coverage delta on changed queries:", flush=True)
    changed_indices = np.where(changed_mask)[0]
    pool_added = 0
    pool_lost = 0
    pool_same = 0
    for qi in changed_indices:
        qid = qids[qi]
        es_pids = {p for p, g in qrels[qid].items() if g >= 2}
        if not es_pids:
            continue
        orig_pool = {faiss_pos_to_pid[int(p)] for p in I_orig_100[qi] if p >= 0}
        corr_pool = {faiss_pos_to_pid[int(p)] for p in I_corrected_100[qi] if p >= 0}
        added = (es_pids & corr_pool) - (es_pids & orig_pool)
        lost = (es_pids & orig_pool) - (es_pids & corr_pool)
        if added:
            pool_added += 1
        if lost:
            pool_lost += 1
        if not added and not lost:
            pool_same += 1
    print(
        f"  among {len(changed_indices):,} changed queries:",
        flush=True,
    )
    print(f"    {pool_added:,} gained relevant qrels in pool")
    print(f"    {pool_lost:,} lost relevant qrels from pool")
    print(f"    {pool_same:,} no change in coverage")

    # ====================================================================
    # Step 2: CC3-50 R@10 with corrected queries (fast SOTA on corrected).
    # Need to encode corrected queries with rerank_a/b/g + load product vecs.
    # ====================================================================
    print("\nencoding corrected queries with rerank_a/b/g...", flush=True)
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
    print(f"  done in {time.time() - t0:.0f}s", flush=True)

    print("loading product vecs...", flush=True)
    pv_a = np.load(os.path.join(INDEX_DIR, "rerank_A.vecs.fp16.npy")).astype(np.float32)
    pv_b = np.load(os.path.join(INDEX_DIR, "rerank_B.vecs.fp16.npy")).astype(np.float32)
    pv_g = np.load(os.path.join(INDEX_DIR, "rerank_G.vecs.fp16.npy")).astype(np.float32)

    K_RET_FAST = 50

    print("\ncomputing CC3-50 metrics on corrected queries...", flush=True)

    def compute_setup(I, qv_a, qv_b, qv_g, K_RET, qids_subset_indices=None):
        rs, ns, e1s, e3s = [], [], [], []
        idx_iter = qids_subset_indices if qids_subset_indices is not None else range(len(qids))
        for qi in idx_iter:
            qid = qids[qi]
            row = I[qi, :K_RET]
            positions = [int(p) for p in row if p >= 0]
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
        return (
            np.mean(rs) if rs else 0.0,
            np.mean(ns) if ns else 0.0,
            np.mean(e1s) if e1s else 0.0,
            np.mean(e3s) if e3s else 0.0,
        )

    # CC3-50 corrected (all queries).
    r, n, e1, e3 = compute_setup(I_corrected_100, qv_a, qv_b, qv_g, K_RET_FAST)
    print(
        f"  CC3-50 corrected (all):     R@10 {r:.2%}  nDCG {n:.4f}  E@1 {e1:.2%}  E@3 {e3:.2%}",
        flush=True,
    )

    # CC3-50 baseline (all): use original cached scores.
    cached_sumsim = np.load(os.path.join(INDEX_DIR, "ce_top100_sumsim.npy"))
    cached_cands = np.load(os.path.join(INDEX_DIR, "ce_top100_candidates.npy"))
    valid_orig = cached_cands >= 0
    rs0, ns0, e1s0, e3s0 = [], [], [], []
    for qi, qid in enumerate(qids):
        s = cached_sumsim[qi].copy()
        s[~valid_orig[qi]] = -np.inf
        s[K_RET_FAST:] = -np.inf  # restrict to top-50
        order = np.argsort(-s)[:K_EVAL]
        ordering = [faiss_pos_to_pid[int(cached_cands[qi, int(j)])] for j in order]
        m = per_query_metrics(ordering, qrels[qid])
        if m is None:
            continue
        r0, n0, e10, e30 = m
        rs0.append(r0)
        ns0.append(n0)
        if e10 is not None and not math.isnan(e10):
            e1s0.append(e10)
            e3s0.append(e30)
    print(
        f"  CC3-50 baseline (all):       R@10 {np.mean(rs0):.2%}  nDCG {np.mean(ns0):.4f}  "
        f"E@1 {np.mean(e1s0):.2%}  E@3 {np.mean(e3s0):.2%}",
        flush=True,
    )

    # CC3-50 on changed-only subset (where the lift should be concentrated).
    if changed_indices.size > 0:
        r, n, e1, e3 = compute_setup(
            I_corrected_100, qv_a, qv_b, qv_g, K_RET_FAST, qids_subset_indices=changed_indices
        )
        print(
            f"  CC3-50 corrected (changed-only, n={changed_indices.size:,}): "
            f"R@10 {r:.2%}  nDCG {n:.4f}  E@1 {e1:.2%}  E@3 {e3:.2%}",
            flush=True,
        )
        # baseline on changed-only subset.
        rs1, ns1, e1s1, e3s1 = [], [], [], []
        for qi in changed_indices:
            qid = qids[qi]
            s = cached_sumsim[qi].copy()
            s[~valid_orig[qi]] = -np.inf
            s[K_RET_FAST:] = -np.inf
            order = np.argsort(-s)[:K_EVAL]
            ordering = [faiss_pos_to_pid[int(cached_cands[qi, int(j)])] for j in order]
            m = per_query_metrics(ordering, qrels[qid])
            if m is None:
                continue
            r1, n1, e11, e31 = m
            rs1.append(r1)
            ns1.append(n1)
            if e11 is not None and not math.isnan(e11):
                e1s1.append(e11)
                e3s1.append(e31)
        print(
            f"  CC3-50 baseline (changed-only):  "
            f"R@10 {np.mean(rs1):.2%}  nDCG {np.mean(ns1):.4f}  "
            f"E@1 {np.mean(e1s1):.2%}  E@3 {np.mean(e3s1):.2%}",
            flush=True,
        )

    if args.skip_ce:
        print("\nskipping CE step (per --skip-ce). exiting.")
        return

    # ====================================================================
    # Step 3: CC4-100 with CE on changed queries only.
    # CE-rescore the corrected top-100 candidates ONLY for changed queries.
    # For unchanged queries, reuse cached CE scores.
    # ====================================================================
    print(
        f"\nCE-rescoring {changed_indices.size:,} changed queries x 100 candidates...", flush=True
    )
    from sentence_transformers import CrossEncoder

    device = (
        "mps"
        if torch.backends.mps.is_available()
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    ce = CrossEncoder("LiYuan/Amazon-Cup-Cross-Encoder-Regression", device=device)

    n_pairs_total = int(changed_indices.size) * 100
    ce_corrected = np.zeros((len(qids), 100), dtype=np.float32)
    sumsim_corrected = np.zeros((len(qids), 100), dtype=np.float32)
    # For unchanged queries: reuse cached.
    ce_top100_orig = np.load(os.path.join(INDEX_DIR, "ce_top100_scores.npy"))
    sumsim_top100_orig = np.load(os.path.join(INDEX_DIR, "ce_top100_sumsim.npy"))
    candidates_top100_orig = np.load(os.path.join(INDEX_DIR, "ce_top100_candidates.npy"))
    final_candidates = candidates_top100_orig.copy()
    for qi in range(len(qids)):
        if not changed_mask[qi]:
            ce_corrected[qi] = ce_top100_orig[qi]
            sumsim_corrected[qi] = sumsim_top100_orig[qi]

    # Compute sumsim for changed queries (over the new corrected candidate set).
    print("  computing sumsim for changed queries...", flush=True)
    for qi in changed_indices:
        positions = I_corrected_100[qi]
        good = positions >= 0
        if not good.any():
            continue
        idx = positions[good]
        sa = pv_a[idx] @ qv_a[qi]
        sb = pv_b[idx] @ qv_b[qi]
        sg = pv_g[idx] @ qv_g[qi]
        sumsim_corrected[qi, good] = (sa + sb + sg) / 3
        final_candidates[qi] = positions

    # CE on changed queries.
    print(f"  CE scoring {n_pairs_total:,} pairs...", flush=True)
    pairs_buf = []
    locs_buf = []
    n_done = 0
    t0 = time.time()
    for qi in changed_indices:
        q = corrected_queries[qi]
        positions = I_corrected_100[qi]
        for j in range(100):
            pos = int(positions[j])
            if pos < 0:
                continue
            pairs_buf.append((q, index_titles[pos]))
            locs_buf.append((qi, j))
        if len(pairs_buf) >= 4096:
            scores = ce.predict(pairs_buf, batch_size=args.ce_batch, show_progress_bar=False)
            for (qi2, j2), sc in zip(locs_buf, scores):
                ce_corrected[qi2, j2] = float(sc)
            n_done += len(pairs_buf)
            elapsed = time.time() - t0
            rate = n_done / max(elapsed, 1e-3)
            eta = (n_pairs_total - n_done) / max(rate, 1e-3)
            print(
                f"    {n_done:,}/{n_pairs_total:,} ({n_done / n_pairs_total:.1%}) "
                f"@ {rate:.0f}/s eta {eta / 60:.1f}m",
                flush=True,
            )
            pairs_buf.clear()
            locs_buf.clear()
    if pairs_buf:
        scores = ce.predict(pairs_buf, batch_size=args.ce_batch, show_progress_bar=False)
        for (qi2, j2), sc in zip(locs_buf, scores):
            ce_corrected[qi2, j2] = float(sc)

    # Compute CC4-100 with corrections.
    print("\ncomputing CC4-100 metrics...", flush=True)
    valid_corr = final_candidates >= 0

    def per_q_minmax(scores, mask):
        out = scores.copy()
        for qi in range(out.shape[0]):
            v = out[qi, mask[qi]]
            if v.size == 0:
                continue
            lo, hi = float(v.min()), float(v.max())
            out[qi, mask[qi]] = (v - lo) / max(hi - lo, 1e-8)
        return out

    nm_sum = per_q_minmax(sumsim_corrected, valid_corr)
    nm_ce = per_q_minmax(ce_corrected, valid_corr)
    fused = 0.75 * nm_sum + 0.25 * nm_ce

    rs, ns, e1s, e3s = [], [], [], []
    for qi, qid in enumerate(qids):
        s = fused[qi].copy()
        s[~valid_corr[qi]] = -np.inf
        order = np.argsort(-s)[:K_EVAL]
        ordering = [faiss_pos_to_pid[int(final_candidates[qi, int(j)])] for j in order]
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
        f"  CC4-100 corrected (all):     R@10 {np.mean(rs):.2%}  nDCG {np.mean(ns):.4f}  "
        f"E@1 {np.mean(e1s):.2%}  E@3 {np.mean(e3s):.2%}",
        flush=True,
    )

    # CC4-100 baseline (all): from cached.
    nm_sum_b = per_q_minmax(sumsim_top100_orig, candidates_top100_orig >= 0)
    nm_ce_b = per_q_minmax(ce_top100_orig, candidates_top100_orig >= 0)
    fused_b = 0.75 * nm_sum_b + 0.25 * nm_ce_b
    rs, ns, e1s, e3s = [], [], [], []
    for qi, qid in enumerate(qids):
        s = fused_b[qi].copy()
        s[~(candidates_top100_orig[qi] >= 0)] = -np.inf
        order = np.argsort(-s)[:K_EVAL]
        ordering = [faiss_pos_to_pid[int(candidates_top100_orig[qi, int(j)])] for j in order]
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
        f"  CC4-100 baseline (all):      R@10 {np.mean(rs):.2%}  nDCG {np.mean(ns):.4f}  "
        f"E@1 {np.mean(e1s):.2%}  E@3 {np.mean(e3s):.2%}",
        flush=True,
    )

    # Changed-only subset metric.
    if changed_indices.size > 0:
        rs, ns, e1s, e3s = [], [], [], []
        for qi in changed_indices:
            qid = qids[qi]
            s = fused[qi].copy()
            s[~valid_corr[qi]] = -np.inf
            order = np.argsort(-s)[:K_EVAL]
            ordering = [faiss_pos_to_pid[int(final_candidates[qi, int(j)])] for j in order]
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
            f"  CC4-100 corrected (changed-only): R@10 {np.mean(rs):.2%} nDCG {np.mean(ns):.4f} "
            f"E@1 {np.mean(e1s):.2%} E@3 {np.mean(e3s):.2%}",
            flush=True,
        )
        rs, ns, e1s, e3s = [], [], [], []
        for qi in changed_indices:
            qid = qids[qi]
            s = fused_b[qi].copy()
            s[~(candidates_top100_orig[qi] >= 0)] = -np.inf
            order = np.argsort(-s)[:K_EVAL]
            ordering = [faiss_pos_to_pid[int(candidates_top100_orig[qi, int(j)])] for j in order]
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
            f"  CC4-100 baseline  (changed-only): R@10 {np.mean(rs):.2%} nDCG {np.mean(ns):.4f} "
            f"E@1 {np.mean(e1s):.2%} E@3 {np.mean(e3s):.2%}",
            flush=True,
        )


if __name__ == "__main__":
    main()
