#!/usr/bin/env python3
"""Phonetic-fallback spell correction probe.

The existing pyspellchecker-based correction misses cases like
'moniter pivo' because the catalog itself contains 'moniter' as a
real (typoed) listing — pyspellchecker treats it as in-vocab. A
phonetic backstop catches these: if a query token has a
phonetic-equivalent catalog token with MUCH higher frequency, swap.

Approach: simple consonant-skeleton encoding (drops vowels except
leading vowel). Two tokens are phonetic-equivalent if their
skeletons match. Trigger swap only when the alternative is at least
20x more frequent than the original AND within edit distance 3.

Pipeline:
  1. Apply the existing pyspellchecker correction (catalog vocab,
     distance 2) — same logic as eval_spell_correct.py.
  2. For tokens still unchanged AND below a frequency threshold,
     try phonetic match against the catalog vocab.
  3. Re-retrieve via bm25s, eval CC3-50.

Usage:
    python evaluation/eval_spell_phonetic.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json  # noqa: E402
import math  # noqa: E402
import os  # noqa: E402
import re  # noqa: E402
import time  # noqa: E402
from collections import defaultdict  # noqa: E402

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import bm25s  # noqa: E402
import numpy as np  # noqa: E402
import Stemmer  # noqa: E402
from spellchecker import SpellChecker  # noqa: E402

SCRIPT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INDEX_DIR = os.path.join(SCRIPT_DIR, "combined_index_us_minilm")
K_EVAL = 10
K_RET = 50
TOK_RE = re.compile(r"[a-z0-9]+")


def phonetic_skeleton(tok):
    """Drop vowels (except leading), collapse repeated consonants. Cheap proxy
    for Metaphone — sufficient for catching typo-pairs like 'moniter'/'monitor',
    'tredmills'/'treadmills', 'inpjone'/'iphone' that share consonant order."""
    if not tok:
        return ""
    out = [tok[0]]
    for c in tok[1:]:
        if c in "aeiouy":
            continue
        if out and out[-1] == c:
            continue
        out.append(c)
    return "".join(out)


def edit_distance(a, b):
    """Standard Levenshtein, capped early at 4."""
    if abs(len(a) - len(b)) > 3:
        return 4
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        curr = [i]
        for j, cb in enumerate(b, 1):
            curr.append(
                min(
                    prev[j] + 1,
                    curr[-1] + 1,
                    prev[j - 1] + (ca != cb),
                )
            )
        prev = curr
        if min(prev) > 3:
            return 4
    return prev[-1]


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
    print(f"  {len(qids):,} eval queries", flush=True)

    # Load catalog spell vocab.
    print("loading catalog vocab + building phonetic index...", flush=True)
    with open(os.path.join(INDEX_DIR, "spell_vocab.json")) as f:
        vocab = json.load(f)
    vocab_set = set(vocab.keys())
    print(f"  {len(vocab):,} catalog tokens", flush=True)

    # Build phonetic index.
    t0 = time.time()
    phon_to_tokens = defaultdict(list)
    for tok, freq in vocab.items():
        if len(tok) <= 3 or any(c.isdigit() for c in tok):
            continue
        skel = phonetic_skeleton(tok)
        if len(skel) < 3:
            continue
        phon_to_tokens[skel].append((tok, freq))
    # Sort each bucket by frequency desc.
    for skel in phon_to_tokens:
        phon_to_tokens[skel].sort(key=lambda x: -x[1])
    print(f"  {len(phon_to_tokens):,} phonetic buckets ({time.time() - t0:.0f}s)", flush=True)

    # Build the spellchecker (same as ship version).
    spell = SpellChecker(language=None, distance=2)
    for tok, freq in vocab.items():
        spell.word_frequency.add(tok, freq)

    # Phonetic-fallback correction.
    PHON_FREQ_RATIO = 20  # alternative must be 20x more frequent than original
    PHON_MIN_FREQ_TOKEN = 100  # don't consider alternatives below 100 occurrences

    def correct_with_phonetic(q):
        tokens = TOK_RE.findall(q.lower())
        if not tokens:
            return q, False
        out = []
        changed = False
        for tok in tokens:
            # Stage 1: catalog edit-distance correction (same as ship version).
            corrected = tok
            if not (
                tok in vocab_set or len(tok) <= 2 or tok.isdigit() or any(c.isdigit() for c in tok)
            ):
                cands = spell.candidates(tok) or set()
                best = None
                best_freq = -1
                for c in cands:
                    if c not in vocab_set:
                        continue
                    f = spell.word_frequency[c]
                    if f > best_freq:
                        best_freq = f
                        best = c
                if best and best != tok:
                    corrected = best

            # Stage 2: phonetic fallback. Only fire if Stage 1 didn't change tok,
            # tok is in vocab (otherwise Stage 1 would have done its job), and
            # tok's frequency is suspiciously low compared to a phonetic peer.
            if corrected == tok and len(tok) > 3 and not any(c.isdigit() for c in tok):
                tok_freq = vocab.get(tok, 0)
                skel = phonetic_skeleton(tok)
                bucket = phon_to_tokens.get(skel, [])
                # bucket is freq-sorted. Top entry is the most likely "real" token.
                if bucket:
                    alt_tok, alt_freq = bucket[0]
                    if (
                        alt_tok != tok
                        and alt_freq >= PHON_MIN_FREQ_TOKEN
                        and alt_freq >= max(tok_freq * PHON_FREQ_RATIO, PHON_MIN_FREQ_TOKEN)
                        and edit_distance(tok, alt_tok) <= 3
                    ):
                        corrected = alt_tok

            if corrected != tok:
                changed = True
            out.append(corrected)
        return " ".join(out), changed

    print("\ncorrecting queries (edit + phonetic)...", flush=True)
    t0 = time.time()
    corrected = []
    n_changed = 0
    for qi, q in enumerate(queries):
        c, ch = correct_with_phonetic(q)
        corrected.append(c)
        if ch:
            n_changed += 1
        if (qi + 1) % 5000 == 0:
            print(f"  {qi + 1:,}/{len(queries):,} ({n_changed:,} changed)", flush=True)
    print(
        f"  done {time.time() - t0:.0f}s; {n_changed:,} ({n_changed / len(queries):.1%}) changed",
        flush=True,
    )

    # Show a few examples that the original spell missed (compare with spell-only).
    spell_corrected = []
    for q in queries:
        tokens = TOK_RE.findall(q.lower())
        out = []
        for tok in tokens:
            if tok in vocab_set or len(tok) <= 2 or tok.isdigit() or any(c.isdigit() for c in tok):
                out.append(tok)
                continue
            cands = spell.candidates(tok) or set()
            best = None
            best_freq = -1
            for c in cands:
                if c not in vocab_set:
                    continue
                f = spell.word_frequency[c]
                if f > best_freq:
                    best_freq = f
                    best = c
            out.append(best if best and best != tok else tok)
        spell_corrected.append(" ".join(out))

    print("\nexamples where phonetic fallback fires (and spell didn't):")
    samples = 0
    for qi in range(len(queries)):
        if corrected[qi] != spell_corrected[qi] and samples < 15:
            print(
                f"  '{queries[qi][:60]}' | spell-only: '{spell_corrected[qi][:60]}' "
                f"| +phon: '{corrected[qi][:60]}'"
            )
            samples += 1

    out_dir = os.path.join(INDEX_DIR, "spell_phon")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "corrected_queries.json"), "w") as f:
        json.dump(corrected, f)

    # Re-retrieve + eval.
    print("\nbm25s retrieve with edit+phon corrections...", flush=True)
    bm25s_idx = bm25s.BM25.load(os.path.join(INDEX_DIR, "bm25s_index"), mmap=False)
    stemmer = Stemmer.Stemmer("english")
    qt = bm25s.tokenize(corrected, stopwords="en", stemmer=stemmer, show_progress=False)
    results, _ = bm25s_idx.retrieve(qt, k=100, show_progress=False)
    I = np.asarray(results, dtype=np.int64)

    print("encoding corrected queries...", flush=True)
    qv_a = encode_subproc(os.path.join(SCRIPT_DIR, "query_model_us_full_6m_mnrl"), corrected)
    qv_b = encode_subproc(os.path.join(SCRIPT_DIR, "query_model_us_qrels_mnrl_hardneg"), corrected)
    qv_g = encode_subproc(os.path.join(SCRIPT_DIR, "query_model_us_esci_supervised"), corrected)
    pv_a = np.load(os.path.join(INDEX_DIR, "rerank_A.vecs.fp16.npy")).astype(np.float32)
    pv_b = np.load(os.path.join(INDEX_DIR, "rerank_B.vecs.fp16.npy")).astype(np.float32)
    pv_g = np.load(os.path.join(INDEX_DIR, "rerank_G.vecs.fp16.npy")).astype(np.float32)

    rs, ns, e1s, e3s = [], [], [], []
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
        r, n, e1, e3 = m
        rs.append(r)
        ns.append(n)
        if e1 is not None and not math.isnan(e1):
            e1s.append(e1)
            e3s.append(e3)
    print(
        f"\nCC3-50 + edit+phon: R@10 {np.mean(rs):.2%}  nDCG {np.mean(ns):.4f}  "
        f"E@1 {np.mean(e1s):.2%}  E@3 {np.mean(e3s):.2%}"
    )
    print("baseline    spell-only: R@10 21.84%, E@1 42.53% (shipped fast SOTA)")
    print("baseline    no-spell  : R@10 21.61%, E@1 42.10%")


if __name__ == "__main__":
    main()
