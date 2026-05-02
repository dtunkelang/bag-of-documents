#!/usr/bin/env python3
"""LLM query rewriting via Qwen3-4B-4bit (mlx_lm) on conversational
queries from the ESCI test set.

Filter: queries that look conversational (>= 8 tokens, or contain
function words like 'I want', 'looking for', 'with a', etc.). Roughly
1-2K queries.

Few-shot prompt: convert conversational query → 2-5 keyword search query.

Re-retrieve via bm25s, eval CC3-50 (no CE for speed). The probe targets
the long-natural-language failures the error analysis surfaced.

Outputs to combined_index_us_minilm/llm/.

Usage:
    python evaluation/eval_llm_rewrite.py
    python evaluation/eval_llm_rewrite.py --max-queries 500  # smoke test
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
from collections import defaultdict  # noqa: E402

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import bm25s  # noqa: E402
import numpy as np  # noqa: E402
import Stemmer  # noqa: E402

SCRIPT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INDEX_DIR = os.path.join(SCRIPT_DIR, "combined_index_us_minilm")
K_EVAL = 10
K_RET = 50

CONVERSATIONAL_MARKERS = [
    "i want",
    "i need",
    "i'm looking",
    "looking for",
    "i am looking",
    "i would like",
    "i'd like",
    "do you have",
    "can you find",
    "does anyone",
    "anyone know",
    "please find",
    "what is the best",
    "best ",
    "something that",
    "thing that",
    "that has",
    "with a",
    "with the",
    "for my",
    " is light",
    "lightweight",
    "but can",
    "but is",
    "and is ",
    "any other",
]


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


def is_conversational(q, min_tokens=8):
    """Heuristic: long queries OR queries with function-word markers."""
    if len(q.split()) >= min_tokens:
        return True
    ql = q.lower()
    return any(m in ql for m in CONVERSATIONAL_MARKERS)


def parse_rewritten(s):
    """Extract a clean rewrite from LLM output. Take the first non-empty line."""
    for line in s.strip().split("\n"):
        line = line.strip()
        # Strip leading 'Search:' or '#' or quotes.
        line = re.sub(r"^(search\s*:?\s*|query\s*:?\s*|[#>*-]+\s*|\"+|\'+)", "", line, flags=re.I)
        line = re.sub(r"(\"+|\'+)\s*$", "", line).strip()
        if line:
            return line
    return ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--model",
        default="mlx-community/Qwen3-4B-4bit",
    )
    ap.add_argument("--max-queries", type=int, default=0, help="0 = all conversational queries")
    ap.add_argument("--max-tokens", type=int, default=24, help="generation length cap")
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

    conv_indices = [qi for qi, q in enumerate(queries) if is_conversational(q)]
    if args.max_queries > 0:
        conv_indices = conv_indices[: args.max_queries]
    print(f"  {len(conv_indices):,}/{len(queries):,} conversational queries to rewrite", flush=True)

    print(f"loading {args.model} via mlx_lm...", flush=True)
    from mlx_lm import generate, load

    t0 = time.time()
    model, tokenizer = load(args.model)
    print(f"  loaded in {time.time() - t0:.0f}s", flush=True)

    prompt_template = """You convert conversational shopping queries into short keyword search queries (2-5 keywords) suitable for a product search engine. Output only the keywords. No explanation.

Query: I'm looking for a wireless mouse for my laptop
Search: wireless mouse laptop

Query: I want a sturdy step ladder for my garage
Search: sturdy step ladder

Query: do you have anything that can hold a phone in the car
Search: phone car holder

Query: something that has a good amd ryzen processor in it
Search: amd ryzen laptop

Query: {q}
Search:"""

    rewritten = list(queries)  # default = original
    print("\nrewriting...", flush=True)
    n_done = 0
    n_changed = 0
    t0 = time.time()
    for qi in conv_indices:
        q = queries[qi]
        prompt = prompt_template.format(q=q)
        try:
            resp = generate(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                max_tokens=args.max_tokens,
                verbose=False,
            )
        except Exception as e:
            print(f"  qi={qi} gen failed: {e}", flush=True)
            continue
        new_q = parse_rewritten(resp)
        # Sanity guards.
        if not new_q:
            continue
        if len(new_q) > 200 or len(new_q.split()) > 12:
            continue
        if new_q.lower() == q.lower():
            continue
        rewritten[qi] = new_q
        n_changed += 1
        n_done += 1
        if n_done % 100 == 0:
            elapsed = time.time() - t0
            rate = n_done / max(elapsed, 1e-3)
            eta = (len(conv_indices) - n_done) / max(rate, 1e-3)
            print(
                f"  {n_done:,}/{len(conv_indices):,} ({n_changed:,} changed) "
                f"@ {rate:.1f}/s eta {eta / 60:.1f}m",
                flush=True,
            )
    print(
        f"\ndone in {time.time() - t0:.0f}s; rewrote {n_changed:,}/{len(conv_indices):,}",
        flush=True,
    )

    print("\nsample rewrites:")
    samples = 0
    for qi in conv_indices:
        if rewritten[qi] != queries[qi] and samples < 12:
            print(f"  '{queries[qi][:80]}' → '{rewritten[qi]}'")
            samples += 1

    llm_dir = os.path.join(INDEX_DIR, "llm")
    os.makedirs(llm_dir, exist_ok=True)
    with open(os.path.join(llm_dir, "rewritten_queries.json"), "w") as f:
        json.dump(rewritten, f)
    changed_mask = np.array([rewritten[qi] != queries[qi] for qi in range(len(queries))])
    np.save(os.path.join(llm_dir, "changed_mask.npy"), changed_mask)

    # Re-retrieve + eval.
    print("\nbm25s retrieve with LLM-rewritten queries...", flush=True)
    bm25s_idx = bm25s.BM25.load(os.path.join(INDEX_DIR, "bm25s_index"), mmap=False)
    stemmer = Stemmer.Stemmer("english")
    qt = bm25s.tokenize(rewritten, stopwords="en", stemmer=stemmer, show_progress=False)
    results, _ = bm25s_idx.retrieve(qt, k=100, show_progress=False)
    I_llm = np.asarray(results, dtype=np.int64)
    np.save(os.path.join(llm_dir, "I_llm_top100.npy"), I_llm)

    print("encoding rewritten queries with rerank_a/b/g...", flush=True)
    qv_a = encode_subproc(os.path.join(SCRIPT_DIR, "query_model_us_full_6m_mnrl"), rewritten)
    qv_b = encode_subproc(os.path.join(SCRIPT_DIR, "query_model_us_qrels_mnrl_hardneg"), rewritten)
    qv_g = encode_subproc(os.path.join(SCRIPT_DIR, "query_model_us_esci_supervised"), rewritten)
    pv_a = np.load(os.path.join(INDEX_DIR, "rerank_A.vecs.fp16.npy")).astype(np.float32)
    pv_b = np.load(os.path.join(INDEX_DIR, "rerank_B.vecs.fp16.npy")).astype(np.float32)
    pv_g = np.load(os.path.join(INDEX_DIR, "rerank_G.vecs.fp16.npy")).astype(np.float32)

    print("computing CC3-50 with rewritten queries...", flush=True)
    rs, ns, e1s, e3s = [], [], [], []
    rs_pq = np.full(len(qids), np.nan, dtype=np.float32)
    e1s_pq = np.full(len(qids), np.nan, dtype=np.float32)
    for qi, qid in enumerate(qids):
        positions = [int(p) for p in I_llm[qi, :K_RET] if p >= 0]
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
    np.save(os.path.join(llm_dir, "llm_per_query_R10.npy"), rs_pq)
    np.save(os.path.join(llm_dir, "llm_per_query_E1.npy"), e1s_pq)
    print(
        f"\nCC3-50 with LLM rewrites: R@10 {np.mean(rs):.2%} nDCG {np.mean(ns):.4f} "
        f"E@1 {np.mean(e1s):.2%} E@3 {np.mean(e3s):.2%}"
    )

    # Changed-only subset.
    if n_changed > 0:
        ci = np.array([qi for qi in conv_indices if rewritten[qi] != queries[qi]])
        rs2, ns2, e1s2, e3s2 = [], [], [], []
        for qi in ci:
            qid = qids[qi]
            positions = [int(p) for p in I_llm[qi, :K_RET] if p >= 0]
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
            rs2.append(r)
            ns2.append(n)
            if e1 is not None and not math.isnan(e1):
                e1s2.append(e1)
                e3s2.append(e3)
        print(
            f"  changed-only subset (n={ci.size:,}): R@10 {np.mean(rs2):.2%} "
            f"nDCG {np.mean(ns2):.4f} E@1 {np.mean(e1s2):.2%}"
        )


if __name__ == "__main__":
    main()
