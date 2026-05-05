#!/usr/bin/env python3
"""Probe: negation-aware query parsing.

For queries containing "without X" / "no X" / "non-X" / "X-free" patterns,
extract the negated tokens and post-filter CC5 candidates that contain those
tokens in their title text.

Tests two modes:
  hard filter: drop any candidate whose title contains a negated token
  soft penalty: subtract a penalty from candidates with negated tokens

Compares R@10 / E@1 on the negation-query subset and on aggregate against
the CC5 baseline (no filtering).

Reads precomputed candidates and CC5 fusion scores from cache; no model
inference. ~5 minute probe.
"""

import json
import os
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INDEX_DIR = os.path.join(SCRIPT_DIR, "combined_index_us_minilm")

# Negation patterns. Capture group 1 is the negated noun phrase (rest of query
# OR explicit token after "without"/"no"). We're loose on tokenization and
# rely on string-contains in titles, so a few extra captured stopwords are OK.
NEG_PATTERNS = [
    # Tight one-noun-after capture; avoid greedy ".+$" which grabs brand names.
    re.compile(r"\bwithout\s+([a-z][a-z0-9-]+)\b", re.IGNORECASE),
    re.compile(r"\bno\s+([a-z][a-z0-9-]+)\b", re.IGNORECASE),
    re.compile(r"\bnon[\s-]([a-z][a-z0-9-]+)\b", re.IGNORECASE),
    re.compile(r"\b([a-z][a-z0-9]+)-free\b", re.IGNORECASE),
    re.compile(r"\bfree of\s+([a-z][a-z0-9-]+)\b", re.IGNORECASE),
    re.compile(r"\bfree from\s+([a-z][a-z0-9-]+)\b", re.IGNORECASE),
]

STOPWORDS = {
    "a",
    "an",
    "the",
    "of",
    "and",
    "or",
    "for",
    "with",
    "in",
    "on",
    "to",
    "is",
    "are",
}


def extract_negated_tokens(query):
    """Return a list of lowercased tokens that should NOT appear in the matching product."""
    tokens = []
    for pat in NEG_PATTERNS:
        for m in pat.finditer(query):
            phrase = m.group(1).lower()
            for tok in re.findall(r"[a-z][a-z0-9-]+", phrase):
                if tok not in STOPWORDS and len(tok) >= 3:
                    tokens.append(tok)
    return tokens


def per_query_minmax(scores):
    lo = scores.min(axis=1, keepdims=True)
    hi = scores.max(axis=1, keepdims=True)
    rng = np.maximum(hi - lo, 1e-9)
    return (scores - lo) / rng


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
    print(f"  {len(qids):,} eligible queries", flush=True)

    # Identify negation queries
    print("\nextracting negated tokens...", flush=True)
    neg_tokens_per_qid = {}
    for qid in qids:
        toks = extract_negated_tokens(queries_all[qid])
        if toks:
            neg_tokens_per_qid[qid] = toks
    print(f"  {len(neg_tokens_per_qid):,} queries match a negation pattern", flush=True)
    print("  examples:", flush=True)
    for qid in list(neg_tokens_per_qid.keys())[:8]:
        print(f"    {queries_all[qid]!r:60s} -> {neg_tokens_per_qid[qid]}", flush=True)

    # Load CC5 cached scores + candidates
    print("\nloading CC5 cached scores...", flush=True)
    cands = np.load(os.path.join(INDEX_DIR, "ce_top100_candidates.npy"))
    sumsim = np.load(os.path.join(INDEX_DIR, "ce_top100_sumsim.npy"))
    liyuan = np.load(os.path.join(INDEX_DIR, "ce_top100_scores.npy"))
    bge = np.load(os.path.join(INDEX_DIR, "bge_rerank/bge_scores_top100_all.npy"))

    test_queries_order = list(queries_all.keys())
    qid_to_row = {qid: i for i, qid in enumerate(test_queries_order)}
    rows = np.array([qid_to_row[qid] for qid in qids if qid in qid_to_row])
    cands = cands[rows]
    sumsim_n = per_query_minmax(sumsim[rows])
    liyuan_n = per_query_minmax(liyuan[rows])
    bge_n = per_query_minmax(bge[rows])

    # CC5 weighted fusion (0.4, 0.2, 0.4)
    fused = 0.4 * sumsim_n + 0.2 * liyuan_n + 0.4 * bge_n
    K = 100
    K_EVAL = 10
    n_q = cands.shape[0]

    def evaluate(scored_fused, sub_cands, label):
        recalls, e1s = [], []
        for i in range(n_q):
            s = scored_fused[i]
            top_idx = np.argpartition(-s, K_EVAL)[:K_EVAL]
            order = top_idx[np.argsort(-s[top_idx])]
            top_pos = sub_cands[i, order]
            top_pids = [
                faiss_pos_to_pid[p] if 0 <= p < len(faiss_pos_to_pid) else None for p in top_pos
            ]
            qr = qrels[qids[i]]
            es = {p for p, g in qr.items() if g >= 2}
            ee = {p for p, g in qr.items() if g == 3}
            if es:
                recalls.append((qids[i], sum(1 for p in top_pids if p in es) / len(es)))
            if ee:
                e1s.append((qids[i], 1.0 if top_pids[0] in ee else 0.0))
        return recalls, e1s

    print("\nbaseline CC5 top-10 (no negation filter)...", flush=True)
    base_r, base_e = evaluate(fused, cands, "baseline")

    # Apply hard negation filter: -inf any candidate whose title contains a negated token
    print("\nhard negation filter (drop candidates with any negated token in title)...", flush=True)
    hard_fused = fused.copy()
    n_filtered = 0
    n_neg_queries = 0
    for i in range(n_q):
        qid = qids[i]
        if qid not in neg_tokens_per_qid:
            continue
        n_neg_queries += 1
        neg_toks = neg_tokens_per_qid[qid]
        for j in range(K):
            pos = int(cands[i, j])
            if pos < 0 or pos >= len(index_titles):
                continue
            title_lc = index_titles[pos].lower()
            if any(re.search(rf"\b{re.escape(t)}\b", title_lc) for t in neg_toks):
                hard_fused[i, j] = -1e9
                n_filtered += 1
    print(
        f"  {n_neg_queries:,} negation queries; filtered {n_filtered:,} (q,doc) pairs total "
        f"(avg {n_filtered / max(1, n_neg_queries):.1f}/query)",
        flush=True,
    )

    hard_r, hard_e = evaluate(hard_fused, cands, "hard")

    # Soft penalty: subtract 0.2 from any candidate with negated token
    print(
        "\nsoft negation penalty (subtract 0.2 from candidates with negated tokens)...", flush=True
    )
    soft_fused = fused.copy()
    for i in range(n_q):
        qid = qids[i]
        if qid not in neg_tokens_per_qid:
            continue
        neg_toks = neg_tokens_per_qid[qid]
        for j in range(K):
            pos = int(cands[i, j])
            if pos < 0 or pos >= len(index_titles):
                continue
            title_lc = index_titles[pos].lower()
            if any(re.search(rf"\b{re.escape(t)}\b", title_lc) for t in neg_toks):
                soft_fused[i, j] -= 0.2

    soft_r, soft_e = evaluate(soft_fused, cands, "soft")

    # Aggregate metrics
    def agg(pairs):
        return statistics.mean([v for _, v in pairs]) if pairs else 0

    # Subset: only the queries that had a negation pattern
    def agg_subset(pairs, qid_set):
        v = [v for q, v in pairs if q in qid_set]
        return statistics.mean(v) if v else 0, len(v)

    neg_set = set(neg_tokens_per_qid.keys())
    print("\n=== AGGREGATE (full eligible set) ===", flush=True)
    print(f"{'setup':<10}  R@10    E@1     n_es")
    for label, rs, es in [
        ("baseline", base_r, base_e),
        ("hard", hard_r, hard_e),
        ("soft", soft_r, soft_e),
    ]:
        print(f"{label:<10}  {agg(rs) * 100:5.2f}  {agg(es) * 100:5.2f}  {len(rs)}")

    print("\n=== NEGATION-ONLY SUBSET ===", flush=True)
    print(f"{'setup':<10}  R@10    E@1     n_es")
    for label, rs, es in [
        ("baseline", base_r, base_e),
        ("hard", hard_r, hard_e),
        ("soft", soft_r, soft_e),
    ]:
        r_avg, n_r = agg_subset(rs, neg_set)
        e_avg, _ = agg_subset(es, neg_set)
        print(f"{label:<10}  {r_avg * 100:5.2f}  {e_avg * 100:5.2f}  {n_r}")

    out = {
        "n_eligible": n_q,
        "n_negation_queries": len(neg_tokens_per_qid),
        "examples": {
            qid: {"query": queries_all[qid], "neg_tokens": toks}
            for qid, toks in list(neg_tokens_per_qid.items())[:30]
        },
    }
    with open("/tmp/negation_probe.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\nsaved negation-probe metadata to /tmp/negation_probe.json", flush=True)


if __name__ == "__main__":
    main()
