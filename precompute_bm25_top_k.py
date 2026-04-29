#!/usr/bin/env python3
"""Precompute BM25 top-100 positions for the 22,458 ESCI test queries.

Reads `combined_index_us_minilm/tantivy_index` (en_stem tokenizer) and writes
`combined_index_us_minilm/bm25_top100.npy` — an (N_QUERIES, 100) int32 array
of FAISS-aligned product positions, with -1 padding when fewer than 100 hits.

Order matches the qid order produced by eval_mnrl_retriever.py so the array
can be loaded directly without re-deriving qids.
"""

import json
import os
import re
import time
from collections import defaultdict

import numpy as np
import tantivy

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INDEX_DIR = os.path.join(SCRIPT_DIR, "combined_index_us_minilm")
TANTIVY_PATH = os.path.join(INDEX_DIR, "tantivy_index")
OUT_PATH = os.path.join(INDEX_DIR, "bm25_top100.npy")

K = 100
PUNCT_RE = re.compile(r"[^\w\s]")


def safe_parse(idx, q):
    try:
        return idx.parse_query(q, ["title"])
    except ValueError:
        cleaned = PUNCT_RE.sub(" ", q).strip()
        if not cleaned:
            return None
        try:
            return idx.parse_query(cleaned, ["title"])
        except ValueError:
            return None


def main():
    print("loading qrels + queries...", flush=True)
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

    qids = [qid for qid in queries_all if qid in qrels and any(g >= 2 for g in qrels[qid].values())]
    queries = [queries_all[qid] for qid in qids]
    print(f"  {len(qids):,} eval queries", flush=True)

    print("loading tantivy index...", flush=True)
    idx = tantivy.Index.open(TANTIVY_PATH)
    idx.reload()
    s = idx.searcher()
    print(f"  {s.num_docs:,} docs", flush=True)

    print("loading titles.json + building title -> first-position map...", flush=True)
    with open(os.path.join(INDEX_DIR, "titles.json")) as f:
        titles = json.load(f)
    title_to_pos = {}
    for i, t in enumerate(titles):
        if t not in title_to_pos:
            title_to_pos[t] = i
    print(f"  {len(titles):,} titles, {len(title_to_pos):,} unique", flush=True)

    out = np.full((len(queries), K), -1, dtype=np.int32)
    n_empty = 0
    n_unparseable = 0
    t0 = time.time()
    for qi, q in enumerate(queries):
        parsed = safe_parse(idx, q)
        if parsed is None:
            n_unparseable += 1
            continue
        hits = s.search(parsed, limit=K).hits
        if not hits:
            n_empty += 1
            continue
        positions = []
        for _, addr in hits:
            title = s.doc(addr)["title"][0]
            pos = title_to_pos.get(title)
            if pos is not None:
                positions.append(pos)
                if len(positions) == K:
                    break
        for j, p in enumerate(positions):
            out[qi, j] = p
        if (qi + 1) % 2000 == 0:
            elapsed = time.time() - t0
            rate = (qi + 1) / elapsed
            eta = (len(queries) - (qi + 1)) / rate
            print(f"  {qi + 1:,}/{len(queries):,}  rate={rate:.0f} q/s  eta={eta:.0f}s", flush=True)

    elapsed = time.time() - t0
    print(
        f"  done {elapsed:.0f}s. unparseable={n_unparseable}, empty={n_empty}",
        flush=True,
    )

    np.save(OUT_PATH, out)
    print(f"wrote {OUT_PATH}  shape={out.shape}", flush=True)


if __name__ == "__main__":
    main()
