#!/usr/bin/env python3
"""Per-bucket BoD lift on ESCI-US — direct counterpart to diagnose_bestbuy_lift.py.

Goal: see whether the BestBuy +17.5pp vs ESCI +4pp gap is explained by the
size of the "base-blind" query subset (queries where base R@10 = 0).

Treats E-grade (relevance=3) qrels as positives, mirroring BestBuy's
clicked-SKU = positive convention. R@10 is "fraction of E-graded products
in top-10 / total E-graded for that query".

Comparison axes mirror the BestBuy diagnostic:
  - base difficulty (0 hits, partial, full)
  - query length

Catalog vectors:
  - base MiniLM:    encoded fresh (cached at combined_index_us_minilm/base_catalog.vecs.fp16.npy)
  - BoD MiniLM:     reuses combined_index_us_minilm/rerank_A.vecs.fp16.npy (the 6M-MNRL retriever)
"""

import json
import os
from collections import defaultdict

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

DATA_DIR = "esci_us_data"
INDEX_DIR = "combined_index_us_minilm"
BASE_CACHE = os.path.join(INDEX_DIR, "base_catalog.vecs.fp16.npy")


def load_or_encode_base(titles, base_name, device):
    if os.path.exists(BASE_CACHE):
        print(f"loading cached base catalog from {BASE_CACHE}...", flush=True)
        return np.load(BASE_CACHE).astype(np.float32)
    print(f"encoding {len(titles):,} titles with {base_name}...", flush=True)
    model = SentenceTransformer(base_name, device=device)
    v = model.encode(
        titles, normalize_embeddings=True, batch_size=128, show_progress_bar=True
    ).astype(np.float32)
    np.save(BASE_CACHE, v.astype(np.float16))
    print(f"  cached at {BASE_CACHE}", flush=True)
    return v


def main():
    print("loading data...", flush=True)
    with open(os.path.join(DATA_DIR, "product_ids.json")) as f:
        pids = json.load(f)
    with open(os.path.join(INDEX_DIR, "titles.json")) as f:
        titles = json.load(f)
    pid_to_idx = {p: i for i, p in enumerate(pids)}

    queries_by_qid = {}
    with open(os.path.join(DATA_DIR, "test_queries.jsonl")) as f:
        for line in f:
            d = json.loads(line)
            queries_by_qid[d["query_id"]] = d["query"]

    e_pos = defaultdict(set)
    es_pos = defaultdict(set)
    with open(os.path.join(DATA_DIR, "test_qrels.jsonl")) as f:
        for line in f:
            r = json.loads(line)
            pid = r["product_id"]
            if pid not in pid_to_idx:
                continue
            i = pid_to_idx[pid]
            if r["relevance"] == 3:
                e_pos[r["query_id"]].add(i)
                es_pos[r["query_id"]].add(i)
            elif r["relevance"] == 2:
                es_pos[r["query_id"]].add(i)

    qids = sorted(queries_by_qid)
    queries = [queries_by_qid[q] for q in qids]
    print(
        f"  catalog={len(pids):,}  queries={len(queries):,}  "
        f"E-pos queries={sum(1 for v in e_pos.values() if v):,}  "
        f"ES-pos queries={sum(1 for v in es_pos.values() if v):,}",
        flush=True,
    )

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    base_pv = load_or_encode_base(titles, "all-MiniLM-L6-v2", device)
    print("loading BoD catalog (rerank_A.vecs.fp16.npy)...", flush=True)
    bod_pv = np.load(os.path.join(INDEX_DIR, "rerank_A.vecs.fp16.npy")).astype(np.float32)

    print("loading models for query encoding...", flush=True)
    base = SentenceTransformer("all-MiniLM-L6-v2", device=device)
    bod = SentenceTransformer("query_model_us_full_6m_mnrl", device=device)
    print("encoding queries (base)...", flush=True)
    base_qv = base.encode(
        queries, normalize_embeddings=True, batch_size=256, show_progress_bar=True
    ).astype(np.float32)
    print("encoding queries (BoD)...", flush=True)
    bod_qv = bod.encode(
        queries, normalize_embeddings=True, batch_size=256, show_progress_bar=True
    ).astype(np.float32)

    # Free the encoders before the heavy matmul phase.
    del base, bod
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

    # Per-query top-10 hits, keyed on E-only (matches BestBuy clicked-only convention).
    # Smaller chunk to stay under 32GB peak RAM (each chunk allocates a
    # chunk × 1.2M fp32 sims matrix; 1024 = 4.6GB, 256 = 1.15GB).
    per_q = []
    chunk = 256
    k = 10
    n_chunks = (len(qids) + chunk - 1) // chunk
    for ci, start in enumerate(range(0, len(qids), chunk)):
        end = min(start + chunk, len(qids))
        bsim = base_qv[start:end] @ base_pv.T
        btopk = np.argpartition(-bsim, k, axis=1)[:, :k]
        del bsim
        dsim = bod_qv[start:end] @ bod_pv.T
        dtopk = np.argpartition(-dsim, k, axis=1)[:, :k]
        del dsim
        for j, gi in enumerate(range(start, end)):
            qid = qids[gi]
            g = e_pos.get(qid, set())
            if not g:
                continue
            bh = len({int(x) for x in btopk[j]} & g)
            dh = len({int(x) for x in dtopk[j]} & g)
            n_tok = len(queries[gi].split())
            per_q.append((qid, queries[gi], len(g), bh, dh, n_tok))
        if (ci + 1) % 10 == 0 or ci + 1 == n_chunks:
            print(
                f"  matmul: {ci + 1}/{n_chunks} chunks ({end:,}/{len(qids):,} queries)",
                flush=True,
            )

    def agg(rows):
        if not rows:
            return None
        n = len(rows)
        base_r = sum(r[3] / r[2] if r[2] else 0 for r in rows) / n
        bod_r = sum(r[4] / r[2] if r[2] else 0 for r in rows) / n
        return n, base_r, bod_r, bod_r - base_r

    print("\n" + "=" * 78)
    print("BoD lift on ESCI-US holdout — per-bucket R@10 (E-only positives)")
    print("=" * 78)

    # By query length.
    print("\n--- by query length (#tokens) ---")
    print(f"  {'len':>5} {'n':>6} {'base':>8} {'BoD':>8} {'Δ':>8}")
    by_len = defaultdict(list)
    for r in per_q:
        by_len[min(r[5], 8)].append(r)  # cap at 8+
    for L in sorted(by_len):
        a = agg(by_len[L])
        label = f"{L}+" if L == 8 else str(L)
        print(f"  {label:>5} {a[0]:>6,} {a[1]:>8.3f} {a[2]:>8.3f} {a[3]:>+8.3f}")

    # By base-difficulty bucket.
    print("\n--- by base difficulty (base R@10 hits / n_pos) ---")
    by_diff = defaultdict(list)
    for r in per_q:
        ratio = r[3] / r[2] if r[2] else 0
        if ratio == 0:
            bucket = "0.0 (base misses entirely)"
        elif ratio < 0.5:
            bucket = "0.0-0.5"
        elif ratio < 1.0:
            bucket = "0.5-1.0"
        else:
            bucket = "1.0 (base perfect)"
        by_diff[bucket].append(r)
    print(f"  {'bucket':<28} {'n':>6} {'base':>8} {'BoD':>8} {'Δ':>8}")
    total = sum(len(v) for v in by_diff.values())
    for k_ in [
        "0.0 (base misses entirely)",
        "0.0-0.5",
        "0.5-1.0",
        "1.0 (base perfect)",
    ]:
        if k_ in by_diff:
            a = agg(by_diff[k_])
            pct = 100.0 * a[0] / total
            print(f"  {k_:<28} {a[0]:>6,} ({pct:>4.1f}%) {a[1]:>8.3f} {a[2]:>8.3f} {a[3]:>+8.3f}")

    a = agg(per_q)
    print(f"\n  overall (n={a[0]:,}): base={a[1]:.3f}  BoD={a[2]:.3f}  Δ={a[3]:+.3f}")


if __name__ == "__main__":
    main()
