#!/usr/bin/env python3
"""Four-way union-oracle: BoD + HyDE + Doc2Query + Algolia drop-in.

Extends Pattern 17's three-way oracle by adding a stronger drop-in base
encoder (Algolia) as a fourth orthogonal lever. Tests:

  - Does Algolia base rescue queries the three trained methods all miss?
  - Does the union-oracle ceiling rise meaningfully?
  - Are the methods disjoint, redundant, or partially overlapping?

For each corpus: encode catalog + queries with Algolia (cached if present),
compute per-query top-K hits, then join with the existing per-query JSONLs
from BoD/HyDE/Doc2Query runs to compute pairwise overlap + union ceiling.

Usage:
    python evaluation/eval_four_way_oracle.py --corpus scifact
    python evaluation/eval_four_way_oracle.py --corpus nfcorpus
"""

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

ALGOLIA_MODEL = "algolia/algolia-large-multilang-generic-v2410"
ALGOLIA_QUERY_PREFIX = "query: "
K = 10
MIN_RELEVANCE = 1

# Per-corpus config: (data_dir, bod_pq, hyde_pq, d2q_pq)
CORPORA = {
    "scifact": (
        "scifact_data",
        "bod_per_query_scifact.jsonl",
        "hyde_per_query_scifact.jsonl",
        "doc2query_per_query_scifact_d2q_full.jsonl",
    ),
    "nfcorpus": (
        "nfcorpus_data",
        "bod_per_query_nfcorpus.jsonl",
        "hyde_per_query_nfcorpus.jsonl",
        "doc2query_per_query_nfcorpus_d2q_oracle_vecavg_fixed.jsonl",
    ),
    "fiqa": (
        "fiqa_data",
        "bod_per_query_fiqa.jsonl",
        "hyde_per_query_fiqa.jsonl",
        "doc2query_per_query_fiqa_d2q_oracle.jsonl",
    ),
    "programmers": (
        "cqadupstack_programmers_data",
        "bod_per_query_programmers.jsonl",
        "hyde_per_query_programmers.jsonl",
        "doc2query_per_query_programmers_d2q_oracle.jsonl",
    ),
    "english": (
        "cqadupstack_english_data",
        "bod_per_query_english.jsonl",
        "hyde_per_query_english.jsonl",
        "doc2query_per_query_english_d2q_oracle.jsonl",
    ),
}


def encode_catalog_chunked(model, titles, out_path, progress_path, chunk_size=10000, batch_size=64):
    """Chunked memmap-resumable encode (same pattern as eval_alt_encoder.py)."""
    dim = model.get_sentence_embedding_dimension()
    target_shape = (len(titles), dim)
    if not out_path.exists():
        np.save(out_path, np.zeros(target_shape, dtype=np.float16))
    vecs = np.lib.format.open_memmap(out_path, mode="r+")
    done = set()
    if progress_path.exists():
        with open(progress_path) as f:
            done = set(json.load(f))
    n_chunks = (len(titles) + chunk_size - 1) // chunk_size
    print(f"  chunks: {n_chunks}  resume: {len(done)} done", flush=True)
    t0 = time.time()
    for ci in range(n_chunks):
        if ci in done:
            continue
        start, end = ci * chunk_size, min((ci + 1) * chunk_size, len(titles))
        chunk_vecs = model.encode(
            titles[start:end],
            normalize_embeddings=True,
            batch_size=batch_size,
            show_progress_bar=False,
        ).astype(np.float16)
        vecs[start:end] = chunk_vecs
        vecs.flush()
        done.add(ci)
        with open(progress_path, "w") as f:
            json.dump(sorted(done), f)
        elapsed = time.time() - t0
        print(
            f"  chunk {ci + 1}/{n_chunks}  ({end:,}/{len(titles):,})  elapsed={elapsed:.0f}s",
            flush=True,
        )
    return np.asarray(vecs)


def compute_algolia_per_query(data_dir: Path, pids, pid_to_idx, titles, queries_by_qid, pos):
    out_pq = data_dir / "algolia_per_query.jsonl"
    if out_pq.exists():
        print(f"  using cached {out_pq}", flush=True)
        with open(out_pq) as f:
            return {json.loads(line)["query_id"]: json.loads(line) for line in f}

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"  loading {ALGOLIA_MODEL}...", flush=True)
    m = SentenceTransformer(ALGOLIA_MODEL, device=device)
    m.max_seq_length = 256

    vec_path = data_dir / "algolia_catalog.vecs.fp16.npy"
    prog_path = data_dir / "algolia_catalog.progress.json"
    target_shape = (len(pids), m.get_sentence_embedding_dimension())
    needs_encode = not vec_path.exists() or np.load(vec_path, mmap_mode="r").shape != target_shape
    if needs_encode:
        print("  encoding catalog with Algolia...", flush=True)
        catalog_vecs = encode_catalog_chunked(m, titles, vec_path, prog_path).astype(np.float32)
    else:
        print(f"  loading cached catalog vecs from {vec_path}", flush=True)
        catalog_vecs = np.load(vec_path).astype(np.float32)

    qids = sorted(queries_by_qid)
    queries = [queries_by_qid[q] for q in qids]
    print(f"  encoding {len(qids):,} queries...", flush=True)
    qv = m.encode(
        [ALGOLIA_QUERY_PREFIX + q for q in queries],
        normalize_embeddings=True,
        batch_size=128,
        show_progress_bar=False,
    ).astype(np.float32)

    print("  scoring...", flush=True)
    sim = qv @ catalog_vecs.T
    top_k = np.argpartition(-sim, K, axis=1)[:, :K]
    per_q = {}
    with open(out_pq, "w") as f:
        for j, qid in enumerate(qids):
            g = pos.get(qid, set())
            if not g:
                continue
            hits = len({int(x) for x in top_k[j]} & g)
            row = {"query_id": qid, "n_gold": len(g), "algolia_hit": hits}
            per_q[qid] = row
            f.write(json.dumps(row) + "\n")
    print(f"  wrote per-query hits to {out_pq}", flush=True)
    return per_q


def load_per_query(path):
    with open(path) as f:
        return {json.loads(line)["query_id"]: json.loads(line) for line in f}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True, choices=list(CORPORA.keys()))
    ap.add_argument("--min-relevance", type=int, default=MIN_RELEVANCE)
    args = ap.parse_args()

    data_dir_name, bod_pq, hyde_pq, d2q_pq = CORPORA[args.corpus]
    data_dir = Path(data_dir_name)
    print(f"\n=== {args.corpus} ===", flush=True)

    # Load corpus
    ids_file = (
        data_dir / "doc_ids.json"
        if (data_dir / "doc_ids.json").exists()
        else data_dir / "product_ids.json"
    )
    with open(ids_file) as f:
        pids = json.load(f)
    pid_to_idx = {p: i for i, p in enumerate(pids)}
    with open(data_dir / "titles.json") as f:
        titles = json.load(f)
    queries_by_qid = {}
    with open(data_dir / "test_queries.jsonl") as f:
        for line in f:
            d = json.loads(line)
            queries_by_qid[d["query_id"]] = d["query"]
    pos = defaultdict(set)
    with open(data_dir / "test_qrels.jsonl") as f:
        for line in f:
            r = json.loads(line)
            if r["relevance"] < args.min_relevance:
                continue
            did = r.get("product_id") or r.get("doc_id")
            if did not in pid_to_idx:
                continue
            pos[r["query_id"]].add(pid_to_idx[did])
    print(f"  catalog={len(pids):,}  queries={len(queries_by_qid):,}", flush=True)

    # Algolia per-query
    print("\n[1/2] computing Algolia per-query hits", flush=True)
    algolia = compute_algolia_per_query(data_dir, pids, pid_to_idx, titles, queries_by_qid, pos)

    # Existing per-query files
    print("\n[2/2] loading BoD / HyDE / Doc2Query per-query files", flush=True)
    bod = load_per_query(data_dir / bod_pq)
    hyde = load_per_query(data_dir / hyde_pq)
    d2q = load_per_query(data_dir / d2q_pq)

    # Join on shared queries
    shared = set(algolia) & set(bod) & set(hyde) & set(d2q)
    print(f"\nshared positive-bearing queries: {len(shared):,}", flush=True)

    rows = []
    for qid in shared:
        a = algolia[qid]
        b = bod[qid]
        h = hyde[qid]
        d = d2q[qid]
        n_gold = a["n_gold"]
        if not (n_gold == b["n_gold"] == h["n_gold"] == d["n_gold"]):
            continue  # qrels disagreement
        rows.append(
            {
                "qid": qid,
                "n_gold": n_gold,
                "base": b["base_hit"],
                "bod": b["bod_hit"],
                "hyde": h["hyde_hit"],
                "d2q": d["d2q_hit"],
                "algolia": a["algolia_hit"],
            }
        )
    n = len(rows)
    if n == 0:
        print("ERROR: no joinable rows")
        return
    print(f"evaluated: n={n:,}", flush=True)

    def mean_r(key):
        return sum(r[key] / r["n_gold"] for r in rows) / n

    methods = ["base", "bod", "hyde", "d2q", "algolia"]
    means = {m: mean_r(m) for m in methods}
    base_r = means["base"]
    union3 = sum(max(r["bod"], r["hyde"], r["d2q"]) / r["n_gold"] for r in rows) / n
    union4 = sum(max(r["bod"], r["hyde"], r["d2q"], r["algolia"]) / r["n_gold"] for r in rows) / n

    print(f"\n  {'method':<14} {'R@10':>7} {'Δ vs base':>10}")
    for m in methods:
        print(f"  {m:<14} {means[m]:>7.4f} {means[m] - base_r:>+10.4f}")
    print(f"  {'UNION-3-way':<14} {union3:>7.4f} {union3 - base_r:>+10.4f}  (BoD+HyDE+D2Q)")
    print(f"  {'UNION-4-way':<14} {union4:>7.4f} {union4 - base_r:>+10.4f}  (+ Algolia)")
    print(f"  4-way headroom over 3-way: {union4 - union3:+.4f}")
    print(f"  4-way headroom over best single: {union4 - max(means.values()):+.4f}")

    # Exclusivity on base-blind subset
    base_blind = [r for r in rows if r["base"] == 0]
    if base_blind:
        only_bod = sum(
            1
            for r in base_blind
            if r["bod"] > 0 and r["hyde"] == 0 and r["d2q"] == 0 and r["algolia"] == 0
        )
        only_hyde = sum(
            1
            for r in base_blind
            if r["hyde"] > 0 and r["bod"] == 0 and r["d2q"] == 0 and r["algolia"] == 0
        )
        only_d2q = sum(
            1
            for r in base_blind
            if r["d2q"] > 0 and r["bod"] == 0 and r["hyde"] == 0 and r["algolia"] == 0
        )
        only_alg = sum(
            1
            for r in base_blind
            if r["algolia"] > 0 and r["bod"] == 0 and r["hyde"] == 0 and r["d2q"] == 0
        )
        all_four = sum(
            1
            for r in base_blind
            if r["bod"] > 0 and r["hyde"] > 0 and r["d2q"] > 0 and r["algolia"] > 0
        )
        none = sum(
            1
            for r in base_blind
            if r["bod"] == 0 and r["hyde"] == 0 and r["d2q"] == 0 and r["algolia"] == 0
        )
        print(f"\n  base-blind exclusivity (n={len(base_blind):,}):")
        print(
            f"    only BoD: {only_bod}   only HyDE: {only_hyde}   only D2Q: {only_d2q}   only Algolia: {only_alg}"
        )
        print(f"    all four: {all_four}   none: {none}")


if __name__ == "__main__":
    main()
