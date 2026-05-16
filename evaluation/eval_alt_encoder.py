#!/usr/bin/env python3
"""Drop-in baseline evaluator for any sentence-transformers-compatible encoder.

Encodes a catalog with chunked checkpointing (memmap + progress file), so
restarts after OOM-kill resume from the last completed chunk.

Compares against the cached MiniLM-L6 base (where present) and reports
R@10 deltas plus per-bucket breakdown by MiniLM-base ratio.

Usage:
    python evaluation/eval_alt_encoder.py \\
        --data-dir bestbuy_acm_data \\
        --queries test_queries_1k.jsonl --qrels test_qrels_1k.jsonl \\
        --model algolia/algolia-large-multilang-generic-v2410 \\
        --query-prefix 'query: ' \\
        --out-name algolia_catalog --baseline-per-query bod_per_query_bestbuy_1k.jsonl
"""

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import SentenceTransformer


def encode_catalog_chunked(
    model: SentenceTransformer,
    titles: list[str],
    out_path: Path,
    progress_path: Path,
    chunk_size: int = 50_000,
    batch_size: int = 64,
):
    """Encode `titles` into `out_path` (fp16 .npy) chunked. Atomic per chunk
    via `progress_path` (JSON list of completed chunk indices)."""
    dim = model.get_sentence_embedding_dimension()
    target_shape = (len(titles), dim)

    if not out_path.exists():
        np.save(out_path, np.zeros(target_shape, dtype=np.float16))
    vecs = np.lib.format.open_memmap(out_path, mode="r+")
    if vecs.shape != target_shape:
        raise RuntimeError(f"existing {out_path} has shape {vecs.shape}, expected {target_shape}")

    done = set()
    if progress_path.exists():
        with open(progress_path) as f:
            done = set(json.load(f))
    n_chunks = (len(titles) + chunk_size - 1) // chunk_size
    print(
        f"  encoder: dim={dim}  chunks: {n_chunks}  chunk_size={chunk_size:,}  "
        f"resume: {len(done)} done",
        flush=True,
    )

    t0 = time.time()
    for ci in range(n_chunks):
        if ci in done:
            continue
        start = ci * chunk_size
        end = min(start + chunk_size, len(titles))
        chunk_titles = titles[start:end]
        t1 = time.time()
        chunk_vecs = model.encode(
            chunk_titles,
            normalize_embeddings=True,
            batch_size=batch_size,
            show_progress_bar=False,
        ).astype(np.float16)
        vecs[start:end] = chunk_vecs
        vecs.flush()
        done.add(ci)
        with open(progress_path, "w") as f:
            json.dump(sorted(done), f)
        dur = time.time() - t1
        elapsed = time.time() - t0
        remaining = n_chunks - len(done)
        avg_per_chunk = elapsed / max(len(done) - (len(done) - 1 if ci in done else 0), 1)
        eta_min = remaining * (dur if avg_per_chunk == 0 else avg_per_chunk) / 60
        print(
            f"  chunk {ci + 1}/{n_chunks}  ({end:,}/{len(titles):,})  "
            f"{dur:.1f}s ({len(chunk_titles) / dur:.0f} docs/s)  ETA {eta_min:.0f}min",
            flush=True,
        )

    return np.asarray(vecs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--titles-file", default="titles.json")
    ap.add_argument(
        "--ids-file", default=None, help="defaults to doc_ids.json then product_ids.json"
    )
    ap.add_argument("--queries", default="test_queries.jsonl")
    ap.add_argument("--qrels", default="test_qrels.jsonl")
    ap.add_argument("--min-relevance", type=int, default=1)
    ap.add_argument("--model", required=True)
    ap.add_argument("--query-prefix", default="")
    ap.add_argument(
        "--out-name", required=True, help="stem for output files (foo → foo.vecs.fp16.npy)"
    )
    ap.add_argument(
        "--baseline-per-query",
        default=None,
        help="path to <method>_per_query_*.jsonl for bucket comparison",
    )
    ap.add_argument("--chunk-size", type=int, default=50_000)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--k", type=int, default=10)
    args = ap.parse_args()

    data = Path(args.data_dir)

    # Resolve doc-id file
    if args.ids_file:
        ids_path = data / args.ids_file
    elif (data / "doc_ids.json").exists():
        ids_path = data / "doc_ids.json"
    else:
        ids_path = data / "product_ids.json"

    with open(ids_path) as f:
        pids = json.load(f)
    pid_to_idx = {p: i for i, p in enumerate(pids)}
    with open(data / args.titles_file) as f:
        titles = json.load(f)
    print(f"catalog: {len(pids):,} from {ids_path}", flush=True)

    queries_by_qid = {}
    with open(data / args.queries) as f:
        for line in f:
            d = json.loads(line)
            queries_by_qid[d["query_id"]] = d["query"]
    pos = defaultdict(set)
    with open(data / args.qrels) as f:
        for line in f:
            r = json.loads(line)
            if r["relevance"] < args.min_relevance:
                continue
            did = r.get("product_id") or r.get("doc_id")
            if did not in pid_to_idx:
                continue
            pos[r["query_id"]].add(pid_to_idx[did])
    qids = sorted(queries_by_qid)
    queries = [queries_by_qid[q] for q in qids]
    print(f"queries: {len(qids):,}", flush=True)

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"\nloading {args.model} on {device}...", flush=True)
    t0 = time.time()
    m = SentenceTransformer(args.model, device=device)
    print(
        f"  loaded in {time.time() - t0:.1f}s  dim={m.get_sentence_embedding_dimension()}  max_seq={m.max_seq_length}",
        flush=True,
    )

    out_vecs = data / f"{args.out_name}.vecs.fp16.npy"
    progress = data / f"{args.out_name}.progress.json"
    print(f"\nencoding catalog → {out_vecs} (chunked, resumable)...", flush=True)
    catalog_vecs = encode_catalog_chunked(
        m, titles, out_vecs, progress, args.chunk_size, args.batch_size
    ).astype(np.float32)
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

    prefixed = [args.query_prefix + q for q in queries] if args.query_prefix else queries
    print(f"\nencoding {len(queries):,} queries (prefix='{args.query_prefix}')...", flush=True)
    query_vecs = m.encode(
        prefixed, normalize_embeddings=True, batch_size=128, show_progress_bar=False
    ).astype(np.float32)

    print("\nbatched retrieval...", flush=True)
    BATCH_Q = 200
    per_q = []
    for qs in range(0, len(qids), BATCH_Q):
        qe = min(qs + BATCH_Q, len(qids))
        sim = query_vecs[qs:qe] @ catalog_vecs.T
        top_k = np.argpartition(-sim, args.k, axis=1)[:, : args.k]
        for j in range(qe - qs):
            qid = qids[qs + j]
            g = pos.get(qid, set())
            if not g:
                continue
            hits = len({int(x) for x in top_k[j]} & g)
            per_q.append((qid, len(g), hits))

    alt_r = float(np.mean([h / g for _, g, h in per_q]))
    print(f"\n{args.model} R@{args.k}: {alt_r:.4f}  (n={len(per_q):,})", flush=True)

    if args.baseline_per_query:
        bl_path = data / args.baseline_per_query
        with open(bl_path) as f:
            baseline = {r["query_id"]: r for r in (json.loads(line) for line in f)}
        ml_base = []
        ml_bod = []
        for qid, g, _ in per_q:
            r = baseline.get(qid)
            if r is None or r["n_gold"] != g:
                continue
            ml_base.append(r["base_hit"] / g)
            if "bod_hit" in r:
                ml_bod.append(r["bod_hit"] / g)
        if ml_base:
            print(
                f"MiniLM base R@{args.k}: {np.mean(ml_base):.4f}  (n={len(ml_base):,})", flush=True
            )
            print(f"  Δ (alt − MiniLM base): {alt_r - np.mean(ml_base):+.4f}", flush=True)
        if ml_bod:
            print(
                f"MiniLM BoD R@{args.k}: {np.mean(ml_bod):.4f}  (fine-tuned, for context)",
                flush=True,
            )

        # Per-bucket Algolia R@10 by MiniLM base ratio
        minilm_by_qid = {
            qid: r["base_hit"] / r["n_gold"] for qid, r in baseline.items() if r["n_gold"] > 0
        }
        buckets = {"miss": [], "0_5": [], "5_10": [], "perfect": []}
        for qid, g, h in per_q:
            b = minilm_by_qid.get(qid)
            if b is None:
                continue
            ratio = h / g
            if b == 0:
                buckets["miss"].append(ratio)
            elif b == 1.0:
                buckets["perfect"].append(ratio)
            elif b <= 0.5:
                buckets["0_5"].append(ratio)
            else:
                buckets["5_10"].append(ratio)
        print(f"\nper-bucket alt R@{args.k} (bucketed by MiniLM base ratio):")
        for k, vs in buckets.items():
            if vs:
                print(f"  {k:>8}  n={len(vs):>5}  alt R@{args.k}={np.mean(vs):.4f}")


if __name__ == "__main__":
    main()
