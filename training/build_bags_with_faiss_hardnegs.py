#!/usr/bin/env python3
"""Augment bags.jsonl with FAISS-mined hard negatives per bag.

Uses the current 6M-MNRL dense retriever to find the products it ranks
highest for each bag's query, then takes the top-N that are NOT in the bag's
positive set. These are the model's residual error: products it confuses with
the bag, which BM25-hardnegs (lexical confusion) miss by construction.

Output: bags_with_faiss_hardnegs.jsonl
  Same format as bags.jsonl + "hardnegs" field (titles list) — drop-in
  replacement for bags_with_hardnegs.jsonl with finetune_with_hardnegs.py.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
from sentence_transformers import SentenceTransformer  # noqa: E402

SCRIPT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INDEX_DIR = os.path.join(SCRIPT_DIR, "combined_index_us_minilm")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bags", default=os.path.join(SCRIPT_DIR, "bags.jsonl"))
    ap.add_argument(
        "--retriever",
        default=os.path.join(SCRIPT_DIR, "query_model_us_full_6m_mnrl"),
        help="SentenceTransformer model used to encode queries",
    )
    ap.add_argument(
        "--catalog-vecs",
        default=os.path.join(INDEX_DIR, "rerank_A.vecs.fp16.npy"),
        help="Precomputed catalog vectors aligned with --titles",
    )
    ap.add_argument("--titles", default=os.path.join(INDEX_DIR, "titles.json"))
    ap.add_argument("--output", default=os.path.join(SCRIPT_DIR, "bags_with_faiss_hardnegs.jsonl"))
    ap.add_argument(
        "--top-k",
        type=int,
        default=200,
        help="Top-K from FAISS to consider per query (positives filtered out)",
    )
    ap.add_argument(
        "--n-hardnegs",
        type=int,
        default=10,
        help="Number of hard negatives per bag (after filtering positives)",
    )
    ap.add_argument(
        "--skip-rank",
        type=int,
        default=0,
        help="Skip the top N FAISS hits (avoid sampling true relevants missed by bag construction). 0 = use top-K from rank 0.",
    )
    ap.add_argument("--query-batch", type=int, default=128)
    args = ap.parse_args()

    print(f"loading titles from {args.titles}...", flush=True)
    with open(args.titles) as f:
        titles = json.load(f)
    print(f"  {len(titles):,} titles", flush=True)

    print(f"loading catalog vecs from {args.catalog_vecs}...", flush=True)
    pv = np.load(args.catalog_vecs).astype(np.float32)
    print(f"  shape={pv.shape}", flush=True)

    print(f"reading bags from {args.bags}...", flush=True)
    bags = []
    with open(args.bags) as f:
        for line in f:
            bags.append(json.loads(line))
    print(f"  {len(bags):,} bags", flush=True)
    queries = [b["query"] for b in bags]

    device = (
        "mps"
        if torch.backends.mps.is_available()
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"\nencoding queries with {args.retriever} on {device}...", flush=True)
    t0 = time.time()
    model = SentenceTransformer(args.retriever, device=device)
    qv = model.encode(
        queries,
        normalize_embeddings=True,
        batch_size=args.query_batch,
        show_progress_bar=True,
    )
    qv = np.asarray(qv, dtype=np.float32)
    print(f"  encode took {time.time() - t0:.0f}s, qv.shape={qv.shape}", flush=True)

    print(
        f"\nbrute top-{args.top_k} (skip {args.skip_rank}) over {len(titles):,} products...",
        flush=True,
    )
    t0 = time.time()
    n_q = qv.shape[0]
    top_pos = np.zeros((n_q, args.top_k + args.skip_rank), dtype=np.int64)
    BATCH = 64
    for start in range(0, n_q, BATCH):
        end = min(start + BATCH, n_q)
        sims = qv[start:end] @ pv.T
        k_total = args.top_k + args.skip_rank
        top = np.argpartition(-sims, k_total, axis=1)[:, :k_total]
        for i in range(end - start):
            row = sims[i, top[i]]
            order = np.argsort(-row)
            top_pos[start + i] = top[i, order]
        if (start // BATCH) % 50 == 0:
            print(f"  {end}/{n_q} ({time.time() - t0:.0f}s)", flush=True)
    print(f"  brute top-k took {time.time() - t0:.0f}s", flush=True)

    print(f"\nwriting {args.output}...", flush=True)
    n_with = 0
    counts = []
    with open(args.output, "w") as fout:
        for bi, bag in enumerate(bags):
            positive_titles = {r["title"] for r in bag.get("results", [])}
            hardnegs = []
            for rank, pos in enumerate(top_pos[bi]):
                if rank < args.skip_rank:
                    continue
                title = titles[int(pos)]
                if title in positive_titles:
                    continue
                hardnegs.append(title)
                if len(hardnegs) >= args.n_hardnegs:
                    break
            bag["hardnegs"] = hardnegs
            if hardnegs:
                n_with += 1
                counts.append(len(hardnegs))
            fout.write(json.dumps(bag) + "\n")

    print(f"\ndone. {len(bags):,} bags written.", flush=True)
    print(f"  {n_with:,} ({n_with / len(bags):.1%}) have >=1 hardneg", flush=True)
    if counts:
        print(
            f"  hardnegs/bag: mean={np.mean(counts):.1f}, "
            f"median={int(np.median(counts))}, max={max(counts)}",
            flush=True,
        )


if __name__ == "__main__":
    main()
