#!/usr/bin/env python3
"""Build BoD bags where te3 picks the docs and bge-small defines the target space.

Distillation recipe:
  1. Encode each train query with text-embedding-3-large (1024d).
  2. Top-K cosine neighbors against te3 catalog -> those K doc ids are the bag.
  3. Bag centroid (= training target) = mean of those K docs in BGE-SMALL space.

That way a bge-small student trained with cos loss learns to map query text to
the point in its own (384-d) space where te3-chosen neighbors cluster — the
supervisor signal switches from "bge top-K" to "te3 top-K" without changing
the student's dimensionality.

Output schema matches build_jobs_bags.py so finetune_query_model.py can consume
it directly.

Usage:
  .venv/bin/python download/build_jobs_bags_te3.py \\
      --data-dir jobs_data_usajobs \\
      --queries-file train_queries.jsonl \\
      --te3-catalog jobs_data_usajobs/te3_large_1024.vecs.fp16.npy \\
      --bge-catalog jobs_data_usajobs/bge_small_en_catalog.vecs.fp16.npy \\
      --te3-model text-embedding-3-large --te3-dim 1024 \\
      --k 20 \\
      --output jobs_data_usajobs/bags_te3.jsonl
"""

import argparse
import datetime
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
from dotenv import load_dotenv  # noqa: E402

load_dotenv(override=True)

SPEND_LEDGER = Path(__file__).resolve().parent.parent / ".api_spend.jsonl"
OPENAI_PRICES_PER_M_TOKENS = {
    "text-embedding-3-large": 0.13,
    "text-embedding-3-small": 0.02,
}


def record_spend(model, tokens, cost_usd, purpose):
    rec = {
        "ts": datetime.datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "provider": "openai",
        "model": model,
        "tokens": int(tokens),
        "cost_usd": round(float(cost_usd), 4),
        "purpose": purpose,
    }
    with open(SPEND_LEDGER, "a") as f:
        f.write(json.dumps(rec) + "\n")


def encode_te3(queries, model, dim, batch_size=512, purpose=""):
    from openai import OpenAI

    client = OpenAI()
    out = []
    tokens = 0
    for i in range(0, len(queries), batch_size):
        sub = queries[i : i + batch_size]
        resp = client.embeddings.create(model=model, input=sub, dimensions=dim if dim else None)
        out.extend([d.embedding for d in resp.data])
        tokens += resp.usage.total_tokens
        if (i // batch_size) % 5 == 0 or i + batch_size >= len(queries):
            print(
                f"  encoded {min(i + batch_size, len(queries)):,}/{len(queries):,}  tokens={tokens:,}",
                flush=True,
            )
    cost = tokens * OPENAI_PRICES_PER_M_TOKENS[model] / 1e6
    record_spend(model, tokens, cost, purpose)
    print(f"  total tokens={tokens:,}  cost=${cost:.4f}", flush=True)
    return np.array(out, dtype=np.float32)


def l2norm(x):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return x / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--queries-file", required=True)
    ap.add_argument("--te3-catalog", required=True)
    ap.add_argument("--bge-catalog", required=True)
    ap.add_argument("--te3-model", default="text-embedding-3-large")
    ap.add_argument("--te3-dim", type=int, default=1024)
    ap.add_argument("--ids-file", default=None)
    ap.add_argument("--titles-file", default="titles.json")
    ap.add_argument("--k", type=int, default=20)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    data = Path(args.data_dir)
    with open(data / (args.ids_file or "doc_ids.json")) as f:
        pids = json.load(f)
    with open(data / args.titles_file) as f:
        titles = json.load(f)
    if len(pids) != len(titles):
        raise SystemExit(f"ids ({len(pids)}) != titles ({len(titles)})")

    te3_cat = np.load(args.te3_catalog, mmap_mode="r").astype(np.float32)
    bge_cat = np.load(args.bge_catalog, mmap_mode="r").astype(np.float32)
    if te3_cat.shape[0] != len(pids):
        raise SystemExit(f"te3_cat rows ({te3_cat.shape[0]}) != ids")
    if bge_cat.shape[0] != len(pids):
        raise SystemExit(f"bge_cat rows ({bge_cat.shape[0]}) != ids")
    print(f"te3_cat: {te3_cat.shape}  bge_cat: {bge_cat.shape}", flush=True)
    te3_cat = l2norm(te3_cat)

    # Load queries
    queries = []
    with open(data / args.queries_file) as f:
        for line in f:
            q = json.loads(line)
            queries.append(q)
    print(f"queries: {len(queries):,}", flush=True)
    qtexts = [q["query"] for q in queries]

    # Encode with te3
    print(f"encoding {len(qtexts):,} queries with {args.te3_model}...", flush=True)
    t0 = time.time()
    qv = encode_te3(
        qtexts,
        args.te3_model,
        args.te3_dim,
        purpose=f"te3 bag-build queries for {args.data_dir}",
    )
    qv = l2norm(qv)
    print(f"  encoded in {time.time() - t0:.1f}s", flush=True)

    # Cosine sim, top-K
    print(f"building bags k={args.k}...", flush=True)
    t0 = time.time()
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_bags = 0
    chunk = 256
    with open(out_path, "w") as fout:
        for i in range(0, len(qv), chunk):
            sub = qv[i : i + chunk]
            scores = sub @ te3_cat.T  # (chunk, n_doc)
            top = np.argpartition(-scores, kth=args.k, axis=1)[:, : args.k]
            for r in range(len(sub)):
                pool = top[r]
                # sort by score desc
                order = np.argsort(-scores[r, pool])
                pool = pool[order]
                # bag centroid in bge space
                centroid = bge_cat[pool].mean(axis=0)
                cnorm = np.linalg.norm(centroid)
                if cnorm > 0:
                    centroid = centroid / cnorm
                # specificity in te3 space: mean intra-bag cosine
                bag_te3 = te3_cat[pool]
                intra = bag_te3 @ bag_te3.T
                np.fill_diagonal(intra, 0.0)
                specificity = float(intra.sum() / (args.k * (args.k - 1)))
                bag = {
                    "query": queries[i + r]["query"],
                    "query_vector": centroid.astype(float).tolist(),
                    "results": [
                        {"product_id": pids[int(d)], "title": titles[int(d)]} for d in pool
                    ],
                    "num_results": int(args.k),
                    "specificity": specificity,
                }
                fout.write(json.dumps(bag) + "\n")
                n_bags += 1
            if (i // chunk) % 4 == 0 or i + chunk >= len(qv):
                rate = n_bags / max(time.time() - t0, 1e-3)
                print(
                    f"  built {n_bags:,}/{len(qv):,}  ({rate:.0f}/s, ETA {(len(qv) - n_bags) / max(rate, 1) / 60:.1f}min)",
                    flush=True,
                )
    print(f"wrote {n_bags:,} bags to {out_path}", flush=True)


if __name__ == "__main__":
    main()
