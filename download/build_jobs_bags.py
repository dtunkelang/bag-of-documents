#!/usr/bin/env python3
"""Build BoD training bags for the jobs corpus.

Methodology (label-independent, avoids eval-leakage):
  For each training query, encode with base MiniLM, retrieve top-K from
  base_catalog.vecs.fp16.npy via cosine similarity, take the K docs as the
  bag. Compute centroid + specificity. No CE filter (we don't have a
  jobs-domain CE; using a foreign-domain CE like ESCI's would inject noise,
  and using an LLM-judge filter would risk eval-circularity).

Output format matches what training/finetune_query_model.py expects:
    {"query": str, "query_vector": [float], "results": [{"title": str}],
     "num_results": int, "specificity": float}

Usage:
  .venv/bin/python download/build_jobs_bags.py \\
      --data-dir jobs_data \\
      --queries-file train_queries.jsonl \\
      --encoder sentence-transformers/all-MiniLM-L6-v2 \\
      --catalog jobs_data/base_catalog.vecs.fp16.npy \\
      --k 20 \\
      --output jobs_data/bags.jsonl
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--queries-file", required=True, help="JSONL with query_id+query")
    ap.add_argument("--encoder", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--catalog", required=True, help="catalog .vecs.fp16.npy")
    ap.add_argument("--ids-file", default=None)
    ap.add_argument("--titles-file", default="titles.json")
    ap.add_argument("--k", type=int, default=20, help="bag size (top-K)")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    data = Path(args.data_dir)
    if args.ids_file:
        ids_path = data / args.ids_file
    elif (data / "doc_ids.json").exists():
        ids_path = data / "doc_ids.json"
    else:
        ids_path = data / "product_ids.json"
    with open(ids_path) as f:
        pids = json.load(f)
    with open(data / args.titles_file) as f:
        titles = json.load(f)

    # Load catalog
    cat = np.load(args.catalog, mmap_mode="r")
    if cat.shape[0] != len(pids):
        raise SystemExit(f"catalog rows ({cat.shape[0]}) != ids ({len(pids)})")
    print(f"catalog: {cat.shape} dtype={cat.dtype}", flush=True)
    cat_f = np.asarray(cat).astype(np.float32)

    # Load queries
    queries = []
    with open(data / args.queries_file) as f:
        for line in f:
            d = json.loads(line)
            queries.append((d["query_id"], d["query"]))
    print(f"queries: {len(queries):,} from {args.queries_file}", flush=True)

    # Encode queries
    from sentence_transformers import SentenceTransformer

    print(f"loading {args.encoder} on {args.device}...", flush=True)
    t0 = time.time()
    model = SentenceTransformer(args.encoder, device=args.device)
    print(f"  loaded in {time.time() - t0:.1f}s", flush=True)
    t0 = time.time()
    qv = model.encode(
        [q for _, q in queries],
        normalize_embeddings=True,
        batch_size=64,
        show_progress_bar=False,
    ).astype(np.float32)
    print(f"  encoded {len(queries)} queries in {time.time() - t0:.1f}s", flush=True)

    # Retrieve top-K + compute centroid/specificity for each bag
    n_q = len(queries)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"\nbuilding bags k={args.k}...", flush=True)
    n_written = 0
    chunk = 256
    t0 = time.time()
    with open(out_path, "w") as fout:
        for s in range(0, n_q, chunk):
            e = min(s + chunk, n_q)
            sims = qv[s:e] @ cat_f.T  # (chunk, N_cat)
            for i in range(e - s):
                qidx = s + i
                # top-k
                tk = np.argpartition(-sims[i], args.k - 1)[: args.k]
                tk = tk[np.argsort(-sims[i, tk])]
                doc_vecs = cat_f[tk]  # (k, dim)
                centroid = doc_vecs.mean(axis=0)
                cnorm = np.linalg.norm(centroid)
                if cnorm > 1e-9:
                    centroid = centroid / cnorm
                # specificity = mean cosine to centroid
                spec = float((doc_vecs @ centroid).mean())
                rec = {
                    "query": queries[qidx][1],
                    "query_vector": centroid.tolist(),
                    "results": [{"title": titles[int(p)]} for p in tk],
                    "num_results": int(args.k),
                    "specificity": spec,
                }
                fout.write(json.dumps(rec) + "\n")
                n_written += 1
            if (s // chunk) % 4 == 0:
                rate = n_written / max(time.time() - t0, 1e-3)
                print(
                    f"  built {n_written:,}/{n_q:,} ({rate:.0f} bags/s, "
                    f"ETA {(n_q - n_written) / max(rate, 1e-3) / 60:.1f}min)",
                    flush=True,
                )

    print(
        f"\nwrote {n_written:,} bags to {out_path} in {time.time() - t0:.0f}s",
        flush=True,
    )


if __name__ == "__main__":
    main()
