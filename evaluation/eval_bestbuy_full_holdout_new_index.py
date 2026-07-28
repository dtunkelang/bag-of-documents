#!/usr/bin/env python3
"""Recompute the headline BestBuy metrics (R@10 binary hit-rate, E@1) on the
RE-INDEXED 1.27M catalog (name+manufacturer+categoryPath+class embeddings).

Metric definitions are copied verbatim in spirit from the original scripts that
produced the README numbers:
  * evaluation/eval_bestbuy_bod_full.py     -> R@10 binary hit-rate
  * evaluation/eval_bestbuy_bod_full_e1.py  -> E@1

  R@10 (binary hit-rate) = fraction of evaluated queries whose top-10 contains
                           at least one gold (clicked) product.
  E@1                    = fraction of evaluated queries whose top-1 result is
                           a gold (clicked) product.
  Evaluated queries      = holdout queries with >=1 gold that maps into the
                           catalog product_ids (n_eval).

Catalog vectors come from /tmp/bestbuy_reindex_output/artifacts (byte-identical
to what is live on the HF dataset). Holdout queries/qrels/product_ids come from
the HF dataset snapshot.
"""

import json
import os
import time
from collections import defaultdict

import numpy as np
import torch
from huggingface_hub import snapshot_download
from sentence_transformers import SentenceTransformer

ARTIFACTS = "/tmp/bestbuy_reindex_output/artifacts"
DATASET_REPO = "dtunkelang/bag-of-documents-bestbuy"
BASE_MODEL = "all-MiniLM-L6-v2"
BOD_MODEL = "dtunkelang/bag-of-documents-bestbuy-minilm"
K = 10
OUT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "results",
    "bestbuy_full_holdout_new_index.json",
)


def load_holdout():
    data_dir = snapshot_download(
        repo_id=DATASET_REPO,
        repo_type="dataset",
        allow_patterns=["holdout_queries.jsonl", "holdout_qrels.jsonl"],
    )
    qids, queries = [], []
    with open(os.path.join(data_dir, "holdout_queries.jsonl")) as f:
        for line in f:
            d = json.loads(line)
            qids.append(d["query_id"])
            queries.append(d["query"])
    qrels = []
    with open(os.path.join(data_dir, "holdout_qrels.jsonl")) as f:
        for line in f:
            qrels.append(json.loads(line))
    return qids, queries, qrels


def topk_metrics(qv, pv, gold, K=K, chunk=None):
    """Return (n_eval, binary_hits@K, e1_hits, frac_recovered_sum)."""
    n_docs = pv.shape[0]
    if chunk is None:
        chunk = max(64, int(2.5e8 // n_docs))
    n_q = qv.shape[0]
    hits = e1 = n_eval = 0
    frac = 0.0
    n_chunks = (n_q + chunk - 1) // chunk
    t0 = time.time()
    for ci, start in enumerate(range(0, n_q, chunk)):
        end = min(start + chunk, n_q)
        sim = qv[start:end] @ pv.T
        part = np.argpartition(-sim, K, axis=1)[:, :K]
        rows = np.arange(end - start)[:, None]
        part_scores = sim[rows, part]
        order = np.argsort(-part_scores, axis=1)
        topk = part[rows, order]
        del sim, part, part_scores, order
        for j, gi in enumerate(range(start, end)):
            g = gold[gi]
            if not g:
                continue
            n_eval += 1
            tk = topk[j]
            tk_set = {int(x) for x in tk}
            inter = tk_set & g
            if inter:
                hits += 1
            if int(tk[0]) in g:
                e1 += 1
            frac += len(inter) / len(g)
        if (ci + 1) % 10 == 0 or ci + 1 == n_chunks:
            el = time.time() - t0
            print(
                f"    chunk {ci + 1}/{n_chunks}  {el:.1f}s elapsed "
                f"({el / (ci + 1) * n_chunks:.0f}s projected)",
                flush=True,
            )
    return n_eval, hits, e1, frac


def main():
    t_start = time.time()
    print("loading product_ids...", flush=True)
    with open(os.path.join(ARTIFACTS, "product_ids.json")) as f:
        pids = json.load(f)
    pid_to_idx = {p: i for i, p in enumerate(pids)}

    print("loading holdout...", flush=True)
    qids, queries, qrels = load_holdout()
    pos = defaultdict(set)
    for r in qrels:
        if r["product_id"] in pid_to_idx:
            pos[r["query_id"]].add(pid_to_idx[r["product_id"]])
    gold = [pos.get(q, set()) for q in qids]
    print(
        f"  catalog={len(pids):,}  queries={len(queries):,}  "
        f"queries_with_gold={sum(1 for g in gold if g):,}",
        flush=True,
    )

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    results = {}

    for tag, model_name, vec_file in [
        ("base", BASE_MODEL, "base_catalog.vecs.fp16.npy"),
        ("bod", BOD_MODEL, "bod_catalog.vecs.fp16.npy"),
    ]:
        print(f"\n=== {tag} ===", flush=True)
        t0 = time.time()
        model = SentenceTransformer(model_name, device=device)
        qv = model.encode(
            queries, normalize_embeddings=True, batch_size=256, show_progress_bar=True
        ).astype(np.float32)
        del model
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        t_enc = time.time() - t0
        print(f"  encoded {qv.shape} in {t_enc:.1f}s", flush=True)

        t0 = time.time()
        pv = np.load(os.path.join(ARTIFACTS, vec_file)).astype(np.float32)
        print(f"  vecs {pv.shape} loaded in {time.time() - t0:.1f}s", flush=True)

        t0 = time.time()
        n_eval, hits, e1, frac = topk_metrics(qv, pv, gold)
        t_search = time.time() - t0
        del pv, qv

        results[tag] = {
            "model": model_name,
            "n_eval": n_eval,
            "r_at_10_binary": hits / n_eval,
            "e_at_1": e1 / n_eval,
            "r_at_10_fraction_recovered": frac / n_eval,
            "encode_seconds": round(t_enc, 1),
            "search_seconds": round(t_search, 1),
        }
        print(
            f"  R@10(binary)={hits / n_eval:.4f}  E@1={e1 / n_eval:.4f}  "
            f"R@10(frac)={frac / n_eval:.4f}  search={t_search:.1f}s",
            flush=True,
        )

    total = time.time() - t_start
    old = {
        "base_r_at_10_binary": 0.3238,
        "bod_r_at_10_binary": 0.5013,
        "base_e_at_1": 0.0926,
        "bod_e_at_1": 0.1589,
    }
    payload = {
        "description": (
            "Headline BestBuy metrics recomputed on the RE-INDEXED 1.27M catalog "
            "(name+manufacturer+categoryPath+class embedding text), full "
            "12,128-query holdout. Metric definitions match "
            "evaluation/eval_bestbuy_bod_full.py and eval_bestbuy_bod_full_e1.py."
        ),
        "catalog_size": len(pids),
        "n_holdout_queries": len(queries),
        "k": K,
        "artifacts_dir": ARTIFACTS,
        "dataset_repo": DATASET_REPO,
        "old_index_numbers": old,
        "new_index_numbers": results,
        "deltas_new_index": {
            "r_at_10_binary_pp": round(
                (results["bod"]["r_at_10_binary"] - results["base"]["r_at_10_binary"]) * 100, 2
            ),
            "e_at_1_pp": round((results["bod"]["e_at_1"] - results["base"]["e_at_1"]) * 100, 2),
        },
        "total_wall_clock_seconds": round(total, 1),
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nwrote {OUT}")
    print(json.dumps(payload["new_index_numbers"], indent=2))
    print(json.dumps(payload["deltas_new_index"], indent=2))
    print(f"total wall clock: {total:.1f}s")


if __name__ == "__main__":
    main()
