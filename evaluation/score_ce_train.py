#!/usr/bin/env python3
"""Score the LiYuan ESCI cross-encoder on a sample of ESCI *train* queries
to produce distillation supervision data without test-set leakage.

For each sampled train query: bm25s @ (k1=0.3, b=0.6) retrieves top-100,
CE scores all 100 (q, title) pairs.

Outputs (under combined_index_us_minilm/):
  ce_train_qids.json           - sampled query_ids
  ce_train_queries.json        - corresponding query strings
  ce_train_candidates.npy      - (n_q, 100) FAISS positions (= title indices)
  ce_train_scores.npy          - (n_q, 100) CE scores

Usage:
    python evaluation/score_ce_train.py --n-queries 10000
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import random  # noqa: E402
import time  # noqa: E402

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import bm25s  # noqa: E402
import numpy as np  # noqa: E402
import Stemmer  # noqa: E402
import torch  # noqa: E402
from sentence_transformers import CrossEncoder  # noqa: E402

SCRIPT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INDEX_DIR = os.path.join(SCRIPT_DIR, "combined_index_us_minilm")

TOP_K_POOL = 100


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-queries", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--ce-model", default="LiYuan/Amazon-Cup-Cross-Encoder-Regression")
    ap.add_argument("--batch-size", type=int, default=64)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    print("loading train queries...", flush=True)
    queries_all = {}
    with open(os.path.join(SCRIPT_DIR, "esci_us_data/train_queries.jsonl")) as f:
        for line in f:
            d = json.loads(line)
            queries_all[d["query_id"]] = d["query"]
    print(f"  {len(queries_all):,} train queries available", flush=True)

    qids_pool = sorted(queries_all.keys())
    sampled_qids = random.sample(qids_pool, min(args.n_queries, len(qids_pool)))
    sampled_qids.sort()
    queries = [queries_all[qid] for qid in sampled_qids]
    print(f"  sampled {len(sampled_qids):,} queries (seed={args.seed})", flush=True)

    print("loading bm25s index...", flush=True)
    bm25s_idx = bm25s.BM25.load(os.path.join(INDEX_DIR, "bm25s_index"), mmap=False)
    stemmer = Stemmer.Stemmer("english")
    print("tokenizing queries...", flush=True)
    qt = bm25s.tokenize(queries, stopwords="en", stemmer=stemmer, show_progress=False)
    print(f"retrieving top-{TOP_K_POOL} per query...", flush=True)
    t0 = time.time()
    results, _ = bm25s_idx.retrieve(qt, k=TOP_K_POOL, show_progress=False)
    candidate_pos = np.asarray(results, dtype=np.int64)
    print(f"  done in {time.time() - t0:.0f}s; shape={candidate_pos.shape}", flush=True)

    print("loading product titles...", flush=True)
    with open(os.path.join(INDEX_DIR, "titles.json")) as f:
        index_titles = json.load(f)

    device = (
        "mps"
        if torch.backends.mps.is_available()
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"loading CE model on {device}...", flush=True)
    ce = CrossEncoder(args.ce_model, device=device)

    valid = candidate_pos >= 0
    n_pairs_total = int(valid.sum())
    print(f"scoring {len(queries):,} queries x {TOP_K_POOL} = {n_pairs_total:,} pairs", flush=True)

    ce_scores = np.zeros((len(queries), TOP_K_POOL), dtype=np.float32)
    pairs_buf = []
    locs_buf = []
    n_done = 0
    t0 = time.time()
    for qi, q in enumerate(queries):
        for j in range(TOP_K_POOL):
            pos = int(candidate_pos[qi, j])
            if pos < 0:
                continue
            pairs_buf.append((q, index_titles[pos]))
            locs_buf.append((qi, j))
        if len(pairs_buf) >= 4096 or qi == len(queries) - 1:
            scores = ce.predict(pairs_buf, batch_size=args.batch_size, show_progress_bar=False)
            for (qi2, j2), sc in zip(locs_buf, scores):
                ce_scores[qi2, j2] = float(sc)
            n_done += len(pairs_buf)
            elapsed = time.time() - t0
            rate = n_done / max(elapsed, 1e-3)
            eta = (n_pairs_total - n_done) / max(rate, 1e-3)
            print(
                f"  {n_done:,}/{n_pairs_total:,} ({n_done / n_pairs_total:.1%}) "
                f"@ {rate:.0f}/s eta {eta / 60:.1f}m",
                flush=True,
            )
            pairs_buf.clear()
            locs_buf.clear()

    print("saving artifacts...", flush=True)
    with open(os.path.join(INDEX_DIR, "ce_train_qids.json"), "w") as f:
        json.dump(sampled_qids, f)
    with open(os.path.join(INDEX_DIR, "ce_train_queries.json"), "w") as f:
        json.dump(queries, f)
    np.save(os.path.join(INDEX_DIR, "ce_train_candidates.npy"), candidate_pos)
    np.save(os.path.join(INDEX_DIR, "ce_train_scores.npy"), ce_scores)
    print(f"done. ce_train_scores shape={ce_scores.shape}", flush=True)


if __name__ == "__main__":
    main()
