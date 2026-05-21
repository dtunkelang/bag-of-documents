#!/usr/bin/env python3
"""Cross-lingual CC-style fusion eval for ESCI-Spanish / ESCI-Japanese.

Tests whether layering BGE-reranker-v2-m3 on top of the Pattern 20 winner
(mE5-small + LoRA-BoD) gives the same ~+1pp R@10 lift that CC5 demonstrated
on English. Shape:

    BM25 top-K           (bm25s, language-specific tokenizer)
        ↓
    score with mE5-BoD   (cosine vs cached fp16 vecs)
        ↓
    score with BGE-CE    (BAAI/bge-reranker-v2-m3, multilingual)
        ↓
    per-query min-max normalize each, fuse at w_bge in {0, 0.25, 0.5, 0.75, 1}

Required artifacts (built by indexing/build_bm25s_cross_lingual.py):
    <data_dir>/bm25s_top200.npy
    <data_dir>/bm25s_qids.json
    <data_dir>/me5_small_lora_bod.vecs.fp16.npy
    <data_dir>/titles.json
    <data_dir>/product_ids.json
    <data_dir>/test_qrels.jsonl, test_queries.jsonl

Outputs:
    <data_dir>/cc_eval/bod_top{K}_scores.npy
    <data_dir>/cc_eval/bge_top{K}_scores.npy
    <data_dir>/cc_eval/bge_progress.json  (checkpoint)
    <data_dir>/cc_eval/results.json

Usage:
    .venv/bin/python evaluation/eval_cc_cross_lingual.py \\
        --data-dir esci_es_data \\
        --bod-model query_model_esci_es_me5_small_lora_bod
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import os  # noqa: E402
import time  # noqa: E402
from collections import defaultdict  # noqa: E402

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np  # noqa: E402
import torch  # noqa: E402
from sentence_transformers import CrossEncoder, SentenceTransformer  # noqa: E402

K_EVAL = 10


def per_query_metrics(retrieved_pids, qrels_q, k=K_EVAL, min_rel=2, exact_rel=3):
    pos_e = {pid for pid, g in qrels_q.items() if g >= exact_rel}
    pos_es = {pid for pid, g in qrels_q.items() if g >= min_rel}
    if not pos_es:
        return None
    top_k = retrieved_pids[:k]
    recall = sum(1 for p in top_k if p in pos_es) / len(pos_es)
    gains = [1.0 if p in pos_e else (0.1 if p in pos_es else 0.0) for p in top_k]
    dcg = sum(g / math.log2(i + 2) for i, g in enumerate(gains))
    ideal = sorted((1.0 if p in pos_e else 0.1 for p in pos_es), reverse=True)[:k]
    idcg = sum(g / math.log2(i + 2) for i, g in enumerate(ideal))
    ndcg = dcg / idcg if idcg > 0 else 0.0
    if pos_e:
        e1 = 1.0 if top_k and top_k[0] in pos_e else 0.0
        e3 = sum(1 for p in top_k[:3] if p in pos_e) / min(3, len(pos_e))
    else:
        e1 = e3 = float("nan")
    return recall, ndcg, e1, e3


def normalize_per_query(scores, valid_mask):
    out = scores.copy()
    for qi in range(out.shape[0]):
        v = out[qi, valid_mask[qi]]
        if v.size == 0:
            continue
        lo, hi = float(v.min()), float(v.max())
        out[qi, valid_mask[qi]] = (v - lo) / max(hi - lo, 1e-8)
    return out


def score_with_bod(model_dir, queries, top_pos, catalog_vecs, query_prefix=""):
    device = (
        "mps"
        if torch.backends.mps.is_available()
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"loading BoD model from {model_dir} on {device}...", flush=True)
    st_kwargs = {}
    if "nomic" in str(model_dir).lower():
        st_kwargs["trust_remote_code"] = True
    m = SentenceTransformer(model_dir, device=device, **st_kwargs)
    t0 = time.time()
    prefixed = [query_prefix + q for q in queries] if query_prefix else queries
    qv = m.encode(
        prefixed, normalize_embeddings=True, batch_size=64, show_progress_bar=False
    ).astype(np.float32)
    print(f"  encoded {len(queries):,} queries in {time.time() - t0:.0f}s", flush=True)

    N, K = top_pos.shape
    scores = np.full((N, K), np.nan, dtype=np.float32)
    cat = np.asarray(catalog_vecs).astype(np.float32)
    valid = top_pos >= 0
    for qi in range(N):
        pos = top_pos[qi]
        ok = valid[qi]
        idxs = pos[ok]
        if idxs.size == 0:
            continue
        cand_vecs = cat[idxs]
        sims = cand_vecs @ qv[qi]
        scores[qi, ok] = sims
    return scores


def score_with_bge(
    model_name,
    queries,
    titles,
    top_pos,
    out_score_path,
    progress_path,
    batch_size=16,
    checkpoint_every_queries=200,
):
    device = (
        "mps"
        if torch.backends.mps.is_available()
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    N, K = top_pos.shape
    if out_score_path.exists():
        scores = np.load(out_score_path)
        if scores.shape != (N, K):
            raise RuntimeError(
                f"existing {out_score_path} has shape {scores.shape}, want ({N},{K})"
            )
    else:
        scores = np.full((N, K), np.nan, dtype=np.float32)
        np.save(out_score_path, scores)
    if progress_path.exists():
        with open(progress_path) as f:
            done_qi = set(json.load(f)["done_qi"])
    else:
        done_qi = set()

    todo = [qi for qi in range(N) if qi not in done_qi]
    print(f"loading CE {model_name} on {device}  (resume: {len(done_qi):,}/{N:,} done)", flush=True)
    ce = CrossEncoder(model_name, device=device)

    t0 = time.time()
    pairs_buf, locs_buf = [], []
    queries_completed = 0

    def flush():
        if not pairs_buf:
            return
        sc = ce.predict(pairs_buf, batch_size=batch_size, show_progress_bar=False)
        for (qi_, j_), s in zip(locs_buf, sc):
            scores[qi_, j_] = float(s)
        pairs_buf.clear()
        locs_buf.clear()

    for n_processed, qi in enumerate(todo, start=1):
        q = queries[qi]
        for j in range(K):
            pos = int(top_pos[qi, j])
            if pos < 0:
                continue
            pairs_buf.append((q, titles[pos]))
            locs_buf.append((qi, j))
        queries_completed += 1
        if len(pairs_buf) >= 2048:
            flush()
        if queries_completed % checkpoint_every_queries == 0:
            flush()
            done_qi.add(qi)
            for k_qi in todo[:n_processed]:
                done_qi.add(k_qi)
            np.save(out_score_path, scores)
            with open(progress_path, "w") as f:
                json.dump({"done_qi": sorted(done_qi)}, f)
            elapsed = time.time() - t0
            rate = queries_completed / max(elapsed, 1e-3)
            remaining = len(todo) - queries_completed
            eta_min = remaining / max(rate, 1e-3) / 60
            print(
                f"  ckpt @ qi={qi}  done={len(done_qi):,}/{N:,}  "
                f"({queries_completed} this run @ {rate:.2f} q/s)  eta {eta_min:.1f}m",
                flush=True,
            )
    flush()
    for qi in todo:
        done_qi.add(qi)
    np.save(out_score_path, scores)
    with open(progress_path, "w") as f:
        json.dump({"done_qi": sorted(done_qi)}, f)
    print(f"BGE scoring done in {time.time() - t0:.0f}s", flush=True)
    return scores


def eval_setups(
    qids,
    qrels,
    top_pos,
    pids_arr,
    score_matrices,
    valid,
    sample_idx=None,
    min_rel=2,
    exact_rel=3,
):
    """score_matrices is dict label -> (N, K) float."""
    pids_arr = np.asarray(pids_arr)
    if sample_idx is None:
        sample_idx = np.arange(len(qids))
    results = {}
    for label, mat in score_matrices.items():
        rs, ns, e1s, e3s = [], [], [], []
        for qi in sample_idx:
            qid = qids[int(qi)]
            s = mat[qi].copy()
            s[~valid[qi]] = -np.inf
            order = np.argsort(-s)[:K_EVAL]
            ordering = [
                pids_arr[int(top_pos[qi, j])] if top_pos[qi, j] >= 0 else None for j in order
            ]
            ordering = [p for p in ordering if p is not None]
            m = per_query_metrics(ordering, qrels[qid], min_rel=min_rel, exact_rel=exact_rel)
            if m is None:
                continue
            r, nd, e1, e3 = m
            rs.append(r)
            ns.append(nd)
            if not math.isnan(e1):
                e1s.append(e1)
                e3s.append(e3)
        out = {
            "r10": float(np.mean(rs)) if rs else 0.0,
            "ndcg10": float(np.mean(ns)) if ns else 0.0,
            "e1": float(np.mean(e1s)) if e1s else 0.0,
            "e3": float(np.mean(e3s)) if e3s else 0.0,
            "n": len(rs),
        }
        results[label] = out
        print(
            f"  {label:<40s} | R@10 {out['r10']:.4f}  nDCG@10 {out['ndcg10']:.4f}  "
            f"E@1 {out['e1']:.4f}  E@3 {out['e3']:.4f}  (n={out['n']})",
            flush=True,
        )
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--bod-model", required=True, help="path to query_model_*_me5_small_lora_bod")
    ap.add_argument(
        "--bod-vecs",
        default="me5_small_lora_bod.vecs.fp16.npy",
        help="filename inside data-dir for cached catalog vecs",
    )
    ap.add_argument("--bge-model", default="BAAI/bge-reranker-v2-m3")
    ap.add_argument("--top-k", type=int, default=100, help="rerank pool size")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--skip-bge", action="store_true", help="skip BGE scoring (use existing)")
    ap.add_argument("--queries-file", default="test_queries.jsonl")
    ap.add_argument("--qrels-file", default="test_qrels.jsonl")
    ap.add_argument(
        "--bm25-suffix",
        default="",
        help="suffix appended to bm25s_top200/bm25s_qids (e.g. '_1k') so this "
        "eval finds the right split's BM25 artifacts",
    )
    ap.add_argument(
        "--out-suffix",
        default="",
        help="suffix appended to bod/bge score caches and results.json under cc_eval/",
    )
    ap.add_argument(
        "--query-prefix",
        default="",
        help="prepend to each query before BoD encoding (e.g. 'search_query: ' for nomic)",
    )
    ap.add_argument(
        "--min-relevance",
        type=int,
        default=2,
        help="qrels relevance threshold for 'relevant' (default 2 = ESCI E+S; "
        "set to 1 for binary qrels like BestBuy)",
    )
    ap.add_argument(
        "--exact-relevance",
        type=int,
        default=3,
        help="qrels relevance threshold for 'exact match' (default 3 = ESCI E only; "
        "set to 1 for binary qrels)",
    )
    args = ap.parse_args()

    data = Path(args.data_dir).resolve()
    out_dir = data / "cc_eval"
    out_dir.mkdir(exist_ok=True)
    print(f"corpus: {data.name}  top_k: {args.top_k}", flush=True)

    qrels = defaultdict(dict)
    with open(data / args.qrels_file) as f:
        for line in f:
            r = json.loads(line)
            qrels[r["query_id"]][r["product_id"]] = r["relevance"]
    queries_all = {}
    with open(data / args.queries_file) as f:
        for line in f:
            d = json.loads(line)
            queries_all[d["query_id"]] = d["query"]
    with open(data / "titles.json") as f:
        titles = json.load(f)
    with open(data / "product_ids.json") as f:
        pids_arr = json.load(f)

    qids_path = data / f"bm25s_qids{args.bm25_suffix}.json"
    top_path = data / f"bm25s_top200{args.bm25_suffix}.npy"
    with open(qids_path) as f:
        eval_qids = json.load(f)
    bm25_top_full = np.load(top_path)
    if bm25_top_full.shape[1] < args.top_k:
        raise SystemExit(
            f"{top_path.name} has only {bm25_top_full.shape[1]} cols; need {args.top_k}"
        )
    top_pos = bm25_top_full[:, : args.top_k].astype(np.int64)
    queries = [queries_all[qid] for qid in eval_qids]
    print(f"  {len(eval_qids):,} eval queries  {len(titles):,} catalog", flush=True)

    valid = top_pos >= 0

    # 1) score with BoD
    bod_path = out_dir / f"bod_top{args.top_k}_scores{args.out_suffix}.npy"
    if bod_path.exists():
        bod_scores = np.load(bod_path)
        print(f"loaded cached BoD scores from {bod_path}", flush=True)
    else:
        catalog_vecs = np.load(data / args.bod_vecs, mmap_mode="r")
        bod_scores = score_with_bod(
            args.bod_model, queries, top_pos, catalog_vecs, query_prefix=args.query_prefix
        )
        np.save(bod_path, bod_scores)
        print(f"saved BoD scores -> {bod_path}", flush=True)

    # 2) score with BGE-reranker
    bge_path = out_dir / f"bge_top{args.top_k}_scores{args.out_suffix}.npy"
    bge_progress = out_dir / f"bge_top{args.top_k}_progress{args.out_suffix}.json"
    if args.skip_bge:
        if not bge_path.exists():
            print(
                f"--skip-bge set but {bge_path} not found; reporting BoD-only "
                f"results then exiting.",
                flush=True,
            )
            bod_valid = valid & ~np.isnan(bod_scores)
            nm_bod = normalize_per_query(np.nan_to_num(bod_scores, nan=0.0), bod_valid)
            score_matrices = {
                "BM25 alone (skip rerank)": np.where(
                    valid, -np.arange(args.top_k)[None, :].astype(np.float32), -np.inf
                ),
                "BoD alone (mE5+LoRA-BoD over BM25 top-K)": np.where(valid, nm_bod, -np.inf),
            }
            print(f"\neval over {len(eval_qids):,} queries:", flush=True)
            results = eval_setups(
                eval_qids,
                qrels,
                top_pos,
                pids_arr,
                score_matrices,
                valid,
                min_rel=args.min_relevance,
                exact_rel=args.exact_relevance,
            )
            with open(out_dir / "results_bod_only.json", "w") as f:
                json.dump(results, f, indent=2)
            return
        bge_scores = np.load(bge_path)
        print(f"loaded cached BGE scores from {bge_path}", flush=True)
    else:
        bge_scores = score_with_bge(
            args.bge_model,
            queries,
            titles,
            top_pos,
            bge_path,
            bge_progress,
            batch_size=args.batch_size,
        )

    # 3) build score matrices + report
    bge_valid = valid & ~np.isnan(bge_scores)
    bod_valid = valid & ~np.isnan(bod_scores)
    nm_bod = normalize_per_query(np.nan_to_num(bod_scores, nan=0.0), bod_valid)
    nm_bge = normalize_per_query(np.nan_to_num(bge_scores, nan=0.0), bge_valid)

    score_matrices = {
        "BM25 alone (skip rerank)": np.where(
            valid, -np.arange(args.top_k)[None, :].astype(np.float32), -np.inf
        ),
        "BoD alone (mE5+LoRA-BoD over BM25 top-K)": np.where(valid, nm_bod, -np.inf),
        "BGE alone (BGE-reranker over BM25 top-K)": np.where(valid, nm_bge, -np.inf),
        "BoD + BGE w=0.25": np.where(valid, 0.75 * nm_bod + 0.25 * nm_bge, -np.inf),
        "BoD + BGE w=0.50": np.where(valid, 0.50 * nm_bod + 0.50 * nm_bge, -np.inf),
        "BoD + BGE w=0.75": np.where(valid, 0.25 * nm_bod + 0.75 * nm_bge, -np.inf),
    }

    print(f"\neval over {len(eval_qids):,} queries (BM25 top-{args.top_k} pool):", flush=True)
    results = eval_setups(
        eval_qids,
        qrels,
        top_pos,
        pids_arr,
        score_matrices,
        valid,
        min_rel=args.min_relevance,
        exact_rel=args.exact_relevance,
    )

    with open(out_dir / f"results{args.out_suffix}.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nsaved results -> {out_dir / 'results.json'}", flush=True)


if __name__ == "__main__":
    main()
