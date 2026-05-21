#!/usr/bin/env python3
"""Build a bm25s index over a non-English ESCI corpus and run a k1/b sweep.

Mirrors indexing/build_bm25s_index.py but parameterized by corpus + language.
Saves the sweep results, the winning final index, and a top-K test-query
cache aligned with the eval qid order produced by Pattern 20-style evals.

Supported languages:
  - es: Snowball Spanish stemmer, Spanish stopwords
  - jp: fugashi morpheme tokenizer (UniDic-lite), no stemmer

Usage:
    .venv/bin/python indexing/build_bm25s_cross_lingual.py \\
        --data-dir esci_es_data --language es

    .venv/bin/python indexing/build_bm25s_cross_lingual.py \\
        --data-dir esci_jp_data --language jp
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

import bm25s  # noqa: E402
import numpy as np  # noqa: E402

SCRIPT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

K_EVAL = 10
SWEEP_K1 = [0.3, 0.5, 0.9, 1.2]
SWEEP_B = [0.3, 0.5, 0.6, 0.75]


def tokenize_en(texts: list[str]):
    import Stemmer

    stemmer = Stemmer.Stemmer("english")
    return bm25s.tokenize(texts, stopwords="en", stemmer=stemmer, show_progress=False)


def tokenize_es(texts: list[str]):
    import Stemmer

    stemmer = Stemmer.Stemmer("spanish")
    return bm25s.tokenize(texts, stopwords="es", stemmer=stemmer, show_progress=False)


def tokenize_jp(texts: list[str]):
    import fugashi

    tagger = fugashi.Tagger()
    tokens_list = []
    for t in texts:
        toks = []
        for w in tagger(t):
            s = w.surface.strip()
            if s:
                toks.append(s.lower())
        tokens_list.append(toks)
    return tokens_list


def eval_topk(top_k_pos: np.ndarray, eval_qids, qrels, pids_arr, k=K_EVAL, min_rel=2):
    pids_arr = np.asarray(pids_arr)
    recalls = []
    for qi, qid in enumerate(eval_qids):
        pos_pids = {p for p, g in qrels[qid].items() if g >= min_rel}
        if not pos_pids:
            continue
        positions = top_k_pos[qi, :k]
        hit_pids = set()
        for pos in positions:
            if pos < 0:
                continue
            hit_pids.add(pids_arr[pos])
        recalls.append(len(hit_pids & pos_pids) / len(pos_pids))
    return float(np.mean(recalls)) if recalls else 0.0


def ndcg_at_k(top_k_pos: np.ndarray, eval_qids, qrels, pids_arr, k=K_EVAL, min_rel=2, exact_rel=3):
    pids_arr = np.asarray(pids_arr)
    out = []
    for qi, qid in enumerate(eval_qids):
        pos_e = {p for p, g in qrels[qid].items() if g >= exact_rel}
        pos_es = {p for p, g in qrels[qid].items() if g >= min_rel}
        if not pos_es:
            continue
        positions = top_k_pos[qi, :k]
        gains = []
        seen = set()
        for pos in positions:
            if pos < 0:
                gains.append(0.0)
                continue
            pid = pids_arr[pos]
            if pid in seen:
                gains.append(0.0)
                continue
            seen.add(pid)
            gains.append(1.0 if pid in pos_e else (0.1 if pid in pos_es else 0.0))
        dcg = sum(g / math.log2(i + 2) for i, g in enumerate(gains))
        ideal = sorted((1.0 if p in pos_e else 0.1 for p in pos_es), reverse=True)[:k]
        idcg = sum(g / math.log2(i + 2) for i, g in enumerate(ideal))
        out.append(dcg / idcg if idcg > 0 else 0.0)
    return float(np.mean(out)) if out else 0.0


def build_index_and_retrieve(title_tokens, query_tokens, k1: float, b: float, top_k: int):
    idx = bm25s.BM25(k1=k1, b=b)
    idx.index(title_tokens, show_progress=False)
    results, _ = idx.retrieve(query_tokens, k=top_k, show_progress=False)
    return idx, np.asarray(results, dtype=np.int64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True, help="esci_es_data or esci_jp_data")
    ap.add_argument("--language", choices=["en", "es", "jp"], required=True)
    ap.add_argument(
        "--queries-file",
        default="test_queries.jsonl",
        help="filename inside data-dir for test queries",
    )
    ap.add_argument(
        "--qrels-file",
        default="test_qrels.jsonl",
        help="filename inside data-dir for test qrels",
    )
    ap.add_argument(
        "--out-suffix",
        default="",
        help="appended to bm25s_top200, bm25s_qids etc (e.g. '_1k') so multiple splits coexist",
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
    ap.add_argument("--top-k-eval", type=int, default=200)
    ap.add_argument("--top-k-sweep", type=int, default=100, help="top-K used in sweep eval")
    ap.add_argument("--skip-sweep", action="store_true", help="reuse winning k1/b if known")
    ap.add_argument("--k1", type=float, default=None, help="if --skip-sweep, use this")
    ap.add_argument("--b", type=float, default=None, help="if --skip-sweep, use this")
    args = ap.parse_args()

    data = Path(args.data_dir).resolve()
    print(f"corpus: {data.name}  language: {args.language}", flush=True)

    with open(data / "titles.json") as f:
        titles = json.load(f)
    with open(data / "product_ids.json") as f:
        pids_arr = json.load(f)
    if len(titles) != len(pids_arr):
        raise SystemExit(f"len(titles)={len(titles)} != len(pids)={len(pids_arr)}")
    print(f"  {len(titles):,} titles", flush=True)

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
    eval_qids = [
        qid
        for qid in queries_all
        if qid in qrels and any(g >= args.min_relevance for g in qrels[qid].values())
    ]
    queries = [queries_all[qid] for qid in eval_qids]
    print(f"  {len(eval_qids):,} eval queries", flush=True)

    print(f"\ntokenizing titles + queries ({args.language})...", flush=True)
    t0 = time.time()
    if args.language == "en":
        title_tokens = tokenize_en(titles)
        query_tokens = tokenize_en(queries)
    elif args.language == "es":
        title_tokens = tokenize_es(titles)
        query_tokens = tokenize_es(queries)
    else:
        title_tokens = tokenize_jp(titles)
        query_tokens = tokenize_jp(queries)
    print(f"  done in {time.time() - t0:.0f}s", flush=True)

    out_index_dir = data / "bm25s_index"
    out_index_dir.mkdir(exist_ok=True)

    if args.skip_sweep:
        if args.k1 is None or args.b is None:
            raise SystemExit("--skip-sweep requires --k1 and --b")
        best_k1, best_b = args.k1, args.b
        print(f"\nskipping sweep — using k1={best_k1}, b={best_b}", flush=True)
    else:
        print(
            f"\nsweep k1 in {SWEEP_K1} x b in {SWEEP_B} = {len(SWEEP_K1) * len(SWEEP_B)} combos",
            flush=True,
        )
        sweep_results = []
        for k1 in SWEEP_K1:
            for b in SWEEP_B:
                t1 = time.time()
                _, top = build_index_and_retrieve(
                    title_tokens, query_tokens, k1, b, args.top_k_sweep
                )
                r10 = eval_topk(
                    top, eval_qids, qrels, pids_arr, k=K_EVAL, min_rel=args.min_relevance
                )
                nd = ndcg_at_k(
                    top,
                    eval_qids,
                    qrels,
                    pids_arr,
                    k=K_EVAL,
                    min_rel=args.min_relevance,
                    exact_rel=args.exact_relevance,
                )
                print(
                    f"  k1={k1}  b={b}  R@10={r10:.4f}  nDCG@10={nd:.4f}  ({time.time() - t1:.0f}s)",
                    flush=True,
                )
                sweep_results.append({"k1": k1, "b": b, "r10": r10, "ndcg10": nd})
        sweep_results.sort(key=lambda r: r["r10"], reverse=True)
        best_k1, best_b = sweep_results[0]["k1"], sweep_results[0]["b"]
        print(f"\nbest: k1={best_k1}  b={best_b}  R@10={sweep_results[0]['r10']:.4f}", flush=True)
        sweep_path = data / f"bm25s_sweep{args.out_suffix}.json"
        with open(sweep_path, "w") as f:
            json.dump(sweep_results, f, indent=2)
        print(f"  saved sweep results to {sweep_path}", flush=True)

    print(f"\nbuilding final index with k1={best_k1}, b={best_b}...", flush=True)
    t0 = time.time()
    idx, top_eval = build_index_and_retrieve(
        title_tokens, query_tokens, best_k1, best_b, args.top_k_eval
    )
    print(f"  done in {time.time() - t0:.0f}s", flush=True)

    idx.save(str(out_index_dir), show_progress=False)
    with open(out_index_dir / "bm25_params.json", "w") as f:
        json.dump(
            {"k1": best_k1, "b": best_b, "language": args.language},
            f,
        )

    out_top_path = data / f"bm25s_top{args.top_k_eval}{args.out_suffix}.npy"
    np.save(out_top_path, top_eval.astype(np.int64))
    print(f"  saved {out_top_path}: shape={top_eval.shape}", flush=True)

    qids_path = data / f"bm25s_qids{args.out_suffix}.json"
    with open(qids_path, "w") as f:
        json.dump(eval_qids, f)
    print(f"  saved {qids_path}", flush=True)

    final_r10 = eval_topk(
        top_eval, eval_qids, qrels, pids_arr, k=K_EVAL, min_rel=args.min_relevance
    )
    final_ndcg = ndcg_at_k(
        top_eval,
        eval_qids,
        qrels,
        pids_arr,
        k=K_EVAL,
        min_rel=args.min_relevance,
        exact_rel=args.exact_relevance,
    )
    print(
        f"\nfinal BM25-alone R@10={final_r10:.4f}  nDCG@10={final_ndcg:.4f}  (k={args.top_k_eval})",
        flush=True,
    )


if __name__ == "__main__":
    main()
