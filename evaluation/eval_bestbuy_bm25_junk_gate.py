#!/usr/bin/env python3
"""Can a BM25 signal over the same catalog remove the BoD encoder's category junk?

Follow-on to evaluation/eval_bestbuy_llm_judge_junkrate.py, which measured a
category "junk-rate" (fraction of the top-10 an LLM judge calls the wrong
product TYPE) of 0.258 for base MiniLM and 0.141 for the BoD-fine-tuned MiniLM
over the full 1.27M-product BestBuy catalog, on 250 sampled holdout queries.
That is a real residual failure mode and it is only weakly correlated with
R@10 / E@1, so the click metrics cannot see it.

Eyeballing the worst queries suggested a specific mechanism: on queries with
rare / misspelled / OOV tokens, BOTH dense models collapse into the CD / DVD /
VHS media long tail that dominates the catalog by count --

    tamrac              -> "Tamiz - CD", "Tambu - CD", "Tambo - CD"
    bravia 40           -> "Brava - CD", "Bravada - CD"
    ford focus          -> 10x "Focus - CD" / "Focus - CASSETTE"
    laptops for teachers-> "Teachers - DVD", "Art for Teachers of Children - VHS"

-- which looks like WordPiece subword overlap plus a short-title length effect,
not semantic relevance.

PHASE 1 (--phase diagnose) tests three competing explanations for the measured
junk, using only cached artifacts (no API spend):

    (a) deceptive lexical overlap  junk titles share whole/partial tokens with
                                   the query despite the wrong category
    (b) pure semantic drift        junk titles have ~zero lexical grounding
    (c) hubness                    a few catalog items recur as junk across
                                   many unrelated queries

PHASE 2 (--phase eval) tests whether a BM25 leg over the same catalog fixes it.
BM25 respects whole-token matching (after stemming) and IDF, so a title that
only "looks like" the query at the subword level scores zero. Two interventions
are scored against the unmodified BoD dense ranking:

    gate   BoD dense top-N, stable-partitioned so that candidates with a
           nonzero BM25 score against the query come first. Pure demotion:
           introduces no BM25-only documents, and is a no-op when the query has
           no in-vocabulary token (so it cannot hurt misspelling queries).
    rrf    RRF(BoD-dense-top-N, BM25-top-N) at a couple of rrf_k values --
           the standard hybrid, which CAN introduce BM25-only documents.

New top-10 entries that were already judged by the junkrate run reuse that
label; only genuinely new (query, product) pairs are sent to gpt-4o-mini, with
the same prompt and the same cost guard.

Phases (cached to --work-dir, resumable):

    --phase pool      BoD dense top-N + BM25 top-N over the full catalog, plus
                      the BM25 score of every already-judged (query, doc) pair
    --phase diagnose  Part 1: mechanism of the junk (no API spend)
    --phase estimate  project OpenAI cost for the new candidates -- spends nothing
    --phase judge     category yes/no over the NEW (query, product) pairs only
    --phase eval      junk-rate + R@10 + E@1 per variant, bootstrap CIs -> JSON

Usage (the repo .venv lacks torch/bm25s; layer them over the job-search venv):
    uv run --no-project --python /Users/dtunkelang/job-search/.venv/bin/python \\
        --with bm25s --with PyStemmer \\
        python evaluation/eval_bestbuy_bm25_junk_gate.py --phase pool
"""

import argparse
import asyncio
import json
import math
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np  # noqa: E402
from dotenv import load_dotenv  # noqa: E402

from evaluation.eval_bestbuy_llm_judge_junkrate import (  # noqa: E402
    CATEGORY_PROMPT,
    DATASET_REPO,
    Usage,
    _chat,
    _load_judged,
    _score_from_logprobs,
    estimate_cost,
    make_client,
    pair_key,
    record_spend,
    resolve_data_dir,
)
from evaluation.eval_bestbuy_llm_judge_rerank import (  # noqa: E402
    load_corpus,
    load_split,
    per_query_metrics,
)
from evaluation.eval_esci_llm_judge_lexical_bias import overlap_metrics, toks  # noqa: E402

load_dotenv(override=True)

K_EVAL = 10

# Hard stop: refuse to start a paid phase whose projection exceeds this.
COST_CEILING_USD = 2.0

# Queries named in the diagnosis; reported by name so the fix can be checked on
# exactly the anecdotes that motivated it rather than on the aggregate only.
WATCH_QUERIES = (
    "tamrac",
    "bravia 40",
    "ford focus",
    "a skylit drive",
    "laptops for teachers",
)


# --------------------------------------------------------------------------
# paths
# --------------------------------------------------------------------------
def work_paths(work_dir, tag, junk_dir, junk_tag):
    w = Path(work_dir)
    w.mkdir(parents=True, exist_ok=True)
    j = Path(junk_dir)
    return {
        # produced here
        "rank": w / f"gate_ranks_{tag}.npz",
        "pairbm25": w / f"gate_pairbm25_{tag}.json",
        "judge": w / f"gate_judge_{tag}.jsonl",
        "judge_meta": w / f"gate_judge_meta_{tag}.json",
        # consumed from the junkrate run
        "junk_pool": j / f"junk_pool_{junk_tag}.json",
        "junk_judge": j / f"junk_judge_{junk_tag}.jsonl",
    }


def load_junk_pool(paths):
    p = paths["junk_pool"]
    if not Path(p).exists():
        raise SystemExit(
            f"missing {p} -- run evaluation/eval_bestbuy_llm_judge_junkrate.py --phase pool first"
        )
    with open(p) as f:
        return json.load(f)


def load_all_judged(paths):
    """Cached junkrate labels, then this run's -- later wins on a repeat pair."""
    out = {}
    out.update(_load_judged(paths["junk_judge"]))
    out.update(_load_judged(paths["judge"]))
    return out


# --------------------------------------------------------------------------
# BM25 (same conventions as evaluation/analyze_esci_bm25_paraphrase_rrf.py)
# --------------------------------------------------------------------------
def build_bm25(titles, k1, b):
    import bm25s
    from Stemmer import Stemmer

    stemmer = Stemmer("english")
    print(f"  tokenizing {len(titles):,} docs (stem=en, stopwords=en)...", flush=True)
    t0 = time.time()
    tok = bm25s.tokenize(titles, stopwords="en", stemmer=stemmer, show_progress=False)
    print(f"    {time.time() - t0:.1f}s", flush=True)
    print(f"  indexing BM25 k1={k1} b={b}...", flush=True)
    t0 = time.time()
    idx = bm25s.BM25(k1=k1, b=b)
    idx.index(tok, show_progress=False)
    print(f"    {time.time() - t0:.1f}s", flush=True)
    return idx, stemmer


def bm25_query_tokens(stemmer, queries):
    """Per-query stemmed, stopworded token lists -- the units BM25 actually matches."""
    import bm25s

    tk = bm25s.tokenize(queries, stopwords="en", stemmer=stemmer, show_progress=False)
    inv = {v: k for k, v in tk.vocab.items()}
    return [[inv[i] for i in ids] for ids in tk.ids]


def bm25_topk(idx, stemmer, queries, k):
    import bm25s

    t0 = time.time()
    qtok = bm25s.tokenize(queries, stopwords="en", stemmer=stemmer, show_progress=False)
    res_idx, res_score = idx.retrieve(qtok, k=k, show_progress=False)
    print(f"  BM25 top-{k} for {len(queries):,} queries in {time.time() - t0:.1f}s", flush=True)
    return np.asarray(res_idx, dtype=np.int64), np.asarray(res_score, dtype=np.float32)


# --------------------------------------------------------------------------
# dense
# --------------------------------------------------------------------------
def topk_over_catalog(qv, catalog_path, k, chunk=100_000):
    """Exact top-k cosine over a memmapped fp16 catalog, in row blocks."""
    catalog = np.load(catalog_path, mmap_mode="r")
    n_docs = catalog.shape[0]
    n_q = qv.shape[0]
    best_scores = np.full((n_q, k), -np.inf, dtype=np.float32)
    best_idx = np.full((n_q, k), -1, dtype=np.int64)
    t0 = time.time()
    for start in range(0, n_docs, chunk):
        end = min(start + chunk, n_docs)
        block = np.asarray(catalog[start:end]).astype(np.float32)
        sims = qv @ block.T
        m_block = min(k, sims.shape[1])
        part = np.argpartition(-sims, m_block - 1, axis=1)[:, :m_block]
        part_scores = np.take_along_axis(sims, part, axis=1)
        cand_scores = np.concatenate([best_scores, part_scores], axis=1)
        cand_idx = np.concatenate([best_idx, part + start], axis=1)
        keep = np.argpartition(-cand_scores, k - 1, axis=1)[:, :k]
        best_scores = np.take_along_axis(cand_scores, keep, axis=1)
        best_idx = np.take_along_axis(cand_idx, keep, axis=1)
        del sims, block
    order = np.argsort(-best_scores, axis=1)
    best_scores = np.take_along_axis(best_scores, order, axis=1)
    best_idx = np.take_along_axis(best_idx, order, axis=1)
    print(f"    top-{k} over {n_docs:,} docs in {time.time() - t0:.0f}s", flush=True)
    return best_idx, best_scores


# --------------------------------------------------------------------------
# phase: pool
# --------------------------------------------------------------------------
def phase_pool(args, paths):
    import torch
    from sentence_transformers import SentenceTransformer

    payload = load_junk_pool(paths)
    rows = payload["rows"]
    queries = [r["query"] for r in rows]
    keys = [r["key"] for r in rows]
    print(f"  {len(rows)} queries reused verbatim from the junkrate pool", flush=True)

    data = Path(payload["data_dir"])
    if not data.exists():
        data = resolve_data_dir(args)
    data, titles, pids = load_corpus(data)
    pid_to_i = {p: i for i, p in enumerate(pids)}

    device = (
        "mps"
        if torch.backends.mps.is_available()
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"  encoding queries with {args.bod_model} on {device}...", flush=True)
    m = SentenceTransformer(args.bod_model, device=device)
    qv = m.encode(queries, normalize_embeddings=True, batch_size=64, show_progress_bar=False)
    del m
    qv = qv.astype(np.float32)
    dense_idx, dense_sim = topk_over_catalog(qv, data / args.bod_vecs, args.depth)

    print("\nbuilding BM25 over the full catalog...", flush=True)
    idx, stemmer = build_bm25(titles, args.k1, args.b)
    bm_idx, bm_score = bm25_topk(idx, stemmer, queries, args.depth)
    qtokens = bm25_query_tokens(stemmer, queries)

    # BM25 score of every (query, doc) pair we will ever need: the dense top-N,
    # the BM25 top-N, and every pair the junkrate run already judged. get_scores
    # is a full-corpus pass, so do exactly one per query.
    print("\nscoring pairs (one full-corpus BM25 pass per query)...", flush=True)
    t0 = time.time()
    pair_bm25 = {}
    for qi, r in enumerate(rows):
        need = set(int(j) for j in dense_idx[qi] if j >= 0)
        need |= {int(j) for j in bm_idx[qi] if j >= 0}
        for model_key in ("base", "bod"):
            for d in r[model_key]:
                di = pid_to_i.get(d["product_id"])
                if di is not None:
                    need.add(di)
        scores = idx.get_scores(qtokens[qi]) if qtokens[qi] else None
        for di in need:
            s = 0.0 if scores is None else float(scores[di])
            pair_bm25[pair_key(r["query"], pids[di])] = s
    print(f"  {len(pair_bm25):,} pair scores in {time.time() - t0:.0f}s", flush=True)

    np.savez_compressed(
        paths["rank"],
        keys=np.asarray(keys, dtype=object),
        queries=np.asarray(queries, dtype=object),
        dense_idx=dense_idx,
        dense_sim=dense_sim,
        bm25_idx=bm_idx,
        bm25_score=bm_score,
        is_manual=np.asarray([r["is_manual"] for r in rows]),
        n_query_tokens=np.asarray([len(t) for t in qtokens]),
        data_dir=np.asarray(str(data), dtype=object),
        depth=np.asarray(args.depth),
    )
    with open(paths["pairbm25"], "w") as f:
        json.dump(
            {
                "k1": args.k1,
                "b": args.b,
                "tokenizer": "bm25s.tokenize(stopwords='en', stemmer=Stemmer('english'))",
                "query_tokens": {r["query"]: qtokens[i] for i, r in enumerate(rows)},
                "scores": pair_bm25,
            },
            f,
        )
    print(f"\nsaved -> {paths['rank']}\nsaved -> {paths['pairbm25']}", flush=True)


def load_ranks(paths):
    if not Path(paths["rank"]).exists():
        raise SystemExit(f"missing {paths['rank']} -- run --phase pool first")
    z = np.load(paths["rank"], allow_pickle=True)
    return {
        "keys": [str(x) for x in z["keys"]],
        "queries": [str(x) for x in z["queries"]],
        "dense_idx": z["dense_idx"],
        "dense_sim": z["dense_sim"],
        "bm25_idx": z["bm25_idx"],
        "bm25_score": z["bm25_score"],
        "is_manual": z["is_manual"],
        "n_query_tokens": z["n_query_tokens"],
        "data_dir": str(z["data_dir"]),
        "depth": int(z["depth"]),
    }


def load_pair_bm25(paths):
    if not Path(paths["pairbm25"]).exists():
        raise SystemExit(f"missing {paths['pairbm25']} -- run --phase pool first")
    with open(paths["pairbm25"]) as f:
        return json.load(f)


# --------------------------------------------------------------------------
# phase: diagnose (Part 1 -- no API spend)
# --------------------------------------------------------------------------
def _char_overlap(query, title):
    """Sub-token similarity: how much of the query survives at CHARACTER level.

    A WordPiece encoder can match "tamrac" to "Tambu" on shared word pieces
    while whole-token matching sees nothing. Two cheap proxies, both taken as
    the best over (query token, title token) pairs and averaged over query
    tokens:

    prefix : len(common prefix) / len(query token)
    lcs    : len(longest common substring) / len(query token)
    """
    q, d = toks(query), toks(title)
    if not q or not d:
        return 0.0, 0.0
    pre, lcs = [], []
    for qt in q:
        best_p = best_l = 0
        for dt in d:
            n = 0
            while n < min(len(qt), len(dt)) and qt[n] == dt[n]:
                n += 1
            best_p = max(best_p, n)
            # longest common substring, O(len(qt) * len(dt)) DP on short tokens
            prev = [0] * (len(dt) + 1)
            for i in range(1, len(qt) + 1):
                cur = [0] * (len(dt) + 1)
                for j in range(1, len(dt) + 1):
                    if qt[i - 1] == dt[j - 1]:
                        cur[j] = prev[j - 1] + 1
                        best_l = max(best_l, cur[j])
                prev = cur
        pre.append(best_p / len(qt))
        lcs.append(best_l / len(qt))
    return float(np.mean(pre)), float(np.mean(lcs))


MEDIA_SUFFIXES = (
    " - cd",
    " - dvd",
    " - vhs",
    " - cassette",
    " - blu-ray",
    " - vinyl",
    " - lp",
)


def _is_media(title):
    t = (title or "").lower()
    return any(s in t for s in MEDIA_SUFFIXES)


def phase_diagnose(args, paths):
    payload = load_junk_pool(paths)
    judged = load_all_judged(paths)
    pb = load_pair_bm25(paths)
    scores = pb["scores"]
    thr = args.junk_threshold

    titles_all = None
    catalog_media_rate = None
    try:
        _d, titles_all, _p = load_corpus(Path(payload["data_dir"]))
        catalog_media_rate = float(np.mean([_is_media(t) for t in titles_all]))
    except Exception:  # diagnosis must still run without the 1.9GB snapshot
        pass

    buckets = {"junk": [], "ok": []}
    junk_pid_queries = defaultdict(set)
    junk_pid_titles = {}
    n_slots = 0
    per_query_junk = []

    for r in payload["rows"]:
        if r["is_manual"]:
            continue
        seen = set()
        for model_key in ("base", "bod"):
            for d in r[model_key]:
                k = pair_key(r["query"], d["product_id"])
                j = judged.get(k)
                if not j or j.get("p_yes") is None:
                    continue
                n_slots += 1
                is_junk = j["p_yes"] < thr
                if k in seen:
                    continue
                seen.add(k)
                cov, jac = overlap_metrics(r["query"], d["title"])
                pre, lcs = _char_overlap(r["query"], d["title"])
                rec = {
                    "query": r["query"],
                    "product_id": d["product_id"],
                    "title": d["title"],
                    "coverage": cov,
                    "jaccard": jac,
                    "prefix": pre,
                    "lcs": lcs,
                    "bm25": scores.get(k),
                    "media": _is_media(d["title"]),
                    "n_title_tokens": len(toks(d["title"])),
                }
                buckets["junk" if is_junk else "ok"].append(rec)
                if is_junk:
                    junk_pid_queries[d["product_id"]].add(r["query"])
                    junk_pid_titles[d["product_id"]] = d["title"]
        per_query_junk.append(
            sum(
                1
                for d in r["bod"]
                if (judged.get(pair_key(r["query"], d["product_id"])) or {}).get("p_yes", 1.0) < thr
            )
        )

    def stats(recs):
        def col(name):
            v = np.asarray(
                [x[name] for x in recs if x[name] is not None],
                dtype=np.float64,
            )
            return v

        out = {"n": len(recs)}
        for name in ("coverage", "jaccard", "prefix", "lcs", "bm25", "n_title_tokens"):
            v = col(name)
            out[f"mean_{name}"] = float(v.mean()) if v.size else float("nan")
            out[f"median_{name}"] = float(np.median(v)) if v.size else float("nan")
        cov = col("coverage")
        out["frac_zero_token_overlap"] = float(np.mean(cov == 0.0)) if cov.size else float("nan")
        out["frac_full_token_coverage"] = float(np.mean(cov >= 1.0)) if cov.size else float("nan")
        bm = col("bm25")
        out["frac_zero_bm25"] = float(np.mean(bm == 0.0)) if bm.size else float("nan")
        pre = col("prefix")
        out["frac_prefix_ge_0.5_and_zero_token"] = (
            float(np.mean((pre >= 0.5) & (cov == 0.0))) if pre.size else float("nan")
        )
        out["frac_media_title"] = float(np.mean([x["media"] for x in recs])) if recs else 0.0
        return out

    diag = {
        "n_judged_slots": n_slots,
        "junk_threshold_p_yes": thr,
        "junk": stats(buckets["junk"]),
        "ok": stats(buckets["ok"]),
        "catalog_media_rate": catalog_media_rate,
    }

    # (c) hubness: are the same catalog items junk for many unrelated queries?
    multi = sorted(junk_pid_queries.items(), key=lambda kv: -len(kv[1]))
    n_junk_slots = len(buckets["junk"])
    n_distinct = len(junk_pid_queries)
    diag["hubness"] = {
        "n_junk_pairs": n_junk_slots,
        "n_distinct_junk_products": n_distinct,
        "mean_queries_per_junk_product": (n_junk_slots / max(n_distinct, 1)),
        "frac_junk_pairs_on_repeat_products": float(
            sum(len(v) for _p, v in multi if len(v) > 1) / max(n_junk_slots, 1)
        ),
        "top_repeat_products": [
            {
                "product_id": p,
                "title": junk_pid_titles[p],
                "n_queries": len(v),
                "queries": sorted(v)[:8],
            }
            for p, v in multi[:15]
            if len(v) > 1
        ],
    }

    # the media long tail specifically
    junk_media = [x for x in buckets["junk"] if x["media"]]
    diag["media_tail"] = {
        "frac_of_junk_that_is_media": float(len(junk_media) / max(n_junk_slots, 1)),
        "media_junk_mean_coverage": float(np.mean([x["coverage"] for x in junk_media]))
        if junk_media
        else float("nan"),
        "media_junk_mean_prefix": float(np.mean([x["prefix"] for x in junk_media]))
        if junk_media
        else float("nan"),
        "media_junk_frac_zero_bm25": float(
            np.mean([x["bm25"] == 0.0 for x in junk_media if x["bm25"] is not None])
        )
        if junk_media
        else float("nan"),
        "media_junk_mean_title_tokens": float(np.mean([x["n_title_tokens"] for x in junk_media]))
        if junk_media
        else float("nan"),
    }

    # how much junk would a zero-BM25 gate even be ABLE to touch?
    junk_bm = [x["bm25"] for x in buckets["junk"] if x["bm25"] is not None]
    ok_bm = [x["bm25"] for x in buckets["ok"] if x["bm25"] is not None]
    diag["gate_headroom"] = {
        "frac_junk_with_zero_bm25": float(np.mean([s == 0.0 for s in junk_bm])) if junk_bm else 0.0,
        "frac_ok_with_zero_bm25": float(np.mean([s == 0.0 for s in ok_bm])) if ok_bm else 0.0,
        "note": "a zero-BM25 gate can only demote the first group; it wrongly demotes the second",
    }

    print(json.dumps(diag, indent=2, default=float), flush=True)
    print("\nworst repeat junk products:", flush=True)
    for e in diag["hubness"]["top_repeat_products"][:10]:
        print(f"  {e['n_queries']:>3}x  {e['title'][:60]:<60} {e['queries'][:4]}", flush=True)
    return diag


# --------------------------------------------------------------------------
# variants
# --------------------------------------------------------------------------
def rrf_merge(rankings_list, top_k, rrf_k=60):
    """Sum of 1/(rrf_k + rank + 1) over the rankings a doc appears in.

    Verbatim construction from evaluation/analyze_esci_bm25_paraphrase_rrf.py.
    """
    scores = defaultdict(float)
    for rankings in rankings_list:
        for rank, doc_idx in enumerate(rankings):
            scores[int(doc_idx)] += 1.0 / (rrf_k + rank + 1)
    return sorted(scores, key=scores.get, reverse=True)[:top_k]


def build_variants(ranks, pair_bm25, pids, args):
    """query index -> {variant name: [doc index, ...] top-K_EVAL}."""
    scores = pair_bm25["scores"]
    depth = ranks["depth"]
    n_gate = args.gate_depth if args.gate_depth else depth
    out = []
    for qi, q in enumerate(ranks["queries"]):
        dense = [int(j) for j in ranks["dense_idx"][qi] if j >= 0]
        bm = [
            int(j)
            for j, s in zip(ranks["bm25_idx"][qi], ranks["bm25_score"][qi])
            if j >= 0 and s > 0
        ]
        v = {"dense": dense[:K_EVAL], "bm25": bm[:K_EVAL]}

        # gate: stable partition of the dense list on "has any BM25 mass".
        for label, nd in (("gate", n_gate), ("gate_deep", depth)):
            pool = dense[:nd]
            keep = [j for j in pool if scores.get(pair_key(q, pids[j]), 0.0) > 0.0]
            drop = [j for j in pool if scores.get(pair_key(q, pids[j]), 0.0) <= 0.0]
            v[label] = (keep + drop)[:K_EVAL]

        for rk in args.rrf_k:
            v[f"rrf{rk}"] = rrf_merge([dense[:depth], bm[:depth]], K_EVAL, rrf_k=rk)
        out.append(v)
    return out


# --------------------------------------------------------------------------
# phase: estimate / judge
# --------------------------------------------------------------------------
def _new_pairs(args, paths):
    ranks = load_ranks(paths)
    pair_bm25 = load_pair_bm25(paths)
    _d, titles, pids = load_corpus(Path(ranks["data_dir"]))
    variants = build_variants(ranks, pair_bm25, pids, args)
    judged = load_all_judged(paths)

    todo = {}
    for qi, q in enumerate(ranks["queries"]):
        for docs in variants[qi].values():
            for j in docs:
                k = pair_key(q, pids[j])
                if k in judged or k in todo:
                    continue
                todo[k] = {
                    "query": q,
                    "product_id": pids[j],
                    "title": titles[j],
                    "prompt": CATEGORY_PROMPT.format(
                        query=q, title=titles[j][: args.max_title_chars]
                    ),
                }
    return ranks, variants, titles, pids, judged, todo


def phase_estimate(args, paths, quiet=False):
    _r, _v, _t, _p, judged, todo = _new_pairs(args, paths)
    CHAT_ENVELOPE_TOKENS = 8
    tin = sum(len(p["prompt"]) / 4.0 + CHAT_ENVELOPE_TOKENS for p in todo.values())
    tout = len(todo)
    cost = estimate_cost(args.model, tin, tout)
    out = {
        "model": args.model,
        "n_cached_labels_reused": len(judged),
        "n_new_pairs": len(todo),
        "est_tokens_in": int(tin),
        "est_tokens_out": int(tout),
        "est_cost_usd_total": cost,
        "ceiling_usd": COST_CEILING_USD,
    }
    if not quiet:
        print(json.dumps(out, indent=2), flush=True)
        print(
            f"\nPROJECTED COST: ${cost:.4f} ({len(todo):,} new calls) "
            f"vs ceiling ${COST_CEILING_USD:.2f}",
            flush=True,
        )
    return out


def _guard_cost(args, paths):
    est = phase_estimate(args, paths, quiet=True)
    c = est["est_cost_usd_total"]
    print(f"[cost guard] projected ${c:.4f} (ceiling ${COST_CEILING_USD:.2f})", flush=True)
    if c > COST_CEILING_USD:
        raise SystemExit(
            f"Refusing to run: projected ${c:.4f} exceeds ceiling ${COST_CEILING_USD:.2f}. "
            f"Lower --depth/--rrf-k or raise --cost-ceiling deliberately."
        )
    return est


async def _run_judge(args, todo, usage, out_f):
    client = make_client()
    sem = asyncio.Semaphore(args.concurrency)

    async def one(item):
        _key, p = item
        ch = await _chat(client, sem, usage, args.model, p["prompt"], 1, logprobs=True)
        try:
            margin, p_yes = _score_from_logprobs(ch)
        except Exception:  # never let one odd token kill a paid run
            usage.errors += 1
            margin, p_yes = float("nan"), float("nan")
        return p, margin, p_yes

    t0 = time.time()
    done = 0
    chunk = 500
    for i in range(0, len(todo), chunk):
        batch = todo[i : i + chunk]
        for p, margin, p_yes in await asyncio.gather(*[one(x) for x in batch]):
            out_f.write(
                json.dumps(
                    {
                        "query": p["query"],
                        "product_id": p["product_id"],
                        "title": p["title"],
                        "margin": None if math.isnan(margin) else margin,
                        "p_yes": None if math.isnan(p_yes) else p_yes,
                    }
                )
                + "\n"
            )
        out_f.flush()
        done += len(batch)
        el = time.time() - t0
        print(
            f"  [judge {done}/{len(todo)}] {done / max(el, 1e-9):.1f} pairs/s "
            f"errors={usage.errors} spent=${estimate_cost(args.model, usage.tin, usage.tout):.4f}",
            flush=True,
        )


def phase_judge(args, paths):
    _guard_cost(args, paths)
    _r, _v, _t, _p, judged, todo = _new_pairs(args, paths)
    items = list(todo.items())
    print(f"  {len(judged):,} cached labels reused, {len(items):,} new pairs to judge", flush=True)
    if not items:
        print("  fully cached; nothing to do", flush=True)
        return

    usage = Usage()
    t0 = time.time()
    try:
        with open(paths["judge"], "a") as out_f:
            asyncio.run(_run_judge(args, items, usage, out_f))
    finally:
        c = estimate_cost(args.model, usage.tin, usage.tout)
        if usage.calls:
            record_spend(args.model, usage.tin, usage.tout, c, "bestbuy BM25 junk gate: judge")

    cost = estimate_cost(args.model, usage.tin, usage.tout)
    meta = {
        "judge_model": args.model,
        "prompt": CATEGORY_PROMPT,
        "mode": "pointwise category yes/no, max_tokens=1, top_logprobs=20",
        "n_new_pairs_judged": len(items),
        "n_cached_labels_reused": len(judged),
        "tokens_in": usage.tin,
        "tokens_out": usage.tout,
        "api_calls": usage.calls,
        "api_errors": usage.errors,
        "cost_usd": cost,
        "wall_clock_s": time.time() - t0,
    }
    with open(paths["judge_meta"], "w") as f:
        json.dump(meta, f, indent=2)
    print(
        f"\njudge done in {meta['wall_clock_s'] / 60:.1f}m calls={usage.calls:,} "
        f"errors={usage.errors} cost=${cost:.4f}",
        flush=True,
    )


# --------------------------------------------------------------------------
# phase: eval
# --------------------------------------------------------------------------
def _boot_mean_ci(values, n_boot=2000, seed=0):
    v = np.asarray([x for x in values if x is not None and not math.isnan(x)], dtype=np.float64)
    if v.size == 0:
        return None, None
    rng = np.random.default_rng(seed)
    means = v[rng.integers(0, v.size, size=(n_boot, v.size))].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def _paired_delta_ci(a, b, n_boot=2000, seed=0):
    """Paired bootstrap over queries of mean(a) - mean(b). Returns (delta, lo, hi)."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    ok = ~(np.isnan(a) | np.isnan(b))
    a, b = a[ok], b[ok]
    if a.size == 0:
        return float("nan"), None, None
    d = a - b
    rng = np.random.default_rng(seed)
    draws = d[rng.integers(0, d.size, size=(n_boot, d.size))].mean(axis=1)
    return (
        float(d.mean()),
        float(np.percentile(draws, 2.5)),
        float(np.percentile(draws, 97.5)),
    )


def phase_eval(args, paths):
    ranks = load_ranks(paths)
    pair_bm25 = load_pair_bm25(paths)
    data = Path(ranks["data_dir"])
    _d, titles, pids = load_corpus(data)
    qrels, _q = load_split(data, args.queries_file, args.qrels_file)
    judged = load_all_judged(paths)
    variants = build_variants(ranks, pair_bm25, pids, args)
    thr = args.junk_threshold

    names = list(variants[0].keys())
    held = [qi for qi in range(len(ranks["queries"])) if not ranks["is_manual"][qi]]

    per_variant = {n: {"junk": [], "recall": [], "e1": [], "ndcg": [], "hit": []} for n in names}
    per_query = {n: {} for n in names}
    n_missing = 0

    for qi in held:
        q = ranks["queries"][qi]
        key = ranks["keys"][qi]
        for n in names:
            docs = variants[qi][n][:K_EVAL]
            labels = []
            for j in docs:
                jr = judged.get(pair_key(q, pids[j]))
                if jr and jr.get("p_yes") is not None:
                    labels.append(jr["p_yes"])
                else:
                    n_missing += 1
            junk = sum(1 for p in labels if p < thr) / len(labels) if labels else float("nan")
            m = per_query_metrics(
                [pids[j] for j in docs],
                qrels[key],
                k=K_EVAL,
                min_rel=args.min_relevance,
                exact_rel=args.exact_relevance,
            )
            recall, ndcg, e1, _e3 = m if m else (float("nan"),) * 4
            gold = {p for p, g in qrels[key].items() if g >= args.min_relevance}
            hit = 1.0 if any(pids[j] in gold for j in docs) else 0.0
            per_variant[n]["junk"].append(junk)
            per_variant[n]["recall"].append(recall)
            per_variant[n]["ndcg"].append(ndcg)
            per_variant[n]["e1"].append(e1)
            per_variant[n]["hit"].append(hit)
            per_query[n][key] = {
                "query": q,
                "junk_rate": junk,
                "recall_at_10": recall,
                "e_at_1": e1,
            }
    if n_missing:
        print(f"  WARNING: {n_missing} (variant, query, doc) slots have no judge label", flush=True)

    def nm(v):
        a = np.asarray(v, dtype=np.float64)
        a = a[~np.isnan(a)]
        return float(a.mean()) if a.size else float("nan")

    summary = {}
    for n in names:
        d = per_variant[n]
        lo, hi = _boot_mean_ci(d["junk"], args.n_boot, args.seed)
        summary[n] = {
            "junk_rate_at_10": nm(d["junk"]),
            "junk_rate_ci95": [lo, hi],
            "queries_with_zero_junk": float(np.mean([x == 0.0 for x in d["junk"]])),
            "queries_majority_junk": float(np.mean([x > 0.5 for x in d["junk"]])),
            "recall_at_10_fraction_recovered": nm(d["recall"]),
            "hit_rate_at_10": nm(d["hit"]),
            "ndcg_at_10": nm(d["ndcg"]),
            "e_at_1": nm(d["e1"]),
            "mean_changed_slots_vs_dense": None,
        }

    for n in names:
        changed = []
        for qi in held:
            base = variants[qi]["dense"][:K_EVAL]
            v = variants[qi][n][:K_EVAL]
            changed.append(len(set(base) ^ set(v)) / 2.0)
        summary[n]["mean_changed_slots_vs_dense"] = float(np.mean(changed))

    deltas = {}
    for n in names:
        if n == "dense":
            continue
        e = {}
        for metric, field in (
            ("junk_rate_at_10", "junk"),
            ("recall_at_10", "recall"),
            ("e_at_1", "e1"),
            ("ndcg_at_10", "ndcg"),
            ("hit_rate_at_10", "hit"),
        ):
            dv, lo, hi = _paired_delta_ci(
                per_variant[n][field], per_variant["dense"][field], args.n_boot, args.seed
            )
            e[metric] = {"delta": dv, "ci95": [lo, hi]}
        deltas[n] = e

    # named-anecdote check
    watch = {}
    qlower = {ranks["queries"][qi].strip().lower(): qi for qi in range(len(ranks["queries"]))}
    for wq in args.watch_queries:
        qi = qlower.get(wq.strip().lower())
        if qi is None:
            continue
        entry = {}
        for n in names:
            docs = variants[qi][n][:K_EVAL]
            lst = []
            for j in docs:
                jr = judged.get(pair_key(ranks["queries"][qi], pids[j]))
                p_yes = jr.get("p_yes") if jr else None
                lst.append(
                    {
                        "title": titles[j],
                        "p_yes": p_yes,
                        "junk": (p_yes is not None and p_yes < thr),
                        "bm25": pair_bm25["scores"].get(pair_key(ranks["queries"][qi], pids[j])),
                    }
                )
            scored = [x for x in lst if x["p_yes"] is not None]
            entry[n] = {
                "junk_rate": sum(1 for x in scored if x["junk"]) / len(scored)
                if scored
                else float("nan"),
                "top_10": lst,
            }
        watch[ranks["queries"][qi]] = entry

    diag = phase_diagnose(args, paths) if args.with_diagnose else None

    judge_meta = {}
    if Path(paths["judge_meta"]).exists():
        with open(paths["judge_meta"]) as f:
            judge_meta = json.load(f)

    print(f"\n=== {len(held)} holdout queries, k={K_EVAL} ===")
    hdr = f"{'variant':<10} {'junk@10':>9} {'R@10':>8} {'E@1':>8} {'nDCG':>8} {'chg':>6}"
    print(hdr)
    for n in names:
        s = summary[n]
        print(
            f"{n:<10} {s['junk_rate_at_10']:>9.4f} "
            f"{s['recall_at_10_fraction_recovered']:>8.4f} {s['e_at_1']:>8.4f} "
            f"{s['ndcg_at_10']:>8.4f} {s['mean_changed_slots_vs_dense']:>6.2f}"
        )
    print("\ndeltas vs dense (paired bootstrap CI95):")
    for n, e in deltas.items():
        for metric in ("junk_rate_at_10", "recall_at_10", "e_at_1"):
            d = e[metric]
            print(
                f"  {n:<8} {metric:<18} {d['delta']:+.4f} "
                f"[{d['ci95'][0]:+.4f}, {d['ci95'][1]:+.4f}]"
            )
    for q, entry in watch.items():
        print(f"\n--- {q!r} ---")
        for n in names:
            print(f"  [{n}] junk-rate {entry[n]['junk_rate']:.2f}")
            for x in entry[n]["top_10"][:5]:
                flag = "JUNK" if x["junk"] else "ok  "
                print(f"      {flag} bm25={(x['bm25'] or 0.0):>6.2f}  {x['title'][:70]}")

    out = {
        "experiment": "BestBuy: does a BM25 leg remove the BoD encoder's category junk?",
        "question": (
            "The BoD MiniLM's top-10 still contains 14.1% wrong-product-TYPE results "
            "(eval_bestbuy_llm_judge_junkrate.py). Does gating or fusing with BM25 over "
            "the same 1.27M-title catalog cut that, WITHOUT costing R@10 or E@1?"
        ),
        "config": {
            "dataset_repo": DATASET_REPO,
            "bod_model": args.bod_model,
            "sample_size": len(held),
            "seed": args.seed,
            "k_eval": K_EVAL,
            "depth": ranks["depth"],
            "gate_depth": args.gate_depth or ranks["depth"],
            "bm25": f"bm25s k1={pair_bm25['k1']} b={pair_bm25['b']}, {pair_bm25['tokenizer']}",
            "rrf_k": list(args.rrf_k),
            "judge_model": args.model,
            "judge_prompt": CATEGORY_PROMPT,
            "junk_threshold_p_yes": thr,
            "n_boot": args.n_boot,
            "label_reuse": (
                "labels from /tmp/bestbuy_junkrate reused for any (query, product) pair "
                "already judged there; only new candidates cost money"
            ),
        },
        "variants": {
            "dense": "BoD MiniLM cosine top-10 over the full catalog (baseline)",
            "bm25": "BM25 top-10 alone (reference leg, not a proposed system)",
            "gate": (
                "BoD dense top-{gd} stable-partitioned: nonzero-BM25 candidates first, "
                "zero-BM25 candidates after. No BM25-only documents introduced; a no-op "
                "when no query token is in the BM25 vocabulary."
            ).format(gd=args.gate_depth or ranks["depth"]),
            "gate_deep": f"same gate, promoting from BoD dense top-{ranks['depth']}",
            **{
                f"rrf{rk}": f"RRF(BoD-dense-top-{ranks['depth']}, BM25-top-{ranks['depth']}), "
                f"rrf_k={rk}"
                for rk in args.rrf_k
            },
        },
        "judge_run": judge_meta,
        "summary": summary,
        "deltas_vs_dense": deltas,
        "watch_queries": watch,
        "diagnosis": diag,
        "per_query": per_query,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=float)
    print(f"\nwrote {out_path}", flush=True)


# --------------------------------------------------------------------------
def main():
    global COST_CEILING_USD
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--phase", required=True, choices=["pool", "diagnose", "estimate", "judge", "eval"]
    )
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--queries-file", default="holdout_queries.jsonl")
    ap.add_argument("--qrels-file", default="holdout_qrels.jsonl")
    ap.add_argument("--bod-model", default="dtunkelang/bag-of-documents-bestbuy-minilm")
    ap.add_argument("--bod-vecs", default="bod_catalog.vecs.fp16.npy")
    ap.add_argument("--depth", type=int, default=200, help="candidate depth per retrieval leg")
    ap.add_argument(
        "--gate-depth",
        type=int,
        default=50,
        help="dense depth the BM25 gate may promote from (0 = --depth)",
    )
    ap.add_argument("--rrf-k", type=int, nargs="*", default=[10, 60])
    ap.add_argument("--k1", type=float, default=1.5)
    ap.add_argument("--b", type=float, default=0.75)
    ap.add_argument("--model", default="gpt-4o-mini", help="OpenAI judge model")
    ap.add_argument("--concurrency", type=int, default=32)
    ap.add_argument("--cost-ceiling", type=float, default=COST_CEILING_USD)
    ap.add_argument("--max-title-chars", type=int, default=300)
    ap.add_argument("--min-relevance", type=int, default=1)
    ap.add_argument("--exact-relevance", type=int, default=1)
    ap.add_argument("--junk-threshold", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--watch-queries", nargs="*", default=list(WATCH_QUERIES))
    ap.add_argument("--with-diagnose", action="store_true", default=True)
    ap.add_argument("--no-diagnose", dest="with_diagnose", action="store_false")
    ap.add_argument("--work-dir", default="/tmp/bestbuy_bm25_gate")
    ap.add_argument("--junk-work-dir", default="/tmp/bestbuy_junkrate")
    ap.add_argument("--tag", default="bestbuy")
    ap.add_argument("--junk-tag", default="bestbuy")
    ap.add_argument("--out", default="evaluation/results/bestbuy_bm25_junk_gate.json")
    args = ap.parse_args()

    COST_CEILING_USD = args.cost_ceiling

    paths = work_paths(args.work_dir, args.tag, args.junk_work_dir, args.junk_tag)
    {
        "pool": lambda: phase_pool(args, paths),
        "diagnose": lambda: phase_diagnose(args, paths),
        "estimate": lambda: phase_estimate(args, paths),
        "judge": lambda: phase_judge(args, paths),
        "eval": lambda: phase_eval(args, paths),
    }[args.phase]()


if __name__ == "__main__":
    main()
