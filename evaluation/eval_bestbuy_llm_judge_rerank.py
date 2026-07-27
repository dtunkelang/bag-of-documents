#!/usr/bin/env python3
"""LLM-as-judge reranking on the BestBuy BoD-retrieve candidate pool.

Question: on the *same* first-stage pool that Pattern 23 uses (BoD bi-encoder
dense retrieval over the full catalog, top-N), can a local LLM judge beat
`BAAI/bge-reranker-v2-m3` as the second-stage scorer?

Architecture (mirrors evaluation/eval_cc_cross_lingual.py architecture "C"):

    BoD-retrieve top-N over the full catalog        (first stage, unchanged)
        |
        +-- score with BoD cosine                   (stream 1)
        +-- score with BGE-reranker-v2-m3 CE        (stream 2, the incumbent)
        +-- score with a local LLM judge            (stream 3, the challenger)
        |
    per-query min-max normalize, fuse at w in {0.25, 0.50, 0.75}

The LLM judge is POINTWISE and *logit-based*: for each (query, title) pair we
run a single forward pass over the chat-formatted prompt and read the
next-token distribution, taking

    score = log p("yes") - log p("no")

over the yes/no token variants. This is the monoT5 / RankGPT-pointwise trick.
It costs one prefill per pair (no autoregressive decoding), it produces a
*continuous* score (no ties, no parse failures), and it is the cheapest
possible LLM-judge configuration -- which is the relevant thing to measure if
the question is "could an LLM judge ever be affordable to serve online?".

A LISTWISE variant (`--phase listwise`) asks the model to emit an ordering of
the pool in one generation call, as a second-pass comparison.

Phases are cached to --work-dir so they can be run under different venvs
(the retrieval/BGE phases need torch + sentence-transformers; the LLM phase
needs mlx-lm; the eval phase needs only numpy):

    --phase pool      encode sampled queries with BoD, build top-N pool
    --phase bge       cross-encoder scores over the pool
    --phase llm       pointwise LLM-judge scores over the pool
    --phase listwise  listwise LLM-judge ordering over the pool
    --phase eval      metrics + evaluation/results/*.json

Usage:
    python evaluation/eval_bestbuy_llm_judge_rerank.py --phase pool \\
        --data-dir <dir with titles.json/product_ids.json/holdout_*.jsonl> \\
        --bod-model <path or hub id> --bod-vecs bod_catalog.vecs.fp16.npy \\
        --sample 250 --top-n 100 --seed 0
    python evaluation/eval_bestbuy_llm_judge_rerank.py --phase bge  ...
    python evaluation/eval_bestbuy_llm_judge_rerank.py --phase llm  ...
    python evaluation/eval_bestbuy_llm_judge_rerank.py --phase eval ...
"""

import argparse
import json
import math
import os
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np  # noqa: E402

K_EVAL = 10

# Pointwise judge prompt. Deliberately terse: prefill length is the entire
# cost of this method, so every token in here is paid 1x per (query, doc).
POINTWISE_PROMPT = """Search query: {query}
Product: {title}

Is this product a relevant result for that search query? Answer yes or no."""

LISTWISE_PROMPT = """You are reranking search results for an online electronics store.

Search query: {query}

Candidate products:
{numbered}

List the numbers of the {k} most relevant products, best first, separated by \
commas. Output only the numbers."""


# --------------------------------------------------------------------------
# shared helpers (metric definitions match evaluation/eval_cc_cross_lingual.py)
# --------------------------------------------------------------------------
def per_query_metrics(retrieved_pids, qrels_q, k=K_EVAL, min_rel=1, exact_rel=1):
    """R@10 (fraction-recovered), nDCG@10, E@1, E@3 -- same defn as Pattern 23."""
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
    """Per-query min-max to [0,1] -- the blending helper Pattern 23 fuses with."""
    out = scores.copy()
    for qi in range(out.shape[0]):
        v = out[qi, valid_mask[qi]]
        if v.size == 0:
            continue
        lo, hi = float(v.min()), float(v.max())
        out[qi, valid_mask[qi]] = (v - lo) / max(hi - lo, 1e-8)
    return out


def load_corpus(data_dir):
    data = Path(data_dir).resolve()
    with open(data / "titles.json") as f:
        titles = json.load(f)
    with open(data / "product_ids.json") as f:
        pids = json.load(f)
    return data, titles, pids


def load_split(data, queries_file, qrels_file):
    qrels = defaultdict(dict)
    with open(data / qrels_file) as f:
        for line in f:
            r = json.loads(line)
            qrels[r["query_id"]][r["product_id"]] = r["relevance"]
    queries_all = {}
    with open(data / queries_file) as f:
        for line in f:
            d = json.loads(line)
            queries_all[d["query_id"]] = d["query"]
    return qrels, queries_all


def work_paths(work_dir, tag):
    w = Path(work_dir)
    w.mkdir(parents=True, exist_ok=True)
    return {
        "sample": w / f"sample_{tag}.json",
        "pool": w / f"pool_{tag}.npy",
        "bod": w / f"bod_scores_{tag}.npy",
        "bge": w / f"bge_scores_{tag}.npy",
        "llm": w / f"llm_scores_{tag}.npy",
        "llm_meta": w / f"llm_meta_{tag}.json",
        "listwise": w / f"listwise_scores_{tag}.npy",
        "listwise_meta": w / f"listwise_meta_{tag}.json",
    }


# --------------------------------------------------------------------------
# phase: pool
# --------------------------------------------------------------------------
def phase_pool(args, paths):
    import torch
    from sentence_transformers import SentenceTransformer

    data, titles, pids = load_corpus(args.data_dir)
    qrels, queries_all = load_split(data, args.queries_file, args.qrels_file)
    pid_set = set(pids)

    eval_qids = sorted(
        qid
        for qid, q in queries_all.items()
        if qid in qrels
        and any(g >= args.min_relevance and p in pid_set for p, g in qrels[qid].items())
    )
    print(f"  {len(eval_qids):,} eval-eligible queries in split", flush=True)

    rng = random.Random(args.seed)
    if args.sample and args.sample < len(eval_qids):
        sample_qids = sorted(rng.sample(eval_qids, args.sample))
        how = f"random.Random({args.seed}).sample over sorted eval-eligible qids"
    else:
        sample_qids = eval_qids
        how = "all eval-eligible queries"
    print(f"  sample: {len(sample_qids):,} queries ({how})", flush=True)

    queries = [queries_all[qid] for qid in sample_qids]

    device = (
        "mps"
        if torch.backends.mps.is_available()
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"encoding {len(queries):,} queries with {args.bod_model} on {device}...", flush=True)
    m = SentenceTransformer(args.bod_model, device=device)
    qv = m.encode(
        queries, normalize_embeddings=True, batch_size=64, show_progress_bar=False
    ).astype(np.float32)
    del m

    catalog = np.load(data / args.bod_vecs, mmap_mode="r")
    n_docs = catalog.shape[0]
    print(f"  catalog {catalog.shape}; scoring in chunks...", flush=True)

    n_q = len(queries)
    top_n = args.top_n
    best_scores = np.full((n_q, top_n), -np.inf, dtype=np.float32)
    best_idx = np.full((n_q, top_n), -1, dtype=np.int64)
    chunk = 100_000
    t0 = time.time()
    for start in range(0, n_docs, chunk):
        end = min(start + chunk, n_docs)
        block = np.asarray(catalog[start:end]).astype(np.float32)
        sims = qv @ block.T  # (n_q, block)
        m_block = min(top_n, sims.shape[1])
        part = np.argpartition(-sims, m_block - 1, axis=1)[:, :m_block]
        part_scores = np.take_along_axis(sims, part, axis=1)
        cand_scores = np.concatenate([best_scores, part_scores], axis=1)
        cand_idx = np.concatenate([best_idx, part + start], axis=1)
        keep = np.argpartition(-cand_scores, top_n - 1, axis=1)[:, :top_n]
        best_scores = np.take_along_axis(cand_scores, keep, axis=1)
        best_idx = np.take_along_axis(cand_idx, keep, axis=1)
        del sims, block
    order = np.argsort(-best_scores, axis=1)
    best_scores = np.take_along_axis(best_scores, order, axis=1)
    best_idx = np.take_along_axis(best_idx, order, axis=1)
    print(f"  retrieval done in {time.time() - t0:.0f}s", flush=True)

    np.save(paths["pool"], best_idx)
    np.save(paths["bod"], best_scores)
    with open(paths["sample"], "w") as f:
        json.dump(
            {
                "qids": sample_qids,
                "queries": queries,
                "seed": args.seed,
                "sample_size": len(sample_qids),
                "n_eval_eligible": len(eval_qids),
                "selection": how,
                "top_n": top_n,
                "catalog_size": int(n_docs),
                "bod_model": args.bod_model,
                "data_dir": str(data),
            },
            f,
            indent=2,
        )
    # pool ceiling
    pids_arr = np.asarray(pids)
    ceil = []
    for i, qid in enumerate(sample_qids):
        gold = {p for p, g in qrels[qid].items() if g >= args.min_relevance}
        got = {pids_arr[j] for j in best_idx[i] if j >= 0}
        ceil.append(len(gold & got) / len(gold))
    print(f"  pool recall@{top_n} ceiling: {np.mean(ceil):.4f}", flush=True)
    print(f"saved pool -> {paths['pool']}", flush=True)


# --------------------------------------------------------------------------
# phase: bge
# --------------------------------------------------------------------------
def phase_bge(args, paths):
    import torch
    from sentence_transformers import CrossEncoder

    data, titles, _pids = load_corpus(args.data_dir)
    with open(paths["sample"]) as f:
        meta = json.load(f)
    queries = meta["queries"]
    pool = np.load(paths["pool"])
    n_q, top_n = pool.shape

    device = (
        "mps"
        if torch.backends.mps.is_available()
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"loading CE {args.bge_model} on {device}...", flush=True)
    ce = CrossEncoder(args.bge_model, device=device)

    scores = np.full((n_q, top_n), np.nan, dtype=np.float32)
    t0 = time.time()
    for qi in range(n_q):
        idxs = pool[qi]
        pairs = [(queries[qi], titles[int(j)]) for j in idxs if j >= 0]
        sc = ce.predict(pairs, batch_size=args.bge_batch_size, show_progress_bar=False)
        k = 0
        for j_pos, j in enumerate(idxs):
            if j >= 0:
                scores[qi, j_pos] = float(sc[k])
                k += 1
        if (qi + 1) % 25 == 0:
            el = time.time() - t0
            print(
                f"  [{qi + 1}/{n_q}] {el / (qi + 1):.2f}s/query  "
                f"eta {(n_q - qi - 1) * el / (qi + 1) / 60:.1f}m",
                flush=True,
            )
            np.save(paths["bge"], scores)
    np.save(paths["bge"], scores)
    print(f"BGE done in {time.time() - t0:.0f}s -> {paths['bge']}", flush=True)


# --------------------------------------------------------------------------
# phase: llm (pointwise, logit-based)
# --------------------------------------------------------------------------
def _yes_no_token_ids(tok):
    yes, no = [], []
    for s, bucket in (("yes", yes), ("Yes", yes), ("YES", yes), ("no", no), ("No", no), ("NO", no)):
        for variant in (s, " " + s):
            ids = tok.encode(variant, add_special_tokens=False)
            if len(ids) == 1:
                bucket.append(ids[0])
    return sorted(set(yes)), sorted(set(no))


def last_token_logits(model, ids, lens):
    """Vocab logits at the final real token of each right-padded row.

    Materializing (B, T, V) logits for a 150K-token vocab is ~1GB per batch, so
    run the transformer body, slice the last real hidden state per row, and
    apply the output head to just those B vectors.
    """
    import mlx.core as mx

    rows = mx.arange(len(lens))
    cols = mx.array([le - 1 for le in lens])
    try:
        h = model.model(ids)  # (B, T, H)
        last_h = h[rows, cols]  # (B, H)
        head = getattr(model, "lm_head", None)
        if head is not None:
            return head(last_h)
        return model.model.embed_tokens.as_linear(last_h)
    except (AttributeError, TypeError):  # pragma: no cover - arch fallback
        logits = model(ids)
        return logits[rows, cols]


def phase_llm(args, paths):
    import mlx.core as mx
    import mlx.nn as mlx_nn
    from mlx_lm import load

    data, titles, _pids = load_corpus(args.data_dir)
    with open(paths["sample"]) as f:
        meta = json.load(f)
    queries = meta["queries"]
    pool = np.load(paths["pool"])
    n_q, top_n = pool.shape
    depth = min(args.judge_depth, top_n)

    print(f"loading MLX judge {args.judge_model}...", flush=True)
    t_load = time.time()
    model, tok = load(args.judge_model)
    print(f"  loaded in {time.time() - t_load:.0f}s", flush=True)

    yes_ids, no_ids = _yes_no_token_ids(tok)
    if not yes_ids or not no_ids:
        raise SystemExit(f"could not resolve yes/no token ids: {yes_ids} {no_ids}")
    print(f"  yes ids={yes_ids} no ids={no_ids}", flush=True)

    scores = np.full((n_q, top_n), np.nan, dtype=np.float32)
    if args.resume and Path(paths["llm"]).exists():
        prev = np.load(paths["llm"])
        if prev.shape == scores.shape:
            scores = prev
            print("  resuming from cached scores", flush=True)

    def build_ids(q, title):
        prompt = POINTWISE_PROMPT.format(query=q, title=title[: args.max_title_chars])
        return tok.apply_chat_template(
            [{"role": "user", "content": prompt}], add_generation_prompt=True
        )

    t0 = time.time()
    total_pairs = 0
    total_tokens = 0
    for qi in range(n_q):
        if args.resume and not np.isnan(scores[qi, :depth]).any():
            continue
        idxs = pool[qi][:depth]
        prompts = [build_ids(queries[qi], titles[int(j)]) for j in idxs if j >= 0]
        positions = [jp for jp, j in enumerate(idxs) if j >= 0]
        out = []
        for bs in range(0, len(prompts), args.judge_batch_size):
            batch = prompts[bs : bs + args.judge_batch_size]
            lens = [len(p) for p in batch]
            maxlen = max(lens)
            # RIGHT padding is safe under a causal mask: padded positions come
            # after every real token, so they cannot influence real positions.
            arr = np.zeros((len(batch), maxlen), dtype=np.int32)
            for r, p in enumerate(batch):
                arr[r, : len(p)] = p
            last = last_token_logits(model, mx.array(arr), lens)
            lp = mlx_nn.log_softmax(last.astype(mx.float32), axis=-1)
            y = mx.logsumexp(lp[:, mx.array(yes_ids)], axis=-1)
            n = mx.logsumexp(lp[:, mx.array(no_ids)], axis=-1)
            mx.eval(y, n)
            out.extend((np.array(y) - np.array(n)).tolist())
            total_tokens += sum(lens)
        for jp, s in zip(positions, out):
            scores[qi, jp] = float(s)
        total_pairs += len(prompts)
        if (qi + 1) % 10 == 0 or qi == 0:
            el = time.time() - t0
            print(
                f"  [{qi + 1}/{n_q}] {el / (qi + 1):.2f}s/query  "
                f"{total_pairs / el:.1f} pairs/s  {total_tokens / el:.0f} tok/s  "
                f"eta {(n_q - qi - 1) * el / (qi + 1) / 60:.1f}m",
                flush=True,
            )
            np.save(paths["llm"], scores)
    elapsed = time.time() - t0
    np.save(paths["llm"], scores)
    judged = scores[:, :depth]
    finite = judged[np.isfinite(judged)]
    stats = {
        "judge_model": args.judge_model,
        "mode": "pointwise-logit",
        "judge_depth": depth,
        "n_queries": n_q,
        "n_pairs": int(total_pairs),
        "wall_clock_s": elapsed,
        "s_per_query": elapsed / max(n_q, 1),
        "pairs_per_s": total_pairs / max(elapsed, 1e-9),
        "prompt_tokens": int(total_tokens),
        "prompt_tokens_per_s": total_tokens / max(elapsed, 1e-9),
        "score_mean": float(finite.mean()) if finite.size else None,
        "score_std": float(finite.std()) if finite.size else None,
        "score_p05": float(np.percentile(finite, 5)) if finite.size else None,
        "score_p95": float(np.percentile(finite, 95)) if finite.size else None,
        "frac_yes": float((finite > 0).mean()) if finite.size else None,
        "n_nan": int(np.isnan(judged).sum()),
        "mean_distinct_scores_per_query": float(
            np.mean([len(np.unique(np.round(r[np.isfinite(r)], 4))) for r in judged])
        ),
    }
    with open(paths["llm_meta"], "w") as f:
        json.dump(stats, f, indent=2)
    print(json.dumps(stats, indent=2), flush=True)


# --------------------------------------------------------------------------
# phase: listwise
# --------------------------------------------------------------------------
def phase_listwise(args, paths):
    import re

    from mlx_lm import generate, load
    from mlx_lm.sample_utils import make_sampler

    data, titles, _pids = load_corpus(args.data_dir)
    with open(paths["sample"]) as f:
        meta = json.load(f)
    queries = meta["queries"]
    pool = np.load(paths["pool"])
    n_q, top_n = pool.shape
    depth = min(args.listwise_depth, top_n)

    print(f"loading MLX judge {args.judge_model}...", flush=True)
    model, tok = load(args.judge_model)
    sampler = make_sampler(temp=0.0)

    scores = np.full((n_q, top_n), np.nan, dtype=np.float32)
    n_parse_fail = 0
    n_partial = 0
    t0 = time.time()
    for qi in range(n_q):
        idxs = pool[qi][:depth]
        valid = [(jp, int(j)) for jp, j in enumerate(idxs) if j >= 0]
        numbered = "\n".join(
            f"{n + 1}. {titles[j][: args.max_title_chars]}" for n, (_jp, j) in enumerate(valid)
        )
        prompt = LISTWISE_PROMPT.format(query=queries[qi], numbered=numbered, k=K_EVAL)
        text = tok.apply_chat_template(
            [{"role": "user", "content": prompt}], add_generation_prompt=True, tokenize=False
        )
        resp = generate(
            model, tok, prompt=text, max_tokens=args.listwise_max_tokens, sampler=sampler
        )
        nums, seen = [], set()
        for tokn in re.findall(r"\d+", resp):
            v = int(tokn)
            if 1 <= v <= len(valid) and v not in seen:
                seen.add(v)
                nums.append(v)
        if not nums:
            n_parse_fail += 1
        elif len(nums) < K_EVAL:
            n_partial += 1
        # rank -> score; anything the model omitted keeps its BoD pool order
        # below every ranked item.
        base = float(len(valid) + K_EVAL)
        for rank, v in enumerate(nums):
            jp = valid[v - 1][0]
            scores[qi, jp] = base - rank
        for jp, _j in valid:
            if np.isnan(scores[qi, jp]):
                scores[qi, jp] = float(len(valid) - jp)
        if (qi + 1) % 10 == 0 or qi == 0:
            el = time.time() - t0
            print(
                f"  [{qi + 1}/{n_q}] {el / (qi + 1):.2f}s/query  "
                f"eta {(n_q - qi - 1) * el / (qi + 1) / 60:.1f}m",
                flush=True,
            )
            np.save(paths["listwise"], scores)
    elapsed = time.time() - t0
    np.save(paths["listwise"], scores)
    stats = {
        "judge_model": args.judge_model,
        "mode": "listwise",
        "listwise_depth": depth,
        "n_queries": n_q,
        "wall_clock_s": elapsed,
        "s_per_query": elapsed / max(n_q, 1),
        "n_parse_failures": n_parse_fail,
        "n_partial_lists": n_partial,
    }
    with open(paths["listwise_meta"], "w") as f:
        json.dump(stats, f, indent=2)
    print(json.dumps(stats, indent=2), flush=True)


# --------------------------------------------------------------------------
# phase: eval
# --------------------------------------------------------------------------
def eval_setups(sample_qids, qrels, pool, pids_arr, score_matrices, valid, min_rel, exact_rel):
    results = {}
    per_query = {}
    for label, mat in score_matrices.items():
        rs, ns, e1s, e3s = [], [], [], []
        pq = []
        for qi, qid in enumerate(sample_qids):
            s = mat[qi].copy()
            s[~valid[qi]] = -np.inf
            order = np.argsort(-s)[:K_EVAL]
            ordering = [pids_arr[int(pool[qi, j])] for j in order if pool[qi, j] >= 0]
            m = per_query_metrics(ordering, qrels[qid], min_rel=min_rel, exact_rel=exact_rel)
            if m is None:
                pq.append(None)
                continue
            r, nd, e1, e3 = m
            rs.append(r)
            ns.append(nd)
            pq.append(r)
            if not math.isnan(e1):
                e1s.append(e1)
                e3s.append(e3)
        results[label] = {
            "r10": float(np.mean(rs)) if rs else 0.0,
            "ndcg10": float(np.mean(ns)) if ns else 0.0,
            "e1": float(np.mean(e1s)) if e1s else 0.0,
            "e3": float(np.mean(e3s)) if e3s else 0.0,
            "n": len(rs),
        }
        per_query[label] = pq
        o = results[label]
        print(
            f"  {label:<44s} | R@10 {o['r10']:.4f}  nDCG@10 {o['ndcg10']:.4f}  "
            f"E@1 {o['e1']:.4f}  E@3 {o['e3']:.4f}  (n={o['n']})",
            flush=True,
        )
    return results, per_query


def phase_eval(args, paths):
    data, titles, pids = load_corpus(args.data_dir)
    qrels, _queries_all = load_split(data, args.queries_file, args.qrels_file)
    with open(paths["sample"]) as f:
        meta = json.load(f)
    sample_qids = meta["qids"]
    pool = np.load(paths["pool"])
    bod_raw = np.load(paths["bod"])
    valid = pool >= 0
    pids_arr = np.asarray(pids)
    n_q, top_n = pool.shape

    nm_bod = normalize_per_query(np.nan_to_num(bod_raw, nan=0.0), valid)
    score_matrices = {
        "BoD-retrieve order (no rerank)": np.where(
            valid, -np.arange(top_n)[None, :].astype(np.float32), -np.inf
        ),
        "BoD alone (cosine rescore)": np.where(valid, nm_bod, -np.inf),
    }

    # The LLM judge is far more expensive per pair than the CE, so it may have
    # been run to a shallower depth. Read that depth back and give the CE a
    # depth-matched twin so the head-to-head is on identical candidate sets.
    judge_depth = top_n
    if Path(paths["llm_meta"]).exists():
        with open(paths["llm_meta"]) as f:
            judge_depth = int(json.load(f).get("judge_depth", top_n))
    depth_mask = np.zeros_like(valid)
    depth_mask[:, :judge_depth] = True

    have_bge = Path(paths["bge"]).exists()
    if have_bge:
        bge_raw = np.load(paths["bge"])
        bge_valid = valid & ~np.isnan(bge_raw)
        nm_bge = normalize_per_query(np.nan_to_num(bge_raw, nan=0.0), bge_valid)
        score_matrices["BGE-CE alone"] = np.where(valid, nm_bge, -np.inf)
        for w in (0.25, 0.50, 0.75):
            score_matrices[f"BoD + BGE-CE w={w:.2f}"] = np.where(
                valid, (1 - w) * nm_bod + w * nm_bge, -np.inf
            )
        if judge_depth < top_n:
            bge_d = bge_valid & depth_mask
            nm_bge_d = normalize_per_query(np.nan_to_num(bge_raw, nan=0.0), bge_d)
            score_matrices[f"BGE-CE alone (top-{judge_depth} only)"] = np.where(
                bge_d, nm_bge_d, -np.inf
            )
            for w in (0.25, 0.50, 0.75):
                score_matrices[f"BoD + BGE-CE w={w:.2f} (top-{judge_depth} only)"] = np.where(
                    valid, (1 - w) * nm_bod + w * np.where(bge_d, nm_bge_d, 0.0), -np.inf
                )

    have_llm = Path(paths["llm"]).exists()
    if have_llm:
        llm_raw = np.load(paths["llm"])
        llm_valid = valid & ~np.isnan(llm_raw)
        nm_llm = normalize_per_query(np.nan_to_num(llm_raw, nan=0.0), llm_valid)
        # candidates the judge never saw (beyond --judge-depth) sink to the
        # bottom of the judge-only ordering but keep BoD order among themselves
        judge_only = np.where(llm_valid, nm_llm, -np.inf)
        score_matrices["LLM-judge alone (pointwise)"] = judge_only
        for w in (0.25, 0.50, 0.75):
            fused = (1 - w) * nm_bod + w * np.where(llm_valid, nm_llm, 0.0)
            score_matrices[f"BoD + LLM-judge w={w:.2f}"] = np.where(valid, fused, -np.inf)
        if have_bge:
            score_matrices["BoD + BGE 0.25 + LLM 0.25"] = np.where(
                valid,
                0.50 * nm_bod + 0.25 * nm_bge + 0.25 * np.where(llm_valid, nm_llm, 0.0),
                -np.inf,
            )

    have_lw = Path(paths["listwise"]).exists()
    if have_lw:
        lw_raw = np.load(paths["listwise"])
        lw_valid = valid & ~np.isnan(lw_raw)
        nm_lw = normalize_per_query(np.nan_to_num(lw_raw, nan=0.0), lw_valid)
        score_matrices["LLM-judge alone (listwise)"] = np.where(lw_valid, nm_lw, -np.inf)
        for w in (0.25, 0.50, 0.75):
            fused = (1 - w) * nm_bod + w * np.where(lw_valid, nm_lw, 0.0)
            score_matrices[f"BoD + LLM-listwise w={w:.2f}"] = np.where(valid, fused, -np.inf)

    print(f"\neval over {n_q:,} sampled queries (BoD-retrieve top-{top_n} pool):", flush=True)
    results, per_query = eval_setups(
        sample_qids,
        qrels,
        pool,
        pids_arr,
        score_matrices,
        valid,
        args.min_relevance,
        args.exact_relevance,
    )

    # pool ceiling
    ceil = []
    for i, qid in enumerate(sample_qids):
        gold = {p for p, g in qrels[qid].items() if g >= args.min_relevance}
        got = {pids_arr[int(j)] for j in pool[i] if j >= 0}
        ceil.append(len(gold & got) / len(gold) if gold else 0.0)
    pool_ceiling = float(np.mean(ceil))
    print(f"\n  pool recall@{top_n} (rerank ceiling): {pool_ceiling:.4f}", flush=True)

    # head-to-head vs the incumbent
    h2h = {}
    inc = (
        f"BoD + BGE-CE w=0.25 (top-{judge_depth} only)"
        if f"BoD + BGE-CE w=0.25 (top-{judge_depth} only)" in per_query
        else "BoD + BGE-CE w=0.25"
    )
    for label in ("BoD + LLM-judge w=0.25", "BoD + LLM-listwise w=0.25"):
        if label in per_query and inc in per_query:
            a, b = per_query[label], per_query[inc]
            pairs = [(x, y) for x, y in zip(a, b) if x is not None and y is not None]
            h2h[f"{label} vs {inc}"] = {
                "wins": sum(1 for x, y in pairs if x > y),
                "losses": sum(1 for x, y in pairs if x < y),
                "ties": sum(1 for x, y in pairs if x == y),
            }
    for k, v in h2h.items():
        print(f"  {k}: +{v['wins']} / -{v['losses']} / ={v['ties']}", flush=True)

    # Paired bootstrap on R@10 deltas -- at n=250 a 1pp delta is inside the
    # noise band, so print the band rather than let point estimates mislead.
    base_label = "BoD-retrieve order (no rerank)"
    rng = np.random.default_rng(args.seed)
    boot = {}
    base_pq = np.array([x for x in per_query[base_label] if x is not None], dtype=np.float64)
    for label, pq in per_query.items():
        if label == base_label:
            continue
        arr = np.array([x for x in pq if x is not None], dtype=np.float64)
        if arr.shape != base_pq.shape:
            continue
        d = arr - base_pq
        idx = rng.integers(0, len(d), size=(2000, len(d)))
        means = d[idx].mean(axis=1)
        boot[label] = {
            "delta_r10_vs_no_rerank": float(d.mean()),
            "ci95_low": float(np.percentile(means, 2.5)),
            "ci95_high": float(np.percentile(means, 97.5)),
        }
    print("\n  paired-bootstrap 95% CI on ΔR@10 vs no-rerank (2000 resamples):", flush=True)
    for label, b in boot.items():
        print(
            f"    {label:<44s} {b['delta_r10_vs_no_rerank']:+.4f} "
            f"[{b['ci95_low']:+.4f}, {b['ci95_high']:+.4f}]",
            flush=True,
        )

    llm_meta = {}
    for key in ("llm_meta", "listwise_meta"):
        if Path(paths[key]).exists():
            with open(paths[key]) as f:
                llm_meta[key] = json.load(f)

    out = {
        "data_dir": str(data),
        "corpus": args.corpus_name,
        "catalog_size": meta["catalog_size"],
        "bod_model": meta["bod_model"],
        "bge_model": args.bge_model if have_bge else None,
        "judge_model": args.judge_model,
        "top_n": top_n,
        "judge_depth": judge_depth,
        "n_queries": n_q,
        "n_eval_eligible": meta["n_eval_eligible"],
        "sample_seed": meta["seed"],
        "sample_selection": meta["selection"],
        "min_relevance": args.min_relevance,
        "exact_relevance": args.exact_relevance,
        "pool_recall_at_top_n": pool_ceiling,
        "judge_runtime": llm_meta,
        "head_to_head": h2h,
        "bootstrap_r10_vs_no_rerank": boot,
        "results": results,
    }
    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    with open(outp, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nsaved -> {outp}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", required=True, choices=["pool", "bge", "llm", "listwise", "eval"])
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--queries-file", default="holdout_queries.jsonl")
    ap.add_argument("--qrels-file", default="holdout_qrels.jsonl")
    ap.add_argument("--bod-model", default="dtunkelang/bag-of-documents-bestbuy-minilm")
    ap.add_argument("--bod-vecs", default="bod_catalog.vecs.fp16.npy")
    ap.add_argument("--bge-model", default="BAAI/bge-reranker-v2-m3")
    ap.add_argument("--judge-model", default="mlx-community/Qwen2.5-7B-Instruct-4bit")
    ap.add_argument("--top-n", type=int, default=100, help="first-stage pool depth")
    ap.add_argument("--judge-depth", type=int, default=100, help="how deep the LLM judge scores")
    ap.add_argument("--listwise-depth", type=int, default=20)
    ap.add_argument("--listwise-max-tokens", type=int, default=80)
    ap.add_argument("--judge-batch-size", type=int, default=16)
    ap.add_argument("--bge-batch-size", type=int, default=32)
    ap.add_argument("--max-title-chars", type=int, default=160)
    ap.add_argument("--sample", type=int, default=250)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min-relevance", type=int, default=1)
    ap.add_argument("--exact-relevance", type=int, default=1)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--work-dir", default="/tmp/bestbuy_llm_judge_eval")
    ap.add_argument("--tag", default="bestbuy")
    ap.add_argument(
        "--corpus-name",
        default="bestbuy_hf_full_catalog",
        help="label written into the results JSON (the data-dir is an HF snapshot hash)",
    )
    ap.add_argument(
        "--out",
        default=str(Path(__file__).resolve().parent / "results" / "bestbuy_llm_judge_rerank.json"),
    )
    args = ap.parse_args()

    paths = work_paths(args.work_dir, args.tag)
    print(f"phase={args.phase} work_dir={args.work_dir} tag={args.tag}", flush=True)
    {
        "pool": phase_pool,
        "bge": phase_bge,
        "llm": phase_llm,
        "listwise": phase_listwise,
        "eval": phase_eval,
    }[args.phase](args, paths)


if __name__ == "__main__":
    main()
