#!/usr/bin/env python3
"""Category-error ("junk") rate in the BestBuy top-10: base MiniLM vs BoD MiniLM.

Motivating anecdote: the query "apple tablet" returns keyboards in the demo's
top-4. That is a *coarse* failure -- wrong product TYPE -- which is a different
question from the one Pattern 27 answered. Pattern 27 asked whether an LLM
judge could rerank better than BGE-CE against BestBuy's click qrels, and hit a
structural ceiling: those qrels average ~2 golds/query among near-identical
SKU colorway variants, so R@10/E@1 only ever credit an exact SKU match and are
close to blind to "is this even the right kind of product?".

So this script measures something Pattern 27's metrics cannot surface:

    junk-rate = mean over queries of
                (# of the model's top-10 that an LLM judge calls a
                 wrong-CATEGORY result) / 10

The judge is deliberately *not* asked "would this be clicked" -- it is asked
whether the product is a plausible TYPE of thing for the query intent, with an
explicit instruction to say no for accessories-when-the-main-product-was-asked-
for and for brand/keyword coattails. Cheap pointwise yes/no over the OpenAI
API, scored from the first-token logprobs (no parse failures, continuous
margin available for sensitivity checks).

This is STEP 1: measurement only. No reranking, no filtering, no intervention.
R@10 / E@1 on the same 250-query sample are computed alongside purely as a
cross-check on whether junk-rate and click-accuracy move together.

Cost control: a (query, product) category judgment does not depend on which
model retrieved the product, so the two models' top-10 lists are DEDUPED and
each unique pair is judged once, then the score is reused for both models'
junk-rate. On this sample that is ~a 25% saving over judging 250 x 20 pairs.

Phases (cached to --work-dir, resumable):

    --phase pool      download artifacts, sample queries, base+BoD top-10
    --phase estimate  project OpenAI cost -- spends nothing
    --phase judge     category yes/no over the deduped (query, product) pairs
    --phase eval      junk-rate + R@10/E@1 cross-check + examples -> results JSON

Usage:
    python evaluation/eval_bestbuy_llm_judge_junkrate.py --phase pool
    python evaluation/eval_bestbuy_llm_judge_junkrate.py --phase estimate
    python evaluation/eval_bestbuy_llm_judge_junkrate.py --phase judge
    python evaluation/eval_bestbuy_llm_judge_junkrate.py --phase eval
"""

import argparse
import asyncio
import datetime
import json
import math
import os
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np  # noqa: E402
from dotenv import load_dotenv  # noqa: E402

from evaluation.eval_bestbuy_llm_judge_rerank import (  # noqa: E402
    load_corpus,
    load_split,
    per_query_metrics,
)

load_dotenv(override=True)

K_EVAL = 10

DATASET_REPO = "dtunkelang/bag-of-documents-bestbuy"
DATASET_FILES = [
    "titles.json",
    "product_ids.json",
    "holdout_queries.jsonl",
    "holdout_qrels.jsonl",
    "base_catalog.vecs.fp16.npy",
    "bod_catalog.vecs.fp16.npy",
]

# $ per 1M tokens. Mirrors evaluation/eval_esci_llm_judge_lexical_bias.py.
PRICES_PER_M_TOKENS = {
    "gpt-4o-mini": {"in": 0.15, "out": 0.60},
    "gpt-4o": {"in": 2.50, "out": 10.00},
    "gpt-4.1-mini": {"in": 0.40, "out": 1.60},
    "gpt-4.1": {"in": 2.00, "out": 8.00},
}
SPEND_LEDGER = Path(__file__).resolve().parent.parent / ".api_spend.jsonl"

# Hard stop: refuse to start a paid phase whose projection exceeds this.
COST_CEILING_USD = 3.0

# The whole point of this eval is in this prompt. It must ask about product
# TYPE, not about purchase intent, or it degenerates into Pattern 27's
# unanswerable "which colorway was clicked" question.
CATEGORY_PROMPT = """Search query: {query}
Product: {title}

Ignore whether this is the exact item someone would buy - judge only whether \
this product is a plausible TYPE of thing for what someone typing this search \
is looking for. Answer "no" if it's a different product category (e.g. an \
accessory when the query names the main product, or an unrelated product \
type), even if it shares a brand name or keyword with the query. Answer yes \
or no."""

# Queries checked by hand regardless of whether they land in the random
# sample. These are the user-reported failures the eval has to reproduce.
DEFAULT_MANUAL_QUERIES = ("apple tablet", "nokia phone")


# --------------------------------------------------------------------------
# OpenAI plumbing (auth + cost pattern from eval_esci_llm_judge_lexical_bias.py)
# --------------------------------------------------------------------------
def estimate_cost(model: str, tokens_in: int, tokens_out: int) -> float:
    p = PRICES_PER_M_TOKENS.get(model)
    if not p:
        return 0.0
    return (tokens_in * p["in"] + tokens_out * p["out"]) / 1_000_000.0


def record_spend(model, tokens_in, tokens_out, cost_usd, purpose):
    rec = {
        "ts": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "provider": "openai",
        "model": model,
        "tokens": int(tokens_in + tokens_out),
        "tokens_in": int(tokens_in),
        "tokens_out": int(tokens_out),
        "cost_usd": round(float(cost_usd), 6),
        "purpose": purpose,
    }
    with open(SPEND_LEDGER, "a") as f:
        f.write(json.dumps(rec) + "\n")
    return rec


def make_client():
    from openai import AsyncOpenAI

    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        raise SystemExit("OPENAI_API_KEY not set (expected in .env)")
    return AsyncOpenAI(api_key=key)


class Usage:
    """Mutable token accumulator shared across coroutines."""

    def __init__(self):
        self.tin = 0
        self.tout = 0
        self.calls = 0
        self.errors = 0


async def _chat(client, sem, usage, model, prompt, max_tokens, logprobs, max_retries=6):
    """One chat completion with exponential backoff. Returns the choice or None."""
    backoff = 1.0
    async with sem:
        for _ in range(max_retries):
            try:
                kw = dict(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=max_tokens,
                    temperature=0.0,
                )
                if logprobs:
                    kw["logprobs"] = True
                    kw["top_logprobs"] = 20
                resp = await client.chat.completions.create(**kw)
                u = resp.usage
                usage.tin += int(u.prompt_tokens or 0)
                usage.tout += int(u.completion_tokens or 0)
                usage.calls += 1
                return resp.choices[0]
            except Exception:  # rate limit / transient API error
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2.0, 30.0)
        usage.errors += 1
        return None


def _score_from_logprobs(choice):
    """(yes/no margin, p_yes) from the first-token distribution.

    margin = log p(yes) - log p(no); p_yes = p(yes) / (p(yes) + p(no)).
    Mass for a target token missing from the top-20 is floored at the smallest
    observed top-20 probability -- a valid upper bound on it.
    """
    if choice is None or not choice.logprobs or not choice.logprobs.content:
        return float("nan"), float("nan")
    top = choice.logprobs.content[0].top_logprobs
    if not top:
        return float("nan"), float("nan")
    floor = math.exp(min(x.logprob for x in top))
    py = sum(math.exp(x.logprob) for x in top if x.token.strip().lower() == "yes")
    pn = sum(math.exp(x.logprob) for x in top if x.token.strip().lower() == "no")
    py = py if py > 0 else floor
    pn = pn if pn > 0 else floor
    return math.log(py) - math.log(pn), py / (py + pn)


# --------------------------------------------------------------------------
# paths / data
# --------------------------------------------------------------------------
def work_paths(work_dir, tag):
    w = Path(work_dir)
    w.mkdir(parents=True, exist_ok=True)
    return {
        "pool": w / f"junk_pool_{tag}.json",
        "judge": w / f"junk_judge_{tag}.jsonl",
        "judge_meta": w / f"junk_judge_meta_{tag}.json",
    }


def resolve_data_dir(args):
    """Local --data-dir if given, else snapshot_download the demo dataset."""
    if args.data_dir:
        return Path(args.data_dir).resolve()
    from huggingface_hub import snapshot_download

    print(f"snapshot_download from {DATASET_REPO} (~1.9GB, cached after first run)...", flush=True)
    return Path(
        snapshot_download(
            repo_id=DATASET_REPO,
            repo_type="dataset",
            allow_patterns=DATASET_FILES,
        )
    ).resolve()


def pair_key(query, pid):
    return f"{query}\t{pid}"


# --------------------------------------------------------------------------
# phase: pool
# --------------------------------------------------------------------------
def _topk_over_catalog(qv, catalog_path, k, chunk=100_000):
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


def phase_pool(args, paths):
    import torch
    from sentence_transformers import SentenceTransformer

    data = resolve_data_dir(args)
    data, titles, pids = load_corpus(data)
    qrels, queries_all = load_split(data, args.queries_file, args.qrels_file)
    pid_set = set(pids)

    eval_qids = sorted(
        qid
        for qid, q in queries_all.items()
        if qid in qrels
        and any(g >= args.min_relevance and p in pid_set for p, g in qrels[qid].items())
    )
    print(f"  {len(eval_qids):,} eval-eligible holdout queries", flush=True)

    rng = random.Random(args.seed)
    if args.sample and args.sample < len(eval_qids):
        sample_qids = sorted(rng.sample(eval_qids, args.sample))
        how = f"random.Random({args.seed}).sample over sorted eval-eligible qids"
    else:
        sample_qids = eval_qids
        how = "all eval-eligible queries"
    print(f"  sample: {len(sample_qids):,} queries ({how})", flush=True)

    sampled_texts = [queries_all[qid] for qid in sample_qids]
    lowered = {q.strip().lower() for q in sampled_texts}
    manual = [q for q in args.manual_queries if q.strip().lower() not in lowered]
    if manual:
        print(f"  + {len(manual)} manual check queries not in sample: {manual}", flush=True)

    rows = [{"key": qid, "query": queries_all[qid], "is_manual": False} for qid in sample_qids] + [
        {"key": f"manual:{q}", "query": q, "is_manual": True} for q in manual
    ]
    queries = [r["query"] for r in rows]

    device = (
        "mps"
        if torch.backends.mps.is_available()
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    for model_key, model_id, vecs_file in (
        ("base", args.base_model, args.base_vecs),
        ("bod", args.bod_model, args.bod_vecs),
    ):
        print(f"  encoding {len(queries)} queries with {model_id} on {device}...", flush=True)
        m = SentenceTransformer(model_id, device=device)
        qv = m.encode(
            queries, normalize_embeddings=True, batch_size=64, show_progress_bar=False
        ).astype(np.float32)
        del m
        idx, sims = _topk_over_catalog(qv, data / vecs_file, args.top_k)
        for i, r in enumerate(rows):
            r[model_key] = [
                {
                    "rank": rank + 1,
                    "product_id": pids[int(j)],
                    "title": titles[int(j)],
                    "sim": float(sims[i, rank]),
                }
                for rank, j in enumerate(idx[i])
                if j >= 0
            ]

    payload = {
        "dataset_repo": DATASET_REPO if not args.data_dir else str(data),
        "data_dir": str(data),
        "base_model": args.base_model,
        "bod_model": args.bod_model,
        "seed": args.seed,
        "sample_size": len(sample_qids),
        "n_eval_eligible": len(eval_qids),
        "selection": how,
        "top_k": args.top_k,
        "catalog_size": len(pids),
        "manual_queries": manual,
        "rows": rows,
    }
    with open(paths["pool"], "w") as f:
        json.dump(payload, f, indent=2)

    pairs = {
        pair_key(r["query"], d["product_id"]) for r in rows for m in ("base", "bod") for d in r[m]
    }
    dup = sum(len(r[m]) for r in rows for m in ("base", "bod"))
    print(
        f"saved pool -> {paths['pool']}  ({dup:,} (model, query, doc) slots -> "
        f"{len(pairs):,} unique (query, product) pairs to judge)",
        flush=True,
    )


# --------------------------------------------------------------------------
# phase: estimate
# --------------------------------------------------------------------------
def _prompt_for(query, title, max_title_chars):
    return CATEGORY_PROMPT.format(query=query, title=title[:max_title_chars])


def _unique_pairs(payload, max_title_chars):
    """Deduped (query, product_id) -> prompt. Judged once, reused by both models."""
    out = {}
    for r in payload["rows"]:
        for model_key in ("base", "bod"):
            for d in r[model_key]:
                k = pair_key(r["query"], d["product_id"])
                if k not in out:
                    out[k] = {
                        "query": r["query"],
                        "product_id": d["product_id"],
                        "title": d["title"],
                        "prompt": _prompt_for(r["query"], d["title"], max_title_chars),
                    }
    return out


def phase_estimate(args, paths, quiet=False):
    with open(paths["pool"]) as f:
        payload = json.load(f)
    pairs = _unique_pairs(payload, args.max_title_chars)

    CHAT_ENVELOPE_TOKENS = 8  # role/format wrapper the API adds

    tin = sum(len(p["prompt"]) / 4.0 + CHAT_ENVELOPE_TOKENS for p in pairs.values())
    tout = len(pairs) * 1  # max_tokens=1
    cost = estimate_cost(args.model, tin, tout)

    n_slots = sum(len(r[m]) for r in payload["rows"] for m in ("base", "bod"))
    breakdown = {
        "model": args.model,
        "n_queries": len(payload["rows"]),
        "n_model_doc_slots": n_slots,
        "n_unique_pairs": len(pairs),
        "dedup_saving_pct": 100.0 * (1 - len(pairs) / max(n_slots, 1)),
        "judge_calls": len(pairs),
        "est_tokens_in": int(tin),
        "est_tokens_out": int(tout),
        "est_cost_usd_total": cost,
        "ceiling_usd": COST_CEILING_USD,
    }
    if not quiet:
        print(json.dumps(breakdown, indent=2), flush=True)
        print(
            f"\nPROJECTED COST: ${cost:.4f} ({len(pairs):,} calls) "
            f"vs ceiling ${COST_CEILING_USD:.2f}",
            flush=True,
        )
        if cost > COST_CEILING_USD:
            print("OVER CEILING -- the judge phase will refuse to run.", flush=True)
    return breakdown


def _guard_cost(args, paths):
    est = phase_estimate(args, paths, quiet=True)
    c = est["est_cost_usd_total"]
    print(f"[cost guard] projected ${c:.4f} (ceiling ${COST_CEILING_USD:.2f})", flush=True)
    if c > COST_CEILING_USD:
        raise SystemExit(
            f"Refusing to run: projected ${c:.4f} exceeds ceiling ${COST_CEILING_USD:.2f}. "
            f"Lower --sample/--top-k or raise --cost-ceiling deliberately."
        )
    return est


# --------------------------------------------------------------------------
# phase: judge
# --------------------------------------------------------------------------
def _load_judged(path):
    done = {}
    if not Path(path).exists():
        return done
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:  # truncated final line from a kill
                continue
            if r.get("margin") is not None and not math.isnan(r["margin"]):
                done[pair_key(r["query"], r["product_id"])] = r
    return done


async def _run_judge(args, todo, usage, out_f):
    client = make_client()
    sem = asyncio.Semaphore(args.concurrency)

    async def one(item):
        key, p = item
        ch = await _chat(client, sem, usage, args.model, p["prompt"], 1, logprobs=True)
        try:
            margin, p_yes = _score_from_logprobs(ch)
        except Exception:  # never let one odd token kill a paid run
            usage.errors += 1
            margin, p_yes = float("nan"), float("nan")
        return key, p, margin, p_yes

    t0 = time.time()
    done = 0
    chunk = 500
    for i in range(0, len(todo), chunk):
        batch = todo[i : i + chunk]
        for _key, p, margin, p_yes in await asyncio.gather(*[one(x) for x in batch]):
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
            f"eta {(len(todo) - done) / max(done / max(el, 1e-9), 1e-9) / 60:.1f}m "
            f"errors={usage.errors} spent=${estimate_cost(args.model, usage.tin, usage.tout):.4f}",
            flush=True,
        )
    return len(todo)


def phase_judge(args, paths):
    with open(paths["pool"]) as f:
        payload = json.load(f)
    pairs = _unique_pairs(payload, args.max_title_chars)
    _guard_cost(args, paths)

    already = _load_judged(paths["judge"]) if args.resume else {}
    todo = [(k, v) for k, v in pairs.items() if k not in already]
    print(f"  {len(already):,} cached, {len(todo):,} to judge", flush=True)
    if not todo:
        print("  fully cached; nothing to do", flush=True)
        return

    usage = Usage()
    t0 = time.time()
    # A crash mid-run must still bank scores AND log the money already spent,
    # otherwise the ledger under-reports the true cost.
    try:
        with open(paths["judge"], "a") as out_f:
            asyncio.run(_run_judge(args, todo, usage, out_f))
    finally:
        c = estimate_cost(args.model, usage.tin, usage.tout)
        if usage.calls:
            record_spend(args.model, usage.tin, usage.tout, c, "bestbuy category junk-rate: judge")

    cost = estimate_cost(args.model, usage.tin, usage.tout)
    meta = {
        "judge_model": args.model,
        "prompt": CATEGORY_PROMPT,
        "mode": "pointwise category yes/no, max_tokens=1, top_logprobs=20; "
        "margin = log p(yes) - log p(no)",
        "n_pairs_judged": len(todo),
        "n_pairs_total": len(pairs),
        "tokens_in": usage.tin,
        "tokens_out": usage.tout,
        "api_calls": usage.calls,
        "api_errors": usage.errors,
        "cost_usd": cost,
        "wall_clock_s": time.time() - t0,
    }
    prev = {}
    if Path(paths["judge_meta"]).exists():
        with open(paths["judge_meta"]) as f:
            prev = json.load(f)
    meta["cost_usd_cumulative"] = round(cost + float(prev.get("cost_usd_cumulative", 0.0)), 6)
    with open(paths["judge_meta"], "w") as f:
        json.dump(meta, f, indent=2)
    print(
        f"\njudge done in {meta['wall_clock_s'] / 60:.1f}m  calls={usage.calls:,} "
        f"errors={usage.errors} cost=${cost:.4f} "
        f"(cumulative ${meta['cost_usd_cumulative']:.4f})",
        flush=True,
    )


# --------------------------------------------------------------------------
# phase: eval
# --------------------------------------------------------------------------
def _bootstrap_ci(values, n_boot=2000, seed=0):
    v = np.asarray([x for x in values if x is not None and not math.isnan(x)], dtype=np.float64)
    if v.size == 0:
        return None, None
    rng = np.random.default_rng(seed)
    means = v[rng.integers(0, v.size, size=(n_boot, v.size))].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def _paired_ci(deltas, n_boot=2000, seed=0):
    return _bootstrap_ci(deltas, n_boot, seed)


def phase_eval(args, paths):
    with open(paths["pool"]) as f:
        payload = json.load(f)
    data = Path(payload["data_dir"])
    qrels, _queries_all = load_split(data, args.queries_file, args.qrels_file)
    judged = _load_judged(paths["judge"])
    print(f"  {len(judged):,} judged pairs loaded", flush=True)

    thr = args.junk_threshold  # p_yes below this = junk (0.5 == argmax no)
    rows = payload["rows"]
    scored_rows = []
    n_missing = 0

    for r in rows:
        entry = {"key": r["key"], "query": r["query"], "is_manual": r["is_manual"]}
        for model_key in ("base", "bod"):
            docs = []
            for d in r[model_key]:
                j = judged.get(pair_key(r["query"], d["product_id"]))
                p_yes = j["p_yes"] if j and j.get("p_yes") is not None else None
                if p_yes is None:
                    n_missing += 1
                docs.append(
                    {
                        **d,
                        "p_yes": p_yes,
                        "margin": j["margin"] if j else None,
                        "junk": (p_yes is not None and p_yes < thr),
                    }
                )
            entry[model_key] = docs
        scored_rows.append(entry)
    if n_missing:
        print(f"  WARNING: {n_missing} (query, doc) slots have no judge score", flush=True)

    holdout_rows = [r for r in scored_rows if not r["is_manual"]]
    manual_rows = [r for r in scored_rows if r["is_manual"]]

    summary = {}
    per_query = {"base": {}, "bod": {}}
    for model_key in ("base", "bod"):
        junk_rates, mean_p_yes, recalls, hits, e1s, ndcgs = [], [], [], [], [], []
        for r in holdout_rows:
            docs = r[model_key][:K_EVAL]
            scored = [d for d in docs if d["p_yes"] is not None]
            jr = sum(1 for d in scored if d["junk"]) / len(scored) if scored else float("nan")
            junk_rates.append(jr)
            mean_p_yes.append(
                float(np.mean([d["p_yes"] for d in scored])) if scored else float("nan")
            )
            m = per_query_metrics(
                [d["product_id"] for d in docs],
                qrels[r["key"]],
                k=K_EVAL,
                min_rel=args.min_relevance,
                exact_rel=args.exact_relevance,
            )
            recall, ndcg, e1, _e3 = m if m else (float("nan"),) * 4
            gold = {p for p, g in qrels[r["key"]].items() if g >= args.min_relevance}
            hit = 1.0 if any(d["product_id"] in gold for d in docs) else 0.0
            recalls.append(recall)
            ndcgs.append(ndcg)
            e1s.append(e1)
            hits.append(hit)
            per_query[model_key][str(r["key"])] = {
                "query": r["query"],
                "junk_rate": jr,
                "recall_at_10": recall,
                "hit_at_10": hit,
                "e_at_1": e1,
            }

        def nm(v):
            a = np.asarray(v, dtype=np.float64)
            a = a[~np.isnan(a)]
            return float(a.mean()) if a.size else float("nan")

        lo, hi = _bootstrap_ci(junk_rates, args.n_boot, args.seed)
        summary[model_key] = {
            "junk_rate_at_10": nm(junk_rates),
            "junk_rate_ci95": [lo, hi],
            "mean_p_yes": nm(mean_p_yes),
            "queries_with_zero_junk": float(np.mean([j == 0.0 for j in junk_rates])),
            "queries_majority_junk": float(np.mean([j > 0.5 for j in junk_rates])),
            "recall_at_10_fraction_recovered": nm(recalls),
            "hit_rate_at_10": nm(hits),
            "ndcg_at_10": nm(ndcgs),
            "e_at_1": nm(e1s),
            "_junk_rates": junk_rates,
        }

    d_junk = [
        summary["bod"]["_junk_rates"][i] - summary["base"]["_junk_rates"][i]
        for i in range(len(holdout_rows))
    ]
    dlo, dhi = _paired_ci(d_junk, args.n_boot, args.seed)
    delta = {
        "junk_rate_bod_minus_base": float(np.nanmean(d_junk)),
        "junk_rate_delta_ci95": [dlo, dhi],
        "recall_at_10_delta": summary["bod"]["recall_at_10_fraction_recovered"]
        - summary["base"]["recall_at_10_fraction_recovered"],
        "hit_rate_at_10_delta": summary["bod"]["hit_rate_at_10"]
        - summary["base"]["hit_rate_at_10"],
        "e_at_1_delta": summary["bod"]["e_at_1"] - summary["base"]["e_at_1"],
    }

    # Is junk-rate telling us anything R@10 doesn't? Correlate them per query.
    corr = {}
    for model_key in ("base", "bod"):
        jr = np.asarray([v["junk_rate"] for v in per_query[model_key].values()])
        rc = np.asarray([v["recall_at_10"] for v in per_query[model_key].values()])
        e1 = np.asarray([v["e_at_1"] for v in per_query[model_key].values()])
        ok = ~(np.isnan(jr) | np.isnan(rc))
        corr[f"{model_key}_pearson_junkrate_vs_recall10"] = (
            float(np.corrcoef(jr[ok], rc[ok])[0, 1]) if ok.sum() > 2 else None
        )
        ok1 = ~(np.isnan(jr) | np.isnan(e1))
        corr[f"{model_key}_pearson_junkrate_vs_e1"] = (
            float(np.corrcoef(jr[ok1], e1[ok1])[0, 1]) if ok1.sum() > 2 else None
        )

    def example(r, model_key):
        return {
            "query": r["query"],
            "junk_rate": sum(1 for d in r[model_key] if d["junk"])
            / max(len([d for d in r[model_key] if d["p_yes"] is not None]), 1),
            "top_10": [
                {
                    "rank": d["rank"],
                    "title": d["title"],
                    "p_yes": d["p_yes"],
                    "junk": d["junk"],
                }
                for d in r[model_key][:K_EVAL]
            ],
        }

    examples = {}
    for model_key in ("base", "bod"):
        ranked = sorted(
            holdout_rows,
            key=lambda r: -sum(1 for d in r[model_key] if d["junk"]),
        )
        examples[model_key] = [example(r, model_key) for r in ranked[: args.n_examples]]

    manual_checks = {r["query"]: {m: example(r, m) for m in ("base", "bod")} for r in manual_rows}
    # If a manual query happened to land in the random sample, still surface it.
    for r in holdout_rows:
        if r["query"].strip().lower() in {q.strip().lower() for q in args.manual_queries}:
            manual_checks[r["query"]] = {m: example(r, m) for m in ("base", "bod")}

    judge_meta = {}
    if Path(paths["judge_meta"]).exists():
        with open(paths["judge_meta"]) as f:
            judge_meta = json.load(f)

    for v in summary.values():
        v.pop("_junk_rates", None)

    print(f"\n=== category junk-rate @ {K_EVAL}, {len(holdout_rows)} holdout queries ===")
    hdr = f"{'model':<8} {'junk@10':>9} {'meanPyes':>9} {'R@10':>8} {'hit@10':>8} {'E@1':>8}"
    print(hdr)
    for model_key in ("base", "bod"):
        s = summary[model_key]
        print(
            f"{model_key:<8} {s['junk_rate_at_10']:>9.4f} {s['mean_p_yes']:>9.4f} "
            f"{s['recall_at_10_fraction_recovered']:>8.4f} {s['hit_rate_at_10']:>8.4f} "
            f"{s['e_at_1']:>8.4f}"
        )
    print(
        f"\nΔ junk-rate (BoD - base): {delta['junk_rate_bod_minus_base']:+.4f} "
        f"CI95 [{dlo:+.4f}, {dhi:+.4f}]"
    )
    print(f"Δ R@10: {delta['recall_at_10_delta']:+.4f}   Δ E@1: {delta['e_at_1_delta']:+.4f}")
    print(f"\njunk-rate vs click-metric correlation: {json.dumps(corr, indent=2)}")

    for q, mc in manual_checks.items():
        print(f"\n--- manual check: {q!r} ---")
        for model_key in ("base", "bod"):
            print(f"  [{model_key}] junk-rate {mc[model_key]['junk_rate']:.2f}")
            for d in mc[model_key]["top_10"]:
                flag = "JUNK" if d["junk"] else "ok  "
                py = "  n/a" if d["p_yes"] is None else f"{d['p_yes']:.3f}"
                print(f"    {d['rank']:>2}. {flag} p_yes={py}  {d['title'][:90]}")

    out = {
        "experiment": "BestBuy top-10 category-error (junk) rate, base MiniLM vs BoD MiniLM",
        "question": (
            "Do these encoders surface outright wrong-CATEGORY products in the top-10, "
            "and can a cheap LLM judge detect that? Distinct from Pattern 27, whose "
            "click-qrel R@10/E@1 metrics only credit exact SKU matches and are near-blind "
            "to product-type errors."
        ),
        "step": "1 of N: measurement only, no reranking or filtering applied",
        "config": {
            "dataset_repo": DATASET_REPO,
            "base_model": payload["base_model"],
            "bod_model": payload["bod_model"],
            "catalog_size": payload["catalog_size"],
            "sample_size": len(holdout_rows),
            "seed": payload["seed"],
            "selection": payload["selection"],
            "top_k": payload["top_k"],
            "k_eval": K_EVAL,
            "judge_model": args.model,
            "judge_prompt": CATEGORY_PROMPT,
            "junk_threshold_p_yes": thr,
            "n_boot": args.n_boot,
            "dedup": "each unique (query, product_id) judged once, reused for both models",
        },
        "judge_run": judge_meta,
        "summary": summary,
        "delta": delta,
        "correlations": corr,
        "examples_highest_junk": examples,
        "manual_checks": manual_checks,
        "per_query": per_query,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {out_path}", flush=True)


# --------------------------------------------------------------------------
def main():
    global COST_CEILING_USD
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--phase", required=True, choices=["pool", "estimate", "judge", "eval"])
    ap.add_argument("--data-dir", default=None, help="local artifact dir; default = HF snapshot")
    ap.add_argument("--queries-file", default="holdout_queries.jsonl")
    ap.add_argument("--qrels-file", default="holdout_qrels.jsonl")
    ap.add_argument("--base-model", default="all-MiniLM-L6-v2")
    ap.add_argument("--bod-model", default="dtunkelang/bag-of-documents-bestbuy-minilm")
    ap.add_argument("--base-vecs", default="base_catalog.vecs.fp16.npy")
    ap.add_argument("--bod-vecs", default="bod_catalog.vecs.fp16.npy")
    ap.add_argument("--model", default="gpt-4o-mini", help="OpenAI judge model")
    ap.add_argument("--concurrency", type=int, default=32)
    ap.add_argument(
        "--cost-ceiling",
        type=float,
        default=COST_CEILING_USD,
        help="refuse to start the judge phase if projected above this (USD)",
    )
    ap.add_argument("--sample", type=int, default=250)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--top-k", type=int, default=K_EVAL)
    ap.add_argument("--max-title-chars", type=int, default=300)
    ap.add_argument("--min-relevance", type=int, default=1)
    ap.add_argument("--exact-relevance", type=int, default=1)
    ap.add_argument("--junk-threshold", type=float, default=0.5, help="p_yes below this = junk")
    ap.add_argument("--n-examples", type=int, default=10)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument(
        "--manual-queries",
        nargs="*",
        default=list(DEFAULT_MANUAL_QUERIES),
        help="always retrieved + judged, reported separately from the random sample",
    )
    ap.add_argument("--resume", action="store_true", default=True)
    ap.add_argument("--no-resume", dest="resume", action="store_false")
    ap.add_argument("--work-dir", default="/tmp/bestbuy_junkrate")
    ap.add_argument("--tag", default="bestbuy")
    ap.add_argument("--out", default="evaluation/results/bestbuy_llm_judge_junkrate.json")
    args = ap.parse_args()

    COST_CEILING_USD = args.cost_ceiling

    paths = work_paths(args.work_dir, args.tag)
    {
        "pool": lambda: phase_pool(args, paths),
        "estimate": lambda: phase_estimate(args, paths),
        "judge": lambda: phase_judge(args, paths),
        "eval": lambda: phase_eval(args, paths),
    }[args.phase]()


if __name__ == "__main__":
    main()
