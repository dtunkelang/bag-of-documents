#!/usr/bin/env python3
"""Query-level story behind the NEW-index aggregate: where do base and BoD differ?

`eval_bestbuy_full_reindex_validation.py` showed that on the re-indexed BestBuy
catalog the BoD-fine-tuned MiniLM beats the base MiniLM in aggregate
(R@10 0.2467 vs 0.1691, junk@10 0.1456 vs 0.2508 over the same 250 seed-0
holdout queries). An aggregate delta says nothing about WHICH queries moved, or
whether the win is broad or carried by a handful of blowouts -- and BoD is
trained on a 2012 click log, so it can plausibly carry idiosyncratic biases the
base model does not share. This script answers:

  1. How different are the two top-10s, per query? (top-10 set overlap)
  2. On the queries where they genuinely diverge, who wins -- and by which
     definition of "wins":
       * categorical relevance  -> per-query junk@10 from the gpt-4o-mini
         "is this a plausible TYPE for the query" judge
       * click accuracy         -> does the top-10 contain the qrels gold
         (actually-clicked) product
  3. Are there real counter-examples where base-NEW beats BoD-NEW?
  4. Brand diversity: a top-10 can be judged 100% on-category and still be a bad
     result page if it is ten near-duplicate variants of one product from one
     manufacturer. Distinct manufacturers in the top-10 is the read on that,
     prompted by a hand-test where base looked fixated on a single brand.

Everything reuses the artifacts the validation run already produced:

  * retrieval  -- /tmp/bestbuy_reindex_output/retrieval_bestbuy.json holds the
    full-corpus top-10 for all four arms (base/bod x old/new) on exactly this
    250-query sample. No re-encoding, no re-retrieval.
  * judge labels -- the same three (query, product) -> p_yes caches the
    validation run used. As of this writing every base_new/bod_new slot is
    already judged, so the judge phase is a $0 no-op; it stays in place so the
    script is honest and resumable if the retrieval set ever changes.

Judge methodology is byte-identical to eval_bestbuy_llm_judge_junkrate.py
(gpt-4o-mini, CATEGORY_PROMPT, max_tokens=1, top_logprobs=20, p_yes < 0.5 =
junk) and the prompt is always built from the OLD shipped title -- the exact
string the cached labels came from -- so a label stays a pure function of
(query, product).

Phases (cached to --work-dir, resumable):
  divergence  overlap between base_new and bod_new top-10; pick the divergent set
  probe       full-corpus top-10 for off-sample queries ("laptop", "windows laptop")
  estimate    what the judge would cost for any unjudged slot (expect $0)
  judge       fill any gaps (append-only JSONL)
  eval        per-query win/loss, overlap distribution, brand diversity, examples

Usage:
  python evaluation/eval_bestbuy_base_vs_bod_divergence.py --phase divergence
  python evaluation/eval_bestbuy_base_vs_bod_divergence.py --phase probe
  python evaluation/eval_bestbuy_base_vs_bod_divergence.py --phase estimate
  python evaluation/eval_bestbuy_base_vs_bod_divergence.py --phase judge
  python evaluation/eval_bestbuy_base_vs_bod_divergence.py --phase eval
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np  # noqa: E402
from dotenv import load_dotenv  # noqa: E402

from evaluation.eval_bestbuy_full_reindex_validation import (  # noqa: E402
    PRIOR_JUDGE_CACHES,
)
from evaluation.eval_bestbuy_llm_judge_junkrate import (  # noqa: E402
    CATEGORY_PROMPT,
    Usage,
    _bootstrap_ci,
    _load_judged,
    _run_judge,
    _topk_over_catalog,
    estimate_cost,
    pair_key,
    record_spend,
)
from evaluation.eval_bestbuy_llm_judge_rerank import (  # noqa: E402
    load_split,
    per_query_metrics,
)

load_dotenv(override=True)

K_EVAL = 10

# Hard stop: refuse to start the judge phase if the projection exceeds this.
COST_CEILING_USD = 2.0

# The validation run's retrieval + judge artifacts. Its own judge JSONL is a
# cache here too -- it already covers every base_new/bod_new slot.
DEFAULT_RETRIEVAL = "/tmp/bestbuy_reindex_output/retrieval_bestbuy.json"
REINDEX_JUDGE_CACHE = "/tmp/bestbuy_reindex_output/reindex_judge_bestbuy.jsonl"

# Per-SKU manufacturer, written by the re-index catalog phase. It is the exact
# field that went into the NEW embedding text, so the brand-diversity read is
# on the same data the index was built from.
CATALOG_FIELDS = "/tmp/bestbuy_reindex_output/catalog_fields_bestbuy.jsonl"
NEW_VECS = {
    "base": "/tmp/bestbuy_reindex_output/artifacts/base_catalog.vecs.fp16.npy",
    "bod": "/tmp/bestbuy_reindex_output/artifacts/bod_catalog.vecs.fp16.npy",
}
NEW_PRODUCT_IDS = "/tmp/bestbuy_reindex_output/artifacts/product_ids.json"
NEW_TITLES = "/tmp/bestbuy_reindex_output/artifacts/titles.json"

# Off-sample queries the user hit by hand in the local demo and reported base
# as "overfixated on one brand/model". Retrieved ad hoc by --phase probe.
DEFAULT_PROBE_QUERIES = ("laptop", "windows laptop")

ARMS = ("base_new", "bod_new")


# --------------------------------------------------------------------------
# paths
# --------------------------------------------------------------------------
def work_paths(work_dir, tag):
    w = Path(work_dir)
    w.mkdir(parents=True, exist_ok=True)
    return {
        "root": w,
        "divergence": w / f"divergence_{tag}.json",
        "judge": w / f"divergence_judge_{tag}.jsonl",
        "judge_meta": w / f"divergence_judge_meta_{tag}.json",
        "probe": w / f"probe_{tag}.json",
    }


def load_prior_labels(paths, resume=True):
    """Every cached (query, pid) -> label this session has produced."""
    out = {}
    for p in (*PRIOR_JUDGE_CACHES, REINDEX_JUDGE_CACHE):
        out.update(_load_judged(p))
    if resume:
        out.update(_load_judged(paths["judge"]))
    return out


def load_retrieval(args):
    with open(args.retrieval) as f:
        payload = json.load(f)
    holdout = [r for r in payload["rows"] if not r["is_manual"]]
    return payload, holdout


# --------------------------------------------------------------------------
# phase: divergence
# --------------------------------------------------------------------------
def _overlap(a_docs, b_docs, k):
    a = {d["product_id"] for d in a_docs[:k]}
    b = {d["product_id"] for d in b_docs[:k]}
    inter = a & b
    union = a | b
    return {
        "n_base": len(a),
        "n_bod": len(b),
        "n_intersection": len(inter),
        "jaccard": len(inter) / max(len(union), 1),
        "overlap_frac": len(inter) / max(min(len(a), len(b)), 1),
        "top1_same": bool(a_docs and b_docs and a_docs[0]["product_id"] == b_docs[0]["product_id"]),
    }


def phase_divergence(args, paths):
    payload, holdout = load_retrieval(args)
    print(f"  {len(holdout)} holdout queries from {args.retrieval}", flush=True)

    rows = []
    for r in holdout:
        ov = _overlap(r["base_new"], r["bod_new"], args.top_k)
        rows.append({"key": r["key"], "query": r["query"], **ov})

    counts = [0] * (args.top_k + 1)
    for row in rows:
        counts[row["n_intersection"]] += 1
    jac = np.array([row["jaccard"] for row in rows], dtype=np.float64)

    divergent = [row["key"] for row in rows if row["n_intersection"] <= args.divergence_max_overlap]
    out = {
        "retrieval_source": args.retrieval,
        "base_model": payload["base_model"],
        "bod_model": payload["bod_model"],
        "seed": payload["seed"],
        "sample_size": payload["sample_size"],
        "top_k": args.top_k,
        "divergence_rule": (
            f"|base_new_top{args.top_k} n bod_new_top{args.top_k}| <= {args.divergence_max_overlap}"
        ),
        "n_divergent": len(divergent),
        "overlap_histogram": {str(i): counts[i] for i in range(args.top_k + 1)},
        "mean_jaccard": float(jac.mean()),
        "median_jaccard": float(np.median(jac)),
        "top1_same_rate": float(np.mean([row["top1_same"] for row in rows])),
        "divergent_keys": divergent,
        "per_query_overlap": rows,
    }
    with open(paths["divergence"], "w") as f:
        json.dump(out, f, indent=2)

    print(
        f"  mean Jaccard {out['mean_jaccard']:.4f}  median {out['median_jaccard']:.4f}", flush=True
    )
    print(f"  top-1 identical on {out['top1_same_rate'] * 100:.1f}% of queries", flush=True)
    print("  |intersection| histogram:", flush=True)
    for i in range(args.top_k + 1):
        bar = "#" * counts[i]
        print(f"    {i:2d}: {counts[i]:4d} {bar}", flush=True)
    print(
        f"  divergent set ({out['divergence_rule']}): {len(divergent)} / {len(rows)} queries",
        flush=True,
    )
    print(f"saved -> {paths['divergence']}", flush=True)
    return out


# --------------------------------------------------------------------------
# brand diversity
# --------------------------------------------------------------------------
def load_brand_map(path=CATALOG_FIELDS):
    """sku -> manufacturer, straight from the re-index catalog phase."""
    p = Path(path)
    if not p.exists():
        raise SystemExit(f"missing {p} -- run the re-index --phase catalog first")
    out = {}
    with open(p) as f:
        for line in f:
            r = json.loads(line)
            out[r["sku"]] = (r.get("manufacturer") or "").strip()
    return out


def _norm_brand(b):
    return " ".join(b.lower().split())


def _norm_title(t):
    return " ".join(t.lower().split())


def diversity_of(docs, brands, top_k=K_EVAL):
    """Brand spread of one top-10.

    distinct_brands       -- how many different manufacturers are represented
    max_brand_share       -- fraction of branded slots held by the single most
                             common brand; 1.0 == total fixation on one brand
    distinct_titles       -- near-duplicate proxy (variants share a title)
    """
    from collections import Counter

    docs = docs[:top_k]
    bs = [_norm_brand(brands.get(d["product_id"], "")) for d in docs]
    branded = [b for b in bs if b]
    c = Counter(branded)
    top_brand, top_n = c.most_common(1)[0] if c else ("", 0)
    return {
        "n_docs": len(docs),
        "n_with_brand": len(branded),
        "distinct_brands": len(c),
        "top_brand": top_brand,
        "top_brand_count": top_n,
        "max_brand_share": (top_n / len(branded)) if branded else float("nan"),
        "distinct_titles": len({_norm_title(d["title_old"]) for d in docs}),
        "brands": [b if b else "<none>" for b in bs],
    }


# --------------------------------------------------------------------------
# phase: probe  (off-sample queries the user tested by hand)
# --------------------------------------------------------------------------
def phase_probe(args, paths):
    """Full-corpus top-10 on the NEW index for --probe-queries, both models."""
    from sentence_transformers import SentenceTransformer

    with open(NEW_PRODUCT_IDS) as f:
        pids = json.load(f)
    with open(NEW_TITLES) as f:
        titles = json.load(f)
    payload, _ = load_retrieval(args)
    queries = list(args.probe_queries)
    print(f"  probing {len(queries)} off-sample queries: {queries}", flush=True)

    rows = [{"key": f"probe:{q}", "query": q, "is_manual": True} for q in queries]
    for key, model_id in (("base", payload["base_model"]), ("bod", payload["bod_model"])):
        print(f"  encoding with {model_id}...", flush=True)
        m = SentenceTransformer(model_id)
        qv = m.encode(queries, normalize_embeddings=True, batch_size=32).astype(np.float32)
        del m
        print(f"    full-corpus top-{args.top_k}: {key}_new", flush=True)
        idx, sims = _topk_over_catalog(qv, NEW_VECS[key], args.top_k)
        for i, r in enumerate(rows):
            r[f"{key}_new"] = [
                {
                    "rank": rank + 1,
                    "row": int(j),
                    "product_id": pids[int(j)],
                    "title_old": titles[int(j)],
                    "sim": float(sims[i, rank]),
                }
                for rank, j in enumerate(idx[i])
                if j >= 0
            ]

    out = {
        "note": (
            "off-sample probe queries, NEW index only; no qrels and no judge labels -- "
            "used for the brand-diversity read, not for junk-rate or click accuracy"
        ),
        "base_model": payload["base_model"],
        "bod_model": payload["bod_model"],
        "top_k": args.top_k,
        "rows": rows,
    }
    with open(paths["probe"], "w") as f:
        json.dump(out, f, indent=2)
    print(f"saved -> {paths['probe']}", flush=True)
    return out


# --------------------------------------------------------------------------
# phase: estimate / judge
# --------------------------------------------------------------------------
def _needed_pairs(holdout, keys, max_title_chars, top_k):
    """(query, product) -> prompt for every base_new/bod_new slot of `keys`."""
    want = set(keys)
    out = {}
    for r in holdout:
        if r["key"] not in want:
            continue
        for arm in ARMS:
            for d in r[arm][:top_k]:
                k = pair_key(r["query"], d["product_id"])
                if k not in out:
                    out[k] = {
                        "query": r["query"],
                        "product_id": d["product_id"],
                        "title": d["title_old"],
                        "prompt": CATEGORY_PROMPT.format(
                            query=r["query"], title=d["title_old"][:max_title_chars]
                        ),
                    }
    return out


def _load_divergence(paths):
    if not Path(paths["divergence"]).exists():
        raise SystemExit(f"missing {paths['divergence']} -- run --phase divergence first")
    with open(paths["divergence"]) as f:
        return json.load(f)


def _judge_scope_keys(args, div, holdout):
    """Which queries to guarantee labels for: the divergent set, or all."""
    if args.judge_all:
        return [r["key"] for r in holdout]
    return list(div["divergent_keys"])


def phase_estimate(args, paths, quiet=False):
    _payload, holdout = load_retrieval(args)
    div = _load_divergence(paths)
    keys = _judge_scope_keys(args, div, holdout)
    pairs = _needed_pairs(holdout, keys, args.max_title_chars, args.top_k)
    have = set(load_prior_labels(paths, args.resume))
    todo = [v for k, v in pairs.items() if k not in have]

    envelope = 8  # role/format wrapper the API adds
    tin = sum(len(p["prompt"]) / 4.0 + envelope for p in todo)
    tout = len(todo) * 1  # max_tokens=1
    cost = estimate_cost(args.model, tin, tout)
    breakdown = {
        "model": args.model,
        "scope": "all holdout queries" if args.judge_all else "divergent queries only",
        "n_queries_in_scope": len(keys),
        "n_unique_pairs": len(pairs),
        "n_reused_from_cache": len(pairs) - len(todo),
        "cache_reuse_pct": 100.0 * (len(pairs) - len(todo)) / max(len(pairs), 1),
        "n_new_pairs_to_judge": len(todo),
        "est_tokens_in": int(tin),
        "est_tokens_out": int(tout),
        "est_cost_usd_total": cost,
        "ceiling_usd": args.cost_ceiling,
    }
    if not quiet:
        print(json.dumps(breakdown, indent=2), flush=True)
        print(
            f"\nPROJECTED COST: ${cost:.4f} ({len(todo):,} new calls) "
            f"vs ceiling ${args.cost_ceiling:.2f}",
            flush=True,
        )
        if cost > args.cost_ceiling:
            print("OVER CEILING -- the judge phase will refuse to run.", flush=True)
    return breakdown, todo


def phase_judge(args, paths):
    est, todo = phase_estimate(args, paths, quiet=True)
    c = est["est_cost_usd_total"]
    print(
        f"[cost guard] projected ${c:.4f} "
        f"({est['n_new_pairs_to_judge']:,} new / {est['n_unique_pairs']:,} pairs, "
        f"{est['cache_reuse_pct']:.1f}% reused) vs ceiling ${args.cost_ceiling:.2f}",
        flush=True,
    )
    if c > args.cost_ceiling:
        raise SystemExit(
            f"Refusing to run: projected ${c:.4f} exceeds ceiling ${args.cost_ceiling:.2f}."
        )
    if not todo:
        print("  fully cached; nothing to judge (cost $0)", flush=True)
        return

    usage = Usage()
    t0 = time.time()
    items = [(pair_key(p["query"], p["product_id"]), p) for p in todo]
    try:
        with open(paths["judge"], "a") as out_f:
            asyncio.run(_run_judge(args, items, usage, out_f))
    finally:
        cost = estimate_cost(args.model, usage.tin, usage.tout)
        if usage.calls:
            record_spend(
                args.model, usage.tin, usage.tout, cost, "bestbuy base-vs-bod divergence: judge"
            )

    cost = estimate_cost(args.model, usage.tin, usage.tout)
    meta = {
        "judge_model": args.model,
        "prompt": CATEGORY_PROMPT,
        "prompt_title_source": "OLD shipped titles.json (matches the cached labels)",
        "n_pairs_judged_this_run": len(todo),
        "n_pairs_reused_from_cache": est["n_reused_from_cache"],
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
def _score_arm(r, arm, judged, thr, top_k):
    docs = []
    for d in r[arm][:top_k]:
        j = judged.get(pair_key(r["query"], d["product_id"]))
        p_yes = j["p_yes"] if j and j.get("p_yes") is not None else None
        docs.append({**d, "p_yes": p_yes, "junk": (p_yes is not None and p_yes < thr)})
    return docs


def _junk_rate(docs):
    ok = [d for d in docs if d["p_yes"] is not None]
    return (sum(1 for d in ok if d["junk"]) / len(ok)) if ok else float("nan")


def _verdict(delta, eps=1e-9):
    """delta = bod-minus-base on a higher-is-better quantity."""
    if delta > eps:
        return "bod"
    if delta < -eps:
        return "base"
    return "tie"


def _doc_view(d):
    return {
        "rank": d["rank"],
        "product_id": d["product_id"],
        "title": d["title_old"],
        "p_yes": d["p_yes"],
        "junk": d["junk"],
        "is_gold": d.get("is_gold", False),
    }


def phase_eval(args, paths):
    payload, holdout = load_retrieval(args)
    div = _load_divergence(paths)
    judged = load_prior_labels(paths, resume=True)
    print(f"  {len(judged):,} cached judge labels", flush=True)
    brands = load_brand_map(args.catalog_fields)
    print(f"  {len(brands):,} sku -> manufacturer entries", flush=True)

    data = Path(payload["data_dir"])
    qrels, _ = load_split(data, args.queries_file, args.qrels_file)

    thr = args.junk_threshold
    divergent_keys = set(div["divergent_keys"])
    overlap_by_key = {row["key"]: row for row in div["per_query_overlap"]}

    n_missing = 0
    per_query = []
    for r in holdout:
        gold = {p for p, g in qrels[r["key"]].items() if g >= args.min_relevance}
        rec = {
            "key": r["key"],
            "query": r["query"],
            "n_intersection": overlap_by_key[r["key"]]["n_intersection"],
            "jaccard": overlap_by_key[r["key"]]["jaccard"],
            "top1_same": overlap_by_key[r["key"]]["top1_same"],
            "is_divergent": r["key"] in divergent_keys,
            "n_gold": len(gold),
        }
        arm_docs = {}
        for arm in ARMS:
            docs = _score_arm(r, arm, judged, thr, args.top_k)
            for d in docs:
                d["is_gold"] = d["product_id"] in gold
                if d["p_yes"] is None:
                    n_missing += 1
            arm_docs[arm] = docs
            pids = [d["product_id"] for d in docs]
            m = per_query_metrics(
                pids,
                qrels[r["key"]],
                k=args.top_k,
                min_rel=args.min_relevance,
                exact_rel=args.exact_relevance,
            )
            recall, ndcg, e1, _ = m if m else (float("nan"),) * 4
            short = "base" if arm == "base_new" else "bod"
            rec[f"{short}_junk_rate"] = _junk_rate(docs)
            rec[f"{short}_mean_p_yes"] = float(
                np.mean([d["p_yes"] for d in docs if d["p_yes"] is not None])
            )
            rec[f"{short}_hit"] = 1.0 if any(d["is_gold"] for d in docs) else 0.0
            rec[f"{short}_recall"] = recall
            rec[f"{short}_ndcg"] = ndcg
            rec[f"{short}_e1"] = e1
            rec[f"{short}_gold_rank"] = next((d["rank"] for d in docs if d["is_gold"]), None)
            div_i = diversity_of(docs, brands, args.top_k)
            rec[f"{short}_distinct_brands"] = div_i["distinct_brands"]
            rec[f"{short}_max_brand_share"] = div_i["max_brand_share"]
            rec[f"{short}_top_brand"] = div_i["top_brand"]
            rec[f"{short}_distinct_titles"] = div_i["distinct_titles"]
            rec[f"{short}_brands"] = div_i["brands"]
        # bod-minus-base, oriented so positive = BoD better
        rec["junk_delta"] = rec["base_junk_rate"] - rec["bod_junk_rate"]
        rec["hit_delta"] = rec["bod_hit"] - rec["base_hit"]
        rec["ndcg_delta"] = rec["bod_ndcg"] - rec["base_ndcg"]
        rec["brand_delta"] = rec["bod_distinct_brands"] - rec["base_distinct_brands"]
        rec["categorical_winner"] = _verdict(rec["junk_delta"])
        rec["click_winner"] = _verdict(rec["hit_delta"])
        rec["diversity_winner"] = _verdict(rec["brand_delta"])
        rec["_docs"] = arm_docs
        per_query.append(rec)

    if n_missing:
        print(f"  WARNING: {n_missing} slots have no judge label", flush=True)

    def agg(rows, label):
        if not rows:
            return {"label": label, "n_queries": 0}
        junk_b = [x["base_junk_rate"] for x in rows]
        junk_d = [x["bod_junk_rate"] for x in rows]
        hit_b = [x["base_hit"] for x in rows]
        hit_d = [x["bod_hit"] for x in rows]
        cat = [x["categorical_winner"] for x in rows]
        clk = [x["click_winner"] for x in rows]
        jd = [x["junk_delta"] for x in rows]
        hd = [x["hit_delta"] for x in rows]
        lo_j, hi_j = _bootstrap_ci(jd, args.n_boot, args.seed)
        lo_h, hi_h = _bootstrap_ci(hd, args.n_boot, args.seed)
        return {
            "label": label,
            "n_queries": len(rows),
            "mean_jaccard": float(np.mean([x["jaccard"] for x in rows])),
            "base_junk_rate_at_10": float(np.nanmean(junk_b)),
            "bod_junk_rate_at_10": float(np.nanmean(junk_d)),
            "junk_delta_base_minus_bod": {
                "delta": float(np.nanmean(jd)),
                "ci95": [lo_j, hi_j],
            },
            "base_hit_rate_at_10": float(np.mean(hit_b)),
            "bod_hit_rate_at_10": float(np.mean(hit_d)),
            "hit_delta_bod_minus_base": {
                "delta": float(np.mean(hd)),
                "ci95": [lo_h, hi_h],
            },
            "base_ndcg_at_10": float(np.nanmean([x["base_ndcg"] for x in rows])),
            "bod_ndcg_at_10": float(np.nanmean([x["bod_ndcg"] for x in rows])),
            "categorical_wins": {
                "bod": cat.count("bod"),
                "base": cat.count("base"),
                "tie": cat.count("tie"),
                "bod_win_rate_excl_ties": (
                    cat.count("bod") / max(cat.count("bod") + cat.count("base"), 1)
                ),
            },
            "base_distinct_brands_at_10": float(np.mean([x["base_distinct_brands"] for x in rows])),
            "bod_distinct_brands_at_10": float(np.mean([x["bod_distinct_brands"] for x in rows])),
            "distinct_brands_delta_bod_minus_base": {
                "delta": float(np.mean([x["brand_delta"] for x in rows])),
                "ci95": list(
                    _bootstrap_ci([x["brand_delta"] for x in rows], args.n_boot, args.seed)
                ),
            },
            "base_max_brand_share": float(np.nanmean([x["base_max_brand_share"] for x in rows])),
            "bod_max_brand_share": float(np.nanmean([x["bod_max_brand_share"] for x in rows])),
            "base_single_brand_top10_rate": float(
                np.mean([x["base_distinct_brands"] <= 1 for x in rows])
            ),
            "bod_single_brand_top10_rate": float(
                np.mean([x["bod_distinct_brands"] <= 1 for x in rows])
            ),
            "base_distinct_titles_at_10": float(np.mean([x["base_distinct_titles"] for x in rows])),
            "bod_distinct_titles_at_10": float(np.mean([x["bod_distinct_titles"] for x in rows])),
            "diversity_wins": {
                "bod": [x["diversity_winner"] for x in rows].count("bod"),
                "base": [x["diversity_winner"] for x in rows].count("base"),
                "tie": [x["diversity_winner"] for x in rows].count("tie"),
            },
            "click_wins": {
                "bod": clk.count("bod"),
                "base": clk.count("base"),
                "tie": clk.count("tie"),
                "tie_both_hit": sum(
                    1 for x in rows if x["click_winner"] == "tie" and x["bod_hit"] == 1.0
                ),
                "tie_both_miss": sum(
                    1 for x in rows if x["click_winner"] == "tie" and x["bod_hit"] == 0.0
                ),
                "bod_win_rate_excl_ties": (
                    clk.count("bod") / max(clk.count("bod") + clk.count("base"), 1)
                ),
            },
        }

    divergent = [x for x in per_query if x["is_divergent"]]
    agreeing = [x for x in per_query if not x["is_divergent"]]
    by_bucket = {}
    for lo, hi, name in ((0, 0, "0"), (1, 3, "1-3"), (4, 6, "4-6"), (7, 10, "7-10")):
        rows = [x for x in per_query if lo <= x["n_intersection"] <= hi]
        by_bucket[name] = agg(rows, f"|intersection| in [{lo},{hi}]")

    summary = {
        "all_250": agg(per_query, "all holdout queries"),
        "divergent": agg(divergent, div["divergence_rule"]),
        "agreeing": agg(agreeing, f"NOT ({div['divergence_rule']})"),
        "by_overlap_bucket": by_bucket,
    }

    # -- brand diversity -------------------------------------------------
    # A model can score zero "junk" and still be a bad experience if all ten
    # slots are near-duplicate variants of one product from one brand. Distinct
    # manufacturers in the top-10 is the cheap read on that.
    def brand_block(rows, label):
        return {
            "label": label,
            "n_queries": len(rows),
            "base_mean_distinct_brands": float(np.mean([x["base_distinct_brands"] for x in rows])),
            "bod_mean_distinct_brands": float(np.mean([x["bod_distinct_brands"] for x in rows])),
            "base_mean_max_brand_share": float(
                np.nanmean([x["base_max_brand_share"] for x in rows])
            ),
            "bod_mean_max_brand_share": float(np.nanmean([x["bod_max_brand_share"] for x in rows])),
            "base_single_brand_rate": float(
                np.mean([x["base_distinct_brands"] <= 1 for x in rows])
            ),
            "bod_single_brand_rate": float(np.mean([x["bod_distinct_brands"] <= 1 for x in rows])),
        }

    corr_rows = [x for x in per_query if not np.isnan(x["base_max_brand_share"])]
    inter = np.array([x["n_intersection"] for x in corr_rows], dtype=np.float64)
    corr = {}
    for name, vals in (
        ("base_distinct_brands", [x["base_distinct_brands"] for x in corr_rows]),
        ("bod_distinct_brands", [x["bod_distinct_brands"] for x in corr_rows]),
        ("brand_delta_bod_minus_base", [x["brand_delta"] for x in corr_rows]),
        ("base_max_brand_share", [x["base_max_brand_share"] for x in corr_rows]),
    ):
        v = np.array(vals, dtype=np.float64)
        corr[f"pearson_r_vs_n_intersection__{name}"] = (
            float(np.corrcoef(inter, v)[0, 1]) if v.std() > 0 else float("nan")
        )

    probe = {}
    if Path(paths["probe"]).exists():
        with open(paths["probe"]) as f:
            pr = json.load(f)
        for r in pr["rows"]:
            entry = {"query": r["query"]}
            for arm in ARMS:
                short = "base" if arm == "base_new" else "bod"
                dv = diversity_of(r[arm], brands, args.top_k)
                entry[short] = {
                    "distinct_brands": dv["distinct_brands"],
                    "max_brand_share": dv["max_brand_share"],
                    "top_brand": dv["top_brand"],
                    "top_brand_count": dv["top_brand_count"],
                    "distinct_titles": dv["distinct_titles"],
                    "brands": dv["brands"],
                    "top_10": [
                        {
                            "rank": d["rank"],
                            "brand": brands.get(d["product_id"], ""),
                            "title": d["title_old"],
                        }
                        for d in r[arm][: args.top_k]
                    ],
                }
            bset = {d["product_id"] for d in r["base_new"][: args.top_k]}
            dset = {d["product_id"] for d in r["bod_new"][: args.top_k]}
            entry["n_intersection"] = len(bset & dset)
            probe[r["query"]] = entry
    else:
        probe = {"note": f"no probe file at {paths['probe']}; run --phase probe"}

    worst_base = sorted(
        divergent,
        key=lambda x: (
            x["base_distinct_brands"] - x["bod_distinct_brands"],
            x["base_distinct_brands"],
        ),
    )[: args.n_examples]
    worst_bod = sorted(
        divergent,
        key=lambda x: (
            x["bod_distinct_brands"] - x["base_distinct_brands"],
            x["bod_distinct_brands"],
        ),
    )[: args.n_examples]

    def brand_example(rec):
        return {
            "query": rec["query"],
            "n_intersection": rec["n_intersection"],
            "base_distinct_brands": rec["base_distinct_brands"],
            "bod_distinct_brands": rec["bod_distinct_brands"],
            "base_top_brand": rec["base_top_brand"],
            "bod_top_brand": rec["bod_top_brand"],
            "base_brands": rec["base_brands"],
            "bod_brands": rec["bod_brands"],
            "base_junk_rate": rec["base_junk_rate"],
            "bod_junk_rate": rec["bod_junk_rate"],
        }

    brand_diversity = {
        "definition": (
            "distinct non-empty manufacturers among the top-10 (from the same "
            "manufacturer field that went into the NEW embedding text); "
            "max_brand_share = share of branded slots held by the single most "
            "common brand, 1.0 = total fixation on one brand"
        ),
        "all_250": brand_block(per_query, "all holdout queries"),
        "divergent": brand_block(divergent, div["divergence_rule"]),
        "agreeing": brand_block(agreeing, f"NOT ({div['divergence_rule']})"),
        # Music/video SKUs carry a record label (or nothing) in `manufacturer`,
        # which makes the brand read noisy. This cut keeps only the queries whose
        # top-10s are mostly real branded hardware -- the regime the hand-test was in.
        "branded_heavy": brand_block(
            [
                x
                for x in per_query
                if sum(1 for b in x["base_brands"] if b != "<none>") >= args.min_branded_slots
                and sum(1 for b in x["bod_brands"] if b != "<none>") >= args.min_branded_slots
            ],
            f">= {args.min_branded_slots}/10 branded slots in BOTH top-10s",
        ),
        "correlation_with_divergence": corr,
        "probe_queries": probe,
        "base_worst_brand_fixation_examples": [brand_example(x) for x in worst_base],
        "bod_worst_brand_fixation_examples": [brand_example(x) for x in worst_bod],
    }

    # -- examples -------------------------------------------------------
    def example(rec, n_sym=6):
        b = rec["_docs"]["base_new"]
        d = rec["_docs"]["bod_new"]
        bset = {x["product_id"] for x in b}
        dset = {x["product_id"] for x in d}
        return {
            "query": rec["query"],
            "key": rec["key"],
            "n_intersection": rec["n_intersection"],
            "jaccard": rec["jaccard"],
            "base_junk_rate": rec["base_junk_rate"],
            "bod_junk_rate": rec["bod_junk_rate"],
            "base_hit": rec["base_hit"],
            "bod_hit": rec["bod_hit"],
            "base_gold_rank": rec["base_gold_rank"],
            "bod_gold_rank": rec["bod_gold_rank"],
            "categorical_winner": rec["categorical_winner"],
            "click_winner": rec["click_winner"],
            "gold_titles": rec["gold_titles"],
            "base_only_top10": [_doc_view(x) for x in b if x["product_id"] not in dset][:n_sym],
            "bod_only_top10": [_doc_view(x) for x in d if x["product_id"] not in bset][:n_sym],
            "base_top10": [_doc_view(x) for x in b],
            "bod_top10": [_doc_view(x) for x in d],
        }

    # gold titles for readability (from the OLD shipped titles seen in retrieval)
    title_by_pid = {}
    for r in holdout:
        for arm in ("base_old", "base_new", "bod_old", "bod_new"):
            for d in r.get(arm, []):
                title_by_pid.setdefault(d["product_id"], d["title_old"])
    for rec in per_query:
        gold = {p for p, g in qrels[rec["key"]].items() if g >= args.min_relevance}
        rec["gold_titles"] = [
            title_by_pid.get(p, f"<pid {p} not in any top-10>") for p in sorted(gold)
        ]

    n_ex = args.n_examples
    bod_big = sorted(divergent, key=lambda x: (-x["junk_delta"], -x["hit_delta"]))[:n_ex]
    base_big = sorted(divergent, key=lambda x: (x["junk_delta"], x["hit_delta"]))[:n_ex]
    click_bod = [x for x in divergent if x["click_winner"] == "bod"]
    click_base = [x for x in divergent if x["click_winner"] == "base"]
    click_bod = sorted(click_bod, key=lambda x: x["bod_gold_rank"] or 99)[:n_ex]
    click_base = sorted(click_base, key=lambda x: x["base_gold_rank"] or 99)[:n_ex]

    examples = {
        "bod_biggest_categorical_wins": [example(x) for x in bod_big],
        "base_biggest_categorical_wins": [example(x) for x in base_big],
        "bod_click_wins": [example(x) for x in click_bod],
        "base_click_wins": [example(x) for x in click_base],
    }

    judge_meta = {}
    if Path(paths["judge_meta"]).exists():
        with open(paths["judge_meta"]) as f:
            judge_meta = json.load(f)

    for rec in per_query:
        rec.pop("_docs", None)

    out = {
        "experiment": (
            "BestBuy NEW index: query-level divergence between base MiniLM and the "
            "BoD-fine-tuned MiniLM, and who wins the queries where they differ."
        ),
        "question": (
            "The aggregate says BoD-NEW > base-NEW (R@10 0.2467 vs 0.1691, junk@10 "
            "0.1456 vs 0.2508). On which queries do the two top-10s actually differ, "
            "and on those queries does BoD win on categorical relevance (LLM judge) "
            "and on click accuracy (qrels gold in top-10)? Any counter-examples where "
            "base is genuinely better?"
        ),
        "config": {
            "retrieval_source": args.retrieval,
            "base_model": payload["base_model"],
            "bod_model": payload["bod_model"],
            "catalog_size": payload["catalog_size"],
            "index": "NEW re-indexed catalog (name + manufacturer + categoryPath + class)",
            "sample_size": len(holdout),
            "seed": payload["seed"],
            "selection": payload["selection"],
            "top_k": args.top_k,
            "judge_model": args.model,
            "judge_prompt": CATEGORY_PROMPT,
            "junk_threshold": thr,
            "min_relevance": args.min_relevance,
            "divergence_rule": div["divergence_rule"],
            "reused_judge_caches": [*PRIOR_JUDGE_CACHES, REINDEX_JUDGE_CACHE],
        },
        "overlap_distribution": {
            "histogram_intersection_count": div["overlap_histogram"],
            "mean_jaccard": div["mean_jaccard"],
            "median_jaccard": div["median_jaccard"],
            "top1_same_rate": div["top1_same_rate"],
            "n_divergent": div["n_divergent"],
            "n_agreeing": len(holdout) - div["n_divergent"],
        },
        "summary": summary,
        "brand_diversity": brand_diversity,
        "judge_run": judge_meta or {"note": "no new pairs judged; fully served from cache ($0)"},
        "n_unjudged_slots": n_missing,
        "examples": examples,
        "per_query": per_query,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)

    # -- console report -------------------------------------------------
    print("\n" + "=" * 74, flush=True)
    print("OVERLAP DISTRIBUTION (base_new vs bod_new top-10)", flush=True)
    print("=" * 74, flush=True)
    for i in range(args.top_k + 1):
        n = div["overlap_histogram"][str(i)]
        print(f"  |intersection| = {i:2d}: {n:4d}  {'#' * n}", flush=True)
    print(
        f"  mean Jaccard {div['mean_jaccard']:.4f}  "
        f"top-1 identical {div['top1_same_rate'] * 100:.1f}%",
        flush=True,
    )

    for name in ("all_250", "divergent", "agreeing"):
        s = summary[name]
        if not s["n_queries"]:
            continue
        print("\n" + "-" * 74, flush=True)
        print(f"{name.upper()}  ({s['n_queries']} queries; {s['label']})", flush=True)
        print(
            f"  junk@10   base {s['base_junk_rate_at_10']:.4f}  "
            f"bod {s['bod_junk_rate_at_10']:.4f}  "
            f"delta {s['junk_delta_base_minus_bod']['delta']:+.4f} "
            f"CI{[round(v, 4) for v in s['junk_delta_base_minus_bod']['ci95']]}",
            flush=True,
        )
        print(
            f"  hit@10    base {s['base_hit_rate_at_10']:.4f}  "
            f"bod {s['bod_hit_rate_at_10']:.4f}  "
            f"delta {s['hit_delta_bod_minus_base']['delta']:+.4f} "
            f"CI{[round(v, 4) for v in s['hit_delta_bod_minus_base']['ci95']]}",
            flush=True,
        )
        c = s["categorical_wins"]
        k = s["click_wins"]
        print(
            f"  categorical: bod {c['bod']} / base {c['base']} / tie {c['tie']}  "
            f"(bod win-rate excl ties {c['bod_win_rate_excl_ties'] * 100:.1f}%)",
            flush=True,
        )
        print(
            f"  click:       bod {k['bod']} / base {k['base']} / tie {k['tie']} "
            f"(both-hit {k['tie_both_hit']}, both-miss {k['tie_both_miss']})  "
            f"(bod win-rate excl ties {k['bod_win_rate_excl_ties'] * 100:.1f}%)",
            flush=True,
        )

    print("\n" + "-" * 74, flush=True)
    print("BY OVERLAP BUCKET", flush=True)
    for name, s in by_bucket.items():
        if not s["n_queries"]:
            continue
        c = s["categorical_wins"]
        print(
            f"  overlap {name:>5}  n={s['n_queries']:3d}  "
            f"junk base {s['base_junk_rate_at_10']:.3f} / bod {s['bod_junk_rate_at_10']:.3f}  "
            f"hit base {s['base_hit_rate_at_10']:.3f} / bod {s['bod_hit_rate_at_10']:.3f}  "
            f"cat wins bod {c['bod']} base {c['base']}",
            flush=True,
        )

    print("\n" + "-" * 74, flush=True)
    print("BRAND DIVERSITY (distinct manufacturers in top-10)", flush=True)
    for name in ("all_250", "divergent", "agreeing", "branded_heavy"):
        b = brand_diversity[name]
        print(
            f"  {name:>13}  n={b['n_queries']:3d}  distinct brands "
            f"base {b['base_mean_distinct_brands']:.2f} / bod {b['bod_mean_distinct_brands']:.2f}  "
            f"max-brand-share base {b['base_mean_max_brand_share']:.3f} / "
            f"bod {b['bod_mean_max_brand_share']:.3f}  "
            f"single-brand top-10 base {b['base_single_brand_rate'] * 100:.1f}% / "
            f"bod {b['bod_single_brand_rate'] * 100:.1f}%",
            flush=True,
        )
    print("  correlation with |intersection|:", flush=True)
    for k, v in corr.items():
        print(f"    {k}: r={v:+.3f}", flush=True)
    if isinstance(probe, dict) and "note" not in probe:
        print("\n  PROBE QUERIES (off-sample, user-reported):", flush=True)
        for q, e in probe.items():
            print(f"    q={q!r}  |intersection|={e['n_intersection']}", flush=True)
            for short in ("base", "bod"):
                s = e[short]
                print(
                    f"      {short:>4}: {s['distinct_brands']} brands, "
                    f"top={s['top_brand']!r} x{s['top_brand_count']}, "
                    f"{s['distinct_titles']} distinct titles -> {s['brands']}",
                    flush=True,
                )

    print("\n" + "-" * 74, flush=True)
    print("BASE-NEW COUNTER-EXAMPLES (biggest categorical wins for base)", flush=True)
    for e in examples["base_biggest_categorical_wins"][:5]:
        print(
            f"\n  q={e['query']!r}  |inter|={e['n_intersection']}  "
            f"junk base {e['base_junk_rate']:.2f} vs bod {e['bod_junk_rate']:.2f}  "
            f"gold rank base {e['base_gold_rank']} bod {e['bod_gold_rank']}",
            flush=True,
        )
        for d in e["base_only_top10"][:3]:
            print(
                f"      base-only #{d['rank']:2d} p={d['p_yes']:.2f} {d['title'][:66]}", flush=True
            )
        for d in e["bod_only_top10"][:3]:
            print(
                f"      bod-only  #{d['rank']:2d} p={d['p_yes']:.2f} {d['title'][:66]}", flush=True
            )

    print(f"\nsaved -> {out_path}", flush=True)


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--phase", required=True, choices=["divergence", "probe", "estimate", "judge", "eval"]
    )
    ap.add_argument("--retrieval", default=DEFAULT_RETRIEVAL)
    ap.add_argument("--catalog-fields", default=CATALOG_FIELDS)
    ap.add_argument("--probe-queries", nargs="*", default=list(DEFAULT_PROBE_QUERIES))
    ap.add_argument(
        "--min-branded-slots",
        type=int,
        default=8,
        help="branded_heavy cut: min slots with a non-empty manufacturer in both top-10s",
    )
    ap.add_argument("--queries-file", default="holdout_queries.jsonl")
    ap.add_argument("--qrels-file", default="holdout_qrels.jsonl")
    ap.add_argument("--model", default="gpt-4o-mini", help="OpenAI judge model")
    ap.add_argument("--concurrency", type=int, default=32)
    ap.add_argument("--cost-ceiling", type=float, default=COST_CEILING_USD)
    ap.add_argument("--top-k", type=int, default=K_EVAL)
    ap.add_argument(
        "--divergence-max-overlap",
        type=int,
        default=3,
        help="a query is 'divergent' if |base n bod| top-10 intersection is <= this",
    )
    ap.add_argument(
        "--judge-all",
        action="store_true",
        help="guarantee labels for all 250 queries, not just the divergent set",
    )
    ap.add_argument("--max-title-chars", type=int, default=300)
    ap.add_argument("--min-relevance", type=int, default=1)
    ap.add_argument("--exact-relevance", type=int, default=1)
    ap.add_argument("--junk-threshold", type=float, default=0.5)
    ap.add_argument("--n-examples", type=int, default=8)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", action="store_true", default=True)
    ap.add_argument("--no-resume", dest="resume", action="store_false")
    ap.add_argument("--work-dir", default="/tmp/bestbuy_divergence")
    ap.add_argument("--tag", default="bestbuy")
    ap.add_argument("--out", default="evaluation/results/bestbuy_base_vs_bod_divergence.json")
    args = ap.parse_args()

    paths = work_paths(args.work_dir, args.tag)
    {
        "divergence": lambda: phase_divergence(args, paths),
        "probe": lambda: phase_probe(args, paths),
        "estimate": lambda: phase_estimate(args, paths),
        "judge": lambda: phase_judge(args, paths),
        "eval": lambda: phase_eval(args, paths),
    }[args.phase]()


if __name__ == "__main__":
    main()
