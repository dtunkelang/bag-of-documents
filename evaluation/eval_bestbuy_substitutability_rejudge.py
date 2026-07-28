#!/usr/bin/env python3
"""Re-judge the BestBuy substitutability benchmark with gpt-4.1 + prompt v2.1.

Why
---
`eval_bestbuy_substitutability_benchmark.py` labelled 2,504 (anchor, candidate)
pairs with gpt-4o-mini and a short E/S/C/I prompt. An audit
(`eval_bestbuy_substitutability_prompt_v2.py`, gold set
`results/bestbuy_substitutability_audit_v2.json`) found those labels unreliable:
S agreed 89.5% with an independent re-labelling, but E/C/I only 40-60%, and the
dominant error was genuine same-brand substitutes labelled **I**. The prompt
experiment's winning condition (E: gpt-4.1 + prompt "v2.1", i.e. the revised
prompt plus a media/content carve-out) hit 0.78 agreement with gold, fixing 32
original errors with zero regressions.

That matters for the benchmark's headline bias result -- "high lexical overlap
non-substitutes get spuriously high cosine similarity" -- because mislabelling
genuine substitutes as I inflates exactly that effect. This script re-runs ONLY
the judging step with the winning judge, then recomputes the identical
concordance and bias analyses (imported from the benchmark module, not
reimplemented) so the delta attributable to label quality alone is visible.

Everything upstream is reused verbatim from the benchmark's cached artifacts:
the anchor sample, the candidate union, the OLD/NEW cosine similarities and the
BM25 scores. No embeddings and no candidate generation are recomputed.

Phases (cached + resumable, same convention as the benchmark):
    --phase estimate   project OpenAI cost -- spends nothing
    --phase judge      gpt-4.1 + JUDGE_PROMPT_V2_1 over every pair (resumable)
    --phase eval       concordance + bias + v1-vs-v2 comparison -> results JSON

Usage:
    uv run --no-project --python /Users/dtunkelang/job-search/.venv/bin/python \\
        python evaluation/eval_bestbuy_substitutability_rejudge.py --phase estimate
    ... --phase judge / eval
"""

import argparse
import asyncio
import json
import math
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np  # noqa: E402
from dotenv import load_dotenv  # noqa: E402

from evaluation.eval_bestbuy_llm_judge_junkrate import (  # noqa: E402
    Usage,
    _chat,
    estimate_cost,
    make_client,
    record_spend,
)

# The metric code is imported, never re-implemented: any change to the
# benchmark's definition of concordance or of the bias buckets propagates here.
from evaluation.eval_bestbuy_substitutability_benchmark import (  # noqa: E402
    COMPARISONS,
    LABELS,
    _label_from_choice,
    _load_json,
    bias_analysis,
    cat_suffix,
    pair_key,
    paired_bootstrap_delta,
    per_anchor_concordance,
    work_paths,
)
from evaluation.eval_bestbuy_substitutability_prompt_v2 import (  # noqa: E402
    JUDGE_PROMPT_V2_1,
)
from evaluation.eval_esci_llm_judge_lexical_bias import bootstrap_ci  # noqa: E402

load_dotenv(override=True)

# The re-judge is ~7x the tokens of the original mini run (longer prompt, 13x
# the input price). ~$3 projected; refuse anything materially above that.
COST_CEILING_USD = 5.0

V1_RESULTS = "evaluation/results/bestbuy_substitutability_benchmark.json"
DEFAULT_OUT = "evaluation/results/bestbuy_substitutability_benchmark_v2.json"


# --------------------------------------------------------------------------
# prompts
# --------------------------------------------------------------------------
def build_prompts(args, cands):
    """[(key, {...prompt...})] for every (anchor, candidate) pair.

    Identical construction to the benchmark's `build_prompts` (same title
    truncation, same category suffix) -- only the template differs, so the
    labels differ for prompt+model reasons and nothing else.
    """
    items = []
    for r in cands["rows"]:
        a_title = r["title"][: args.max_title_chars]
        a_cat = cat_suffix(r, args.with_category)
        for c in r["candidates"]:
            prompt = JUDGE_PROMPT_V2_1.format(
                a_title=a_title,
                a_cat=a_cat,
                c_title=c["title"][: args.max_title_chars],
                c_cat=cat_suffix(c, args.with_category),
            )
            items.append(
                (
                    pair_key(r["product_id"], c["product_id"]),
                    {
                        "anchor_id": r["product_id"],
                        "candidate_id": c["product_id"],
                        "prompt": prompt,
                    },
                )
            )
    return items


# --------------------------------------------------------------------------
# phase: estimate
# --------------------------------------------------------------------------
def phase_estimate(args, paths, quiet=False):
    cands = _load_json(paths["cands"], "candidates")
    items = build_prompts(args, cands)
    # ~4 chars/token, the same crude ratio the other judge scripts use.
    tin = sum(len(p["prompt"]) for _, p in items) / 4.0
    tout = len(items) * 1.0
    cost = estimate_cost(args.model, tin, tout)
    breakdown = {
        "model": args.model,
        "prompt": "v2.1_media",
        "n_anchors": cands["n_anchors"],
        "n_pairs": len(items),
        "judge_calls": len(items),
        "est_tokens_in": int(tin),
        "est_tokens_out": int(tout),
        "est_cost_usd_total": cost,
        "ceiling_usd": args.cost_ceiling,
    }
    if not quiet:
        print(json.dumps(breakdown, indent=2), flush=True)
        print(
            f"\nPROJECTED COST: ${cost:.4f} ({len(items):,} calls) "
            f"vs ceiling ${args.cost_ceiling:.2f}",
            flush=True,
        )
        if cost > args.cost_ceiling:
            print("OVER CEILING -- the judge phase will refuse to run.", flush=True)
    return breakdown


def _guard_cost(args, paths):
    est = phase_estimate(args, paths, quiet=True)
    c = est["est_cost_usd_total"]
    print(f"[cost guard] projected ${c:.4f} (ceiling ${args.cost_ceiling:.2f})", flush=True)
    if c > args.cost_ceiling:
        raise SystemExit(
            f"Refusing to run: projected ${c:.4f} exceeds ceiling "
            f"${args.cost_ceiling:.2f}. Raise --cost-ceiling deliberately."
        )
    return est


# --------------------------------------------------------------------------
# phase: judge
# --------------------------------------------------------------------------
def _load_judged(path):
    """Cached judgements keyed by pair. Tolerates a truncated final line."""
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
            except json.JSONDecodeError:
                continue
            if r.get("label") in LABELS:
                done[pair_key(r["anchor_id"], r["candidate_id"])] = r
    return done


async def _run_judge(args, todo, usage, out_f):
    client = make_client()
    sem = asyncio.Semaphore(args.concurrency)
    done_n = [0]

    async def one(item):
        _key, p = item
        ch = await _chat(client, sem, usage, args.model, p["prompt"], 4, logprobs=True)
        try:
            label, conf = _label_from_choice(ch)
        except Exception:  # never let one odd token kill a paid run
            usage.errors += 1
            label, conf = None, float("nan")
        if label is None:
            usage.errors += 1
            return
        rec = {
            "anchor_id": p["anchor_id"],
            "candidate_id": p["candidate_id"],
            "label": label,
            "p_label": None if math.isnan(conf) else round(conf, 4),
        }
        out_f.write(json.dumps(rec) + "\n")
        done_n[0] += 1
        if done_n[0] % 250 == 0:
            out_f.flush()
            print(f"    {done_n[0]:,}/{len(todo):,} judged", flush=True)

    await asyncio.gather(*(one(it) for it in todo))
    out_f.flush()


def phase_judge(args, paths):
    _guard_cost(args, paths)
    cands = _load_json(paths["cands"], "candidates")
    items = build_prompts(args, cands)
    done = _load_judged(paths["judge"]) if args.resume else {}
    todo = [(k, p) for k, p in items if k not in done]
    print(f"{len(items):,} pairs, {len(done):,} cached, {len(todo):,} to judge", flush=True)
    if not todo:
        print("nothing to do", flush=True)
        return

    usage = Usage()
    t0 = time.time()
    mode = "a" if (args.resume and Path(paths["judge"]).exists()) else "w"
    with open(paths["judge"], mode) as f:
        asyncio.run(_run_judge(args, todo, usage, f))
    cost = estimate_cost(args.model, usage.tin, usage.tout)
    record_spend(
        args.model,
        usage.tin,
        usage.tout,
        cost,
        "bestbuy substitutability re-judge (gpt-4.1 + prompt v2.1)",
    )
    meta = {
        "model": args.model,
        "prompt": "v2.1_media",
        "n_pairs_total": len(items),
        "n_judged_now": usage.calls,
        "errors": usage.errors,
        "tokens_in": usage.tin,
        "tokens_out": usage.tout,
        "cost_usd": cost,
        "seconds": time.time() - t0,
    }
    with open(paths["judge_meta"], "w") as f:
        json.dump(meta, f, indent=2)
    print(json.dumps(meta, indent=2), flush=True)


# --------------------------------------------------------------------------
# phase: eval
# --------------------------------------------------------------------------
def _concordance_block(rows, labels_by_pair, args):
    per, results = {}, {}
    for tag, sim_key in (("old", "sim_old"), ("new", "sim_new")):
        per[tag], pooled = per_anchor_concordance(rows, labels_by_pair, sim_key)
        block = {}
        for name, _, _ in COMPARISONS:
            vals = list(per[tag][name].values())
            block[name] = {
                "n_anchors_qualifying": len(vals),
                "macro_mean": float(np.mean(vals)) if vals else None,
                "macro_ci95": bootstrap_ci(vals, args.n_boot, args.seed),
                "micro_pooled": (pooled[name][0] / pooled[name][1]) if pooled[name][1] else None,
                "n_pairs": pooled[name][1],
            }
        results[tag] = block
    results["delta_new_minus_old"] = {
        name: paired_bootstrap_delta(per["old"][name], per["new"][name], args.n_boot, args.seed)
        for name, _, _ in COMPARISONS
    }
    return results


def _label_flow(v1_labels, v2_labels):
    """v1 -> v2 transition counts plus the headline agreement rate."""
    keys = [k for k in v2_labels if k in v1_labels]
    flow = Counter((v1_labels[k], v2_labels[k]) for k in keys)
    agree = sum(v for (a, b), v in flow.items() if a == b)
    return {
        "n_compared": len(keys),
        "agreement": agree / len(keys) if keys else None,
        "transitions": {f"{a}->{b}": v for (a, b), v in sorted(flow.items())},
    }


def _bias_deltas(v1_bias, v2_bias):
    """The numbers that decide how much of the bias finding was a label artifact."""
    out = {}
    keys = [
        "spearman_sim_vs_gain_old",
        "spearman_sim_vs_gain_new",
        "pearson_coverage_vs_z_old",
        "pearson_coverage_vs_z_new",
        "spearman_coverage_vs_gain",
        "within_CI_pearson_coverage_vs_z_old",
        "within_CI_pearson_coverage_vs_z_new",
    ]
    for k in keys:
        a, b = v1_bias.get(k), v2_bias.get(k)
        out[k] = {
            "v1": a,
            "v2": b,
            "delta": (b - a) if (a is not None and b is not None) else None,
        }
    for bucket in ("CI_high_overlap", "CI_low_overlap", "S_high_overlap", "S_low_overlap", "E_all"):
        a, b = v1_bias["buckets"].get(bucket, {}), v2_bias["buckets"].get(bucket, {})
        out[f"bucket_{bucket}"] = {
            "n": {"v1": a.get("n"), "v2": b.get("n")},
            "mean_z_old": {"v1": a.get("mean_z_old"), "v2": b.get("mean_z_old")},
            "mean_z_new": {"v1": a.get("mean_z_new"), "v2": b.get("mean_z_new")},
        }
    for tag in ("old", "new"):
        for bucket in ("high_overlap", "low_overlap"):
            k = f"nonsub_beats_substitute_{bucket}_{tag}"
            a, b = v1_bias.get(k, {}), v2_bias.get(k, {})
            out[k] = {
                "v1": a.get("rate"),
                "v2": b.get("rate"),
                "delta": (b.get("rate") - a.get("rate"))
                if (a.get("rate") is not None and b.get("rate") is not None)
                else None,
                "n_pairs": {"v1": a.get("n_pairs"), "v2": b.get("n_pairs")},
            }
        k = f"nonsub_beats_substitute_matched_{tag}"
        a, b = v1_bias.get(k, {}), v2_bias.get(k, {})
        out[k] = {
            "v1_gap": a.get("gap_high_minus_low"),
            "v2_gap": b.get("gap_high_minus_low"),
            "delta_gap": (b.get("gap_high_minus_low") - a.get("gap_high_minus_low"))
            if (a.get("gap_high_minus_low") is not None and b.get("gap_high_minus_low") is not None)
            else None,
            "n_anchors": {"v1": a.get("n_anchors"), "v2": b.get("n_anchors")},
        }
    return out


def _label_controlled_bias(raw, high_overlap):
    """The bias head-to-head under v1 vs v2 labels on the SAME anchors.

    Relabelling changes which anchors qualify for the head-to-head (an anchor
    needs >=1 S and >=1 high- and low-overlap non-substitute), so the raw v1-vs-v2
    comparison confounds "the effect shrank" with "a different set of anchors is
    being averaged". This restricts both label sets to the anchors that qualify
    under BOTH, which is the only like-for-like reading of the delta.
    """

    def hi(p):
        return bool(p.get("shared_brand")) or (p.get("coverage") or 0.0) >= high_overlap

    by_anchor = defaultdict(list)
    for p in raw:
        if p.get("label_v1") is None or "coverage" not in p:
            continue
        by_anchor[p["anchor_id"]].append(p)

    def matched(labkey, tag, anchors=None):
        acc = {"high_overlap": [0.0, 0], "low_overlap": [0.0, 0]}
        seen = set()
        for aid, g in by_anchor.items():
            if anchors is not None and aid not in anchors:
                continue
            subs = [x for x in g if x[labkey] == "S"]
            jhi = [x for x in g if x[labkey] in ("C", "I") and hi(x)]
            jlo = [x for x in g if x[labkey] in ("C", "I") and not hi(x)]
            if not subs or not jhi or not jlo:
                continue
            seen.add(aid)
            for bucket, junk in (("high_overlap", jhi), ("low_overlap", jlo)):
                for s in subs:
                    for j in junk:
                        acc[bucket][1] += 1
                        if j[f"sim_{tag}"] > s[f"sim_{tag}"]:
                            acc[bucket][0] += 1.0
                        elif j[f"sim_{tag}"] == s[f"sim_{tag}"]:
                            acc[bucket][0] += 0.5
        rates = {b: (v[0] / v[1] if v[1] else None) for b, v in acc.items()}
        gap = rates["high_overlap"] - rates["low_overlap"] if None not in rates.values() else None
        return seen, rates, gap

    out = {}
    for tag in ("old", "new"):
        s1, _, _ = matched("label_v1", tag)
        s2, _, _ = matched("label", tag)
        common = s1 & s2
        _, r1, g1 = matched("label_v1", tag, common)
        _, r2, g2 = matched("label", tag, common)
        out[f"matched_common_anchors_{tag}"] = {
            "n_anchors_common": len(common),
            "n_anchors_v1_only_qualifying": len(s1),
            "n_anchors_v2_only_qualifying": len(s2),
            "v1": {**r1, "gap_high_minus_low": g1},
            "v2": {**r2, "gap_high_minus_low": g2},
            "delta_gap": (g2 - g1) if (g1 is not None and g2 is not None) else None,
        }
    return out


def phase_eval(args, paths):
    cands = _load_json(paths["cands"], "candidates")
    judged = _load_judged(paths["judge"])
    if not judged:
        raise SystemExit(f"no judgements in {paths['judge']} -- run --phase judge")
    labels_by_pair = {k: v["label"] for k, v in judged.items()}
    print(f"{len(labels_by_pair):,} judged pairs, {cands['n_anchors']} anchors", flush=True)

    rows = cands["rows"]
    results = _concordance_block(rows, labels_by_pair, args)
    bias, pair_recs = bias_analysis(rows, labels_by_pair, args)

    src_labels = defaultdict(Counter)
    for r in rows:
        for c in r["candidates"]:
            lab = labels_by_pair.get(pair_key(r["product_id"], c["product_id"]))
            if not lab:
                continue
            for s in c["sources"]:
                src_labels[s][lab] += 1
            src_labels["|".join(c["sources"])][lab] += 1
    source_label_mix = {k: dict(v) for k, v in sorted(src_labels.items())}

    v1 = _load_json(args.v1_results, "v1 benchmark results")
    v1_labels = {pair_key(p["anchor_id"], p["candidate_id"]): p["label"] for p in v1["raw_pairs"]}

    raw = []
    for r in rows:
        for c in r["candidates"]:
            key = pair_key(r["product_id"], c["product_id"])
            j = judged.get(key)
            if not j:
                continue
            raw.append(
                {
                    "anchor_id": r["product_id"],
                    "anchor_title": r["title"],
                    "anchor_class": r["class"],
                    "anchor_manufacturer": r["manufacturer"],
                    "candidate_id": c["product_id"],
                    "candidate_title": c["title"],
                    "candidate_class": c["class"],
                    "candidate_manufacturer": c["manufacturer"],
                    "label": j["label"],
                    "label_v1": v1_labels.get(key),
                    "p_label": j.get("p_label"),
                    "sources": c["sources"],
                    "sim_old": c["sim_old"],
                    "sim_new": c["sim_new"],
                    "bm25_score": c["bm25_score"],
                }
            )
    cov = {p["anchor_id"] + "\t" + p["candidate_id"]: p for p in pair_recs}
    for p in raw:
        f = cov.get(p["anchor_id"] + "\t" + p["candidate_id"])
        if f:
            p["coverage"] = round(f["coverage"], 4)
            p["jaccard"] = round(f["jaccard"], 4)
            p["shared_brand"] = f["shared_brand"]
            p["same_class"] = f["same_class"]

    anchors_meta = _load_json(paths["anchors"], "anchors")
    jmeta = {}
    if Path(paths["judge_meta"]).exists():
        jmeta = _load_json(paths["judge_meta"], "judge meta")

    label_dist = dict(Counter(labels_by_pair.values()))
    comparison = {
        "v1_source": str(args.v1_results),
        "v1_judge": {"model": v1["config"]["judge_model"], "prompt": "v1_original"},
        "v2_judge": {"model": args.model, "prompt": "v2.1_media"},
        "label_distribution": {
            lab: {
                "v1": v1["label_distribution"].get(lab, 0),
                "v2": label_dist.get(lab, 0),
                "delta": label_dist.get(lab, 0) - v1["label_distribution"].get(lab, 0),
            }
            for lab in LABELS
        },
        "label_flow_v1_to_v2": _label_flow(v1_labels, labels_by_pair),
        "concordance": {
            tag: {
                name: {
                    "v1_macro": v1["concordance"][tag][name]["macro_mean"],
                    "v2_macro": results[tag][name]["macro_mean"],
                    "delta": results[tag][name]["macro_mean"]
                    - v1["concordance"][tag][name]["macro_mean"],
                    "v1_ci95": v1["concordance"][tag][name]["macro_ci95"],
                    "v2_ci95": results[tag][name]["macro_ci95"],
                }
                for name, _, _ in COMPARISONS
            }
            for tag in ("old", "new")
        },
        "lexical_bias": _bias_deltas(v1["lexical_bias"], bias),
        "lexical_bias_anchor_matched": _label_controlled_bias(raw, args.high_overlap),
    }

    out = {
        "benchmark": (
            "bestbuy product-product substitutability geometry (E>S>{C,I}) "
            "-- re-judged with gpt-4.1 + prompt v2.1"
        ),
        "config": {
            "judge_model": args.model,
            "judge_prompt": "v2.1_media (eval_bestbuy_substitutability_prompt_v2.JUDGE_PROMPT_V2_1)",
            "reused_from_v1": [
                "anchor sample",
                "candidate union (dense_old / dense_new / bm25)",
                "sim_old",
                "sim_new",
                "bm25_score",
            ],
            "old_dir": cands["old_dir"],
            "new_dir": cands["new_dir"],
            "top_k_per_source": cands["top_k"],
            "candidate_sources": ["dense_old", "dense_new", "bm25"],
            "high_overlap_coverage_threshold": args.high_overlap,
            "n_boot": args.n_boot,
            "seed": args.seed,
            "with_category_in_prompt": args.with_category,
        },
        "anchors": {
            "n": anchors_meta["n_anchors"],
            "class_counts": anchors_meta["class_counts"],
            "stratum_counts": anchors_meta["stratum_counts"],
        },
        "candidates": {
            "n_pairs": cands["n_pairs"],
            "mean_per_anchor": cands["mean_candidates_per_anchor"],
            "source_counts": cands["source_counts"],
            "n_judged": len(judged),
        },
        "judge_cost": jmeta,
        "label_distribution": label_dist,
        "source_label_mix": source_label_mix,
        "concordance": results,
        "lexical_bias": bias,
        "comparison_vs_v1": comparison,
        "prompt_v2_1": JUDGE_PROMPT_V2_1,
        "raw_pairs": raw,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)

    _print_report(v1, results, bias, comparison)
    print(f"\nwrote {out_path}", flush=True)


def _print_report(v1, results, bias, comparison):
    print("\n=== label distribution (v1 gpt-4o-mini -> v2 gpt-4.1/v2.1) ===", flush=True)
    for lab in LABELS:
        d = comparison["label_distribution"][lab]
        print(f"  {lab}  {d['v1']:>5} -> {d['v2']:>5}  ({d['delta']:+d})", flush=True)
    lf = comparison["label_flow_v1_to_v2"]
    print(f"  agreement v1 vs v2: {lf['agreement']:.4f} over {lf['n_compared']:,} pairs")
    big = sorted(
        ((k, v) for k, v in lf["transitions"].items() if k[0] != k[-1]),
        key=lambda kv: -kv[1],
    )[:6]
    print("  biggest flips: " + ", ".join(f"{k} {v}" for k, v in big), flush=True)

    print("\n=== concordance (macro over qualifying anchors) ===", flush=True)
    for tag in ("old", "new"):
        for name, _, _ in COMPARISONS:
            c = comparison["concordance"][tag][name]
            print(
                f"  {tag.upper():<4} {name:<9} v1 {c['v1_macro']:.4f} {c['v1_ci95']}  "
                f"v2 {c['v2_macro']:.4f} {c['v2_ci95']}  delta {c['delta']:+.4f}",
                flush=True,
            )
    print("  NEW-minus-OLD deltas under the v2 labels:", flush=True)
    for name, _, _ in COMPARISONS:
        d = results["delta_new_minus_old"][name]
        print(f"    {name:<9} {d['delta']:+.4f} {d['ci95']}  p>0 {d['p_gt_0']:.3f}", flush=True)

    print("\n=== lexical bias: buckets (mean z of cosine within anchor) ===", flush=True)
    for k in ("CI_high_overlap", "CI_low_overlap", "S_high_overlap", "S_low_overlap", "E_all"):
        b = bias["buckets"][k]
        a = v1["lexical_bias"]["buckets"][k]
        if not b.get("n"):
            continue
        print(
            f"  {k:<17} n {a['n']:>5}->{b['n']:<5} "
            f"z_OLD {a['mean_z_old']:+.3f}->{b['mean_z_old']:+.3f}  "
            f"z_NEW {a['mean_z_new']:+.3f}->{b['mean_z_new']:+.3f}",
            flush=True,
        )
    print("\n=== lexical bias: non-substitute outranks a true substitute ===", flush=True)
    for tag in ("old", "new"):
        for bucket in ("high_overlap", "low_overlap"):
            d = comparison["lexical_bias"][f"nonsub_beats_substitute_{bucket}_{tag}"]
            if d["v2"] is None:
                continue
            print(
                f"  {tag.upper():<4} {bucket:<13} v1 {d['v1']:.4f} -> v2 {d['v2']:.4f} "
                f"({d['delta']:+.4f})  n_pairs {d['n_pairs']['v1']:,}->{d['n_pairs']['v2']:,}",
                flush=True,
            )
        m = comparison["lexical_bias"][f"nonsub_beats_substitute_matched_{tag}"]
        if m["v2_gap"] is not None:
            print(
                f"  {tag.upper():<4} matched-anchor gap (high-low): "
                f"v1 {m['v1_gap']:+.4f} -> v2 {m['v2_gap']:+.4f} ({m['delta_gap']:+.4f}), "
                f"anchors {m['n_anchors']['v1']}->{m['n_anchors']['v2']}",
                flush=True,
            )
    print("\n=== same head-to-head, restricted to anchors qualifying under BOTH ===", flush=True)
    for tag in ("old", "new"):
        m = comparison["lexical_bias_anchor_matched"][f"matched_common_anchors_{tag}"]
        if m["v2"]["gap_high_minus_low"] is None:
            continue
        print(
            f"  {tag.upper():<4} n_anchors {m['n_anchors_common']} "
            f"(v1 qualifying {m['n_anchors_v1_only_qualifying']}, "
            f"v2 {m['n_anchors_v2_only_qualifying']})  "
            f"gap v1 {m['v1']['gap_high_minus_low']:+.4f} -> v2 "
            f"{m['v2']['gap_high_minus_low']:+.4f} ({m['delta_gap']:+.4f})",
            flush=True,
        )
    for k in (
        "within_CI_pearson_coverage_vs_z_old",
        "within_CI_pearson_coverage_vs_z_new",
        "pearson_coverage_vs_z_old",
        "pearson_coverage_vs_z_new",
        "spearman_sim_vs_gain_old",
        "spearman_sim_vs_gain_new",
    ):
        d = comparison["lexical_bias"][k]
        print(f"  {k:<38} v1 {d['v1']:+.4f} -> v2 {d['v2']:+.4f} ({d['delta']:+.4f})", flush=True)


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--phase", required=True, choices=["estimate", "judge", "eval"])
    ap.add_argument("--model", default="gpt-4.1", help="OpenAI judge model")
    ap.add_argument("--concurrency", type=int, default=32)
    ap.add_argument("--max-title-chars", type=int, default=200)
    ap.add_argument("--with-category", action="store_true", default=True)
    ap.add_argument("--no-category", dest="with_category", action="store_false")
    ap.add_argument("--cost-ceiling", type=float, default=COST_CEILING_USD)
    ap.add_argument("--high-overlap", type=float, default=0.5)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", action="store_true", default=True)
    ap.add_argument("--no-resume", dest="resume", action="store_false")
    ap.add_argument("--work-dir", default="/tmp/bestbuy_substitutability")
    ap.add_argument("--tag", default="bestbuy", help="tag of the CACHED upstream artifacts")
    ap.add_argument(
        "--judge-tag",
        default="bestbuy_v21_gpt41",
        help="tag for THIS run's judgements -- must differ from --tag so the "
        "original gpt-4o-mini labels are not overwritten",
    )
    ap.add_argument("--v1-results", default=V1_RESULTS)
    ap.add_argument("--out", default=DEFAULT_OUT)
    args = ap.parse_args()

    # Upstream artifacts come from the benchmark's tag; judgements are written
    # under a separate tag so the v1 labels stay intact for the comparison.
    paths = work_paths(args.work_dir, args.tag)
    jpaths = work_paths(args.work_dir, args.judge_tag)
    paths["judge"] = jpaths["judge"]
    paths["judge_meta"] = jpaths["judge_meta"]

    {
        "estimate": lambda: phase_estimate(args, paths),
        "judge": lambda: phase_judge(args, paths),
        "eval": lambda: phase_eval(args, paths),
    }[args.phase]()


if __name__ == "__main__":
    main()
