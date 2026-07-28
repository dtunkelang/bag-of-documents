#!/usr/bin/env python3
"""Validate a revised E/S/C/I judge prompt against a fresh gold set.

Background
----------
`eval_bestbuy_substitutability_benchmark.py` labelled 2,504 (anchor, candidate)
BestBuy pairs with gpt-4o-mini using a short E/S/C/I prompt. A manual audit found
two systematic failures:

1.  Same-brand, same-category products that are genuine alternatives (two
    17" Samsung monitors, two Panasonic camcorders, a ThinkPad tablet vs a
    ThinkPad notebook) get labelled **I** instead of **S** -- the judge conflates
    "not identical" with "not a valid substitute".
2.  A shared brand token alone manufactures a false **C** (Sony PlayStation2 vs
    a Sony CLIE keyboard) even when the two products cannot be used together.

This script tests a revised prompt that names both failure modes explicitly and
adds boundary-case few-shots, against a fresh, persisted gold set
(`results/bestbuy_substitutability_audit_v2.json`).

Conditions
----------
A  baseline   gpt-4o-mini + ORIGINAL prompt   (labels reused from the benchmark)
B  revised    gpt-4o-mini + REVISED prompt    (new calls)
C  stronger   gpt-4.1     + REVISED prompt    (new calls)
   extra models can be added with --extra-model.

Usage
-----
    python eval_bestbuy_substitutability_prompt_v2.py --phase estimate
    python eval_bestbuy_substitutability_prompt_v2.py --phase run
    python eval_bestbuy_substitutability_prompt_v2.py --phase report
"""

from __future__ import annotations

import argparse
import asyncio
import collections
import json
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
GOLD_PATH = HERE / "results" / "bestbuy_substitutability_audit_v2.json"
OUT_PATH = HERE / "results" / "bestbuy_substitutability_prompt_v2.json"
RAW_PATH = Path("/tmp/bestbuy_subst_prompt_v2_raw.jsonl")
CANDS_PATH = Path("/tmp/bestbuy_substitutability/candidates_bestbuy.json")

LABELS = ("E", "S", "C", "I")
COST_CEILING_USD = 1.00

PRICES_PER_M_TOKENS = {
    "gpt-4o-mini": {"in": 0.15, "out": 0.60},
    "gpt-4o": {"in": 2.50, "out": 10.00},
    "gpt-4.1": {"in": 2.00, "out": 8.00},
    "gpt-4.1-mini": {"in": 0.40, "out": 1.60},
    "gpt-5-mini": {"in": 0.25, "out": 2.00},
    "gpt-5.4-mini": {"in": 0.25, "out": 2.00},
}

# --------------------------------------------------------------------------
# prompts
# --------------------------------------------------------------------------
# verbatim from eval_bestbuy_substitutability_benchmark.py (condition A)
JUDGE_PROMPT_V1 = """Anchor product: {a_title}{a_cat}
Candidate product: {c_title}{c_cat}

Label the candidate's relationship to the anchor with ONE letter:
E = same core product, differing only in a trivial variant (color, storage \
size, bundle, refurbished) - a buyer would treat them as interchangeable
S = a different product that could reasonably serve the SAME buying need as \
an alternative (different brand or generation of the same product type)
C = used WITH the anchor, not instead of it (accessory, case, charger, cable, \
mount, warranty, add-on)
I = different category or use case; neither a reasonable alternative nor a \
complement

An accessory for the anchor is always C, never S, even if it names the same \
brand. Answer with one letter only."""

# condition B / C
JUDGE_PROMPT_V2 = """Anchor product: {a_title}{a_cat}
Candidate product: {c_title}{c_cat}

Label the candidate's relationship to the anchor with ONE letter:

E = same core product, differing only in a trivial variant (color, storage \
size, bundle, refurbished, connectivity option) - a buyer would treat them as \
interchangeable.
S = a DIFFERENT product that serves the SAME buying need, i.e. a genuine \
alternative the buyer would weigh against the anchor.
C = used WITH the anchor, not instead of it: an accessory, case, charger, \
cable, mount, dock, warranty or add-on that actually fits or works with THIS \
anchor product.
I = different category or use case; neither a reasonable alternative nor a \
usable complement.

Two rules that are easy to get wrong:

1. "Different model" does NOT mean Irrelevant. Two products of the same type \
that a buyer would genuinely cross-shop are S even when they are different \
models, sizes, capacities, generations, or trim levels, and even when they \
come from the same brand. Two different-sized LCD monitors from the same \
brand are S, not I. Two camcorders from one product line are S, not I. A \
tablet and a notebook from the same laptop family are S, not I. Reserve I for \
products that solve a genuinely different problem.

2. A shared brand or manufacturer NEVER by itself makes something a \
Complement. Ask: can this candidate actually be attached to, installed on, or \
used together with THIS specific anchor? If the accessory is scoped to a \
different model or product line, the answer is no, and the label is I, not C. \
An accessory that does fit the anchor is always C, never S.

Examples:
Anchor: Samsung - 17" LCD Flat-Panel Monitor / Candidate: Samsung - 19" \
Widescreen LCD Monitor -> S
Anchor: Sony - PlayStation2 / Candidate: Sony - Battery Adapter for Sony CLIE \
-> I
Anchor: Apple - iPad 2 with Wi-Fi - 32GB / Candidate: Belkin - Folio Case for \
Apple iPad 2 -> C

Answer with one letter only."""

# condition D: v2 plus a media-title carve-out. Rule 1 in v2 over-generalises to
# content goods -- two unrelated CDs or two unrelated Xbox titles came back S.
_MEDIA_CLAUSE = """
3. Media and content goods (music CDs, movies on DVD/VHS/Blu-ray, books, video \
game titles) are bought for their specific content, not for a product spec. \
Two of them are S only when they are the same title, the same \
series/franchise, or another edition of the same content. Two unrelated titles \
that merely share a format, platform, or genre are I, not S. Rule 1 above is \
about hardware specs and does not license calling unrelated titles substitutes.
"""

JUDGE_PROMPT_V2_1 = JUDGE_PROMPT_V2.replace("\nExamples:", _MEDIA_CLAUSE + "\nExamples:").replace(
    "Anchor: Apple - iPad 2 with Wi-Fi - 32GB / Candidate: Belkin - Folio Case for \
Apple iPad 2 -> C",
    "Anchor: Apple - iPad 2 with Wi-Fi - 32GB / Candidate: Belkin - Folio Case for \
Apple iPad 2 -> C\nAnchor: Sonic Generations - Xbox 360 / Candidate: F.E.A.R. 3 \
- Xbox 360 -> I",
)


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def estimate_cost(model: str, tokens_in: float, tokens_out: float) -> float:
    p = PRICES_PER_M_TOKENS.get(model)
    if p is None:
        return 0.0
    return tokens_in * p["in"] / 1e6 + tokens_out * p["out"] / 1e6


def load_gold():
    with open(GOLD_PATH) as f:
        return json.load(f)["pairs"]


def category_lookup():
    """product_id -> (category_leaf, class) from the benchmark candidate file.

    The benchmark ran with --with-category, so the judge saw a category suffix.
    Reproduce it exactly so conditions B/C differ from A only in the prompt.
    """
    if not CANDS_PATH.exists():
        raise SystemExit(
            f"missing {CANDS_PATH}: rerun the benchmark's candidate phase, or "
            "fall back to the class field stored in the gold set."
        )
    with open(CANDS_PATH) as f:
        cands = json.load(f)
    out = {}
    for row in cands["rows"]:
        out[row["product_id"]] = (row.get("category_leaf", ""), row.get("class", ""))
        for c in row["candidates"]:
            out[c["product_id"]] = (c.get("category_leaf", ""), c.get("class", ""))
    return out


def cat_suffix(pid, lut):
    leaf, cls = lut.get(pid, ("", ""))
    bits = [b for b in (leaf, cls) if b]
    return f" ({' / '.join(bits)})" if bits else ""


def build_prompt(pair, lut, template, max_title_chars=200):
    return template.format(
        a_title=pair["anchor_title"][:max_title_chars],
        a_cat=cat_suffix(pair["anchor_id"], lut),
        c_title=pair["candidate_title"][:max_title_chars],
        c_cat=cat_suffix(pair["candidate_id"], lut),
    )


def _label_from_text(text):
    for ch in (text or "").strip().upper():
        if ch in LABELS:
            return ch
    return None


# --------------------------------------------------------------------------
# API
# --------------------------------------------------------------------------
def make_client():
    from openai import AsyncOpenAI

    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY is not set")
    return AsyncOpenAI()


class Usage:
    def __init__(self):
        self.tin = 0
        self.tout = 0
        self.errors = 0


async def _one(client, sem, usage, model, prompt):
    kwargs = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
    }
    if model.startswith(("gpt-5", "o3", "o4")):
        kwargs["max_completion_tokens"] = 2048
        kwargs["reasoning_effort"] = "low"
    else:
        kwargs["max_tokens"] = 4
        kwargs["temperature"] = 0.0
    async with sem:
        for attempt in range(4):
            try:
                r = await client.chat.completions.create(**kwargs)
                break
            except Exception:
                if attempt == 3:
                    usage.errors += 1
                    return None
                await asyncio.sleep(2**attempt)
    if r.usage:
        usage.tin += r.usage.prompt_tokens
        usage.tout += r.usage.completion_tokens
    return _label_from_text(r.choices[0].message.content)


async def run_condition(model, template_name, template, pairs, lut, concurrency):
    client = make_client()
    sem = asyncio.Semaphore(concurrency)
    usage = Usage()
    prompts = [build_prompt(p, lut, template) for p in pairs]
    labels = await asyncio.gather(*(_one(client, sem, usage, model, pr) for pr in prompts))
    cost = estimate_cost(model, usage.tin, usage.tout)
    print(
        f"  {model} / {template_name}: "
        f"in={usage.tin:,} out={usage.tout:,} errors={usage.errors} "
        f"cost=${cost:.4f}",
        flush=True,
    )
    return labels, {
        "model": model,
        "prompt": template_name,
        "tokens_in": usage.tin,
        "tokens_out": usage.tout,
        "errors": usage.errors,
        "cost_usd": round(cost, 5),
    }


# --------------------------------------------------------------------------
# phases
# --------------------------------------------------------------------------
def phase_estimate(args):
    pairs = load_gold()
    lut = category_lookup()
    tin = sum(len(build_prompt(p, lut, JUDGE_PROMPT_V2)) for p in pairs) / 4.0
    total = 0.0
    for model in [args.model_b, args.model_c] + list(args.extra_model or []):
        c = estimate_cost(model, tin, len(pairs) * 1.0)
        total += c
        print(f"  {model:16s} ~{int(tin):,} tok in -> ${c:.4f}")
    print(f"  TOTAL ~${total:.4f} (ceiling ${COST_CEILING_USD:.2f})")
    if total > COST_CEILING_USD:
        raise SystemExit("estimated cost exceeds ceiling")
    return total


def phase_run(args):
    total = phase_estimate(args)
    if total > COST_CEILING_USD:
        raise SystemExit("cost ceiling")
    pairs = load_gold()
    lut = category_lookup()

    conds = [
        ("B_revised_mini", args.model_b, "v2_revised", JUDGE_PROMPT_V2),
        ("C_revised_strong", args.model_c, "v2_revised", JUDGE_PROMPT_V2),
    ]
    for m in args.extra_model or []:
        conds.append((f"X_revised_{m}", m, "v2_revised", JUDGE_PROMPT_V2))
    if args.with_v21:
        conds.append(("D_v21_mini", args.model_b, "v2.1_media", JUDGE_PROMPT_V2_1))
        conds.append(("E_v21_strong", args.model_c, "v2.1_media", JUDGE_PROMPT_V2_1))

    results, meta = {}, {}
    for key, model, tname, tmpl in conds:
        print(f"running {key} ...", flush=True)
        labels, m = asyncio.run(run_condition(model, tname, tmpl, pairs, lut, args.concurrency))
        results[key] = labels
        meta[key] = m

    with open(RAW_PATH, "w") as f:
        json.dump({"results": results, "meta": meta}, f)
    print(f"wrote {RAW_PATH}")


def _confusion(pairs, preds):
    cm = collections.Counter()
    for p, q in zip(pairs, preds):
        cm[(p["gold_label"], q)] += 1
    return {f"{g}->{q}": n for (g, q), n in sorted(cm.items())}


def _agreement(pairs, preds, subset=None):
    idx = range(len(pairs)) if subset is None else subset
    idx = [i for i in idx if preds[i] is not None]
    if not idx:
        return 0.0, 0
    n = sum(1 for i in idx if pairs[i]["gold_label"] == preds[i])
    return n / len(idx), len(idx)


def phase_report(args):
    pairs = load_gold()
    with open(RAW_PATH) as f:
        raw = json.load(f)
    results = {"A_baseline_v1": [p["baseline_label"] for p in pairs]}
    results.update(raw["results"])

    report = {
        "benchmark": "bestbuy substitutability judge prompt v2 validation",
        "gold_set": str(GOLD_PATH),
        "n_pairs": len(pairs),
        "gold_distribution": dict(collections.Counter(p["gold_label"] for p in pairs)),
        "conditions": {
            "A_baseline_v1": {
                "model": "gpt-4o-mini",
                "prompt": "v1_original",
                "cost_usd": 0.0,
                "note": "labels reused from the original benchmark",
            },
            **raw["meta"],
        },
        "prompt_v1": JUDGE_PROMPT_V1,
        "prompt_v2": JUDGE_PROMPT_V2,
        "prompt_v2_1": JUDGE_PROMPT_V2_1,
    }

    strata = collections.defaultdict(list)
    for i, p in enumerate(pairs):
        strata[p["stratum"]].append(i)
    by_gold = collections.defaultdict(list)
    for i, p in enumerate(pairs):
        by_gold[p["gold_label"]].append(i)
    by_baseline = collections.defaultdict(list)
    for i, p in enumerate(pairs):
        by_baseline[p["baseline_label"]].append(i)

    # the two named error patterns
    err_sub = [
        i for i, p in enumerate(pairs) if p["gold_label"] == "S" and p["baseline_label"] == "I"
    ]
    err_falsec = [
        i for i, p in enumerate(pairs) if p["gold_label"] == "I" and p["baseline_label"] == "C"
    ]

    per_cond = {}
    base = results["A_baseline_v1"]
    for key, preds in results.items():
        acc, n = _agreement(pairs, preds)
        entry = {
            "overall_agreement": round(acc, 4),
            "n_scored": n,
            "by_stratum": {
                s: round(_agreement(pairs, preds, idx)[0], 4) for s, idx in sorted(strata.items())
            },
            "by_gold_label": {
                g: round(_agreement(pairs, preds, idx)[0], 4) for g, idx in sorted(by_gold.items())
            },
            "by_baseline_label": {
                g: round(_agreement(pairs, preds, idx)[0], 4)
                for g, idx in sorted(by_baseline.items())
            },
            "confusion_gold_to_pred": _confusion(pairs, preds),
            "error_pattern_1_gold_S_baseline_I": {
                "n": len(err_sub),
                "recovered_to_S": sum(1 for i in err_sub if preds[i] == "S"),
            },
            "error_pattern_2_gold_I_baseline_C": {
                "n": len(err_falsec),
                "recovered_to_I": sum(1 for i in err_falsec if preds[i] == "I"),
            },
        }
        if key != "A_baseline_v1":
            bc_rc = bc_rw = bw_rc = bw_rw = 0
            regressions, fixes = [], []
            for i, p in enumerate(pairs):
                if preds[i] is None:
                    continue
                b_ok = base[i] == p["gold_label"]
                r_ok = preds[i] == p["gold_label"]
                if b_ok and r_ok:
                    bc_rc += 1
                elif b_ok and not r_ok:
                    bc_rw += 1
                    regressions.append(
                        {
                            "idx": p["idx"],
                            "anchor": p["anchor_title"],
                            "candidate": p["candidate_title"],
                            "gold": p["gold_label"],
                            "baseline": base[i],
                            "revised": preds[i],
                        }
                    )
                elif not b_ok and r_ok:
                    bw_rc += 1
                    fixes.append(
                        {
                            "idx": p["idx"],
                            "anchor": p["anchor_title"],
                            "candidate": p["candidate_title"],
                            "gold": p["gold_label"],
                            "baseline": base[i],
                            "revised": preds[i],
                        }
                    )
                else:
                    bw_rw += 1
            entry["vs_baseline"] = {
                "baseline_correct_revised_correct": bc_rc,
                "baseline_correct_revised_wrong_REGRESSIONS": bc_rw,
                "baseline_wrong_revised_correct_FIXES": bw_rc,
                "baseline_wrong_revised_wrong": bw_rw,
                "net_gain": bw_rc - bc_rw,
            }
            entry["regression_examples"] = regressions[:12]
            entry["fix_examples"] = fixes[:12]
        per_cond[key] = entry

    report["results"] = per_cond
    with open(OUT_PATH, "w") as f:
        json.dump(report, f, indent=1)

    print(
        f"\n{'condition':22s} {'overall':>8s} {'errP1':>8s} {'errP2':>8s} "
        f"{'fixes':>6s} {'regr':>6s} {'net':>5s}"
    )
    for key, e in per_cond.items():
        v = e.get("vs_baseline", {})
        p1 = e["error_pattern_1_gold_S_baseline_I"]
        p2 = e["error_pattern_2_gold_I_baseline_C"]
        print(
            f"{key:22s} {e['overall_agreement']:8.3f} "
            f"{p1['recovered_to_S']:>3d}/{p1['n']:<4d} "
            f"{p2['recovered_to_I']:>3d}/{p2['n']:<4d} "
            f"{v.get('baseline_wrong_revised_correct_FIXES', 0):6d} "
            f"{v.get('baseline_correct_revised_wrong_REGRESSIONS', 0):6d} "
            f"{v.get('net_gain', 0):5d}"
        )
    print(f"\nwrote {OUT_PATH}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", required=True, choices=("estimate", "run", "report"))
    ap.add_argument("--model-b", default="gpt-4o-mini")
    ap.add_argument("--model-c", default="gpt-4.1")
    ap.add_argument("--extra-model", action="append")
    ap.add_argument("--with-v21", action="store_true")
    ap.add_argument("--concurrency", type=int, default=16)
    args = ap.parse_args()
    {"estimate": phase_estimate, "run": phase_run, "report": phase_report}[args.phase](args)


if __name__ == "__main__":
    main()
