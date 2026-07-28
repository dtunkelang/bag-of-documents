#!/usr/bin/env python3
"""Which individual pairs are behind the BestBuy S > {C,I} concordance failures,
and are they usable as hard-negative training data?

Context
-------
`eval_bestbuy_substitutability_benchmark.py` measures whether embedding cosine
similarity respects a substitutability hierarchy around a product anchor:
Exact > Substitute > {Complement, Irrelevant}. `eval_bestbuy_substitutability_
rejudge.py` re-labelled the same 2,504 (anchor, candidate) pairs with gpt-4.1 +
prompt v2.1 (`..._benchmark_v2.json`); those v2 labels are the trustworthy ones
and are what this script uses. Under them, the NEW (re-indexed) embedding scores
S_gt_CI = 0.757 macro -- i.e. roughly a quarter of the time a non-substitute
sits ABOVE a genuine substitute for the same anchor.

This script stops looking at that aggregate and enumerates the individual
failures behind it.

Framing (per the 3-class collapse)
----------------------------------
Complement is treated identically to Irrelevant. Every candidate is E, S, or
NS (= C or I). A VIOLATION is one NS candidate whose cosine similarity to the
anchor strictly exceeds that of at least one S candidate of the same anchor.
Ties are recorded but do not create a violation (the catalog has exact-duplicate
rows, so ties are real and counting them as failures would be unfair; the
pairwise rate below scores them 0.5 to stay consistent with the benchmark).

Two rates, deliberately kept separate
-------------------------------------
* pairwise discordance = 1 - S_gt_CI concordance, over all (S, NS) pairs. This
  is the number that should reproduce the benchmark's ~24%; the script asserts
  it against the stored value as a self-check.
* candidate-level violation rate = fraction of NS candidates that beat >= 1 S.
  This is the quantity that matters for "how many hard negatives do I get",
  and it is necessarily HIGHER than the pairwise rate: one badly placed NS that
  beats a single weak S counts fully here and only fractionally there.

Characterisation, always against a base rate
--------------------------------------------
"78% of violations share a brand token with the anchor" means nothing on its
own if 78% of ALL candidates do -- the candidate pool is built from dense/BM25
top-k, which is already brand-heavy. So every rate is reported alongside the
same rate over (a) all NS candidates in qualifying anchors -- the correct
denominator, since those are exactly the candidates that COULD have violated --
and (b) all candidates of any label. Over-representation is reported as a
ratio plus a Fisher exact test on the 2x2 (violating vs non-violating NS) x
(feature true vs false).

Same-class vs different-class splits the failure mode: a same-class NS beating
an S is a near-miss variant confusion (the model got the category right and the
substitutability wrong), a different-class one is closer to pure lexical/brand
pull.

OLD vs NEW
----------
The whole characterisation runs on `sim_old` and `sim_new` independently, and
the violation SETS are diffed (shared / new-only / old-only). The aggregate
S_gt_CI improved under re-indexing, but the profile of what still fails is the
interesting part.

Usage
-----
    python evaluation/analyze_bestbuy_substitutability_violations.py
    python evaluation/analyze_bestbuy_substitutability_violations.py \
        --benchmark evaluation/results/bestbuy_substitutability_benchmark_v2.json \
        --out evaluation/results/bestbuy_substitutability_violations.json

Pure post-hoc analysis of already-computed similarities and labels: no API
calls, no encoding, no index access.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict

import numpy as np
from scipy.stats import fisher_exact

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_BENCHMARK = os.path.join(
    REPO, "evaluation", "results", "bestbuy_substitutability_benchmark_v2.json"
)
DEFAULT_OUT = os.path.join(
    REPO, "evaluation", "results", "bestbuy_substitutability_violations.json"
)

NS_LABELS = ("C", "I")
SIM_KEYS = ("sim_new", "sim_old")

# BestBuy BoD MNRL recipe as recorded in evaluation/CHS_RESULTS.md (Pattern 20 /
# the click-volume crossover study): 48,516 bags x 5 triplets = 242,580.
BESTBUY_MNRL_TRIPLETS = 242_580
BESTBUY_MNRL_BAGS = 48_516


# --------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------
def load_pairs(path):
    """raw_pairs from a substitutability benchmark JSON, grouped by anchor."""
    with open(path) as f:
        blob = json.load(f)
    by_anchor = defaultdict(list)
    for p in blob["raw_pairs"]:
        by_anchor[p["anchor_id"]].append(p)
    return blob, dict(by_anchor)


# --------------------------------------------------------------------------
# violation extraction
# --------------------------------------------------------------------------
def anchor_groups(by_anchor):
    """{anchor_id: (subs, nonsubs)} for anchors with >=1 S and >=1 NS."""
    out = {}
    for aid, pairs in by_anchor.items():
        subs = [p for p in pairs if p["label"] == "S"]
        nonsubs = [p for p in pairs if p["label"] in NS_LABELS]
        if subs and nonsubs:
            out[aid] = (subs, nonsubs)
    return out


def extract_violations(groups, sim_key):
    """Every NS candidate that outranks >= 1 S candidate of the same anchor.

    Returns (violations, pairwise, per_anchor_rate).
      violations   -- one record per (anchor, offending NS candidate)
      pairwise     -- pooled (S, NS) comparison counts, for the sanity check
      per_anchor   -- {anchor_id: concordance} to reproduce the macro mean
    """
    violations = []
    pooled_conc, pooled_tot = 0.0, 0
    per_anchor = {}

    for aid, (subs, nonsubs) in sorted(groups.items()):
        conc, tot = 0.0, 0
        for s in subs:
            for ns in nonsubs:
                tot += 1
                if s[sim_key] > ns[sim_key]:
                    conc += 1.0
                elif s[sim_key] == ns[sim_key]:
                    conc += 0.5
        per_anchor[aid] = conc / tot
        pooled_conc += conc
        pooled_tot += tot

        sub_sims = np.array([s[sim_key] for s in subs], dtype=np.float64)
        all_sims = sorted(
            (p[sim_key] for p in subs + nonsubs),
            reverse=True,
        )
        for ns in nonsubs:
            sim = ns[sim_key]
            beaten = [s for s in subs if sim > s[sim_key]]
            tied = [s for s in subs if sim == s[sim_key]]
            if not beaten:
                continue
            best_beaten = max(beaten, key=lambda s: s[sim_key])
            violations.append(
                {
                    "anchor_id": aid,
                    "anchor_title": ns["anchor_title"],
                    "anchor_class": ns["anchor_class"],
                    "anchor_manufacturer": ns["anchor_manufacturer"],
                    # the offending non-substitute
                    "ns_id": ns["candidate_id"],
                    "ns_title": ns["candidate_title"],
                    "ns_class": ns["candidate_class"],
                    "ns_manufacturer": ns["candidate_manufacturer"],
                    "ns_label": ns["label"],
                    "ns_label_v1": ns.get("label_v1"),
                    "ns_sim": round(sim, 6),
                    "ns_coverage": ns["coverage"],
                    "ns_jaccard": ns["jaccard"],
                    "ns_shared_brand": bool(ns["shared_brand"]),
                    "ns_same_class": bool(ns["same_class"]),
                    "ns_sources": ns["sources"],
                    "ns_rank_among_candidates": all_sims.index(sim) + 1,
                    # what it beat
                    "n_subs_for_anchor": len(subs),
                    "n_subs_beaten": len(beaten),
                    "n_subs_tied": len(tied),
                    "frac_subs_beaten": round(len(beaten) / len(subs), 4),
                    "margin_vs_best_beaten": round(sim - best_beaten[sim_key], 6),
                    "margin_vs_top_sub": round(sim - float(sub_sims.max()), 6),
                    "beaten_subs": [
                        {
                            "id": s["candidate_id"],
                            "title": s["candidate_title"],
                            "class": s["candidate_class"],
                            "manufacturer": s["candidate_manufacturer"],
                            "sim": round(s[sim_key], 6),
                            "coverage": s["coverage"],
                            "jaccard": s["jaccard"],
                            "shared_brand": bool(s["shared_brand"]),
                            "same_class": bool(s["same_class"]),
                        }
                        for s in sorted(beaten, key=lambda s: -s[sim_key])
                    ],
                }
            )

    pairwise = {
        "n_pairs": pooled_tot,
        "concordant_weighted": round(pooled_conc, 2),
        "micro_concordance": pooled_conc / pooled_tot if pooled_tot else None,
        "micro_discordance": 1 - pooled_conc / pooled_tot if pooled_tot else None,
    }
    return violations, pairwise, per_anchor


# --------------------------------------------------------------------------
# characterisation
# --------------------------------------------------------------------------
def _rate(records, key):
    if not records:
        return None
    return sum(1 for r in records if r[key]) / len(records)


def feature_profile(violations, ns_pool, all_pairs):
    """Violation feature rates vs the base rates that make them interpretable.

    ns_pool   -- every NS candidate belonging to a qualifying anchor (the set a
                 violation could have been drawn from). This is the honest
                 denominator.
    all_pairs -- every judged candidate of any label, for the wider base rate.
    """
    out = {}
    for feat, vkey, bkey in (
        ("shared_brand", "ns_shared_brand", "shared_brand"),
        ("same_class", "ns_same_class", "same_class"),
    ):
        v_rate = _rate(violations, vkey)
        pool_rate = _rate(ns_pool, bkey)
        all_rate = _rate(all_pairs, bkey)
        # 2x2: (violating NS vs non-violating NS) x (feature true/false)
        viol_ids = {(v["anchor_id"], v["ns_id"]) for v in violations}
        non_viol = [p for p in ns_pool if (p["anchor_id"], p["candidate_id"]) not in viol_ids]
        a = sum(1 for v in violations if v[vkey])
        b = len(violations) - a
        c = sum(1 for p in non_viol if p[bkey])
        d = len(non_viol) - c
        odds, pval = fisher_exact([[a, b], [c, d]]) if (a + b) and (c + d) else (None, None)
        out[feat] = {
            "violation_rate": v_rate,
            "base_rate_ns_pool": pool_rate,
            "base_rate_all_candidates": all_rate,
            "lift_vs_ns_pool": (v_rate / pool_rate) if pool_rate else None,
            "contingency_violating_vs_not": {
                "v_true": a,
                "v_false": b,
                "nv_true": c,
                "nv_false": d,
            },
            "fisher_odds_ratio": float(odds) if odds is not None else None,
            "fisher_p": float(pval) if pval is not None else None,
        }

    def _mean(recs, key):
        return float(np.mean([r[key] for r in recs])) if recs else None

    out["lexical_overlap"] = {
        "violations_mean_coverage": _mean(violations, "ns_coverage"),
        "ns_pool_mean_coverage": _mean(ns_pool, "coverage"),
        "violations_mean_jaccard": _mean(violations, "ns_jaccard"),
        "ns_pool_mean_jaccard": _mean(ns_pool, "jaccard"),
    }
    out["label_mix"] = {
        "violations": {lb: sum(1 for v in violations if v["ns_label"] == lb) for lb in NS_LABELS},
        "ns_pool": {lb: sum(1 for p in ns_pool if p["label"] == lb) for lb in NS_LABELS},
    }
    # cross-tab of the two failure modes
    out["mode_crosstab"] = {
        "same_class_and_shared_brand": sum(
            1 for v in violations if v["ns_same_class"] and v["ns_shared_brand"]
        ),
        "same_class_only": sum(
            1 for v in violations if v["ns_same_class"] and not v["ns_shared_brand"]
        ),
        "shared_brand_only": sum(
            1 for v in violations if v["ns_shared_brand"] and not v["ns_same_class"]
        ),
        "neither": sum(
            1 for v in violations if not v["ns_shared_brand"] and not v["ns_same_class"]
        ),
    }
    return out


def per_class_profile(violations, groups, per_anchor, by_anchor):
    """Which anchor product classes generate the violations."""
    cls_of = {aid: by_anchor[aid][0]["anchor_class"] for aid in groups}
    stats = defaultdict(
        lambda: {
            "n_qualifying_anchors": 0,
            "n_ns_candidates": 0,
            "n_violations": 0,
            "anchors_with_violation": 0,
            "concordances": [],
        }
    )
    viol_by_anchor = defaultdict(int)
    for v in violations:
        viol_by_anchor[v["anchor_id"]] += 1
    for aid, (_subs, nonsubs) in groups.items():
        s = stats[cls_of[aid]]
        s["n_qualifying_anchors"] += 1
        s["n_ns_candidates"] += len(nonsubs)
        s["n_violations"] += viol_by_anchor.get(aid, 0)
        s["anchors_with_violation"] += 1 if viol_by_anchor.get(aid) else 0
        s["concordances"].append(per_anchor[aid])
    out = {}
    for cls, s in stats.items():
        out[cls] = {
            "n_qualifying_anchors": s["n_qualifying_anchors"],
            "n_ns_candidates": s["n_ns_candidates"],
            "n_violations": s["n_violations"],
            "violation_rate_per_ns_candidate": (
                s["n_violations"] / s["n_ns_candidates"] if s["n_ns_candidates"] else None
            ),
            "violations_per_anchor": s["n_violations"] / s["n_qualifying_anchors"],
            "anchors_with_violation": s["anchors_with_violation"],
            "macro_S_gt_CI": float(np.mean(s["concordances"])),
        }
    return dict(sorted(out.items(), key=lambda kv: -kv[1]["violations_per_anchor"]))


# --------------------------------------------------------------------------
# illustrative examples
# --------------------------------------------------------------------------
def _example_score(v):
    """Rank clarity, not just severity: beating many substitutes by a wide
    margin from a high absolute rank is the least ambiguous kind of failure."""
    return (
        v["frac_subs_beaten"] * 2.0
        + min(v["margin_vs_best_beaten"], 0.15) * 4.0
        + (0.3 if v["margin_vs_top_sub"] > 0 else 0.0)
    )


def pick_examples(violations, n_target=12):
    """Top violations, one per anchor, with every failure mode represented."""
    buckets = {
        "shared_brand_diff_class": lambda v: v["ns_shared_brand"] and not v["ns_same_class"],
        "same_class_near_miss": lambda v: v["ns_same_class"],
        "no_brand_no_class": lambda v: not v["ns_shared_brand"] and not v["ns_same_class"],
    }
    quota = {
        "shared_brand_diff_class": max(1, n_target // 2),
        "same_class_near_miss": max(1, n_target // 4),
        "no_brand_no_class": max(1, n_target // 4),
    }
    ranked = sorted(violations, key=_example_score, reverse=True)
    used_anchors, chosen = set(), []
    for bucket, pred in buckets.items():
        taken = 0
        for v in ranked:
            if taken >= quota[bucket]:
                break
            if v["anchor_id"] in used_anchors or not pred(v):
                continue
            used_anchors.add(v["anchor_id"])
            chosen.append({"failure_mode": bucket, **_trim_example(v)})
            taken += 1
    # top up with the best remaining, whatever the mode
    for v in ranked:
        if len(chosen) >= n_target:
            break
        if v["anchor_id"] in used_anchors:
            continue
        used_anchors.add(v["anchor_id"])
        mode = (
            next(k for k, p in buckets.items() if p(v))
            if any(p(v) for p in buckets.values())
            else "other"
        )
        chosen.append({"failure_mode": mode, **_trim_example(v)})
    return sorted(chosen, key=lambda v: -_example_score(v))


def _trim_example(v):
    """Example record: the anchor, the offender, and the best substitute it beat."""
    out = {k: v[k] for k in v if k != "beaten_subs"}
    out["best_sub_beaten"] = v["beaten_subs"][0]
    out["worst_sub_beaten"] = v["beaten_subs"][-1]
    return out


# --------------------------------------------------------------------------
# hard-negative triplet accounting
# --------------------------------------------------------------------------
def triplet_yield(violations):
    """(anchor, hard_negative, hard_positive) triplets implied by the violations.

    The training signal a violation licenses is symmetric: pull the anchor
    toward the substitute it should have ranked above, push it away from the
    non-substitute that outranked it. One triplet per (violation, beaten
    substitute) is therefore the natural unit.
    """
    triplets = sum(len(v["beaten_subs"]) for v in violations)
    return {
        "n_triplets_anchor_neg_pos": triplets,
        "n_unique_anchors": len({v["anchor_id"] for v in violations}),
        "n_unique_hard_negatives": len({(v["anchor_id"], v["ns_id"]) for v in violations}),
        "n_unique_hard_positives": len(
            {(v["anchor_id"], s["id"]) for v in violations for s in v["beaten_subs"]}
        ),
        "n_unique_negative_products": len({v["ns_id"] for v in violations}),
        "mean_triplets_per_anchor": (
            triplets / len({v["anchor_id"] for v in violations}) if violations else 0.0
        ),
        "comparison_bestbuy_mnrl": {
            "bestbuy_bod_bags": BESTBUY_MNRL_BAGS,
            "bestbuy_bod_triplets": BESTBUY_MNRL_TRIPLETS,
            "ratio_of_existing_training_set": triplets / BESTBUY_MNRL_TRIPLETS,
            "source": "evaluation/CHS_RESULTS.md (48,516 BestBuy bags x 5 triplets/bag)",
        },
    }


# --------------------------------------------------------------------------
# per-sim-key driver
# --------------------------------------------------------------------------
def analyze(sim_key, by_anchor, groups, all_pairs):
    violations, pairwise, per_anchor = extract_violations(groups, sim_key)
    ns_pool = [p for aid in groups for p in groups[aid][1]]
    viol_anchors = {v["anchor_id"] for v in violations}
    return {
        "sim_key": sim_key,
        "summary": {
            "n_qualifying_anchors": len(groups),
            "n_ns_candidates_in_qualifying_anchors": len(ns_pool),
            "n_violations": len(violations),
            "violation_rate_per_ns_candidate": len(violations) / len(ns_pool) if ns_pool else None,
            "n_anchors_with_violation": len(viol_anchors),
            "frac_anchors_with_violation": len(viol_anchors) / len(groups),
            "pairwise": pairwise,
            "macro_S_gt_CI": float(np.mean(list(per_anchor.values()))),
            "macro_discordance": 1 - float(np.mean(list(per_anchor.values()))),
        },
        "feature_profile": feature_profile(violations, ns_pool, all_pairs),
        "per_anchor_class": per_class_profile(violations, groups, per_anchor, by_anchor),
        "triplet_yield": triplet_yield(violations),
        "violations": sorted(
            violations, key=lambda v: (-v["frac_subs_beaten"], -v["margin_vs_best_beaten"])
        ),
        "_per_anchor": per_anchor,
    }


def compare_old_new(res_new, res_old):
    kn = {(v["anchor_id"], v["ns_id"]) for v in res_new["violations"]}
    ko = {(v["anchor_id"], v["ns_id"]) for v in res_old["violations"]}
    an = {a for a, _ in kn}
    ao = {a for a, _ in ko}
    return {
        "n_violations_new": len(kn),
        "n_violations_old": len(ko),
        "shared": len(kn & ko),
        "new_only": len(kn - ko),
        "old_only": len(ko - kn),
        "jaccard_violation_sets": len(kn & ko) / len(kn | ko) if (kn | ko) else None,
        "anchors_shared": len(an & ao),
        "anchors_new_only": len(an - ao),
        "anchors_old_only": len(ao - an),
        "profile_delta_new_minus_old": {
            feat: {
                "violation_rate_new": res_new["feature_profile"][feat]["violation_rate"],
                "violation_rate_old": res_old["feature_profile"][feat]["violation_rate"],
                "delta": res_new["feature_profile"][feat]["violation_rate"]
                - res_old["feature_profile"][feat]["violation_rate"],
            }
            for feat in ("shared_brand", "same_class")
        },
        "macro_S_gt_CI_new": res_new["summary"]["macro_S_gt_CI"],
        "macro_S_gt_CI_old": res_old["summary"]["macro_S_gt_CI"],
    }


def sanity_check(blob, results):
    """Reproduce the benchmark's stored S_gt_CI numbers from scratch."""
    checks = {}
    for sim_key, tag in (("sim_new", "new"), ("sim_old", "old")):
        stored = blob.get("concordance", {}).get(tag, {}).get("S_gt_CI")
        if not stored:
            continue
        got = results[sim_key]["summary"]
        checks[tag] = {
            "stored_macro": stored["macro_mean"],
            "recomputed_macro": got["macro_S_gt_CI"],
            "macro_abs_diff": abs(stored["macro_mean"] - got["macro_S_gt_CI"]),
            "stored_micro": stored["micro_pooled"],
            "recomputed_micro": got["pairwise"]["micro_concordance"],
            "micro_abs_diff": abs(stored["micro_pooled"] - got["pairwise"]["micro_concordance"]),
            "stored_n_pairs": stored["n_pairs"],
            "recomputed_n_pairs": got["pairwise"]["n_pairs"],
            "stored_n_anchors": stored["n_anchors_qualifying"],
            "recomputed_n_anchors": got["n_qualifying_anchors"],
            "ok": abs(stored["macro_mean"] - got["macro_S_gt_CI"]) < 1e-9
            and stored["n_pairs"] == got["pairwise"]["n_pairs"],
        }
    return checks


# --------------------------------------------------------------------------
# reporting
# --------------------------------------------------------------------------
def print_report(out):
    def pct(x):
        return "n/a" if x is None else f"{100 * x:.1f}%"

    print("=" * 78)
    print("BestBuy substitutability -- S > {C,I} VIOLATIONS (v2 / gpt-4.1 labels)")
    print("=" * 78)

    print("\n-- sanity check vs stored benchmark concordance --")
    for tag, c in out["sanity_check"].items():
        print(
            f"  {tag}: macro stored {c['stored_macro']:.4f} / recomputed "
            f"{c['recomputed_macro']:.4f} (diff {c['macro_abs_diff']:.2e})  "
            f"pairs {c['recomputed_n_pairs']}=={c['stored_n_pairs']}  ok={c['ok']}"
        )

    for sim_key in SIM_KEYS:
        r = out[sim_key]
        s = r["summary"]
        print(f"\n{'=' * 78}\n{sim_key.upper()}\n{'=' * 78}")
        print(
            f"  qualifying anchors      : {s['n_qualifying_anchors']}  "
            f"({pct(s['frac_anchors_with_violation'])} have >=1 violation)"
        )
        print(f"  NS candidates in pool   : {s['n_ns_candidates_in_qualifying_anchors']}")
        print(
            f"  violations (NS beats >=1 S): {s['n_violations']}  "
            f"= {pct(s['violation_rate_per_ns_candidate'])} of NS candidates"
        )
        print(
            f"  pairwise discordance    : {pct(s['pairwise']['micro_discordance'])} micro / "
            f"{pct(s['macro_discordance'])} macro   [the ~24% headline]"
        )

        fp = r["feature_profile"]
        print("\n  feature            violations   NS base   all-cand base   lift   fisher p")
        for feat in ("shared_brand", "same_class"):
            f = fp[feat]
            print(
                f"  {feat:<18} {pct(f['violation_rate']):>9}  "
                f"{pct(f['base_rate_ns_pool']):>8}  {pct(f['base_rate_all_candidates']):>13}  "
                f"{f['lift_vs_ns_pool']:>5.2f}  {f['fisher_p']:.3g}"
            )
        ct = fp["mode_crosstab"]
        print(
            f"  modes: brand+class {ct['same_class_and_shared_brand']} | "
            f"class only {ct['same_class_only']} | brand only {ct['shared_brand_only']} | "
            f"neither {ct['neither']}"
        )
        lex = fp["lexical_overlap"]
        print(
            f"  coverage: violations {lex['violations_mean_coverage']:.3f} vs "
            f"NS pool {lex['ns_pool_mean_coverage']:.3f}   "
            f"jaccard: {lex['violations_mean_jaccard']:.3f} vs "
            f"{lex['ns_pool_mean_jaccard']:.3f}"
        )
        print(f"  NS label mix in violations: {fp['label_mix']['violations']}")

        print("\n  top anchor classes by violations/anchor:")
        for cls, c in list(r["per_anchor_class"].items())[:8]:
            print(
                f"    {cls:<22} {c['violations_per_anchor']:>5.2f}/anchor  "
                f"({c['n_violations']:>3} viol / {c['n_ns_candidates']:>3} NS, "
                f"{c['n_qualifying_anchors']} anchors, S>CI {c['macro_S_gt_CI']:.3f})"
            )

        t = r["triplet_yield"]
        print(
            f"\n  triplets: {t['n_triplets_anchor_neg_pos']} (anchor,neg,pos) from "
            f"{t['n_unique_anchors']} anchors / {t['n_unique_hard_negatives']} unique negatives "
            f"= {t['comparison_bestbuy_mnrl']['ratio_of_existing_training_set']:.4%} of the "
            f"{BESTBUY_MNRL_TRIPLETS:,}-triplet BestBuy MNRL set"
        )

    print(f"\n{'=' * 78}\nOLD vs NEW violation sets\n{'=' * 78}")
    c = out["old_vs_new"]
    print(
        f"  new {c['n_violations_new']} | old {c['n_violations_old']} | shared {c['shared']} "
        f"| new-only {c['new_only']} | old-only {c['old_only']} "
        f"(jaccard {c['jaccard_violation_sets']:.3f})"
    )
    for feat, d in c["profile_delta_new_minus_old"].items():
        print(
            f"  {feat}: new {pct(d['violation_rate_new'])} vs old "
            f"{pct(d['violation_rate_old'])} (delta {100 * d['delta']:+.1f}pp)"
        )

    print(f"\n{'=' * 78}\nILLUSTRATIVE VIOLATIONS (NEW embedding)\n{'=' * 78}")
    for i, v in enumerate(out["examples"], 1):
        print(f"\n[{i}] {v['failure_mode']}   anchor class {v['anchor_class']}")
        print(f"    ANCHOR   {v['anchor_title']}")
        print(
            f"    NS ({v['ns_label']}) sim={v['ns_sim']:.4f} brand={v['ns_shared_brand']} "
            f"same_class={v['ns_same_class']} cov={v['ns_coverage']}"
        )
        print(f"             {v['ns_title']}  [{v['ns_class']}]")
        b = v["best_sub_beaten"]
        print(f"    beat S   sim={b['sim']:.4f}  {b['title']}  [{b['class']}]")
        print(
            f"    ... and {v['n_subs_beaten']}/{v['n_subs_for_anchor']} substitutes total, "
            f"margin over best beaten {v['margin_vs_best_beaten']:+.4f}"
        )


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--benchmark", default=DEFAULT_BENCHMARK)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--n-examples", type=int, default=12)
    ap.add_argument(
        "--max-violations-stored",
        type=int,
        default=0,
        help="truncate the stored full violation list (0 = store all)",
    )
    args = ap.parse_args()

    blob, by_anchor = load_pairs(args.benchmark)
    all_pairs = blob["raw_pairs"]
    groups = anchor_groups(by_anchor)

    results = {k: analyze(k, by_anchor, groups, all_pairs) for k in SIM_KEYS}
    for r in results.values():
        r.pop("_per_anchor", None)

    out = {
        "analysis": "BestBuy substitutability S > {C,I} violation enumeration "
        "(3-class collapse: C treated as I)",
        "config": {
            "benchmark": os.path.relpath(args.benchmark, REPO),
            "labels": blob.get("config", {}).get("judge_model"),
            "judge_prompt": blob.get("config", {}).get("judge_prompt"),
            "violation_definition": "non-substitute (C or I) whose cosine similarity to the "
            "anchor strictly exceeds that of >=1 substitute (S) of the same anchor; ties "
            "excluded from violations, scored 0.5 in the pairwise rate",
            "n_pairs_total": len(all_pairs),
            "n_anchors_total": len(by_anchor),
        },
        "sanity_check": sanity_check(blob, results),
        "examples": pick_examples(results["sim_new"]["violations"], args.n_examples),
        "old_vs_new": compare_old_new(results["sim_new"], results["sim_old"]),
    }
    for k in SIM_KEYS:
        r = dict(results[k])
        if args.max_violations_stored:
            r["violations"] = r["violations"][: args.max_violations_stored]
        out[k] = r

    print_report(out)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
