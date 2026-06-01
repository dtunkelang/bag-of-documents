#!/usr/bin/env python3
"""Quality check for slug -> industry labels, run AFTER the confidence gate.

Why this exists: the education_higher bucket shipped ~75% wrong because nobody sampled
it. This tool makes that mistake loud and cheap to catch. It:

  1. Reports coverage before/after the gate (slugs and job-postings).
  2. Per industry, measures how much of the bucket rests on PROPAGATION near the floor
     -- the "attractor" signature that broke education_higher -- and RED-FLAGS buckets
     that look contaminated.
  3. Emits a stratified random sample (N slugs per industry, with sample job titles) to a
     TSV for human / model review, so correctness is actually eyeballed each refresh.

It imports the SAME gate push_docs.py applies (industry_filter), so the numbers here are
the numbers that ship. Exit code is nonzero if any RED FLAG fires, so refresh.py (or CI)
can treat a contaminated bucket as a failed build.

Usage:
  python qc_industry_labels.py --labels unified_jobs_daily/slug_industry_labels_round2.csv \\
      --meta unified_jobs_daily/metadata.jsonl --out unified_jobs_daily/qc_industry_sample.tsv
"""

import argparse
import csv
import json
import os
import random
import re
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from industry_filter import (  # noqa: E402
    DEFAULT_SIM_FLOOR,
    DROP_METHODS,
    TRUSTED_METHODS,
    _to_float,
    accept,
)

# A bucket is flagged if too much of it is propagated labels sitting just above the floor
# (the attractor signature), or if it's mostly propagation at all.
NEAR_FLOOR_BAND = 0.05  # "just above the floor" = [floor, floor+band)
FLAG_NEAR_FLOOR_SHARE = 0.50  # >=50% of kept slugs near the floor -> suspect attractor
FLAG_PROPAGATED_SHARE = 0.80  # >=80% of kept slugs propagated (few seeds anchoring it)
# Machine-generated junk slugs only: a uuid block, or TWO separate 3+ digit runs (e.g.
# 'world4822stri986des'). A single trailing/embedded digit run is almost always a real
# brand (trading212, amazon-lab126, stanley1913, la2028) -> NOT flagged.
_GARBAGE_SLUG = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}|\d{3,}\D+\d{3,}")


def load_rows(path: str) -> list[dict]:
    with open(path) as f:
        return list(csv.DictReader(f))


def sample_titles(meta_path: str, slugs: set[str], per_slug: int = 2) -> dict[str, list[str]]:
    """One pass over metadata.jsonl, collecting up to `per_slug` titles for `slugs`."""
    out: dict[str, list[str]] = defaultdict(list)
    if not meta_path or not os.path.exists(meta_path):
        return out
    with open(meta_path) as f:
        for line in f:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            slug = rec.get("source_slug") or ""
            if slug in slugs and len(out[slug]) < per_slug:
                t = (rec.get("title") or "").strip().replace("\t", " ")
                if t:
                    out[slug].append(t[:70])
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", required=True, help="slug_industry_labels_round2.csv")
    ap.add_argument("--meta", default="", help="metadata.jsonl (for sample job titles)")
    ap.add_argument("--floor", type=float, default=DEFAULT_SIM_FLOOR)
    ap.add_argument("--per-industry", type=int, default=15, help="sample size per industry")
    ap.add_argument("--out", default="qc_industry_sample.tsv")
    ap.add_argument("--seed", type=int, default=0, help="RNG seed (reproducible samples)")
    args = ap.parse_args()

    rows = load_rows(args.labels)
    rng = random.Random(args.seed)

    # ---- coverage before/after the gate ----
    def docs(rs):
        return sum(int(r.get("n_jobs") or 0) for r in rs)

    raw = [r for r in rows if (r.get("industry") or "") not in ("", "unclassified")]
    kept = [
        r
        for r in raw
        if accept((r.get("method") or "").strip(), _to_float(r.get("top1_sim")), args.floor)
    ]
    print(
        f"floor = {args.floor:g}   (TRUSTED={sorted(TRUSTED_METHODS)}  DROP={sorted(DROP_METHODS)})"
    )
    print(
        f"slugs:  {len(raw):>6} labeled -> {len(kept):>6} kept "
        f"({100 * len(kept) / max(1, len(raw)):.0f}%)   "
        f"docs: {docs(raw):>7} -> {docs(kept):>7} ({100 * docs(kept) / max(1, docs(raw)):.0f}%)"
    )

    # ---- per-industry contamination signature ----
    by_ind: dict[str, list[dict]] = defaultdict(list)
    for r in kept:
        by_ind[r["industry"]].append(r)

    print(f"\n{'industry':28} {'slugs':>6} {'docs':>7} {'prop%':>6} {'nearfloor%':>10}  flags")
    hard_flags: list[str] = []  # contamination -> fail the build
    warnings: list[str] = []  # ugly-but-maybe-real -> review, don't fail
    for ind, rs in sorted(by_ind.items(), key=lambda kv: -docs(kv[1])):
        prop = [r for r in rs if (r.get("method") or "") not in TRUSTED_METHODS]
        near = [
            r
            for r in prop
            if args.floor <= _to_float(r.get("top1_sim")) < args.floor + NEAR_FLOOR_BAND
        ]
        prop_share = len(prop) / max(1, len(rs))
        near_share = len(near) / max(1, len(rs))
        f = ""
        if near_share >= FLAG_NEAR_FLOOR_SHARE:
            f += "ATTRACTOR "
            hard_flags.append(f"{ind}: {near_share:.0%} of kept slugs sit just above the floor")
        if prop_share >= FLAG_PROPAGATED_SHARE and len(rs) >= 20:
            f += "THIN-SEED "
            hard_flags.append(f"{ind}: {prop_share:.0%} propagated, few seeds anchoring it")
        garbage = [r["slug"] for r in rs if _GARBAGE_SLUG.search(r["slug"])]
        if garbage:
            f += f"garbage({len(garbage)}) "
            warnings.append(f"{ind}: machine-generated slug(s) e.g. {garbage[:3]}")
        print(f"{ind:28} {len(rs):>6} {docs(rs):>7} {prop_share:>5.0%} {near_share:>9.0%}   {f}")

    # ---- stratified sample for review ----
    sample: list[dict] = []
    for rs in by_ind.values():
        pick = rs if len(rs) <= args.per_industry else rng.sample(rs, args.per_industry)
        sample.extend(pick)
    titles = sample_titles(args.meta, {r["slug"] for r in sample})
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(
            ["industry", "slug", "n_jobs", "method", "anchor_seed", "top1_sim", "sample_titles"]
        )
        for r in sorted(sample, key=lambda x: (x["industry"], x["slug"])):
            w.writerow(
                [
                    r["industry"],
                    r["slug"],
                    r.get("n_jobs", ""),
                    r.get("method", ""),
                    r.get("top1_seed", ""),
                    r.get("top1_sim", ""),
                    " | ".join(titles.get(r["slug"], [])),
                ]
            )
    print(f"\nwrote {len(sample)} sampled labels for review -> {args.out}")

    if warnings:
        print(f"\n{len(warnings)} warning(s) (review, won't fail the build):")
        for msg in warnings:
            print(f"  - {msg}")
    if hard_flags:
        print(
            f"\n*** {len(hard_flags)} RED FLAG(S) — bucket looks contaminated, fix before shipping ***"
        )
        for msg in hard_flags:
            print(f"  - {msg}")
        return 1
    print("\nno red flags.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
