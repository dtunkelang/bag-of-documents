#!/usr/bin/env python3
"""Print sample jobs with their facet assignments for hand-eyeball review.

- 1 random job per role_family bucket (stratified)
- plus 10 fully-random samples (distribution-weighted)
"""

import json
import random
from collections import defaultdict
from pathlib import Path

META = Path("/Users/dtunkelang/bagofdocs/unified_jobs/metadata.jsonl")
FACETS = Path("/Users/dtunkelang/bagofdocs/jobs_search_demo/facets/facets.jsonl")


def main() -> None:
    print("loading facets...")
    facets_by_idx: dict[int, dict] = {}
    family_buckets: dict[str, list[int]] = defaultdict(list)
    with open(FACETS) as f:
        for line in f:
            d = json.loads(line)
            facets_by_idx[d["idx"]] = d
            family_buckets[d["role_family"]].append(d["idx"])

    rng = random.Random(2026)
    print(f"loaded {len(facets_by_idx):,} facet records, {len(family_buckets)} role buckets")

    # Stratified: one job per bucket. For "other" pull a couple since it's the
    # bucket we'd most want to confirm is actually 'unclassifiable'.
    stratified = []
    for fam in sorted(family_buckets):
        idxs = family_buckets[fam]
        rng.shuffle(idxs)
        n = 2 if fam == "other" else 1
        for i in idxs[:n]:
            stratified.append(("stratified", fam, i))

    # Random: 10 fully-random jobs (will hit popular buckets again)
    all_idxs = sorted(facets_by_idx)
    randsamp = [("random", facets_by_idx[i]["role_family"], i) for i in rng.sample(all_idxs, 10)]

    samples = stratified + randsamp
    print(f"{len(samples)} samples ({len(stratified)} stratified + {len(randsamp)} random)\n")

    # Pull titles + brief descriptions
    want_idx = {s[2] for s in samples}
    rec_by_idx: dict[int, dict] = {}
    with open(META) as f:
        for i, line in enumerate(f):
            if i in want_idx:
                rec_by_idx[i] = json.loads(line)
            if len(rec_by_idx) == len(want_idx):
                break

    for kind, fam_label, idx in samples:
        rec = rec_by_idx[idx]
        fac = facets_by_idx[idx]
        title = (rec.get("title") or "").strip()[:100]
        desc = (rec.get("description") or "").strip()[:220].replace("\n", " ")
        src = fac.get("role_family_source", "?")
        print(f"--- [{kind:>10}] idx={idx} [{fam_label}] ({src}) ---")
        print(f"  title: {title}")
        print(f"  desc:  {desc}{'...' if len(desc) >= 220 else ''}")
        print(f"  role_family:    {fac['role_family']}")
        print(f"  seniority:      {fac['seniority']}")
        print(f"  remote_mode:    {fac['remote_mode']}")
        print(
            f"  location:       {fac.get('location_country', '')}/{fac.get('location_state', '')}/{fac.get('location_city', '')}"
        )
        print(f"  posted_bucket:  {fac['posted_bucket']}")
        print(f"  salary_band:    {fac['salary_band_usd_annual']}")
        if fac.get("tech_stack"):
            print(f"  tech_stack:     {fac['tech_stack']}")
        print()


if __name__ == "__main__":
    main()
