#!/usr/bin/env python3
"""Generate role_family overrides for Swedish JobTech jobs from occupation_field.

Every Arbetsförmedlingen (JobTech) ad carries the Swedish occupation taxonomy
pre-attached; we already store its broad `occupation_field` bucket as the
`department` field at fetch time (cached in jobs_data_jobtech/raw/). That bucket
is the authoritative occupation label, so we map it straight to a role_family via
jobtech_role_family.role_family_for_field -- open-weight, deterministic, no LLM,
no SSYK reconstruction.

Emits role_family_jobtech_overrides.json: {unified_id -> role_family}, restricted
to fields that resolve to a real (non-"other") family. push_docs applies it with
precedence emb > llm > rome > jobtech > esco, only where the title heuristic left
"other". The unified id equals the raw parquet id ("jobtech:<id>"), so no join is
needed.

Usage:
  python classify_other_jobtech.py                     # default paths
  python classify_other_jobtech.py --raw DIR --out FILE
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)  # repo root holds the jobs_data_* corpora
DEFAULT_RAW = os.path.join(ROOT, "jobs_data_jobtech", "raw")
DEFAULT_OUT = os.path.join(HERE, "role_family_jobtech_overrides.json")

sys.path.insert(0, HERE)
from jobtech_role_family import role_family_for_field  # noqa: E402


def build(raw_dir: str, out_path: str) -> dict[str, str]:
    import pandas as pd

    files = sorted(glob.glob(os.path.join(raw_dir, "*.parquet")))
    if not files:
        raise SystemExit(f"no parquet files under {raw_dir}")

    overrides: dict[str, str] = {}
    fam_counts: Counter[str] = Counter()
    seen = 0
    fielded = 0
    for fp in files:
        df = pd.read_parquet(fp, columns=["id", "department"])
        for doc_id, field in zip(df["id"], df["department"]):
            seen += 1
            if not doc_id or not field:
                continue
            fielded += 1
            fam = role_family_for_field(field)
            if fam == "other":
                continue
            # first non-other wins (ids can repeat across crawl shards)
            overrides.setdefault(str(doc_id), fam)
            fam_counts[fam] += 1

    with open(out_path, "w") as f:
        json.dump(overrides, f, ensure_ascii=False, indent=0)

    print(
        f"JobTech raw rows={seen:,} with-field={fielded:,} "
        f"-> {len(overrides):,} non-other overrides written to {out_path}",
        file=sys.stderr,
    )
    for fam, n in fam_counts.most_common():
        print(f"  {fam:<32} {n:,}", file=sys.stderr)
    return overrides


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=DEFAULT_RAW, help="dir of JobTech raw parquet")
    ap.add_argument("--out", default=DEFAULT_OUT, help="output overrides JSON")
    args = ap.parse_args()
    build(args.raw, args.out)


if __name__ == "__main__":
    main()
