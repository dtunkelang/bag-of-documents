#!/usr/bin/env python3
"""Generate role_family overrides for Adzuna jobs from their category label.

Every Adzuna posting carries a category from Adzuna's own taxonomy; we store its
localized label as the `department` field at fetch time. That label is the
source-assigned occupation bucket, so we map it straight to a role_family via
adzuna_role_family.role_family_for_adzuna_category -- open-weight, deterministic,
no LLM, no re-crawl.

NON-ENGLISH ONLY (deliberate). Adzuna's category is an algorithmic guess, not an
authoritative employment-service code like ROME/SSYK; its precision (~85% on
non-English, spot-checked) is fine where it is the *only* taxonomy signal we
have, but English jobs already get classified by the strong native title /
embedding / LLM pipeline, so an English 'other' residual is precisely where those
better signals abstained -- overwriting it with a noisier category is a bad trade
(e.g. Adzuna files a babysitter under "Legal Jobs"). So we emit overrides for
lang != en only, reading the staging metadata (which carries `lang` + the
captured `department`).

Emits role_family_adzuna_overrides.json: {unified_id -> role_family}, restricted
to non-English docs whose category resolves to a real (non-"other") family.
push_docs applies it with precedence emb > llm > rome > jobtech > adzuna > esco,
only where the title heuristic left "other".

Usage:
  python classify_other_adzuna.py                     # default paths
  python classify_other_adzuna.py --meta FILE --out FILE
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)  # repo root holds the staging catalog
DEFAULT_META = os.path.join(ROOT, "unified_jobs_daily", "metadata.jsonl")
DEFAULT_OUT = os.path.join(HERE, "role_family_adzuna_overrides.json")

sys.path.insert(0, HERE)
from adzuna_role_family import role_family_for_adzuna_category  # noqa: E402


def build(meta_path: str, out_path: str) -> dict[str, str]:
    overrides: dict[str, str] = {}
    fam_counts: Counter[str] = Counter()
    seen = 0
    eligible = 0
    with open(meta_path) as mf:
        for line in mf:
            rec = json.loads(line)
            seen += 1
            if rec.get("source") != "adzuna" or (rec.get("lang") or "en") == "en":
                continue
            doc_id = rec.get("id")
            cat = rec.get("department")
            if not doc_id or not cat:
                continue
            eligible += 1
            fam = role_family_for_adzuna_category(cat)
            if fam == "other":
                continue
            overrides.setdefault(str(doc_id), fam)
            fam_counts[fam] += 1

    with open(out_path, "w") as f:
        json.dump(overrides, f, ensure_ascii=False, indent=0)

    print(
        f"Adzuna meta rows={seen:,} non-en-with-category={eligible:,} "
        f"-> {len(overrides):,} non-other overrides written to {out_path}",
        file=sys.stderr,
    )
    for fam, n in fam_counts.most_common():
        print(f"  {fam:<32} {n:,}", file=sys.stderr)
    return overrides


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta", default=DEFAULT_META, help="staging metadata.jsonl")
    ap.add_argument("--out", default=DEFAULT_OUT, help="output overrides JSON")
    args = ap.parse_args()
    build(args.meta, args.out)


if __name__ == "__main__":
    main()
