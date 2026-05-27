#!/usr/bin/env python3
"""Audit autocomplete phrases where role_family='other' dominates Solr title matches
despite a viable non-'other' runner-up — i.e. candidates for new heuristic patterns.

Reads /Users/dtunkelang/bagofdocs/unified_jobs/te3_cache_canonical.json (the
production autocomplete pool), runs one Solr facet query per phrase against
title:"<phrase>", and emits a TSV sorted by potential reclassification impact.
"""

from __future__ import annotations

import json
import sys
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

SOLR = "http://127.0.0.1:8983/solr/jobs/select"
PHRASES_PATH = "/Users/dtunkelang/bagofdocs/unified_jobs/te3_cache_canonical.json"
OUT_PATH = "/Users/dtunkelang/bagofdocs/solr_jobs_demo/facets/audit_other_in_titles.tsv"

MIN_TOTAL = 10
MIN_OTHER_FRAC = 0.5
MIN_RUNNERUP_FRAC = 0.10


def solr_role_family_counts(phrase: str) -> dict[str, int]:
    params = {
        "q": f'title:"{phrase}"',
        "rows": "0",
        "facet": "true",
        "facet.field": "role_family",
        "facet.limit": "30",
        "facet.mincount": "1",
        "wt": "json",
    }
    url = SOLR + "?" + urllib.parse.urlencode(params)
    with urllib.request.urlopen(url, timeout=10) as r:
        d = json.load(r)
    f = d["facet_counts"]["facet_fields"]["role_family"]
    return dict(zip(f[::2], f[1::2], strict=False))


def audit_phrase(phrase: str) -> tuple[str, dict[str, int]] | None:
    try:
        return phrase, solr_role_family_counts(phrase)
    except Exception:
        return None


def main() -> None:
    with open(PHRASES_PATH) as f:
        phrases = sorted(json.load(f).keys())
    print(f"auditing {len(phrases):,} phrases against Solr", flush=True)

    rows: list[tuple[float, int, int, str, int, str]] = []
    t0 = time.time()
    done = 0
    with ThreadPoolExecutor(max_workers=16) as ex:
        for res in as_completed(ex.submit(audit_phrase, p) for p in phrases):
            r = res.result()
            done += 1
            if done % 2000 == 0:
                print(f"  {done:,}/{len(phrases):,} in {time.time() - t0:.0f}s", flush=True)
            if r is None:
                continue
            phrase, counts = r
            total = sum(counts.values())
            if total < MIN_TOTAL:
                continue
            other = counts.get("other", 0)
            if other / total < MIN_OTHER_FRAC:
                continue
            non_other = [(k, n) for k, n in counts.items() if k != "other"]
            if not non_other:
                continue
            non_other.sort(key=lambda x: -x[1])
            ru_role, ru_count = non_other[0]
            if ru_count / total < MIN_RUNNERUP_FRAC:
                continue
            impact = other  # docs that *would* be reclassified if heuristic adopted ru_role
            rows.append((impact, total, other, ru_role, ru_count, phrase))

    rows.sort(reverse=True)
    with open(OUT_PATH, "w") as fout:
        fout.write("impact\ttotal\tother\trunner_up_role\trunner_up_count\tphrase\n")
        for r in rows:
            fout.write("\t".join(str(x) for x in r) + "\n")
    print(f"wrote {len(rows):,} candidates to {OUT_PATH} in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    sys.exit(main())
