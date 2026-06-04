#!/usr/bin/env python3
"""Build an id -> apply_url map for the indexed jobs corpus WITHOUT re-crawling.

Two recovery paths, in priority order:
  1. Join: the OpenApply (greenhouse/lever/ashby) raw files on disk carry the
     authoritative `apply_url`. Use it verbatim where the indexed id still exists
     in the current crawl.
  2. Template: reconstruct from the id `{source}:{slug}:{native_id}` for sources
     whose public posting URL is a deterministic function of the id.

Sources whose URL is a tokenised/ephemeral redirect (adzuna, jooble) or an
arbitrary employer landing page (themuse, findwork, recruitee, reed) are NOT
reconstructible offline -> reported as "no link" (re-fetch would be required).

Usage:
  python build_apply_urls.py            # audit only: print coverage, write nothing
  python build_apply_urls.py --write OUT # also write {id: apply_url} JSON to OUT
"""

import argparse
import collections
import json
import os
import sys
import urllib.parse
from pathlib import Path

ROOT = Path("/Users/dtunkelang/bagofdocs")
OA_RAW = ["openapply_repo/jobs.jsonl", "openapply_repo/jobs_extra.jsonl"]
# The live index is built from the nightly refresh staging dir (unified_jobs_daily),
# NOT the stale manual-rebuild dir (unified_jobs). Default to the staging push_docs
# reads from; override with JOBS_STAGE to match a specific run.
STAGE = Path(os.environ.get("JOBS_STAGE", str(ROOT / "unified_jobs_daily")))
UNIFIED = STAGE / "metadata.jsonl"


def load_oa_join() -> dict:
    m = {}
    for fn in OA_RAW:
        p = ROOT / fn
        if not p.exists():
            continue
        with open(p) as f:
            for line in f:
                try:
                    d = json.loads(line)
                except json.JSONDecodeError:
                    continue
                u = d.get("apply_url")
                if u:
                    m[d["id"]] = u
    return m


def template(jid: str) -> str | None:
    """Deterministic public-posting URL from the id, or None if not derivable."""
    parts = jid.split(":")
    if len(parts) < 2:
        return None
    src = parts[0]
    # native id is everything after the source (+ slug for the slug-bearing sources)
    if src in (
        "greenhouse",
        "lever",
        "ashby",
        "smartrecruiters",
        "workable",
        "usajobs",
        "recruitee",
    ):
        if len(parts) < 3:
            return None
        slug, nid = parts[1], ":".join(parts[2:])
        q = urllib.parse.quote(slug, safe="")
        if src == "greenhouse":  # standard hosted board; resolves/redirects for most
            return f"https://job-boards.greenhouse.io/{q}/jobs/{nid}"
        if src == "lever":
            return f"https://jobs.lever.co/{q}/{nid}"
        if src == "ashby":
            return f"https://jobs.ashbyhq.com/{q}/{nid}"
        if src == "smartrecruiters":
            return f"https://jobs.smartrecruiters.com/{q}/{nid}"
        if src == "workable":
            return f"https://apply.workable.com/{q}/j/{nid}/"
        if src == "usajobs":
            return f"https://www.usajobs.gov/job/{nid}"
        return None  # recruitee needs a title-slug, not just the offer id
    # single-token native id sources
    nid = ":".join(parts[1:])
    if src == "francetravail":
        return f"https://candidat.francetravail.fr/offres/recherche/detail/{nid}"
    if src == "jobtech":
        return f"https://arbetsformedlingen.se/platsbanken/annonser/{nid}"
    if src == "remoteok":
        return f"https://remoteok.com/remote-jobs/{nid}"
    return None  # adzuna, jooble, themuse, findwork, reed: not reconstructible


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", metavar="OUT", help="write {id: apply_url} JSON here")
    args = ap.parse_args()

    join = load_oa_join()
    print(f"OpenApply join map: {len(join):,} ids on disk", file=sys.stderr)

    out = {}
    by_src_total = collections.Counter()
    by_src_join = collections.Counter()
    by_src_tmpl = collections.Counter()
    by_src_none = collections.Counter()

    with open(UNIFIED) as f:
        for line in f:
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            jid = d.get("id")
            if not jid:
                continue
            corpus = d.get("source_corpus") or d.get("source") or "?"
            by_src_total[corpus] += 1
            u = join.get(jid)
            if u:
                out[jid] = u
                by_src_join[corpus] += 1
                continue
            u = template(jid)
            if u:
                out[jid] = u
                by_src_tmpl[corpus] += 1
            else:
                by_src_none[corpus] += 1

    total = sum(by_src_total.values())
    resolved = len(out)
    print(f"\n{'corpus':28s} {'docs':>9s} {'join':>9s} {'tmpl':>9s} {'none':>9s} {'cover%':>7s}")
    for c in sorted(by_src_total, key=lambda x: -by_src_total[x]):
        t = by_src_total[c]
        cov = 100 * (by_src_join[c] + by_src_tmpl[c]) / t if t else 0
        print(
            f"{c:28s} {t:9,d} {by_src_join[c]:9,d} {by_src_tmpl[c]:9,d} {by_src_none[c]:9,d} {cov:6.1f}%"
        )
    print(
        f"{'TOTAL':28s} {total:9,d} {sum(by_src_join.values()):9,d} "
        f"{sum(by_src_tmpl.values()):9,d} {sum(by_src_none.values()):9,d} "
        f"{100 * resolved / total if total else 0:6.1f}%"
    )

    if args.write:
        with open(args.write, "w") as f:
            json.dump(out, f)
        print(f"\nwrote {resolved:,} id->apply_url entries to {args.write}", file=sys.stderr)


if __name__ == "__main__":
    main()
