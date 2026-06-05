#!/usr/bin/env python3
"""ATS board-metadata fallback for employer display names.

The slug-anchored description extractor (build_employer_names.py) recovers a company
name only when the posting's prose names the company. The residual is ~10k single-token
slugs the prettifier mangles ('getwingapp' -> 'Getwingapp', 'onthegosystems' -> nothing
useful) because the name never appears verbatim in the description. For the four ATS
sources that expose a per-board public endpoint we can fetch the real name directly:

  * greenhouse   boards-api JSON          name        ('samsara'    -> 'Samsara')
  * smartrecruiters postings JSON         company.name ('munsonhealthcare1' -> 'Munson Healthcare')
  * ashby        job-board HTML <title>   'X Jobs'     ('ajax'       -> 'Ajax')
  * lever        job-board HTML <title>   'X'          ('getwingapp' -> 'Wing Assistant')

DISPLAY-ONLY, like the extractor: the raw slug stays the filter key; this only feeds
app.py _pretty_employer, consulted AFTER curated (employer_names.json) and the
description extraction (employer_names_extracted.json), both of which WIN. No re-index.

Per-slug results (incl. misses) are cached to .employer_ats_cache.json so reruns are
incremental and a long crawl is resumable. We keep a fetched name only when it improves
on the prettifier AND on what the extractor already produced.
"""

import argparse
import html
import json
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from build_employer_names import _normalize_case, _prettify

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
META = ROOT / "unified_jobs_daily" / "metadata.jsonl"
CURATED = HERE / "space" / "employer_names.json"
EXTRACTED = HERE / "space" / "employer_names_extracted.json"
DEST = HERE / "space" / "employer_names_ats.json"
CACHE = HERE / ".employer_ats_cache.json"

UA = {"User-Agent": "Mozilla/5.0 (jobs-demo employer-name backfill)"}
TIMEOUT = 12
_TITLE = re.compile(r"<title[^>]*>([^<]{1,120})</title>", re.I)
# board-page title chrome to strip: 'Ajax Jobs', 'Foo Careers', 'Bar - Jobs', 'Baz | Job Board'
_TITLE_CHROME = re.compile(
    r"\s*(?:[-|]\s*)?(?:jobs|careers|job board|open positions|hiring|we'?re hiring)\s*$", re.I
)
# leading board chrome: 'Careers at Kokua' -> 'Kokua', 'Join us at Foo' -> 'Foo'
_LEAD_CHROME = re.compile(r"^(?:careers|jobs|work|working|join(?:\s+us)?)\s+(?:at|with)\s+", re.I)
# trailing board chrome, applied repeatedly: 'X Experienced Hiring Job Board' -> 'X',
# 'Y - Early Careers' -> 'Y', 'Z New Job Board' -> 'Z'
_TRAIL_CHROME = re.compile(
    r"\s*(?:[-|–]\s*)?(?:new\s+|experienced\s+|early\s+)*"
    r"(?:job board|jobs|careers|hiring|openings|open roles|talent)\s*$",
    re.I,
)
# generic / internal-board titles that name no company -> drop, let the prettifier titlecase
_JUNK_NAMES = {
    "job board",
    "internal job board",
    "internal board",
    "alumni network",
    "alumni network job board",
    "careers",
    "jobs",
    "home",
    "welcome",
    "login",
    "not found",
    "page not found",
    "404",
    "untitled",
    "internal",
    "external",
}
_INTERNAL = re.compile(
    r"internal\s+(?:job\s+)?board|internal posts only|hourly paid employees", re.I
)


def _clean_board_name(name: str) -> str | None:
    """Post-process an HTML-title-derived name: strip leading/trailing board chrome,
    drop generic/internal-board titles and over-long board descriptions (taglines, not
    company names). Returns the cleaned name or None to fall back to the prettifier."""
    n = _LEAD_CHROME.sub("", name.strip()).strip()
    prev = None
    while prev != n:  # peel repeated trailing chrome ('X Careers Job Board')
        prev = n
        n = _TRAIL_CHROME.sub("", n).strip(" -|–")
    if not n or n.lower() in _JUNK_NAMES or _INTERNAL.search(n) or len(n) > 40:
        return None
    return n


def _single_token(slug: str) -> bool:
    """A slug the prettifier can't split (no separator) -- the only ones worth fetching;
    hyphenated slugs already de-hyphenate + titlecase to a readable name."""
    return bool(slug) and not re.search(r"[-_.\s]", slug)


def _title_name(url: str) -> str | None:
    """Company name from a board page's <title>, with the 'X Jobs/Careers' chrome stripped."""
    try:
        r = requests.get(url, headers=UA, timeout=TIMEOUT)
        if r.status_code != 200:
            return None
        m = _TITLE.search(r.text)
        if not m:
            return None
        t = _TITLE_CHROME.sub("", html.unescape(m.group(1)).strip()).strip()
        return t or None
    except requests.RequestException:
        return None


def _fetch(slug: str, source: str) -> str | None:
    """The board's display name for a slug, or None (404 / no name / network error)."""
    try:
        if source == "greenhouse":
            r = requests.get(
                f"https://boards-api.greenhouse.io/v1/boards/{slug}", headers=UA, timeout=TIMEOUT
            )
            return (r.json().get("name") or "").strip() or None if r.status_code == 200 else None
        if source == "smartrecruiters":
            r = requests.get(
                f"https://api.smartrecruiters.com/v1/companies/{slug}/postings?limit=1",
                headers=UA,
                timeout=TIMEOUT,
            )
            if r.status_code != 200:
                return None
            content = r.json().get("content") or []
            return (
                ((content[0].get("company") or {}).get("name") or "").strip() or None
                if content
                else None
            )
        if source == "ashby":
            return _title_name(f"https://jobs.ashbyhq.com/{slug}")
        if source == "lever":
            return _title_name(f"https://jobs.lever.co/{slug}")
    except (requests.RequestException, ValueError, KeyError, IndexError):
        return None
    return None


# id-prefix -> the ATS we know how to query
SOURCES = {"greenhouse", "ashby", "lever", "smartrecruiters"}


def _targets() -> dict[str, str]:
    """{slug: source} for single-token slugs of a fetchable ATS, minus the slugs that
    curated or the description extractor already name (those take precedence)."""
    skip = set()
    for p in (CURATED, EXTRACTED):
        if p.exists():
            with open(p) as f:
                skip |= {k for k in json.load(f) if not k.startswith("_")}
    out: dict[str, str] = {}
    with open(META) as f:
        for line in f:
            r = json.loads(line)
            slug = (r.get("source_slug") or "").strip()
            if not slug or slug in skip or slug in out or not _single_token(slug):
                continue
            src = (r.get("id") or "").split(":", 1)[0]
            if src in SOURCES:
                out[slug] = src
    return out


def build(workers=8, limit=0, audit=0):
    targets = _targets()
    if limit:
        targets = dict(list(targets.items())[:limit])
    cache: dict[str, str | None] = {}
    if CACHE.exists():
        with open(CACHE) as f:
            cache = json.load(f)
    todo = [(s, src) for s, src in targets.items() if s not in cache]
    print(
        f"targets {len(targets):,} | cached {len(targets) - len(todo):,} | to fetch {len(todo):,}",
        flush=True,
    )

    done = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_fetch, s, src): s for s, src in todo}
        for fut in as_completed(futs):
            slug = futs[fut]
            cache[slug] = fut.result()
            done += 1
            if done % 200 == 0:
                with open(CACHE, "w") as f:
                    json.dump(cache, f, ensure_ascii=False)
                print(f"  fetched {done:,}/{len(todo):,}", flush=True)
    with open(CACHE, "w") as f:
        json.dump(cache, f, ensure_ascii=False)

    # keep only names that improve on the prettifier (and survive case normalization)
    out: dict[str, str] = {}
    by_src = Counter()
    for slug, src in targets.items():
        name = cache.get(slug)
        if not name:
            continue
        name = _clean_board_name(name)  # strip board chrome; drop generic/internal/overlong
        if not name:
            continue
        name = _normalize_case(name)
        if not name:
            continue
        pf = _prettify(slug)
        if name == pf:
            continue
        # Drop a name that only differs from the prettified slug by casing AND adds no
        # internal-capital signal ('adaption' vs 'Adaption' -> keep the titlecase). A
        # name with an internal capital ('Aim4Hire') or different letters ('Inscribe'
        # <- 'InscribeAI') is a real improvement and stays.
        if name.casefold() == pf.casefold() and not any(c.isupper() for c in name[1:]):
            continue
        out[slug] = name
        by_src[src] += 1
    print(f"\nkept {len(out):,} names | by source: {by_src.most_common()}", flush=True)

    if audit:
        import random

        random.seed(13)
        items = list(out.items())
        random.shuffle(items)
        print(f"\n--- {min(audit, len(items))} samples (slug -> name | was prettify) ---")
        for slug, name in items[:audit]:
            print(f"  {slug:26s} [{targets[slug]:14s}] -> {name!r:30s} (was {_prettify(slug)!r})")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="write the ATS-names file")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0, help="cap targets (for a test run)")
    ap.add_argument("--audit", type=int, default=0)
    args = ap.parse_args()
    out = build(workers=args.workers, limit=args.limit, audit=args.audit)
    if args.apply:
        with open(DEST, "w") as f:
            json.dump(out, f, indent=0, sort_keys=True, ensure_ascii=False)
        print(f"\nwrote {len(out):,} ATS names -> {DEST}")


if __name__ == "__main__":
    main()
