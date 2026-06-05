#!/usr/bin/env python3
"""Recover correctly-cased company names from job descriptions (open-weight, no LLM).

The `employer` Solr field stores only the ATS board slug (e.g. "toyotaconnected",
"3imembers", "15five"). app.py `_pretty_employer` de-hyphenates + titlecases, which
reads well for hyphenated slugs but CANNOT split a concatenated single-token slug
-> "Toyotaconnected", "3imembers", "15Five". The real, correctly-cased name almost
always sits near the top of the posting body ("Toyota Connected is expanding...",
"About 3i Members", "15Five is the AI-powered...").

SLUG-ANCHORED extraction (precision-first, safe): scan the contiguous word spans at
the start of the description; the span whose lowercased-alphanumeric concatenation
EQUALS the slug (optionally minus a trailing corp suffix like inc/llc/gmbh) is the
company name in its own casing/spacing. Anchoring to the slug is what makes this
safe -- a random org name in the text won't concatenate to this board's slug. We
vote across the board's postings and keep the modal extraction. Output is a
slug->name map (employer_names_extracted.json) consulted by _pretty_employer AFTER
the hand-curated employer_names.json, so curation always wins.

DISPLAY ONLY: the raw slug stays the Solr filter key (the company-pivot link), so
no re-index is needed -- this is an app.py-side change.
"""

from __future__ import annotations

import argparse
import html
import json
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
META = ROOT / "unified_jobs_daily" / "metadata.jsonl"
CURATED = HERE / "space" / "employer_names.json"
DEST = HERE / "space" / "employer_names_extracted.json"

SCAN_TOKENS = 40  # company name sits at the very top; cap the scan window
MAX_NAME_TOKENS = 6  # company names are short; bounds the span search
MIN_VOTES = 1  # a slug needs this many agreeing extractions to be kept
# trailing legal-entity suffixes the slug may carry but the prose usually omits
# ("143studiosinc" <- "143 Studios"); tried as a fallback when the full slug misses.
CORP_SUFFIX = (
    "incorporated",
    "inc",
    "corp",
    "corporation",
    "llc",
    "llp",
    "ltd",
    "limited",
    "plc",
    "gmbh",
    "srl",
    "sarl",
    "spa",
    "sa",
    "ag",
    "bv",
    "co",
    "company",
    "group",
    "holding",
    "holdings",
)

_TAG = re.compile(r"<[^>]+>")
_WS = re.compile(r"\s+")


def _alnum(s: str) -> str:
    """lowercase, strip accents, keep only [a-z0-9] -> the slug's character class."""
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    return re.sub(r"[^a-z0-9]+", "", s.lower())


def _clean_desc(d: str) -> str:
    """HTML -> text: drop tags, unescape entities, normalize NBSP/BOM/whitespace."""
    d = _TAG.sub(" ", d or "")
    d = html.unescape(d).replace(" ", " ").replace("﻿", "")
    return _WS.sub(" ", d).strip()


def _targets(slug: str) -> list[str]:
    """slug variants to anchor on: the full slug, then the slug minus one trailing
    corp suffix (so '143studiosinc' can match the prose '143 Studios')."""
    out = [slug]
    for suf in CORP_SUFFIX:
        if slug.endswith(suf) and len(slug) > len(suf) + 1:
            out.append(slug[: -len(suf)])
            break
    return out


def extract(slug: str, desc: str) -> str | None:
    """The earliest, shortest contiguous word span at the top of `desc` whose
    lowercased-alnum concatenation equals `slug` (or slug-minus-corp-suffix).
    Returns the span in its ORIGINAL casing/spacing, or None."""
    targets = _targets(slug)
    raw = _clean_desc(desc).split()[:SCAN_TOKENS]
    # original token + its alnum form (skip tokens that contribute no alnum)
    toks = [(t, _alnum(t)) for t in raw]
    toks = [(t, a) for t, a in toks if a]
    for i in range(len(toks)):
        acc = ""
        for j in range(i, min(i + MAX_NAME_TOKENS, len(toks))):
            acc += toks[j][1]
            if acc in targets:
                span = [toks[k][0].strip(".,;:!?()[]{}\"'") for k in range(i, j + 1)]
                return " ".join(w for w in span if w)
            if not any(t.startswith(acc) for t in targets):
                break  # this start can't extend to any target -> abandon
    return None


def _prettify(slug: str) -> str:
    """Mirror of app.py _pretty_employer's fallback (de-hyphenate + titlecase), so we
    only emit an extraction that actually IMPROVES on what the prettifier produces."""
    s = re.sub(r"[-.]com$", "", slug.strip())
    s = re.sub(r"-\d+$", "", s)
    words = [w for w in re.split(r"[-_.\s]+", s) if w]
    return " ".join(w[:1].upper() + w[1:] for w in words)


def _normalize_case(name: str) -> str | None:
    """Tame ALL-CAPS extractions, which come from shouting headers ('MERLIN IS
    HIRING') and read worse than the prettifier's titlecase. A short single token is
    kept as a genuine acronym (YNAB, CMA); a long single token is dropped so the
    prettifier titlecases it ('MERLIN' -> 'Merlin'); a shouting multi-word name is
    titlecased here ('HERSCHEL SUPPLY' -> 'Herschel Supply'), since the prettifier
    can't split the concatenated slug at all. Mixed-case names pass through."""
    if not name.isupper():
        return name
    words = name.split()
    if len(words) == 1:
        return name if len(name) <= 5 else None  # acronym vs shouting single word
    return name.title()


def build(meta_path=META, curated_path=CURATED, min_votes=MIN_VOTES, audit=0):
    curated = set()
    if Path(curated_path).exists():
        with open(curated_path) as f:
            curated = {k for k in json.load(f) if not k.startswith("_")}

    votes: dict[str, Counter] = defaultdict(Counter)
    with open(meta_path) as f:
        for line in f:
            r = json.loads(line)
            slug = (r.get("source_slug") or "").strip()
            if not slug or slug in curated:
                continue
            name = extract(slug, r.get("description") or "")
            if name:
                votes[slug][name] += 1

    out: dict[str, str] = {}
    for slug, c in votes.items():
        name, n = c.most_common(1)[0]
        if n < min_votes:
            continue
        name = _normalize_case(name)
        if not name or name == _prettify(slug):
            continue  # adds nothing over the prettifier
        # A SINGLE-token extraction is only trustworthy when it carries an internal
        # capital -- a mixed-case brand ('iHerb', '15Five', 'TrueCare') or an acronym
        # ('YNAB', 'VTEX'). A plain-lowercase / plain-titlecase single token is either
        # the prettifier's job anyway or a common-word FALSE POSITIVE (a slug that
        # happens to be an English word -- 'acquisition', 'futures' -- matching that
        # word in the prose); drop it and let the prettifier titlecase the slug.
        # Multi-word splits are kept regardless (the prettifier can't make them).
        if " " not in name and not any(c.isupper() for c in name[1:]):
            continue
        out[slug] = name

    if audit:
        import random

        random.seed(13)
        items = list(out.items())
        random.shuffle(items)
        print(f"\n--- {min(audit, len(items))} sample extractions (slug -> name | prettify) ---")
        for slug, name in items[:audit]:
            print(f"  {slug:28s} -> {name!r:32s} (was {_prettify(slug)!r})")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="write the extracted-names file")
    ap.add_argument("--audit", type=int, default=0, help="print N sample extractions")
    ap.add_argument("--min-votes", type=int, default=MIN_VOTES)
    args = ap.parse_args()

    out = build(min_votes=args.min_votes, audit=args.audit)

    # coverage in documents (how many postings get a better employer label)
    covered_docs = 0
    with open(META) as f:
        for line in f:
            slug = (json.loads(line).get("source_slug") or "").strip()
            if slug in out:
                covered_docs += 1
    print(f"\nslugs with a recovered name: {len(out):,}")
    print(f"documents covered: {covered_docs:,}")

    if args.apply:
        with open(DEST, "w") as f:
            json.dump(out, f, indent=0, sort_keys=True, ensure_ascii=False)
        print(f"wrote {len(out):,} extracted names -> {DEST}")


if __name__ == "__main__":
    main()
