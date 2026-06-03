#!/usr/bin/env python3
"""Harvest the ESCO multilingual occupation backbone from the open ESCO API.

ESCO (European Skills, Competences, Qualifications and Occupations) is the EU's
free occupation taxonomy in 28 languages. It is the grounding backbone for the
de/nl/es/it (and, via SSYK<->ISCO, sv) language lanes — the multilingual analogue
of the ROME data that grounds the French related-search lane (build_fr_related.py).

Why the open API and not the bulk CSV: the official CSV bundle is email-gated
(form -> email -> one-time link), so it can't be re-fetched autonomously. The open
API (https://ec.europa.eu/esco/api, no auth) returns the SAME data — and crucially
gives one occupation's full record (all 28 preferredLabels + ~20 altLabel langs +
ISCO code + essential/optional skill links) in a single call. So we harvest once
into a local cache and rebuild per-language bundles offline, exactly like the cached
ROME zip (.rome_opendata.zip). Re-run only on an ESCO version bump.

KEY STRUCTURAL FACT: ESCO has NO occupation->occupation mobility graph like ROME's
`mobilite professionnelle`. Occupation relatedness must be DERIVED from shared skills
(occupation -> essential/optional skills -> other occupations). The skill URIs
captured here are the raw material for that relatedness graph (built downstream).

Crosswalks: ESCO<->ISCO is built in (every occupation carries an ISCO `code`). That
ISCO code is the bridge to SSYK (SSYK-2012 is ISCO-08 based) and to ROME (via an
ISCO mapping); ESCO does NOT publish a direct ROME or SSYK crosswalk.

Two resumable phases, both cached so a re-run is cheap / interruptible:
  Phase A  BFS the ISCO hierarchy (C0..C9 -> sub-groups -> ... -> occupation leaves)
           to enumerate every occupation URI            -> .esco_occ_uris.json
  Phase B  fetch each occupation's full record (labels + ISCO code + skills),
           appending to a JSONL so a crash/Ctrl-C resumes -> .esco_records.jsonl

Finally prints per-language label coverage for the lanes we care about so we can
verify de/nl/es/it/sv are well populated before building anything on top.

Usage:
  .venv/bin/python build_esco_backbone.py            # resume/continue harvest
  .venv/bin/python build_esco_backbone.py --stats     # just re-print coverage from cache
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

API = "https://ec.europa.eu/esco/api"
VERSION = "v1.2.0"  # pin so a harvest is reproducible across ESCO bumps
HERE = os.path.dirname(os.path.abspath(__file__))
URIS_CACHE = os.path.join(HERE, ".esco_occ_uris.json")
RECORDS_CACHE = os.path.join(HERE, ".esco_records.jsonl")

# The lanes we are standing up (plus en/fr already shipped, for a sanity baseline).
LANES = ("de", "nl", "es", "it", "sv", "fr", "en")

ISCO_ROOTS = [f"C{i}" for i in range(10)]  # the 10 ISCO-08 major groups
_OCC_MARK = "/occupation/"
_ISCO_MARK = "/isco/"


def _get(path: str, params: dict, retries: int = 4) -> dict:
    """GET an ESCO API resource as JSON, with backoff. Returns {} on hard failure."""
    qs = urllib.parse.urlencode(params)
    url = f"{API}{path}?{qs}"
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"Accept": "application/json"})
            with urllib.request.urlopen(req, timeout=45) as r:
                return json.loads(r.read())
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError) as e:
            if attempt == retries - 1:
                print(f"  ! giving up on {url}: {e}", file=sys.stderr)
                return {}
            time.sleep(1.5 * (attempt + 1))
    return {}


# ----------------------------------------------------------------------------- #
# Phase A: enumerate occupation URIs by walking the ISCO hierarchy.
# ----------------------------------------------------------------------------- #
def _children(uri: str) -> list[str]:
    """All narrower URIs of a concept. ISCO groups expose sub-GROUPS via
    `narrowerConcept` but their leaf occupations via `narrowerOccupation`; ESCO
    occupations themselves can have narrower (sub-)occupations under the same key."""
    d = _get(
        "/resource/concept",
        {"uri": uri, "language": "en", "selectedVersion": VERSION},
    )
    links = d.get("_links", {}) or {}
    out = []
    for key in ("narrowerConcept", "narrowerOccupation"):
        out += [c.get("uri", "") for c in (links.get(key) or []) if c.get("uri")]
    return out


def enumerate_occupations() -> list[str]:
    if os.path.exists(URIS_CACHE):
        with open(URIS_CACHE) as f:
            uris = json.load(f)
        print(f"[A] {len(uris):,} occupation URIs (cached)", file=sys.stderr)
        return uris

    occ: set[str] = set()
    # BFS the ISCO group tree down to occupation leaves, then keep recursing through
    # occupations (which can have narrower sub-occupations). ISCO groups never re-add.
    frontier = [f"http://data.europa.eu/esco/isco/{g}" for g in ISCO_ROOTS]
    seen: set[str] = set()
    depth = 0
    while frontier:
        depth += 1
        next_frontier: list[str] = []
        with ThreadPoolExecutor(max_workers=12) as ex:
            futs = {ex.submit(_children, u): u for u in frontier if u not in seen}
            seen.update(futs.values())
            for fut in as_completed(futs):
                for child in fut.result():
                    if _OCC_MARK in child:
                        if child not in occ:
                            occ.add(child)
                            next_frontier.append(child)  # may have sub-occupations
                    elif _ISCO_MARK in child and child not in seen:
                        next_frontier.append(child)
        print(
            f"[A] depth {depth}: frontier {len(next_frontier)}, {len(occ):,} occupations so far",
            file=sys.stderr,
        )
        frontier = next_frontier

    uris = sorted(occ)
    with open(URIS_CACHE, "w") as f:
        json.dump(uris, f)
    print(f"[A] enumerated {len(uris):,} occupation URIs -> {URIS_CACHE}", file=sys.stderr)
    return uris


# ----------------------------------------------------------------------------- #
# Phase B: fetch each occupation's full record (resumable JSONL append).
# ----------------------------------------------------------------------------- #
def _slim(d: dict) -> dict:
    """Keep only what the language lanes need from a full occupation record."""
    links = d.get("_links", {}) or {}

    def skill_uris(key: str) -> list[str]:
        return [s.get("uri", "") for s in (links.get(key) or []) if s.get("uri")]

    return {
        "uri": d.get("uri", ""),
        "isco": d.get("code", ""),
        "preferredLabel": d.get("preferredLabel", {}),  # {lang: "label"}
        "altLabels": d.get("alternativeLabel", {}),  # {lang: ["a","b",...]}
        "essential": skill_uris("hasEssentialSkill"),
        "optional": skill_uris("hasOptionalSkill"),
    }


def _fetch_one(uri: str) -> dict | None:
    d = _get(
        "/resource/occupation",
        {"uri": uri, "selectedVersion": VERSION},
    )
    if not d.get("uri"):
        return None
    return _slim(d)


def harvest(uris: list[str]) -> None:
    done: set[str] = set()
    if os.path.exists(RECORDS_CACHE):
        with open(RECORDS_CACHE) as f:
            for line in f:
                try:
                    done.add(json.loads(line)["uri"])
                except (json.JSONDecodeError, KeyError):
                    pass
    todo = [u for u in uris if u not in done]
    print(f"[B] {len(done):,} cached, {len(todo):,} to fetch", file=sys.stderr)
    if not todo:
        return

    n = 0
    t0 = time.time()
    # append-as-we-go so a Ctrl-C / crash loses at most the in-flight batch.
    with open(RECORDS_CACHE, "a") as out, ThreadPoolExecutor(max_workers=10) as ex:
        futs = {ex.submit(_fetch_one, u): u for u in todo}
        for fut in as_completed(futs):
            rec = fut.result()
            if rec:
                out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n += 1
                if n % 200 == 0:
                    out.flush()
                    rate = n / (time.time() - t0)
                    eta = (len(todo) - n) / rate / 60 if rate else 0
                    print(f"[B] {n:,}/{len(todo):,}  {rate:.1f}/s  ETA {eta:.1f}m", file=sys.stderr)
    print(f"[B] fetched {n:,} records -> {RECORDS_CACHE}", file=sys.stderr)


# ----------------------------------------------------------------------------- #
# Verify multilingual coverage.
# ----------------------------------------------------------------------------- #
def stats() -> None:
    if not os.path.exists(RECORDS_CACHE):
        print("no records cache yet", file=sys.stderr)
        return
    total = 0
    pref = Counter()
    alt = Counter()
    isco_missing = 0
    skill_any = 0
    with open(RECORDS_CACHE) as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            total += 1
            if not r.get("isco"):
                isco_missing += 1
            if r.get("essential") or r.get("optional"):
                skill_any += 1
            for lang in LANES:
                if r.get("preferredLabel", {}).get(lang):
                    pref[lang] += 1
                if r.get("altLabels", {}).get(lang):
                    alt[lang] += 1
    print(f"\n=== ESCO backbone coverage ({total:,} occupations, {VERSION}) ===")
    print(f"  ISCO code present : {total - isco_missing:,}/{total:,}")
    print(f"  has any skill link: {skill_any:,}/{total:,}  (relatedness source)")
    print(f"  {'lang':<5}{'preferredLabel':>16}{'altLabels':>12}")
    for lang in LANES:
        print(f"  {lang:<5}{pref[lang]:>16,}{alt[lang]:>12,}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stats", action="store_true", help="just print coverage from cache")
    args = ap.parse_args()
    if args.stats:
        stats()
        return
    uris = enumerate_occupations()
    harvest(uris)
    stats()


if __name__ == "__main__":
    main()
