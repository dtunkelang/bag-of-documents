#!/usr/bin/env python3
"""Loader + relatedness over the harvested ESCO occupation backbone.

Reads .esco_records.jsonl (produced by build_esco_backbone.py) and exposes the
three primitives the de/nl/es/it (and SSYK-bridged sv) language lanes need:

  1. label lookup    : a folded occupation label in language L -> ESCO occupation
  2. ISCO bridge     : ISCO code <-> occupations  (the crosswalk to ROME/SSYK, which
                       ESCO does not map directly but both align to ISCO-08)
  3. relatedness     : occupation -> ranked related occupations

ESCO has NO occupation->occupation mobility graph (unlike ROME's `mobilite`). So
relatedness is DERIVED from shared skills: two occupations are related to the extent
their (essential + optional) skill sets overlap. We weight essential skills higher
than optional, and divide by the smaller skill set (overlap coefficient, not Jaccard)
so a specialised occupation with few skills can still be strongly related to a broad
one it is fully contained in — the lateral-move semantics the French lane wants.

This module is corpus-agnostic backbone; the per-language display vocabulary and the
query/corpus-role -> occupation resolution live in the per-language lane builders.
"""

from __future__ import annotations

import json
import os
import re
import unicodedata
from collections import defaultdict
from functools import lru_cache

HERE = os.path.dirname(os.path.abspath(__file__))
RECORDS = os.path.join(HERE, ".esco_records.jsonl")

_WS = re.compile(r"\s+")


def fold(s: str) -> str:
    """Accent/punct-insensitive label key (matches build_fr_related._csv_fold). German ß
    is mapped to 'ss' BEFORE the punct strip (NFKD leaves ß intact, and [^a-z0-9] would
    otherwise turn it into a space), so 'Schweißer'/'schweisser' share a key with the
    corpus and the app-side fold."""
    nfkd = unicodedata.normalize("NFKD", (s or "").lower().replace("ß", "ss"))
    base = "".join(c for c in nfkd if not unicodedata.combining(c))
    return _WS.sub(" ", re.sub(r"[^a-z0-9]+", " ", base)).strip()


class EscoBackbone:
    def __init__(self, path: str = RECORDS):
        self.by_uri: dict[str, dict] = {}
        self.isco2uris: dict[str, list[str]] = defaultdict(list)
        # folded label (per language) -> occupation URIs
        self.label2uris: dict[str, dict[str, set[str]]] = {}  # lang -> {key -> {uri}}
        self._skill_owners: dict[str, set[str]] = defaultdict(set)  # skill uri -> occ uris
        self._load(path)

    def _load(self, path: str) -> None:
        with open(path) as f:
            for line in f:
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                uri = r["uri"]
                self.by_uri[uri] = r
                isco = (r.get("isco") or "").split(".")[0]  # 4-digit unit group
                if isco:
                    self.isco2uris[isco].append(uri)
                for skill in r.get("essential", []) + r.get("optional", []):
                    self._skill_owners[skill].add(uri)
        # build per-language folded label index (preferred + alts)
        for uri, r in self.by_uri.items():
            for lang, lab in (r.get("preferredLabel") or {}).items():
                self.label2uris.setdefault(lang, defaultdict(set))[fold(lab)].add(uri)
            for lang, alts in (r.get("altLabels") or {}).items():
                idx = self.label2uris.setdefault(lang, defaultdict(set))
                for a in alts if isinstance(alts, list) else [alts]:
                    idx[fold(a)].add(uri)

    # ---- label lookup ----
    def resolve(self, text: str, lang: str) -> list[str]:
        """Folded-exact occupation URIs for a label in `lang` (preferred or alt)."""
        return sorted(self.label2uris.get(lang, {}).get(fold(text), set()))

    def label(self, uri: str, lang: str) -> str:
        r = self.by_uri.get(uri, {})
        return (r.get("preferredLabel") or {}).get(lang) or (r.get("preferredLabel") or {}).get(
            "en", ""
        )

    # ---- relatedness via shared skills (overlap coefficient, essential-weighted) ----
    def _skills(self, uri: str, w_ess: float = 1.0, w_opt: float = 0.5) -> dict[str, float]:
        r = self.by_uri.get(uri, {})
        s = {sk: w_ess for sk in r.get("essential", [])}
        for sk in r.get("optional", []):
            s.setdefault(sk, w_opt)
        return s

    def related(
        self, uri: str, top: int = 12, min_overlap: float = 0.20
    ) -> list[tuple[str, float]]:
        """Ranked related occupation URIs by weighted skill-overlap coefficient.
        Candidates come from the skill inverted index (occupations sharing >=1 skill)."""
        sa = self._skills(uri)
        if not sa:
            return []
        denom_a = sum(sa.values())
        cand: set[str] = set()
        for sk in sa:
            cand |= self._skill_owners.get(sk, set())
        cand.discard(uri)
        scored = []
        for c in cand:
            sb = self._skills(c)
            shared = sum(min(sa[k], sb[k]) for k in sa.keys() & sb.keys())
            if not shared:
                continue
            # overlap coefficient against the SMALLER skill mass -> containment-friendly
            score = shared / min(denom_a, sum(sb.values()))
            if score >= min_overlap:
                scored.append((c, score))
        scored.sort(key=lambda t: -t[1])
        return scored[:top]


@lru_cache(maxsize=1)
def backbone() -> EscoBackbone:
    return EscoBackbone()


if __name__ == "__main__":
    bb = backbone()
    print(f"loaded {len(bb.by_uri):,} occupations, {len(bb._skill_owners):,} distinct skills")
    # sanity probe: a few seed occupations -> related (shown in EN for readability)
    seeds = ["plumber", "registered nurse", "software developer", "cook", "electrician"]
    for seed in seeds:
        uris = bb.resolve(seed, "en")
        if not uris:
            print(f"\n[{seed}] not found")
            continue
        u = uris[0]
        print(f"\n[{seed}]  ISCO {bb.by_uri[u].get('isco')}  (de: {bb.label(u, 'de')})")
        for c, sc in bb.related(u, top=8):
            print(f"   {sc:.2f}  {bb.label(c, 'en')}   |de {bb.label(c, 'de')}")
