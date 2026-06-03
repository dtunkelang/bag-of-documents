#!/usr/bin/env python3
"""Mine clean, canonical French role suggestions from France Travail titles.

France Travail titles are noisy: gender markers (H/F, F/H, H/F/X, (H/F), ·e,
(e)/(ne)/(se)), location parentheticals and #codes, "Copy of"/"Groupe X -"
prefixes, and dual-gender slash forms ("Logisticien / Logisticienne"). This
strips that down to a canonical masculine head, frequency-ranks the cleaned
forms, and emits the top-N as a curated French autocomplete tier (fr_roles.json),
shipped alongside app.py and merged into the suggestion corpus at load time.
"""

import json
import os
import re
import sys
import unicodedata
from collections import Counter

import requests

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "space"))
from suggest_lib import degender_fr  # noqa: E402  (collapse feminine head variants)

# First tokens that are NOT occupations: locations, employment-status / structural words,
# and English tokens from the occasional English posting. They recur widely enough to clear
# the head-promotion bar but must not become standalone role suggestions.
_NONROLE_HEAD = {
    "lyon",
    "paris",
    "secteur",
    "service",
    "poste",
    "offre",
    "cours",
    "contrat",
    "recherche",
    "pole",
    "industrie",
    "futur",
    "premier",
    "second",
    "un",
    "personnel",
    "extra",
    "formation",
    "maintenance",
    "apprentissage",
    "apprenti",
    "alternant",
    "etudiant",
    "independant",
    "franchise",
    "data",
    "business",
    "support",
    "lead",
    "leader",
    "runner",
    "assitant",
}

SOLR = "http://localhost:8983/solr/jobs"
FQ = "source_corpus:jobs_data_francetravail"
TOP_N = 700
MIN_COUNT = 8  # a cleaned form must recur this often to be a "canonical" role

# ---- cleaning regexes (applied in order) ----
# gender markers anywhere: H/F, F/H, H/F/X, M/F, (H/F), [H/F], with optional spaces
_GENDER = re.compile(r"\(?\[?\s*\b[hfmx](?:\s*/\s*[hfmx])+\b\s*\]?\)?", re.I)
# inclusive-writing feminine suffixes: ·e ·ne ·se (middot/period/hyphen), e.g. "ingénieur-e",
# "agent·e", "employé.e" -> masculine head
_INCLUSIVE = re.compile(r"[·.\-]\s*(?:e|ne|se|ère|euse|rice|trice)\b", re.I)
_PAREN_SUFFIX = re.compile(r"\(\s*(?:e|ne|se|ère|euse|rice|trice)\s*\)", re.I)
# req/ref codes: #TET13254, ref 12345
_CODE = re.compile(r"#\S+|\bref\.?\s*\S+", re.I)
# location/anything in parens or brackets (Rodez), (12230), [remote]
_PARENS = re.compile(r"[\(\[][^\)\]]*[\)\]]")
# leading company/prefix junk: "Copy of ", "Groupe RAGT - ", "ENTREPRISE - "
_PREFIX = re.compile(r"^(?:copy of\s+|[A-ZÀ-Ý][\w&'.\- ]{1,30}\s+[-–—]\s+)", re.I)
# trailing " - <freetext>" segment (location/contract noise after a dash)
_TRAIL_DASH = re.compile(r"\s+[-–—]\s+.*$")
# dual-gender / alternative-role list: keep the masculine head before the first "/" or
# "," ("Animateur/Animatrice loisirs" -> "Animateur", "Factrice, distributrice" ->
# "Factrice"). The spaced " / " form is covered by this too.
_VARIANT_CUT = re.compile(r"\s*[/,].*$")
_WS = re.compile(r"\s+")


# non-role noise that survives cleaning (contract types, work arrangements)
_STOP = {
    "cdi",
    "cdd",
    "cdief",
    "alternance",
    "stage",
    "stagiaire",
    "interim",
    "intérim",
    "freelance",
    "temps plein",
    "temps partiel",
    "saisonnier",
    "h/f",
    "f/h",
    "débutant accepté",
    "urgent",
    # English tokens that leak in from the occasional English-language FT posting
    "project",
    "sales",
    "tech",
    "senior",
    "junior",
    # verbs / meta words that survive cleaning as a bare token but aren't roles
    "devenez",
    "emploi",
    "salon",
    "nuit",
    "un",
    "technico",
}


_STOP_FOLDED = {
    s.replace("'", "").replace("’", "")
    for s in (
        unicodedata.normalize("NFKD", x).encode("ascii", "ignore").decode().lower() for x in _STOP
    )
}


def fold(s: str) -> str:
    """Lowercase + strip diacritics and apostrophes (for dedup + accent-
    insensitive matching). FT strips curly apostrophes inconsistently, so
    folding them out merges "d'orientation"/"dorientation" into one key."""
    nfkd = unicodedata.normalize("NFKD", s)
    base = "".join(c for c in nfkd if not unicodedata.combining(c)).lower()
    return base.replace("'", "").replace("’", "")


def clean_title(t: str) -> str:
    s = t.strip()
    s = _PREFIX.sub("", s)
    s = _GENDER.sub(" ", s)
    s = _CODE.sub(" ", s)
    s = _INCLUSIVE.sub("", s)
    s = _PAREN_SUFFIX.sub("", s)
    s = _PARENS.sub(" ", s)
    s = _VARIANT_CUT.sub("", s)  # drop dual-gender / alternative-role list after first / or ,
    s = _TRAIL_DASH.sub("", s)
    s = _WS.sub(" ", s).strip(" -–—·.,\"'")
    return s.lower()


def fetch_titles() -> list[str]:
    out, mark = [], "*"
    while True:
        r = requests.get(
            f"{SOLR}/select",
            params={
                "q": "*:*",
                "fq": FQ,
                "fl": "title_display",
                "rows": "5000",
                "sort": "id asc",
                "cursorMark": mark,
                "wt": "json",
            },
            timeout=60,
        )
        r.raise_for_status()
        d = r.json()
        docs = d["response"]["docs"]
        out += [doc.get("title_display", "") for doc in docs if doc.get("title_display")]
        nxt = d.get("nextCursorMark")
        if not docs or nxt == mark:
            break
        mark = nxt
    return out


def main():
    titles = fetch_titles()
    print(f"fetched {len(titles):,} FT titles", file=sys.stderr)
    # count by folded key, but keep the most common accented surface form
    counts: Counter[str] = Counter()
    surface: dict[str, Counter[str]] = {}
    # Bare-head counts: most FT dev/lawyer titles are qualified ("développeur web",
    # "avocat droit social"), so no single cleaned form hits MIN_COUNT and the bare head
    # ("développeur", "avocat") never makes the list — yet it's the most useful
    # autocomplete + resume key. Accumulate the leading token across all titles and emit
    # heads that recur widely and span several distinct full forms (a real role family,
    # not a one-off). variants[head] tracks distinct continuations for that diversity test.
    head_counts: Counter[str] = Counter()
    head_surface: dict[str, Counter[str]] = {}
    head_variants: dict[str, set[str]] = {}
    for t in titles:
        c = clean_title(t)
        if not (2 <= len(c) <= 48):
            continue
        if not re.search(r"[a-zà-ÿ]", c):
            continue
        if len(c.split()) > 6:
            continue
        k = fold(c)
        if k in _STOP_FOLDED:
            continue
        counts[k] += 1
        surface.setdefault(k, Counter())[c] += 1
        toks = c.split()
        head, fhead = toks[0], k.split()[0]
        if len(fhead) >= 4 and fhead not in _STOP_FOLDED and re.search(r"[a-zà-ÿ]", fhead):
            head_counts[fhead] += 1
            head_surface.setdefault(fhead, Counter())[head] += 1
            head_variants.setdefault(fhead, set()).add(k)
    HEAD_MIN, HEAD_VARIANTS = 12, 3  # bar for promoting a bare head to a standalone role
    for fhead, n in head_counts.items():
        if n < HEAD_MIN or len(head_variants[fhead]) < HEAD_VARIANTS:
            continue
        if fhead in _NONROLE_HEAD:
            continue
        if fhead[-1] in "sx" and fhead[:-1] in head_counts:  # plural -> singular present
            continue
        dg = degender_fr(fhead)  # feminine -> masculine present (vendeuse->vendeur)
        if dg != fhead and (dg in head_counts or dg in counts):
            continue
        # boost to the aggregate head count so the bare role outranks its own extensions
        # (it's the better autocomplete/resume key); covers heads that never occurred bare
        # AND bare heads too rare on their own to clear MIN_COUNT.
        counts[fhead] = max(counts.get(fhead, 0), n)
        surface.setdefault(fhead, head_surface[fhead])
    ranked = [
        (surface[k].most_common(1)[0][0], n) for k, n in counts.most_common() if n >= MIN_COUNT
    ][:TOP_N]
    print(f"{len(ranked)} canonical FR roles (>= {MIN_COUNT} occurrences)", file=sys.stderr)
    out = [{"text": role, "n": n} for role, n in ranked]
    with open("space/fr_roles.json", "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=0)
    print("wrote space/fr_roles.json", file=sys.stderr)
    for role, n in ranked[:30]:
        print(f"  {n:>5}  {role}", file=sys.stderr)


if __name__ == "__main__":
    main()
