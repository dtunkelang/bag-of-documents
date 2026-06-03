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
import re
import sys
import unicodedata
from collections import Counter

import requests

SOLR = "http://localhost:8983/solr/jobs"
FQ = "source_corpus:jobs_data_francetravail"
TOP_N = 500
MIN_COUNT = 8  # a cleaned form must recur this often to be a "canonical" role

# ---- cleaning regexes (applied in order) ----
# gender markers anywhere: H/F, F/H, H/F/X, M/F, (H/F), [H/F], with optional spaces
_GENDER = re.compile(r"\(?\[?\s*\b[hfmx](?:\s*/\s*[hfmx])+\b\s*\]?\)?", re.I)
# inclusive-writing feminine suffixes: ·e ·ne ·se (middot/period), (e) (ne) (se) (trice) (rice)
_INCLUSIVE = re.compile(r"[·.]\s*(?:e|ne|se|ère|euse|rice|trice)\b", re.I)
_PAREN_SUFFIX = re.compile(r"\(\s*(?:e|ne|se|ère|euse|rice|trice)\s*\)", re.I)
# req/ref codes: #TET13254, ref 12345
_CODE = re.compile(r"#\S+|\bref\.?\s*\S+", re.I)
# location/anything in parens or brackets (Rodez), (12230), [remote]
_PARENS = re.compile(r"[\(\[][^\)\]]*[\)\]]")
# leading company/prefix junk: "Copy of ", "Groupe RAGT - ", "ENTREPRISE - "
_PREFIX = re.compile(r"^(?:copy of\s+|[A-ZÀ-Ý][\w&'.\- ]{1,30}\s+[-–—]\s+)", re.I)
# trailing " - <freetext>" segment (location/contract noise after a dash)
_TRAIL_DASH = re.compile(r"\s+[-–—]\s+.*$")
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
    # dual-gender slash: keep the masculine head before " / " when the slash
    # separates two role-word variants (heuristic: both sides start with letters)
    if " / " in s:
        head = s.split(" / ", 1)[0]
        if head.strip():
            s = head
    s = _TRAIL_DASH.sub("", s)
    s = _WS.sub(" ", s).strip(" -–—·.,")
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
