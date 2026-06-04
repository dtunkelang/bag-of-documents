#!/usr/bin/env python3
"""Mine clean, canonical German role suggestions from the German-language corpus
(Adzuna Germany, tagged lang=de) — the German sibling of mine_sv_roles.py.

The dominant German noise is the inclusive gender tag that almost every German ad
appends to the role: "Softwareentwickler (m/w/d)", "Pflegefachkraft (w/m/d)",
"Elektriker m/w/d", "Erzieher (gn)", "Koch (m/w/divers)". Stripping that tag is the
single biggest win. Beyond it the shape is like Swedish: many roles are a single
compound noun (Kraftfahrzeugmechatroniker, Sachbearbeiter, Lagerlogistiker) where the
bare leading token IS the role, but titles also lead with a MODIFIER ("Leitender
Oberarzt", "Erfahrener Bauleiter", "Examinierte Pflegefachkraft") or with staffing /
job-type boilerplate. This strips the gender tag, locations, codes and dash/prep noise,
frequency-ranks the cleaned forms, promotes recurring bare role heads, seeds essential
compound roles, and emits the top-N as a German autocomplete tier (de_roles.json),
shipped alongside app.py and merged into the suggestion corpus at load time.
"""

import json
import re
import sys
import unicodedata
from collections import Counter

import requests

SOLR = "http://localhost:8983/solr/jobs"
FQ = "lang:de"  # every German-language doc (Adzuna Germany today; source-agnostic)
TOP_N = 500
MIN_COUNT = 5  # a cleaned form must recur this often to be a "canonical" role

# Essential bare roles that title-leading-head promotion CANNOT recover, because in
# German they appear overwhelmingly as a SINGLE compound word ("Softwareentwickler",
# "Industriemechaniker", "Pflegefachkraft") or as the TAIL of one, rather than as a
# leading token. Each is included only if it independently clears MIN_COUNT as a title
# term in the live corpus (so the suggestion is grounded and always returns results).
SEED_ROLES = [
    "ingenieur",
    "entwickler",
    "softwareentwickler",
    "techniker",
    "elektriker",
    "mechaniker",
    "mechatroniker",
    "krankenpfleger",
    "pflegefachkraft",
    "pflegekraft",
    "erzieher",
    "verkäufer",
    "kaufmann",
    "koch",
    "kraftfahrer",
    "lagerist",
    "buchhalter",
    "sachbearbeiter",
    "projektleiter",
    "bauleiter",
    "berater",
    "disponent",
    "monteur",
    "fachkraft",
    "facharzt",
    "arzt",
]

# Leading tokens that are NOT occupations: seniority/quality modifiers, employment-type
# and structural words, and English tokens from ads posted in English. They recur widely
# enough to clear the head-promotion bar but must not become standalone role suggestions.
# (A genuine 2-word role led by one of these, e.g. "examinierte pflegefachkraft", still
# survives via full-title counting.)
_NONROLE_HEAD = {
    "senior",
    "junior",
    "leitender",
    "leitende",
    "stellvertretender",
    "stellvertretende",
    "erfahrener",
    "erfahrene",
    "examinierter",
    "examinierte",
    "ausgebildeter",
    "ausgebildete",
    "gelernter",
    "gelernte",
    "technischer",
    "technische",
    "kaufmännischer",
    "kaufmännische",
    "freiberuflicher",
    "selbständiger",
    "selbstständiger",
    "engagierter",
    "engagierte",
    "motivierter",
    "motivierte",
    "neuer",
    "neue",
    "weiterer",
    "weitere",
    "duales",
    "dualer",
    "pädagogische",
    "pädagogischer",
    "medizinische",
    "medizinischer",
    # English modifiers/tokens leaking in from English-language ads
    "experienced",
    "the",
    "part",
    "full",
}

# Non-role noise that survives cleaning as a bare token (job-type, work arrangement,
# meta words), German + the English leakage.
_STOP = {
    "stellenangebot",
    "stellenangebote",
    "stellenanzeige",
    "vollzeit",
    "teilzeit",
    "minijob",
    "midijob",
    "nebenjob",
    "aushilfe",
    "aushilfen",
    "praktikum",
    "praktikant",
    "werkstudent",
    "ausbildung",
    "ausbildungsplatz",
    "trainee",
    "gesucht",
    "sofort",
    "arbeit",
    "job",
    "jobs",
    "stelle",
    "stellen",
    "mitarbeiter",  # bare "employee" is too generic to be a role
    "fachkraft",  # bare; real roles are compounds (pflege-/elektrofachkraft) — kept via seeds
    "kraft",
    "personal",
    "team",
    "diverse",
    "verschiedene",
    # dangling heads of dash-coordinated compounds ("Maschinen- und Anlagenführer",
    # "Gesundheits- und Krankenpfleger"): the bare prefix is a fragment, not a role.
    "maschinen",
    "gesundheits",
    "festanstellung",
    "homeoffice",
    "remote",
    "quereinsteiger",  # career-changer, an applicant type not a role
    "studenten",
    "studentin",
    "student",
    # leading city names that survive as a bare token
    "berlin",
    "münchen",
    "muenchen",
    "hamburg",
    "köln",
    "koeln",
    "frankfurt",
    "stuttgart",
    "düsseldorf",
    "duesseldorf",
    "leipzig",
    "dresden",
    "hannover",
    "nürnberg",
    "nuernberg",
    # English leakage
    "intern",
    "internship",
    "data",
    "product",
    "project",
    "software",
    "customer",
    "business",
    "sales",
    "account",
    "head",
    "manager",
}


def fold(s: str) -> str:
    """Lowercase + strip diacritics (for dedup + accent-insensitive matching). German
    ä/ö/ü fold to a/o/u via NFKD; ß does NOT decompose, so map it to 'ss' explicitly
    (matching app.py._fold). Linguistically lossy but correct for the accent-insensitive
    autocomplete prefix match (a user typing 'elektr' or 'müll'/'mull' both reach the
    accented role)."""
    nfkd = unicodedata.normalize("NFKD", s.replace("ß", "ss"))
    base = "".join(c for c in nfkd if not unicodedata.combining(c)).lower()
    return base.replace("'", "").replace("’", "")


_STOP_FOLDED = {fold(s) for s in _STOP}
_NONROLE_HEAD_FOLDED = {fold(s) for s in _NONROLE_HEAD}

# ---- cleaning regexes (applied in order) ----
# The inclusive gender tag in all its German variants. Two shapes: a parenthesized
# single marker — "(gn)", "(gn*)", "(divers)", "(all genders)", "(d)", "(a*)" — or a
# slash/star sequence with or without parens — "(m/w/d)", "w/m/d", "(m/w/divers)",
# "(m/w/x)", "(d/m/w)", "(m/f/d)". A sequence part may itself be the word "divers"/"gn".
_GENDER = re.compile(
    r"\(\s*(?:gn|all\s+genders|divers|[mwfdax])\s*\*?\s*\)"
    r"|\(?\s*[mwfdaxg](?:\s*[/*]\s*(?:divers|gn|[mwfdaxg]))+\s*\*?\s*\)?",
    re.I,
)
_CODE = re.compile(r"#\S+|\bref\.?\s*\S+|\bkennziffer\s*\S+", re.I)  # req/ref codes
_PARENS = re.compile(r"[\(\[][^\)\]]*[\)\]]")  # (Berlin), [remote], (12230)
# NOTE: no leading-company-prefix strip (unlike the Swedish miner). Adzuna carries the
# company in a separate field, so a German title is the bare role; a leading "X - " is
# far more often a role followed by a location/specialty (handled by _TRAIL_DASH) than a
# company prefix, so stripping the front would eat the role ("Oberarzt - Kardiologie").
_LEAD_ART = re.compile(r"^(?:ein|eine|einen|der|die|das|den)\s+", re.I)  # "eine Pflegekraft"
_COMMA_CUT = re.compile(r"\s*,.*$")  # "Pflegekraft, Nachtdienst" -> "Pflegekraft"
_SLASH_CUT = re.compile(r"\s*/.*$")  # "Koch/Köchin" -> "Koch" (after gender tag stripped)
_TRAIL_DASH = re.compile(r"\s+[-–—]\s+.*$")  # trailing " - <location/freetext>"
# trailing German complement: "Verkäufer in Teilzeit", "Ingenieur für Maschinenbau",
# "Pfleger im Nachtdienst", "Berater (m/w/d) als Quereinsteiger" -> keep the head + role.
_TRAIL_PREP = re.compile(
    r"\s+(?:in|im|für|fuer|als|bei|mit|zur|zum|von|am|an|auf|gesucht|ab\b)\b.*$", re.I
)
_WS = re.compile(r"\s+")


def clean_title(t: str) -> str:
    s = t.strip()
    s = _GENDER.sub(" ", s)  # strip the (m/w/d)-family tag FIRST
    s = _LEAD_ART.sub("", s)
    s = _CODE.sub(" ", s)
    s = _PARENS.sub(" ", s)
    s = _COMMA_CUT.sub("", s)
    s = _SLASH_CUT.sub("", s)
    s = _TRAIL_DASH.sub("", s)
    s = _TRAIL_PREP.sub("", s)
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


def fetch_term_count(term: str) -> int:
    """How many German docs carry `term` as a title token — the grounding test for a
    seed role (its search must return results)."""
    r = requests.get(
        f"{SOLR}/select",
        params={"q": f'title:"{term}"', "fq": FQ, "rows": "0", "wt": "json"},
        timeout=30,
    )
    r.raise_for_status()
    return r.json()["response"]["numFound"]


def main():
    titles = fetch_titles()
    print(f"fetched {len(titles):,} German titles", file=sys.stderr)
    counts: Counter[str] = Counter()
    surface: dict[str, Counter[str]] = {}
    # Bare-head counts: many German role titles lead with the role head before a modifier
    # or complement ("Verkäufer in Teilzeit", "Ingenieur Maschinenbau"); accumulate the
    # leading token across all titles and emit heads that recur widely and span several
    # distinct full forms (a real role family).
    head_counts: Counter[str] = Counter()
    head_surface: dict[str, Counter[str]] = {}
    head_variants: dict[str, set[str]] = {}
    for t in titles:
        c = clean_title(t)
        if not (2 <= len(c) <= 48):
            continue
        if not re.search(r"[a-zà-ÿäöüß]", c):
            continue
        if re.search(r"\d", c):  # drop dates/grades/codes
            continue
        if len(c.split()) > 6:
            continue
        k = fold(c)
        if k in _STOP_FOLDED:
            continue
        counts[k] += 1
        surface.setdefault(k, Counter())[c] += 1
        toks = c.split()
        head, fhead = toks[0].strip("-·."), k.split()[0].strip("-·.")
        if len(fhead) >= 4 and fhead not in _STOP_FOLDED and re.search(r"[a-zà-ÿäöüß]", fhead):
            head_counts[fhead] += 1
            head_surface.setdefault(fhead, Counter())[head] += 1
            head_variants.setdefault(fhead, set()).add(k)
    HEAD_MIN, HEAD_VARIANTS = 8, 3  # bar for promoting a bare head to a standalone role
    for fhead, n in head_counts.items():
        if n < HEAD_MIN or len(head_variants[fhead]) < HEAD_VARIANTS:
            continue
        if fhead in _NONROLE_HEAD_FOLDED:
            continue
        counts[fhead] = max(counts.get(fhead, 0), n)
        surface.setdefault(fhead, head_surface[fhead])
    # Seed essential compound roles (ingenieur/softwareentwickler/...) head-promotion can't
    # reach. Include each only if it independently clears MIN_COUNT as a live title term.
    for role in SEED_ROLES:
        fk = fold(role)
        if counts.get(fk, 0) >= MIN_COUNT:
            continue
        n = fetch_term_count(role)
        if n >= MIN_COUNT:
            counts[fk] = max(counts.get(fk, 0), n)
            surface.setdefault(fk, Counter())[role] += n
    ranked = [
        (surface[k].most_common(1)[0][0], n) for k, n in counts.most_common() if n >= MIN_COUNT
    ][:TOP_N]
    print(f"{len(ranked)} canonical DE roles (>= {MIN_COUNT} occurrences)", file=sys.stderr)
    out = [{"text": role, "n": n} for role, n in ranked]
    with open("space/de_roles.json", "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=0)
    print("wrote space/de_roles.json", file=sys.stderr)
    for role, n in ranked[:30]:
        print(f"  {n:>5}  {role}", file=sys.stderr)


if __name__ == "__main__":
    main()
