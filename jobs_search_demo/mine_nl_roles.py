#!/usr/bin/env python3
"""Mine clean, canonical Dutch role suggestions from the Dutch-language corpus
(Adzuna Netherlands, tagged lang=nl) — the Dutch sibling of mine_de_roles.py.

The dominant Dutch noise is the inclusive gender tag almost every Dutch ad appends to
the role: "Verpleegkundige (m/v)", "Monteur (v/m)", "Lasser m/v/x", "Kok (m/f/d)".
Stripping that tag is the single biggest win. Beyond it the shape is like German: many
roles are a single compound noun (Vrachtwagenchauffeur, Magazijnmedewerker,
Productiemedewerker) where the bare leading token IS the role, but titles also lead with
a MODIFIER ("Senior Monteur", "Ervaren Verpleegkundige", "Allround Elektricien") or with
staffing / job-type boilerplate. This strips the gender tag, locations, codes and
dash/prep noise, frequency-ranks the cleaned forms, promotes recurring bare role heads,
seeds essential compound roles, and emits the top-N as a Dutch autocomplete tier
(nl_roles.json), shipped alongside app.py and merged into the suggestion corpus at load
time.
"""

import json
import re
import sys
import unicodedata
from collections import Counter

import requests

SOLR = "http://localhost:8983/solr/jobs"
FQ = "lang:nl"  # every Dutch-language doc (Adzuna Netherlands today; source-agnostic)
TOP_N = 500
MIN_COUNT = 5  # a cleaned form must recur this often to be a "canonical" role

# Essential bare roles that title-leading-head promotion CANNOT recover, because in
# Dutch they appear overwhelmingly as a SINGLE compound word ("Vrachtwagenchauffeur",
# "Magazijnmedewerker", "Softwareontwikkelaar") or as the TAIL of one, rather than as a
# leading token. Each is included only if it independently clears MIN_COUNT as a title
# term in the live corpus (so the suggestion is grounded and always returns results).
SEED_ROLES = [
    "verpleegkundige",
    "verzorgende",
    "monteur",
    "chauffeur",
    "vrachtwagenchauffeur",
    "elektricien",
    "loodgieter",
    "timmerman",
    "kok",
    "verkoper",
    "verkoopmedewerker",
    "magazijnmedewerker",
    "productiemedewerker",
    "heftruckchauffeur",
    "lasser",
    "projectleider",
    "boekhouder",
    "accountant",
    "schoonmaker",
    "docent",
    "leraar",
    "ingenieur",
    "ontwikkelaar",
    "softwareontwikkelaar",
    "technicus",
    "beveiliger",
    "receptionist",
    "consulent",
    "adviseur",
    "begeleider",
    "operator",
    "planner",
]

# Leading tokens that are NOT occupations: seniority/quality modifiers, employment-type
# and structural words, and English tokens from ads posted in English. They recur widely
# enough to clear the head-promotion bar but must not become standalone role suggestions.
# (A genuine 2-word role led by one of these, e.g. "administratief medewerker", still
# survives via full-title counting.)
_NONROLE_HEAD = {
    "senior",
    "junior",
    "medior",
    "ervaren",
    "allround",
    "aankomend",
    "aankomende",
    "beginnend",
    "beginnende",
    "zelfstandig",
    "zelfstandige",
    "gediplomeerd",
    "gediplomeerde",
    "gemotiveerd",
    "gemotiveerde",
    "leidinggevend",
    "leidinggevende",
    "technisch",
    "technische",
    "commercieel",
    "commerciële",
    "administratief",
    "administratieve",
    "medisch",
    "medische",
    "pedagogisch",
    "pedagogische",
    "enthousiast",
    "enthousiaste",
    "nieuwe",
    "nieuw",
    "meewerkend",  # "meewerkend voorman" — adjective head, real role survives full-title
    "financieel",  # "financieel medewerker/adviseur"
    "persoonlijk",  # "persoonlijk begeleider"
    "eerste",  # "eerste monteur"
    # English modifiers/tokens leaking in from English-language ads
    "experienced",
    "the",
    "part",
    "full",
}

# Non-role noise that survives cleaning as a bare token (job-type, work arrangement,
# meta words), Dutch + the English leakage.
_STOP = {
    "vacature",
    "vacatures",
    "baan",
    "banen",
    "bijbaan",
    "vakantiebaan",
    "fulltime",
    "parttime",
    "deeltijd",
    "voltijd",
    "oproepkracht",
    "uitzendkracht",
    "stage",
    "stageplaats",
    "stagiair",
    "stagiaire",
    "werkstudent",
    "trainee",
    "leerling",
    "gezocht",
    "gevraagd",
    "direct",
    "werk",
    "werken",
    "job",
    "jobs",
    "functie",
    "medewerker",  # bare "employee" is too generic to be a role — kept via compounds/seeds
    "personeel",
    "team",
    "diverse",
    "verschillende",
    "regio",
    # job-type / applicant-type / domain fragments that recur as bare tokens but aren't roles
    "logistiek",  # adjective/domain; "logistiek medewerker" is the role
    "expeditie",  # shipping dept; "expeditie medewerker" is the role
    "assemblage",  # "assemblage medewerker" is the role
    "schoonmaak",  # "schoonmaker" is the role
    "service",
    "vakantiewerk",
    "vakantiekracht",
    "herintreder",  # returner — an applicant type
    "oppassen",
    "veva",  # Dutch military pre-vocational program, not a role
    "ernst",  # name fragment leaking from ad text
    "thuiswerk",
    "thuiswerken",
    "homeoffice",
    "remote",
    "freelance",
    "freelancer",
    "zzp",
    "zzper",
    "interim",
    "festanstellung",
    "quereinsteiger",
    # dangling heads of dash-coordinated compounds: the bare prefix is a fragment.
    "machine",
    "gezondheids",
    # leading city names that survive as a bare token
    "amsterdam",
    "rotterdam",
    "utrecht",
    "eindhoven",
    "groningen",
    "tilburg",
    "almere",
    "breda",
    "nijmegen",
    "haarlem",
    "arnhem",
    "amersfoort",
    "apeldoorn",
    "enschede",
    "zwolle",
    "leiden",
    "maastricht",
    "dordrecht",
    "venlo",
    "delft",
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
    """Lowercase + strip diacritics (for dedup + accent-insensitive matching). Dutch
    ë/ï/é fold to e/i/e via NFKD; Dutch has no ß. Linguistically lossy but correct for
    the accent-insensitive autocomplete prefix match."""
    nfkd = unicodedata.normalize("NFKD", s)
    base = "".join(c for c in nfkd if not unicodedata.combining(c)).lower()
    return base.replace("'", "").replace("’", "")


_STOP_FOLDED = {fold(s) for s in _STOP}
_NONROLE_HEAD_FOLDED = {fold(s) for s in _NONROLE_HEAD}

# ---- cleaning regexes (applied in order) ----
# The inclusive gender tag in its Dutch variants. Dutch uses m=man, v=vrouw (with x for
# non-binary); English ads leak m/f/d. Two shapes: a parenthesized single marker —
# "(v)", "(m)", "(x)" — or a slash/star sequence with or without parens — "(m/v)",
# "v/m", "(m/v/x)", "(m/f/d)", "(v/m/x)".
_GENDER = re.compile(
    r"\(\s*[mvfdxw]\s*\*?\s*\)"
    r"|\(?\s*[mvfdxw](?:\s*[/*]\s*[mvfdxw])+\s*\*?\s*\)?",
    re.I,
)
_CODE = re.compile(r"#\S+|\bref\.?\s*\S+|\bvacaturenummer\s*\S+", re.I)  # req/ref codes
_PARENS = re.compile(r"[\(\[][^\)\]]*[\)\]]")  # (Amsterdam), [remote], (12230)
# NOTE: no leading-company-prefix strip (like the German miner). Adzuna carries the
# company in a separate field, so a Dutch title is the bare role; a leading "X - " is far
# more often a role followed by a location/specialty (handled by _TRAIL_DASH).
_LEAD_ART = re.compile(r"^(?:een|de|het)\s+", re.I)  # "een Verpleegkundige"
# cut at the first comma OR colon: "Monteur, nachtdienst" / "Oppassen: zoeken een..." -> head
_COMMA_CUT = re.compile(r"\s*[,:].*$")
_SLASH_CUT = re.compile(r"\s*/.*$")  # "Verkoper/Verkoopster" -> "Verkoper" (post gender)
_TRAIL_DASH = re.compile(r"\s+[-–—]\s+.*$")  # trailing " - <location/freetext>"
# trailing Dutch complement: "Verpleegkundige in deeltijd", "Monteur voor Amsterdam",
# "Docent wiskunde gezocht", "Adviseur bij ...", -> keep the head + role.
_TRAIL_PREP = re.compile(
    r"\s+(?:in|voor|bij|te|op|als|met|naar|aan|om|gezocht|gevraagd|regio|binnen|vanaf)\b.*$",
    re.I,
)
# trailing Dutch driving-license class on a driver role ("Chauffeur C", "Vrachtwagen-
# chauffeur CE"): a standalone 1-2 letter category, not part of the role name.
_TRAIL_LICENSE = re.compile(r"\s+(?:be|ce|[bcde])$", re.I)
_WS = re.compile(r"\s+")


def clean_title(t: str) -> str:
    s = t.strip()
    s = _GENDER.sub(" ", s)  # strip the (m/v)-family tag FIRST
    s = _LEAD_ART.sub("", s)
    s = _CODE.sub(" ", s)
    s = _PARENS.sub(" ", s)
    s = _COMMA_CUT.sub("", s)
    s = _SLASH_CUT.sub("", s)
    s = _TRAIL_DASH.sub("", s)
    s = _TRAIL_PREP.sub("", s)
    s = _TRAIL_LICENSE.sub("", s)
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
    """How many Dutch docs carry `term` as a title token — the grounding test for a
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
    print(f"fetched {len(titles):,} Dutch titles", file=sys.stderr)
    counts: Counter[str] = Counter()
    surface: dict[str, Counter[str]] = {}
    # Bare-head counts: many Dutch role titles lead with the role head before a modifier
    # or complement ("Verpleegkundige in deeltijd", "Monteur Amsterdam"); accumulate the
    # leading token across all titles and emit heads that recur widely and span several
    # distinct full forms (a real role family).
    head_counts: Counter[str] = Counter()
    head_surface: dict[str, Counter[str]] = {}
    head_variants: dict[str, set[str]] = {}
    for t in titles:
        c = clean_title(t)
        if not (2 <= len(c) <= 48):
            continue
        if not re.search(r"[a-zà-ÿ]", c):
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
        if len(fhead) >= 4 and fhead not in _STOP_FOLDED and re.search(r"[a-zà-ÿ]", fhead):
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
    # Seed essential compound roles (verpleegkundige/softwareontwikkelaar/...) head-
    # promotion can't reach. Include each only if it independently clears MIN_COUNT as a
    # live title term.
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
    print(f"{len(ranked)} canonical NL roles (>= {MIN_COUNT} occurrences)", file=sys.stderr)
    out = [{"text": role, "n": n} for role, n in ranked]
    with open("space/nl_roles.json", "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=0)
    print("wrote space/nl_roles.json", file=sys.stderr)
    for role, n in ranked[:30]:
        print(f"  {n:>5}  {role}", file=sys.stderr)


if __name__ == "__main__":
    main()
