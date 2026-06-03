#!/usr/bin/env python3
"""Mine clean, canonical Swedish role suggestions from JobTech (Arbetsförmedlingen)
titles — the Swedish sibling of mine_fr_roles.py.

Swedish occupational nouns are largely gender-neutral (sjuksköterska, lärare,
ingenjör), so unlike French there is NO degendering. The noise here is different:
~most roles are single compound words (förskollärare, socialsekreterare,
undersköterska), so the bare leading token IS usually the role — but titles
frequently lead with a MODIFIER ("Senior utvecklare", "Personlig assistent",
"Erfaren sjuksköterska", "Legitimerad läkare") or staffing-agency / job-type
boilerplate ("Sommarjobb", "Almia söker ..."). This strips locations/codes/dash
noise, frequency-ranks the cleaned forms, promotes recurring bare role heads, and
emits the top-N as a Swedish autocomplete tier (sv_roles.json), shipped alongside
app.py and merged into the suggestion corpus at load time.
"""

import json
import re
import sys
import unicodedata
from collections import Counter

import requests

SOLR = "http://localhost:8983/solr/jobs"
FQ = "source_corpus:jobs_data_jobtech"
TOP_N = 500
MIN_COUNT = 8  # a cleaned form must recur this often to be a "canonical" role

# Essential bare roles that title-leading-head promotion CANNOT recover, because in
# Swedish they appear overwhelmingly as the TAIL of a single compound word
# ("specialistläkare", "systemutvecklare", "automationsingenjör") rather than as a
# leading token. Each is included only if it independently clears MIN_COUNT as a title
# term in the live corpus (so the suggestion is grounded and always returns results).
SEED_ROLES = [
    "läkare",
    "ingenjör",
    "utvecklare",
    "systemutvecklare",
    "konsult",
    "designer",
    "arkitekt",
    "analytiker",
    "koordinator",
    "programmerare",
    "projektledare",
    "produktägare",
]

# Leading tokens that are NOT occupations: seniority/quality modifiers, employment-type
# and structural words, staffing-agency names, and English tokens from the ads posted in
# English (foodora etc.). They recur widely enough to clear the head-promotion bar but
# must not become standalone role suggestions. (A genuine 2-word role led by one of these,
# e.g. "personlig assistent", still survives via full-title counting.)
_NONROLE_HEAD = {
    "senior",
    "junior",
    "erfaren",
    "erfarna",
    "teknisk",
    "tekniska",
    "personlig",
    "personliga",
    "legitimerad",
    "legitimerade",
    "leg",
    "vikarierande",
    "extra",
    "ny",
    "nya",
    "timanstalld",
    "sommarjobb",
    "sommarjobba",
    "sommarvikarie",  # season/job-type, not a role
    "almia",  # staffing agency, leads "Almia söker ..."
    "bemanning",
    "framtidens",
    "blivande",
    "duktig",
    "duktiga",
    "engagerad",
    "engagerade",
    "ansvarig",  # "ansvarig" alone is a modifier; real roles keep their head ("enhetschef")
    "bitradande",  # "biträdande rektor/chef" = deputy X; bare it's a modifier
    "medicinsk",  # "medicinsk sekreterare" survives as a full title; bare it's a modifier
    "kvinnlig",  # "kvinnlig personlig assistent" -> the gender qualifier isn't the role
    "manlig",
    "studie",  # "studie- och yrkesvägledare" survives; bare "studie-" is a fragment
    "resande",
    "operativ",
    # English modifiers/tokens leaking in from English-language ads
    "experienced",
    "the",
    "food",
    "part",
    "full",
}

# Non-role noise that survives cleaning as a bare token (job-type, work arrangement,
# verbs/meta words), Swedish + the English leakage.
_STOP = {
    "sommarjobb",
    "sommarjobba",
    "extrajobb",
    "vikariat",
    "vikarie",
    "timvikarie",
    "heltid",
    "deltid",
    "sökes",
    "sokes",
    "soker",
    "söker",
    "jobb",
    "jobba",
    "anstallning",
    "tjanst",
    "uppdrag",
    "extra",
    "omgaende",
    "diverse",
    "sommar",
    "sommarpersonal",
    "sommarvikarie",
    "sommarvikarier",
    "sommarvikariat",
    "vikarier",
    "fast",
    "host",
    "omgaende start",
    "strategisk",
    "saljande",
    "teacher",
    "service",
    "mynanny",  # babysitting brand names that lead their ads
    "allakando",
    "vill",
    "arbeta",
    "driven",
    "interim",
    "medarbetare",  # bare "coworker/employee" is too generic to be a role
    "specialist",  # bare; real roles are compounds (specialistläkare/-sjuksköterska)
    "studenter",
    "kvinna",
    "dig",
    "leg",
    "studie",
    # leading city names that survive as a bare token
    "goteborg",
    "stockholm",
    "malmo",
    "uppsala",
    # English leakage
    "job",
    "jobs",
    "intern",
    "internship",
    "remote",
    "data",
    "product",
    "project",
    "software",
    "customer",
    "business",
    "sales",
    "account",
    "head",
    "part time",
    "full time",
}


def fold(s: str) -> str:
    """Lowercase + strip diacritics (for dedup + accent-insensitive matching).
    Swedish å/ä/ö fold to a/a/o — linguistically lossy but correct for the
    accent-insensitive autocomplete prefix match (a user typing 'lara' or 'lärа'
    both reach 'lärare'), matching app.py._fold."""
    nfkd = unicodedata.normalize("NFKD", s)
    base = "".join(c for c in nfkd if not unicodedata.combining(c)).lower()
    return base.replace("'", "").replace("’", "")


_STOP_FOLDED = {fold(s) for s in _STOP}
_NONROLE_HEAD_FOLDED = {fold(s) for s in _NONROLE_HEAD}

# ---- cleaning regexes (applied in order) ----
_CODE = re.compile(r"#\S+|\bref\.?\s*\S+", re.I)  # req/ref codes
_PARENS = re.compile(r"[\(\[][^\)\]]*[\)\]]")  # (Stockholm), [remote], (12230)
# leading company/prefix junk: "Almia AB - ", "Region Skåne - ", "Företag X – "
_PREFIX = re.compile(r"^[A-ZÅÄÖ][\w&'.\- ]{1,30}\s+[-–—]\s+")
# agency/we-search prefix: "Almia söker sjuksköterska", "Vi söker c-chaufförer" -> the role
# after "söker". Lazy up to ~30 chars so it only eats a short leading subject, not a role.
_SEARCH_PREFIX = re.compile(r"^.{0,30}?\bs[öo]ker\s+", re.I)
_LEG = re.compile(r"\bleg\.?\s+", re.I)  # "leg." / "leg" = legitimerad abbrev modifier
_LEAD_ART = re.compile(r"^(?:en|ett|den|det|de|vår|var|nya?)\s+", re.I)  # "en lärare" -> "lärare"
_COMMA_CUT = re.compile(r"\s*,.*$")  # "Sjuksköterska, natt" -> "Sjuksköterska"
_SLASH_CUT = re.compile(r"\s*/.*$")  # "sjuksköterska/distriktssköterska" -> first role
_TRAIL_DASH = re.compile(r"\s+[-–—]\s+.*$")  # trailing " - <location/freetext>"
# trailing Swedish complement: "Sjuksköterska till akutmottagningen", "Lärare i matematik",
# "Utvecklare inom AI", "Säljare på heltid" -> keep the head + role, drop the complement.
_TRAIL_PREP = re.compile(r"\s+(?:till|i|på|pa|inom|för|for|med|sökes|sokes|som)\b.*$", re.I)
_WS = re.compile(r"\s+")


def clean_title(t: str) -> str:
    s = t.strip()
    s = _PREFIX.sub("", s)
    s = _SEARCH_PREFIX.sub("", s)
    s = _LEAD_ART.sub("", s)
    s = _LEG.sub("", s)
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
    """How many JobTech docs carry `term` as a title token — the grounding test for a
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
    print(f"fetched {len(titles):,} JobTech titles", file=sys.stderr)
    # count by folded key, keeping the most common accented surface form
    counts: Counter[str] = Counter()
    surface: dict[str, Counter[str]] = {}
    # Bare-head counts: most Swedish role titles are qualified ("Sjuksköterska till X",
    # "Lärare i matematik") or compound single words, so the bare role head is the most
    # useful autocomplete key. Accumulate the leading token across all titles and emit
    # heads that recur widely and span several distinct full forms (a real role family).
    head_counts: Counter[str] = Counter()
    head_surface: dict[str, Counter[str]] = {}
    head_variants: dict[str, set[str]] = {}
    for t in titles:
        c = clean_title(t)
        if not (2 <= len(c) <= 48):
            continue
        if not re.search(r"[a-zà-ÿåäö]", c):
            continue
        if re.search(r"\d", c):  # drop dates/grades/codes ("höst 2026", "lärare åk 4-6")
            continue
        if len(c.split()) > 6:
            continue
        k = fold(c)
        if k in _STOP_FOLDED:
            continue
        counts[k] += 1
        surface.setdefault(k, Counter())[c] += 1
        toks = c.split()
        # strip stray edge punctuation off the head token so "studie-" folds to "studie"
        head, fhead = toks[0].strip("-·."), k.split()[0].strip("-·.")
        if len(fhead) >= 4 and fhead not in _STOP_FOLDED and re.search(r"[a-zà-ÿåäö]", fhead):
            head_counts[fhead] += 1
            head_surface.setdefault(fhead, Counter())[head] += 1
            head_variants.setdefault(fhead, set()).add(k)
    HEAD_MIN, HEAD_VARIANTS = 12, 3  # bar for promoting a bare head to a standalone role
    for fhead, n in head_counts.items():
        if n < HEAD_MIN or len(head_variants[fhead]) < HEAD_VARIANTS:
            continue
        if fhead in _NONROLE_HEAD_FOLDED:
            continue
        # boost to the aggregate head count so the bare role outranks its own extensions
        # (it's the better autocomplete key); covers heads too rare on their own to clear
        # MIN_COUNT and heads that never occurred as a standalone title.
        counts[fhead] = max(counts.get(fhead, 0), n)
        surface.setdefault(fhead, head_surface[fhead])
    # Seed essential compound-tail roles (läkare/utvecklare/...) that head-promotion can't
    # reach. Include each only if it independently clears MIN_COUNT as a live title term.
    for role in SEED_ROLES:
        fk = fold(role)
        if counts.get(fk, 0) >= MIN_COUNT:  # already qualifies on its own -> keep mined count
            continue
        n = fetch_term_count(role)
        if n >= MIN_COUNT:
            counts[fk] = max(counts.get(fk, 0), n)
            surface.setdefault(fk, Counter())[role] += n
    ranked = [
        (surface[k].most_common(1)[0][0], n) for k, n in counts.most_common() if n >= MIN_COUNT
    ][:TOP_N]
    print(f"{len(ranked)} canonical SV roles (>= {MIN_COUNT} occurrences)", file=sys.stderr)
    out = [{"text": role, "n": n} for role, n in ranked]
    with open("space/sv_roles.json", "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=0)
    print("wrote space/sv_roles.json", file=sys.stderr)
    for role, n in ranked[:30]:
        print(f"  {n:>5}  {role}", file=sys.stderr)


if __name__ == "__main__":
    main()
