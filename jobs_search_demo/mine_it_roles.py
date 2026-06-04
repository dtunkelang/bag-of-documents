#!/usr/bin/env python3
"""Mine clean, canonical Italian role suggestions from the Italian-language corpus
(Adzuna Italy, tagged lang=it) — the Italian sibling of mine_es_roles.py / mine_de_roles.py.

The dominant Italian noise is the inclusive gender marker almost every Italian ad appends
to the role: the slash suffix "Cameriere/a", "Addetto/a", "Operaio/a" and the parenthesized
tag "(m/f)", "(f/m)", "(m/f/d)", or the leaked German "(m/w/d)". Stripping those is the
single biggest win. Beyond it the shape mirrors Spanish, NOT German/Dutch: an Italian role
is usually a HEAD plus a complement introduced by "di"/"alle"/"agli" ("responsabile di
sala", "addetto alle vendite", "tecnico di laboratorio", "capo cantiere"), so — unlike the
Dutch miner — we must NOT strip a trailing "di ..."/"alle ..." (that IS the role). We strip
the gender markers, the "cercasi/cerchiamo" lead, locations, codes and prep/location tails,
frequency-rank the cleaned forms, promote recurring bare role heads, seed essential roles,
and emit the top-N as an Italian autocomplete tier (it_roles.json), shipped alongside app.py
and merged into the suggestion corpus at load time.
"""

import json
import re
import sys
import unicodedata
from collections import Counter

import requests

SOLR = "http://localhost:8983/solr/jobs"
FQ = "lang:it"  # every Italian-language doc (Adzuna Italy today; source-agnostic)
TOP_N = 500
MIN_COUNT = 5  # a cleaned form must recur this often to be a "canonical" role

# Essential roles that title-leading-head promotion may miss (they appear as the TAIL of a
# complement, or below the head-promotion variant bar). Each is included only if it
# independently clears MIN_COUNT as a title term in the live corpus (so the suggestion is
# grounded and always returns results).
SEED_ROLES = [
    "cameriere",
    "cuoco",
    "aiuto cuoco",
    "barista",
    "pizzaiolo",
    "commesso",
    "magazziniere",
    "carrellista",
    "autista",
    "operaio",
    "infermiere",
    "operatore socio sanitario",
    "fisioterapista",
    "farmacista",
    "medico",
    "elettricista",
    "idraulico",
    "saldatore",
    "muratore",
    "falegname",
    "meccanico",
    "manutentore",
    "addetto alle vendite",
    "addetto vendite",
    "addetto mensa",
    "addetto pulizie",
    "receptionist",
    "segretaria",
    "impiegato",
    "contabile",
    "programmatore",
    "sviluppatore",
    "ingegnere",
    "perito",
    "parrucchiere",
    "estetista",
    "panettiere",
    "macellaio",
    "cassiere",
]

# Leading tokens that are NOT occupations: seniority/quality modifiers and structural words.
# They recur widely enough to clear the head-promotion bar but must not become standalone
# role suggestions. (A genuine 2-word role led by one of these still survives via full-title
# counting.)
_NONROLE_HEAD = {
    "senior",
    "junior",
    "responsabile",  # "responsabile di sala" is the role; bare head is a fragment
    "capo",  # "capo cantiere" is the role; bare "capo" head is a fragment
    "assistente",  # "assistente alla poltrona" is the role; bare head is a fragment
    "importante",  # "importante azienda ..." boilerplate
    "primaria",
    "prestigiosa",
    "nuova",
    "nuovo",
    "buon",
    "buona",
    "esperto",
    "esperta",
    "bravo",
    "brava",
    "valido",
    "valida",
    "ottimo",
    "ottima",
    # English modifiers/tokens leaking in from English-language ads
    "experienced",
    "the",
    "part",
    "full",
}

# Non-role noise that survives cleaning as a bare token (job-type, work arrangement, meta
# words), Italian + the English leakage.
_STOP = {
    "lavoro",
    "lavori",
    "impiego",
    "offerta",
    "offerte",
    "posizione",
    "posizioni",
    "vacante",
    "vacanti",
    "contratto",
    "tempo",
    "determinato",
    "indeterminato",
    "completa",
    "parziale",
    "stage",
    "tirocinio",
    "apprendistato",
    "azienda",
    "aziende",
    "settore",
    "zona",
    "sede",
    "turno",
    "turni",
    "urgente",
    "cercasi",
    "cerchiamo",
    "ricerca",
    "ricerchiamo",
    "selezione",
    "selezioniamo",
    "seleziona",
    "candidato",
    "candidata",
    "candidatura",
    "candidature",
    "persona",
    "persone",
    "personale",
    "risorsa",
    "risorse",
    "gente",
    "team",
    "gruppo",
    "esperienza",
    "automunito",
    "automunita",
    "diploma",
    "mansioni",
    "stipendio",
    "retribuzione",
    "neolaureato",
    "neodiplomato",
    "ambosessi",
    "requisiti",
    "spontanea",
    "candidatura spontanea",  # "spontaneous application" boilerplate (full phrase)
    "autocandidatura",
    "smart",
    "working",
    "remoto",
    "freelance",
    "stagionale",
    # boilerplate fragments / template heads that recur as bare tokens or full phrases
    "richieste",
    "richiesta",
    "richiesto",
    "richiesti",
    "nostri",
    "nostra",
    "nostre",
    "clienti",
    "figura",
    "figure",
    "un",
    "addett",  # truncated "addetto" (PDF/encoding cut)
    "opportunita",
    "opportunita di lavoro",
    "opportunita di lavoro flessibile",
    "assistenza",
    # bare English fragment heads from English-language ads (real roles like "sales
    # assistant"/"back office"/"graphic designer" are multi-word and survive full-title)
    "back",
    "front",
    "vice",
    "quality",
    "social",
    "graphic",
    "assistant",
    "director",
    "supervisor",
    # Spanish / Portuguese cross-posts leak into the Italy index (close to it and not
    # distinguished by the gate, so they tag lang=it). Drop the high-frequency es/pt heads.
    "camarero",
    "cocinero",
    "vendedor",
    "comercial",
    "mozo",
    "limpiador",
    "engenheiro",
    "executivo",
    "pessoa",
    "vendas",
    # English meta/fragment tokens that aren't roles (real English ROLES like "software
    # engineer" survive — they're present in Italy's market and harmless in autocomplete)
    "growth",
    "store",
    "inside",
    "field",
    "operations",
    "marketing",
    "brand",
    "tech",
    "content",
    "digital",
    "talent",
    "officer",
    "lead",
    "associate",
    "staff",
    "backend",
    "technical",
    "mid",
    "level",
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
    "remote",
    # leading city/region names that survive as a bare token
    "milano",
    "roma",
    "torino",
    "napoli",
    "palermo",
    "genova",
    "bologna",
    "firenze",
    "bari",
    "catania",
    "venezia",
    "verona",
    "messina",
    "padova",
    "trieste",
    "brescia",
    "parma",
    "modena",
    "prato",
    "cagliari",
    "livorno",
    "perugia",
    "salerno",
    "vicenza",
    "bergamo",
    "monza",
    "latina",
    "rimini",
    "ferrara",
    "pescara",
    "ravenna",
    "lecce",
    "udine",
    "treviso",
    "varese",
    "como",
    "novara",
    "pisa",
    "italia",
}


def fold(s: str) -> str:
    """Lowercase + strip diacritics (for dedup + accent-insensitive matching). Italian
    à/è/é/ì/ò/ù fold to a/e/i/o/u via NFKD. Linguistically lossy but correct for the
    accent-insensitive autocomplete prefix match."""
    nfkd = unicodedata.normalize("NFKD", s)
    base = "".join(c for c in nfkd if not unicodedata.combining(c)).lower()
    return base.replace("'", "").replace("’", "")


_STOP_FOLDED = {fold(s) for s in _STOP}
_NONROLE_HEAD_FOLDED = {fold(s) for s in _NONROLE_HEAD}

# ---- cleaning regexes (applied in order) ----
# The inclusive gender tag in its Italian variants: a parenthesized single marker "(m)",
# "(f)" or a slash/star sequence with or without parens — "(m/f)", "f/m", "(m/f/d)",
# "(m/w/d)" (leaked German). Markers: m=maschio/uomo, f=femmina/donna, plus leaked w/d/x.
_GENDER = re.compile(
    r"\(\s*[mfdwx]\s*\*?\s*\)"
    r"|\(?\s*[mfdwx](?:\s*[/*]\s*[mfdwx])+\s*\*?\s*\)?",
    re.I,
)
_CODE = re.compile(r"#\S+|\brif\.?\s*\S+|\bn°?\s*\S*\d\S*", re.I)  # req/ref codes
_PARENS = re.compile(r"[\(\[][^\)\]]*[\)\]]")  # (Milano), [remoto], (12230)
_LEAD_SEEK = re.compile(
    r"^(?:cercasi|cerchiamo|ricerchiamo|selezioniamo|si\s+(?:ricerca|ricercano|seleziona|"
    r"selezionano|cerca|cercano)|azienda\s+(?:cerca|ricerca|seleziona))\s+",
    re.I,
)  # "Cercasi cameriere" / "Si ricerca cuoco" -> the role
# Recruiting-agency template ("i nostri clienti hanno richiesto fotografi" -> "fotografi")
# and the internship "stage X" lead ("stage addetto" -> "addetto"); strip so the real role
# surfaces (or the residue is dropped by _STOP / length filters).
_LEAD_BOILER = re.compile(
    r"^(?:(?:i\s+)?nostri\s+clienti\s+hanno\s+richiest\w+\s+|stage\s+)",
    re.I,
)
_LEAD_ART = re.compile(r"^(?:un|uno|una|il|lo|la|i|gli|le)\s+", re.I)  # "un Cameriere"
# cut at the first comma OR colon: "Cameriere, fine settimana" / "Offerta: cuoco" -> head
_COMMA_CUT = re.compile(r"\s*[,:].*$")
# the gender SLASH suffix only: "Cameriere/a", "Addetto/a", "Operaio/a" -> head. We cut at
# the FIRST slash; this is safe in Italian because a real "X/Y" alternative role is rare,
# whereas "/a" gender is pervasive. (Done AFTER _GENDER strips the parenthesized form.)
_SLASH_CUT = re.compile(r"\s*/.*$")
_TRAIL_DASH = re.compile(r"\s+[-–—]\s+.*$")  # trailing " - <location/freetext>"
# trailing Italian complement/location: "Cameriere a Milano", "Cuoco per ristorante",
# "Operaio con esperienza". CRITICAL: 'di'/'alle'/'agli'/'alla'/'al'/'ai'/'allo' are
# deliberately EXCLUDED — an Italian role is usually a head + such a complement ("addetto
# alle vendite", "responsabile di sala", "tecnico di laboratorio"), so they ARE the role,
# not a tail to strip.
_TRAIL_PREP = re.compile(
    r"\s+(?:in|per|con|senza|presso|su|da|zona|sede|automunito)\b.*$",
    re.I,
)
# trailing EU driving-license class on a driver role ("Autista C", "Autista C+E"): a
# standalone 1-2 letter category, not part of the role name.
_TRAIL_LICENSE = re.compile(r"\s+(?:c\s*\+\s*e|be|ce|[bcde])$", re.I)
_WS = re.compile(r"\s+")


def clean_title(t: str) -> str:
    s = t.strip()
    s = _GENDER.sub(" ", s)  # strip the (m/f)-family tag FIRST
    s = _LEAD_SEEK.sub("", s)
    s = _LEAD_BOILER.sub("", s)
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
    """How many Italian docs carry `term` as a title phrase — the grounding test for a
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
    print(f"fetched {len(titles):,} Italian titles", file=sys.stderr)
    counts: Counter[str] = Counter()
    surface: dict[str, Counter[str]] = {}
    # Bare-head counts: many Italian role titles lead with the role head before a complement
    # ("Cameriere a Milano", "Operaio con esperienza"); accumulate the leading token across
    # all titles and emit heads that recur widely and span several distinct full forms.
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
    # Seed essential roles head-promotion can't reach. Include each only if it independently
    # clears MIN_COUNT as a live title term.
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
    print(f"{len(ranked)} canonical IT roles (>= {MIN_COUNT} occurrences)", file=sys.stderr)
    out = [{"text": role, "n": n} for role, n in ranked]
    with open("space/it_roles.json", "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=0)
    print("wrote space/it_roles.json", file=sys.stderr)
    for role, n in ranked[:30]:
        print(f"  {n:>5}  {role}", file=sys.stderr)


if __name__ == "__main__":
    main()
