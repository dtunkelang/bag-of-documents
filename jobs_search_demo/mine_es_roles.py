#!/usr/bin/env python3
"""Mine clean, canonical Spanish role suggestions from the Spanish-language corpus
(Adzuna Spain, tagged lang=es) — the Spanish sibling of mine_nl_roles.py / mine_de_roles.py.

The dominant Spanish noise is the inclusive gender marker almost every Spanish ad appends
to the role: the slash suffix "Camarero/a", "Técnico/a", "Enfermero/a" and the parenthesized
tag "(H/M)" (hombre/mujer), "(M/H)", "(H/M/X)", or the leaked "(m/f)". Stripping those is the
single biggest win. Beyond it the shape is unlike German/Dutch: a Spanish role is usually a
HEAD plus a "de"-complement ("jefe de cocina", "técnico de mantenimiento", "auxiliar de
enfermería"), so — unlike the Dutch miner — we must NOT strip a trailing "de ..." (that IS
the role). We strip the gender markers, the "se busca/necesita" lead, locations, codes and
prep/location tails, frequency-rank the cleaned forms, promote recurring bare role heads,
seed essential roles, and emit the top-N as a Spanish autocomplete tier (es_roles.json),
shipped alongside app.py and merged into the suggestion corpus at load time.
"""

import json
import re
import sys
import unicodedata
from collections import Counter

import requests

SOLR = "http://localhost:8983/solr/jobs"
FQ = "lang:es"  # every Spanish-language doc (Adzuna Spain today; source-agnostic)
TOP_N = 500
MIN_COUNT = 5  # a cleaned form must recur this often to be a "canonical" role

# Essential roles that title-leading-head promotion may miss (they appear as the TAIL of a
# "de"-complement, or below the head-promotion variant bar). Each is included only if it
# independently clears MIN_COUNT as a title term in the live corpus (so the suggestion is
# grounded and always returns results).
SEED_ROLES = [
    "camarero",
    "cocinero",
    "ayudante de cocina",
    "enfermero",
    "auxiliar de enfermeria",
    "electricista",
    "fontanero",
    "soldador",
    "conductor",
    "mecanico",
    "carpintero",
    "albañil",
    "recepcionista",
    "dependiente",
    "vendedor",
    "comercial",
    "administrativo",
    "contable",
    "programador",
    "ingeniero",
    "profesor",
    "limpiador",
    "mozo de almacen",
    "carretillero",
    "repartidor",
    "teleoperador",
    "operario",
    "cajero",
    "panadero",
    "carnicero",
    "peluquero",
    "fisioterapeuta",
    "farmaceutico",
    "medico",
    "camarero de pisos",
]

# Leading tokens that are NOT occupations: seniority/quality modifiers and structural words.
# They recur widely enough to clear the head-promotion bar but must not become standalone
# role suggestions. (A genuine 2-word role led by one of these still survives via full-title
# counting.)
_NONROLE_HEAD = {
    "senior",
    "junior",
    "jefe",  # "jefe de cocina" is the role; bare "jefe" head is a fragment (survives full-title)
    "jefa",
    "responsable",  # "responsable de tienda" is the role; bare head is a fragment
    "gran",
    "nuevo",
    "nueva",
    "buen",
    "buena",
    "importante",  # "importante empresa ..." boilerplate
    "primer",
    "primera",
    "experimentado",
    "experimentada",
    "cualificado",
    "cualificada",
    "titulado",
    "titulada",
    "nuevos",
    "nuevas",
    # English modifiers/tokens leaking in from English-language ads
    "experienced",
    "the",
    "part",
    "full",
}

# Non-role noise that survives cleaning as a bare token (job-type, work arrangement, meta
# words), Spanish + the English leakage.
_STOP = {
    "empleo",
    "empleos",
    "trabajo",
    "trabajos",
    "vacante",
    "vacantes",
    "oferta",
    "ofertas",
    "puesto",
    "puestos",
    "contrato",
    "jornada",
    "completa",
    "parcial",
    "media",
    "indefinido",
    "temporal",
    "practicas",
    "beca",
    "becario",
    "becaria",
    "autonomo",
    "autonoma",
    "freelance",
    "teletrabajo",
    "remoto",
    "urgente",
    "busca",
    "buscamos",
    "solicita",
    "necesita",
    "necesitamos",
    "precisamos",
    "gente",
    "personal",
    "equipo",
    "sector",
    "empresa",
    "zona",
    "turno",
    "incorporacion",
    "trabajador",
    "trabajadora",
    "candidato",
    "candidata",
    "persona",
    "personas",
    "diversos",
    "diversas",
    "varios",
    "varias",
    "stage",
    "es",
    "jefe",  # bare "jefe"; the real roles "jefe de obra"/"jefe de cocina" survive full-title
    # non-role fragments (domain/material/service nouns that recur as bare titles)
    "banco",
    "alquiler",
    "tejados",
    "derribos",
    "aislamiento",
    "aislamientos",
    "pavimentos",
    "reparacion",
    "candidatura",
    "espontanea",
    "spontanea",
    # Portuguese cross-posts leak into the Spain index (pt is close to es and not in the
    # detector's gate set, so pt docs tag lang=es). Drop the high-frequency pt role heads.
    "executivo",
    "engenheiro",
    "engenharia",
    "pessoa",
    "coordenador",
    "desenvolvedor",
    "assistente",
    "cientista",
    "estagio",
    "vendas",
    # Italian / Polish / Hungarian cross-posts (same mechanism)
    "magazziniere",
    "pracownik",
    "értékesítő",
    "ertekesito",
    "agenti",
    # English meta/fragment tokens that aren't roles (real English ROLES like "software
    # engineer" survive — they're present in Spain's market and harmless in autocomplete)
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
    "paid",
    "officer",
    "lead",
    "associate",
    "staff",
    "salesforce",
    "backend",
    "technical",
    "mid",
    "level",
    # leading city/region names that survive as a bare token
    "madrid",
    "barcelona",
    "valencia",
    "sevilla",
    "zaragoza",
    "malaga",
    "murcia",
    "palma",
    "bilbao",
    "alicante",
    "cordoba",
    "valladolid",
    "vigo",
    "gijon",
    "granada",
    "coruña",
    "vitoria",
    "elche",
    "oviedo",
    "cartagena",
    "terrassa",
    "jerez",
    "sabadell",
    "mostoles",
    "pamplona",
    "almeria",
    "santander",
    "donostia",
    "burgos",
    "salamanca",
    "huelva",
    "lleida",
    "tarragona",
    "leon",
    "cadiz",
    "españa",
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
    "remote",
}


def fold(s: str) -> str:
    """Lowercase + strip diacritics (for dedup + accent-insensitive matching). Spanish
    á/é/í/ó/ú/ü fold to a/e/i/o/u; ñ folds to n via NFKD. Linguistically lossy but correct
    for the accent-insensitive autocomplete prefix match."""
    nfkd = unicodedata.normalize("NFKD", s)
    base = "".join(c for c in nfkd if not unicodedata.combining(c)).lower()
    return base.replace("'", "").replace("’", "")


_STOP_FOLDED = {fold(s) for s in _STOP}
_NONROLE_HEAD_FOLDED = {fold(s) for s in _NONROLE_HEAD}

# ---- cleaning regexes (applied in order) ----
# The inclusive gender tag in its Spanish variants: a parenthesized single marker "(H)",
# "(M)" or a slash/star sequence with or without parens — "(H/M)", "H/M", "(M/H/X)",
# "(m/f)", "(h/m/d)". Markers: h=hombre, m=mujer, plus leaked f/d/x.
_GENDER = re.compile(
    r"\(\s*[hmfdx]\s*\*?\s*\)"
    r"|\(?\s*[hmfdx](?:\s*[/*]\s*[hmfdx])+\s*\*?\s*\)?",
    re.I,
)
_CODE = re.compile(r"#\S+|\bref\.?\s*\S+|\bnº?\s*\S*\d\S*", re.I)  # req/ref codes
_PARENS = re.compile(r"[\(\[][^\)\]]*[\)\]]")  # (Madrid), [remoto], (12230)
_LEAD_SEEK = re.compile(
    r"^(?:se\s+(?:busca|buscan|necesita|necesitan|precisa|precisan|ofrece|solicita)|"
    r"buscamos|necesitamos|precisamos|solicitamos|se\s+selecciona|seleccionamos)\s+",
    re.I,
)  # "Se busca camarero" / "Buscamos cocinero" -> the role
_LEAD_ART = re.compile(r"^(?:un|una|unos|unas|el|la|los|las)\s+", re.I)  # "un Camarero"
# cut at the first comma OR colon: "Camarero, fines de semana" / "Oferta: cocinero" -> head
_COMMA_CUT = re.compile(r"\s*[,:].*$")
# the gender SLASH suffix only: "Camarero/a", "Técnico/a", "Enfermero/a" -> head. We cut at
# the FIRST slash; this is safe in Spanish because a real "X/Y" alternative role is rare,
# whereas "/a" gender is pervasive. (Done AFTER _GENDER strips the parenthesized form.)
_SLASH_CUT = re.compile(r"\s*/.*$")
_TRAIL_DASH = re.compile(r"\s+[-–—]\s+.*$")  # trailing " - <location/freetext>"
# trailing Spanish complement/location: "Camarero en Madrid", "Cocinero para hotel",
# "Vendedor con experiencia". CRITICAL: 'de' is deliberately EXCLUDED — a Spanish role is
# usually a head + "de"-complement ("jefe de cocina", "auxiliar de enfermería"), so 'de ...'
# IS the role, not a tail to strip.
_TRAIL_PREP = re.compile(
    r"\s+(?:en|para|con|sin|por|según|desde|hasta|zona|urge|incorporación|incorporacion)\b.*$",
    re.I,
)
# trailing EU driving-license class on a driver role ("Conductor C", "Conductor C+E"): a
# standalone 1-2 letter category, not part of the role name.
_TRAIL_LICENSE = re.compile(r"\s+(?:c\s*\+\s*e|be|ce|[bcde])$", re.I)
_WS = re.compile(r"\s+")


def clean_title(t: str) -> str:
    s = t.strip()
    s = _GENDER.sub(" ", s)  # strip the (H/M)-family tag FIRST
    s = _LEAD_SEEK.sub("", s)
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
    """How many Spanish docs carry `term` as a title phrase — the grounding test for a
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
    print(f"fetched {len(titles):,} Spanish titles", file=sys.stderr)
    counts: Counter[str] = Counter()
    surface: dict[str, Counter[str]] = {}
    # Bare-head counts: many Spanish role titles lead with the role head before a complement
    # ("Camarero en Madrid", "Vendedor con experiencia"); accumulate the leading token across
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
    print(f"{len(ranked)} canonical ES roles (>= {MIN_COUNT} occurrences)", file=sys.stderr)
    out = [{"text": role, "n": n} for role, n in ranked]
    with open("space/es_roles.json", "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=0)
    print("wrote space/es_roles.json", file=sys.stderr)
    for role, n in ranked[:30]:
        print(f"  {n:>5}  {role}", file=sys.stderr)


if __name__ == "__main__":
    main()
