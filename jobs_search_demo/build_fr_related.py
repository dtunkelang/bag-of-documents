#!/usr/bin/env python3
"""Build the grounded French *related-searches* bundle (space/fr_related.json).

French related searches can't ride the e5-small-v2 role suggester: on French it
ranks by MORPHOLOGY, not meaning (plombier->plongeur, jardinier->juriste,
developpeur->educateur), well below the coherence floor. The grounded fix is the
ROME occupation taxonomy (France Travail's national job classification):

  query --(appellation)--> ROME code --(mobilite professionnelle)--> related ROMEs
        --(corpus-mined role)--> a French role label that EXISTS in our inventory

So every suggestion is a genuine, France-Travail-validated occupational move
(monteur/poseur for a plombier, arboriste/macon-paysage for a jardinier) AND is
grounded in a role we actually have postings for, so it always returns results.

Data sources (both fully reproducible, no creds needed at serve time):
  * ROME 4.0 open data (data.gouv.fr "Toutes les donnees du ROME", v460):
      - unix_referentiel_appellation_*  : 13k job titles -> ROME code   (query->ROME)
      - unix_rubrique_mobilite_*        : 15k ranked ROME->ROME moves    (ROME->related)
  * Our live Solr index: France Travail titles -> corpus-mined French role vocab
    with counts (the ONLY display vocabulary, so suggestions are grounded).

Emits space/fr_related.json consumed by suggest_lib.FrRelatedSuggester. Re-run
after a corpus refresh (so display labels track live inventory) or a ROME bump.

Usage:
  ./run.sh start                 # local Solr must be up on :8983
  .venv/bin/python build_fr_related.py
"""

import io
import json
import os
import re
import sys
import unicodedata
import urllib.request
import zipfile
from collections import Counter, defaultdict

SOLR = "http://localhost:8983/solr/jobs"
FQ = "source_corpus:jobs_data_francetravail"
MIN_COUNT = 6  # a cleaned title must recur this often to be a "real" corpus role
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "space", "fr_related.json")

# data.gouv.fr "Toutes les donnees du ROME" — the full ROME 4.0 export (zip of CSVs).
ROME_ZIP_URL = "https://api.francetravail.fr/api-nomenclatureemploi/v1/open-data/csv"
ROME_CACHE = os.path.join(HERE, ".rome_opendata.zip")

# reuse the exact title cleaner used for the autocomplete tier so the related-search
# display vocabulary is identical in form to fr_roles.json.
sys.path.insert(0, HERE)
from mine_fr_roles import _STOP_FOLDED, clean_title, fetch_titles, fold  # noqa: E402

_WS = re.compile(r"\s+")


def _csv_fold(s: str) -> str:
    """Accent/punct-insensitive key for matching a query to a ROME appellation.
    Looser than mine_fr_roles.fold: also collapses punctuation to spaces so
    'aide-soignant' and 'aide soignant' share a key."""
    nfkd = unicodedata.normalize("NFKD", s.lower())
    base = "".join(c for c in nfkd if not unicodedata.combining(c))
    return _WS.sub(" ", re.sub(r"[^a-z0-9]+", " ", base)).strip()


# inclusive/feminine parenthetical right after a word: "Developpeur(euse)" -> "Developpeur"
_PAREN_GENDER = re.compile(r"\((?:e|se|ne|euse|rice|trice|ère|ière|er|ne|ve|sse)\)", re.I)
# dual-gender form "Masculin / Feminine <complement>" -> "Masculin <complement>": keep the
# masculine head + the shared complement, dropping only the feminine word. NOT a naive
# split (which would emit a bare "Chauffeur" and wrongly match the highest-volume ROME).
_DUAL = re.compile(r"^(\S+)\s*/\s*\S+(.*)$")


def degender(label: str) -> str:
    """ROME appellations are written 'Plombier / Plombiere', 'Chauffeur / Chauffeuse de
    poids lourd', 'Developpeur(euse) materiaux'. Collapse to the clean masculine form
    ('Plombier', 'Chauffeur de poids lourd', 'Developpeur materiaux') so the query key is
    the searchable role, not a stray gender token."""
    s = _PAREN_GENDER.sub("", label)
    m = _DUAL.match(s)
    if m:
        s = (m.group(1) + m.group(2)).strip()
    return _WS.sub(" ", s).strip()


def load_rome_zip() -> zipfile.ZipFile:
    if not os.path.exists(ROME_CACHE):
        print(f"downloading ROME open data -> {ROME_CACHE}", file=sys.stderr)
        req = urllib.request.Request(ROME_ZIP_URL, headers={"Accept": "application/zip"})
        with urllib.request.urlopen(req, timeout=120) as r:
            data = r.read()
        with open(ROME_CACHE, "wb") as f:
            f.write(data)
    return zipfile.ZipFile(ROME_CACHE)


def _read_csv(zf: zipfile.ZipFile, prefix: str) -> list[dict]:
    name = next(n for n in zf.namelist() if n.startswith(prefix))
    import csv

    with zf.open(name) as fh:
        return list(csv.DictReader(io.TextIOWrapper(fh, encoding="utf-8")))


def main() -> None:
    zf = load_rome_zip()

    # ---- 1. appellation / metier label -> candidate ROME codes (query resolution) ----
    # Both the masculine and feminine halves of "Plombier / Plombiere" become keys.
    # A label can fold to several ROMEs ("vendeur" appears under generalist vente AND
    # niche cosmetics); collisions are resolved to the most corpus-present ROME below,
    # so a bare "vendeur" lands on the generalist family, not a niche appellation.
    appel = _read_csv(zf, "unix_referentiel_appellation")
    label_romes: dict[str, set[str]] = defaultdict(set)
    for r in appel:
        rome = r["code_rome"]
        for lab in (r["libelle_appellation_long"], r["libelle_appellation_court"]):
            k = _csv_fold(degender(lab))
            if k:
                label_romes[k].add(rome)

    # (The cr_gd_dp arborescence file is deliberately NOT used: its row schema mixes
    # 2-digit domaine codes into the rome columns, and suggestions are driven by the
    # corpus role vocabulary below, never by a metier-category label.)

    # ---- 2. mobilite professionnelle: ROME -> [target ROMEs, by official order] ----
    mob_raw: dict[str, list[tuple[int, str]]] = defaultdict(list)
    for r in _read_csv(zf, "unix_rubrique_mobilite"):
        try:
            ordre = int(r["numero_ordre"])
        except (ValueError, KeyError):
            ordre = 999
        mob_raw[r["code_rome"]].append((ordre, r["code_rome_cible"]))
    mobilite = {k: [t for _, t in sorted(v)] for k, v in mob_raw.items()}
    print(
        f"ROME: {len(label_romes):,} query keys, {len(mobilite):,} ROMEs with mobilites",
        file=sys.stderr,
    )

    # ---- 3. corpus-mined French role vocab (the grounded display labels) ----
    titles = fetch_titles()
    print(f"fetched {len(titles):,} FT titles", file=sys.stderr)
    counts: Counter[str] = Counter()
    surface: dict[str, Counter[str]] = {}
    for t in titles:
        c = clean_title(t)
        if not (2 <= len(c) <= 48) or len(c.split()) > 6:
            continue
        if not re.search(r"[a-zà-ÿ]", c):
            continue
        k = fold(c)
        if k in _STOP_FOLDED:
            continue
        counts[k] += 1
        surface.setdefault(k, Counter())[c] += 1

    # map each corpus role -> ROME, keep best (accented) surface form + count. When a
    # role's label folds to several ROMEs, attribute it to the lexicographically first
    # (collisions among mapped roles are rare); query-side collisions are resolved by
    # corpus weight below, which is the case that actually matters for ambiguous terms.
    rome_roles: dict[str, list[dict]] = defaultdict(list)
    rome_weight: Counter[str] = Counter()
    mapped = unmapped = 0
    for k, n in counts.most_common():
        if n < MIN_COUNT:
            continue
        text = surface[k].most_common(1)[0][0]
        romes = label_romes.get(_csv_fold(text))
        if not romes:
            unmapped += 1
            continue
        rome = min(romes)
        mapped += 1
        rome_roles[rome].append({"text": text, "n": n})
        rome_weight[rome] += n
    for rome in rome_roles:
        rome_roles[rome].sort(key=lambda d: -d["n"])

    # collapse label -> single ROME: on collision prefer the most corpus-present ROME,
    # so a bare "vendeur" resolves to the generalist vente family, not niche cosmetics.
    label2rome = {
        k: (min(romes) if len(romes) == 1 else max(romes, key=lambda r: (rome_weight[r], r)))
        for k, romes in label_romes.items()
    }
    print(
        f"corpus roles: {mapped:,} mapped to {len(rome_roles):,} ROMEs, {unmapped:,} unmapped",
        file=sys.stderr,
    )

    # ---- 4. same-domaine fallback: domaine (ROME[:3]) -> ROMEs we have roles for ----
    dom2romes: dict[str, list[str]] = defaultdict(list)
    for rome in rome_roles:
        dom2romes[rome[:3]].append(rome)
    for dom in dom2romes:
        dom2romes[dom].sort(key=lambda r: -rome_weight[r])

    bundle = {
        "label2rome": label2rome,
        "rome_weight": dict(rome_weight),
        "mobilite": mobilite,
        "rome_roles": dict(rome_roles),
        "dom2romes": dict(dom2romes),
    }
    with open(OUT, "w") as f:
        json.dump(bundle, f, ensure_ascii=False)
    sz = os.path.getsize(OUT) / 1024
    print(f"wrote {OUT} ({sz:.0f} KB)", file=sys.stderr)


if __name__ == "__main__":
    main()
