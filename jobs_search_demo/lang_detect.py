"""Lightweight language detection for the jobs demo.

One shared helper used on BOTH sides of the pipeline:
  * index time (refresh.py stage_unify) — tag every doc with a 2-letter `lang`.
  * serve time (space/app.py) — detect the query language to gate retrieval.

Backed by py3langid (pure-python, model bundled, no network). We restrict the
classifier to the languages the corpus actually contains in volume (English +
French from France Travail).

`detect_lang` (used on long doc text, where the classifier is reliable) returns
("en"|"fr", prob). `query_lang_mode` is for SHORT queries, where a bare classifier
is NOT reliable: real English job titles ("python developer", "real estate agent",
".net developer") score French with high confidence because short Latinate phrases
look French to a char-n-gram model. So the query gate is HIGH-PRECISION: it flips to
French only when the query carries a POSITIVE French signal (a French diacritic or a
French structural function word) AND the classifier agrees. Measured on a 92k-volume
real query workload, this dropped English false positives from 843 to 0 while still
firing on genuinely-French queries. Missing a pure-ASCII French content query (e.g.
"comptable") is the safe direction — it falls back to English retrieval, which
already returns French docs well for French queries, rather than stranding a user.
"""

from __future__ import annotations

import re
from functools import lru_cache

# Languages we gate on. The index is overwhelmingly English plus France Travail
# French, JobTech (Arbetsförmedlingen) Swedish, Adzuna Germany German, and Adzuna
# Netherlands Dutch; restricting the classifier to this set makes long doc text
# resolve cleanly.
GATE_LANGS = ("en", "fr", "sv", "de", "nl", "es", "it")

# Positive French signals for the query gate.
_FR_DIACRITIC = re.compile(r"[àâäéèêëîïôöùûüÿçœæ]", re.I)
# French structural/function words that an English job-title query won't contain.
# Deliberately excludes Latinate content words ("agent", "manager") that are shared
# with English. Catches accent-stripped French traffic ("charge de recrutement")
# where diacritics were normalized away.
_FR_WORDS = frozenset(
    {
        "de",
        "des",
        "du",
        "la",
        "le",
        "les",
        "un",
        "une",
        "et",
        "en",
        "dans",
        "pour",
        "avec",
        "chez",
        "au",
        "aux",
        "recherche",
        "emploi",
        "emplois",
        "offre",
        "offres",
        "poste",
        "metier",
        "metiers",
        "stage",
        "alternance",
        "charge",
        "chargee",
        "adjoint",
    }
)
_TOKEN = re.compile(r"[a-zàâäéèêëîïôöùûüÿçœæåñáíóú]+", re.I)

# Positive Swedish signals for the query gate. 'å' is uniquely Scandinavian; 'ä'/'ö'
# are shared with (rare) French, but the gate checks French FIRST and the Swedish
# branch still requires the classifier to confidently return 'sv' (measured 1.0 on
# single-word role queries like "sjuksköterska"/"ingenjör"), so a French ä/ö word
# resolves to 'fr' before it can reach here. The function/role words below additionally
# catch accent-free Swedish traffic.
_SV_DIACRITIC = re.compile(r"[åäö]", re.I)
_SV_WORDS = frozenset(
    {
        "jobb",
        "jobben",
        "lediga",
        "ledig",
        "sökes",
        "söker",
        "tjänst",
        "tjänster",
        "anställning",
        "arbete",
        "och",
        "för",
        "inom",
        "samt",
        "heltid",
        "deltid",
        "vikariat",
    }
)

# Positive German signals for the query gate. 'ß' is uniquely German; 'ü' is German
# (shared only with rare French, which is checked FIRST). 'ä'/'ö' are shared with
# Swedish so they are NOT German diacritic signals on their own — German must instead
# carry 'ü'/'ß' OR a German function/role word (and the classifier must still confidently
# return 'de'). German is checked BEFORE Swedish, but a Swedish ä/ö word never carries a
# German signal, so it falls through to the Swedish branch.
_DE_DIACRITIC = re.compile(r"[üß]", re.I)
_DE_WORDS = frozenset(
    {
        "der",
        "die",
        "das",
        "den",
        "dem",
        "ein",
        "eine",
        "einen",
        "und",
        "oder",
        "für",
        "fur",
        "mit",
        "im",
        "bei",
        "als",
        "von",
        "zur",
        "zum",
        "stelle",
        "stellen",
        "stellenangebot",
        "stellenangebote",
        "stellenanzeige",
        "arbeit",
        "beruf",
        "ausbildung",
        "gesucht",
        "vollzeit",
        "teilzeit",
        "mitarbeiter",
        "fachkraft",
    }
)

# Positive Dutch signals for the query gate. Dutch has no diacritic of its own (ë/ï are
# shared with French, which is checked FIRST and guarded by classifier agreement), so the
# signal is purely structural/function/role words an English (or French/German) job-title
# query won't carry. Shared short articles ("de", "en") are deliberately omitted — they
# collide with French and add no Dutch-specific evidence; the words below ("vacature",
# "gezocht", "een", "het", "medewerker") are unambiguously Dutch. A bare Dutch cognate role
# ("verpleegkundige") carries no signal and falls to 'en' — the safe direction, mirroring
# French ("comptable"); the related-search router recovers it via the ESCO resolver.
_NL_WORDS = frozenset(
    {
        "een",
        "het",
        "van",
        "voor",
        "bij",
        "met",
        "naar",
        "aan",
        "uit",
        "om",
        "vacature",
        "vacatures",
        "baan",
        "banen",
        "werk",
        "werken",
        "gezocht",
        "gevraagd",
        "medewerker",
        "medewerkster",
        "ervaren",
        "regio",
        "deeltijd",
        "voltijd",
        "dienstverband",
        "stage",
        "zzp",
    }
)

# Positive Spanish signals for the query gate. 'ñ' is uniquely Spanish; the acute-accent
# vowels 'á'/'í'/'ó'/'ú' do NOT occur in French (French uses é/è/ê/à/â/î/ô/û/ç — checked
# FIRST), so they are Spanish-distinguishing here. 'é'/'ü' ARE shared with French and so
# are deliberately NOT Spanish diacritic signals on their own (mirrors German excluding the
# Swedish-shared ä/ö) — a Spanish 'é'-word ("médico") instead relies on the function/role
# words below or falls to 'en', the safe direction. Shared short articles ("un"/"de"/"la")
# are omitted — they collide with French and add no Spanish-specific evidence.
_ES_DIACRITIC = re.compile(r"[ñáíóú]", re.I)
_ES_WORDS = frozenset(
    {
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
        "busca",
        "buscamos",
        "buscando",
        "solicita",
        "solicitamos",
        "requiere",
        "requieren",
        "necesita",
        "necesitamos",
        "contrato",
        "jornada",
        "experiencia",
        "salario",
        "sueldo",
        "para",
        "con",
        "del",
        "los",
        "las",
        "una",
        "empresa",
        "sector",
    }
)


# Positive Italian signals for the query gate. Italian has NO unique diacritic — its
# accented vowels à/è/é/ì/ò/ù all occur in French (checked FIRST and guarded by classifier
# agreement), so — exactly like Dutch — the signal is purely structural/function/role words
# an English (or French/Spanish) job-title query won't carry. Shared short articles/preps
# ("di", "in", "e", "la", "le", "un", "una", "con") are deliberately omitted — they collide
# with French/Spanish and add no Italian-specific evidence. The words below ("lavoro",
# "azienda", "cercasi", "assunzione", "tirocinio", "della", "presso", "addetto") are
# unambiguously Italian. A bare Italian cognate role ("infermiere", "elettricista") carries
# no signal and falls to 'en' — the safe direction, mirroring French/Dutch; the related-
# search router recovers it via the ESCO resolver.
_IT_WORDS = frozenset(
    {
        "della",
        "delle",
        "degli",
        "dello",
        "nella",
        "nelle",
        "sulla",
        "presso",
        "anche",
        "oppure",
        "nostra",
        "nostro",
        "per",
        "lavoro",
        "lavori",
        "impiego",
        "posizione",
        "posizioni",
        "mansione",
        "mansioni",
        "azienda",
        "assunzione",
        "tirocinio",
        "apprendistato",
        "stipendio",
        "retribuzione",
        "offresi",
        "cercasi",
        "cerchiamo",
        "ricerca",
        "ricerchiamo",
        "selezioniamo",
        "selezione",
        "esperienza",
        "automunito",
        "richiesto",
        "richiesta",
        "addetto",
        "addetta",
        "addetti",
        "indeterminato",
        "determinato",
        "turni",
    }
)


@lru_cache(maxsize=1)
def _identifier():
    from py3langid.langid import MODEL_FILE, LanguageIdentifier

    idf = LanguageIdentifier.from_pickled_model(MODEL_FILE, norm_probs=True)
    idf.set_languages(list(GATE_LANGS))
    return idf


def detect_lang(text: str) -> tuple[str, float]:
    """Classify `text` (a doc's title+description) into one of GATE_LANGS. Returns
    (lang, probability). Empty text -> ("en", 0.0). Reliable on long text; for short
    queries use query_lang_mode instead."""
    text = (text or "").strip()
    if not text:
        return ("en", 0.0)
    lang, prob = _identifier().classify(text[:2000])
    return (str(lang), float(prob))


def _has_fr_signal(query: str) -> bool:
    """True if the query carries an unambiguous French signal (a French diacritic or
    a French structural function word as a whole token)."""
    if _FR_DIACRITIC.search(query):
        return True
    return bool({t.lower() for t in _TOKEN.findall(query)} & _FR_WORDS)


def _has_sv_signal(query: str) -> bool:
    """True if the query carries an unambiguous Swedish signal (the Scandinavian-only
    letter 'å' or a Swedish structural/role word as a whole token)."""
    if _SV_DIACRITIC.search(query):
        return True
    return bool({t.lower() for t in _TOKEN.findall(query)} & _SV_WORDS)


def _has_de_signal(query: str) -> bool:
    """True if the query carries an unambiguous German signal (the German-only letters
    'ü'/'ß' or a German structural/role word as a whole token). 'ä'/'ö' are deliberately
    NOT German signals — they are shared with Swedish, which has its own branch."""
    if _DE_DIACRITIC.search(query):
        return True
    return bool({t.lower() for t in _TOKEN.findall(query)} & _DE_WORDS)


def _has_nl_signal(query: str) -> bool:
    """True if the query carries an unambiguous Dutch signal (a Dutch structural/function/
    role word as a whole token). Dutch has no unique diacritic, so this is word-only."""
    return bool({t.lower() for t in _TOKEN.findall(query)} & _NL_WORDS)


def _has_es_signal(query: str) -> bool:
    """True if the query carries an unambiguous Spanish signal (the Spanish-distinguishing
    letters 'ñ'/'á'/'í'/'ó'/'ú' — none of which occur in French — or a Spanish structural/
    role word as a whole token). 'é'/'ü' are deliberately NOT Spanish signals: they are
    shared with French, which has its own branch checked first."""
    if _ES_DIACRITIC.search(query):
        return True
    return bool({t.lower() for t in _TOKEN.findall(query)} & _ES_WORDS)


def _has_it_signal(query: str) -> bool:
    """True if the query carries an unambiguous Italian signal (an Italian structural/
    function/role word as a whole token). Italian has no unique diacritic, so this is
    word-only (mirrors Dutch)."""
    return bool({t.lower() for t in _TOKEN.findall(query)} & _IT_WORDS)


def query_lang_mode(
    query: str,
    fr_floor: float = 0.90,
    sv_floor: float = 0.90,
    de_floor: float = 0.90,
    nl_floor: float = 0.90,
    es_floor: float = 0.90,
    it_floor: float = 0.90,
) -> str:
    """Map a search query to a gate mode ('fr' | 'de' | 'sv' | 'nl' | 'es' | 'it' | 'en'): a
    non-English mode only when the query carries that language's positive signal AND the
    classifier confidently agrees (prob >= floor); else 'en'. High-precision by design (see
    module docstring) — short ambiguous/English queries stay 'en' so we never scope a user
    to the wrong-language inventory. French is checked first (highest-volume), then German,
    then Swedish, then Dutch, then Spanish, then Italian; the diacritic signals are disjoint
    (fr accents / ü,ß / å / none for nl / ñ,á,í,ó,ú for es / none for it) so order only
    matters for shared-function-word edge cases, which the classifier-agreement guard
    resolves anyway."""
    if not query or not query.strip():
        return "en"
    if _has_fr_signal(query):
        lang, prob = detect_lang(query)
        if lang == "fr" and prob >= fr_floor:
            return "fr"
    if _has_de_signal(query):
        lang, prob = detect_lang(query)
        if lang == "de" and prob >= de_floor:
            return "de"
    if _has_sv_signal(query):
        lang, prob = detect_lang(query)
        if lang == "sv" and prob >= sv_floor:
            return "sv"
    if _has_nl_signal(query):
        lang, prob = detect_lang(query)
        if lang == "nl" and prob >= nl_floor:
            return "nl"
    if _has_es_signal(query):
        lang, prob = detect_lang(query)
        if lang == "es" and prob >= es_floor:
            return "es"
    if _has_it_signal(query):
        lang, prob = detect_lang(query)
        if lang == "it" and prob >= it_floor:
            return "it"
    return "en"
