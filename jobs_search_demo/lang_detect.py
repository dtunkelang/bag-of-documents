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
# French and JobTech (Arbetsförmedlingen) Swedish; restricting the classifier to this
# set makes long doc text resolve cleanly.
GATE_LANGS = ("en", "fr", "sv")

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
_TOKEN = re.compile(r"[a-zàâäéèêëîïôöùûüÿçœæå]+", re.I)

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


def query_lang_mode(query: str, fr_floor: float = 0.90, sv_floor: float = 0.90) -> str:
    """Map a search query to a gate mode ('fr' | 'sv' | 'en'): a non-English mode only
    when the query carries that language's positive signal AND the classifier confidently
    agrees (prob >= floor); else 'en'. High-precision by design (see module docstring) —
    short ambiguous/English queries stay 'en' so we never scope a user to the wrong-language
    inventory. French is checked before Swedish (it is far higher-volume in the index)."""
    if not query or not query.strip():
        return "en"
    if _has_fr_signal(query):
        lang, prob = detect_lang(query)
        if lang == "fr" and prob >= fr_floor:
            return "fr"
    if _has_sv_signal(query):
        lang, prob = detect_lang(query)
        if lang == "sv" and prob >= sv_floor:
            return "sv"
    return "en"
