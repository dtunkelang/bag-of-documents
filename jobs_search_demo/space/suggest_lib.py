"""Query-context SUGGESTED SEARCHES — semantic role reformulation.

Given a query, suggest NARROW (software engineer -> ML engineer) or LATERAL
(-> data engineer) role moves drawn from the offline corpus role vocabulary
(build_role_vocab.py). Deliberately NOT suggested: synonyms (-> software developer,
cut by the upper similarity bound) and level-only variants (-> senior SWE, collapsed
out of the vocab at build time + filtered here) — both are redundant or unexciting.

Mechanism (all phrase<->phrase e5-small cosine, which has real dynamic range):
  1. similarity BAND: LOW < cos < HIGH  (HIGH cuts synonyms/self, LOW cuts unrelated)
  2. level/self filter: drop candidates that reduce to the query's own role phrase
  3. MMR: trade relevance vs mutual diversity so the 4 picks aren't near-duplicates
  4. grounding: vocab is corpus-mined with a min count, so every pick has results
"""

from __future__ import annotations

import json
import os
import re
import unicodedata

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))

# Band tuned on e5-small-v2 query<->query cosine (see measure_suggest.py).
SIM_HIGH = 0.965  # above this = synonym or the query itself -> not a useful move
SIM_LOW = 0.86  # below this = unrelated role -> not a reformulation of the query
MMR_LAMBDA = 0.62  # relevance vs diversity in the MMR re-rank
DEFAULT_K = 4

_SENIORITY = re.compile(
    r"\b(?:senior|sr\.?|junior|jr\.?|lead|principal|staff|chief|head|entry[- ]?level|"
    r"associate|assistant|mid[- ]?level|experienced|expert|master|apprentice|trainee|"
    r"intern|graduate|grad)\b",
    re.I,
)
_WS = re.compile(r"\s+")

# Interchangeable role-head synonyms: a candidate that differs from the query ONLY by
# one of these is a synonym (software developer ~ software engineer), not a move -- it
# returns ~the same results, so it's filtered alongside level-variants.
_SYN_HEAD = {
    "developer": "engineer",
    "programmer": "engineer",
    "engineering": "engineer",
    "rep": "representative",
    "salesperson": "representative",
    "coder": "engineer",
}

# Generic qualifier words that rarely distinguish one role MOVE from another: two phrases
# differing only by one of these are equivalent for dedup ('full stack software engineer'
# ~ 'full stack engineer'). Kept minimal to avoid over-collapsing distinct roles.
_FILLER = {"software"}
_PUNCT = re.compile(r"[-/]+")
# Glued compounds that also occur spaced ('fullstack' ~ 'full stack'); normalize so the
# two spellings share a key. Word-bounded to avoid mangling longer tokens.
_GLUED_MAP = {"fullstack": "full stack", "frontend": "front end", "backend": "back end"}
_GLUED = re.compile(r"\b(?:{})\b".format("|".join(_GLUED_MAP)))


def _role_key(phrase: str) -> str:
    """Collapse a phrase to its level-, punctuation-, filler-, and head-synonym-agnostic
    core so 'senior data engineer'/'data engineer', 'full-stack'/'fullstack'/'full stack',
    and 'full stack software engineer'/'full stack engineer' all compare equal (level,
    synonym, hyphenation, spacing, and filler differences are redundant, not real moves)."""
    p = _PUNCT.sub(" ", phrase.lower())  # 'full-stack' -> 'full stack'
    p = _GLUED.sub(lambda m: _GLUED_MAP[m.group(0)], p)  # 'fullstack' -> 'full stack'
    p = _SENIORITY.sub(" ", p)
    toks = [t for t in _WS.sub(" ", p).strip().split() if t not in _FILLER]
    if toks:
        toks[-1] = _SYN_HEAD.get(toks[-1], toks[-1])
    return " ".join(toks)


class RoleSuggester:
    def __init__(self, vocab_path: str | None = None, emb_path: str | None = None):
        vocab_path = vocab_path or os.path.join(HERE, "role_vocab.json")
        emb_path = emb_path or os.path.join(HERE, "role_vocab_emb.npy")
        with open(vocab_path) as f:
            self.vocab = json.load(f)
        self.emb = np.load(emb_path)  # [N,384], L2-normalized, row-aligned to vocab
        self.phrases = [v["phrase"] for v in self.vocab]
        self.displays = [v["display"] for v in self.vocab]
        self.keys = [_role_key(p) for p in self.phrases]

    def suggest(
        self,
        query: str,
        qv: np.ndarray,
        k: int = DEFAULT_K,
        sim_low: float = SIM_LOW,
        sim_high: float = SIM_HIGH,
        mmr_lambda: float = MMR_LAMBDA,
    ) -> list[dict]:
        """`qv` is the L2-normalized e5-small-v2 embedding of the query
        ('query: ' prefix, computed by the caller). Returns up to k suggestions
        [{display, phrase, count, sim}], MMR-diversified, best first."""
        qkey = _role_key(query.strip())
        sims = self.emb @ qv.astype(np.float32)  # cosine (both normalized)
        cand: list[int] = []
        for i, s in enumerate(sims):
            if not (sim_low < s < sim_high):
                continue
            key = self.keys[i]
            # drop the query's own role and pure level-variants of it
            if key == qkey or key in qkey or qkey in key:
                continue
            cand.append(i)
        cand.sort(key=lambda i: -sims[i])
        cand = cand[: max(k * 8, 24)]  # MMR over a shortlist
        if not cand:
            return []

        chosen: list[int] = []
        chosen_keys: set[str] = set()
        while cand and len(chosen) < k:
            if not chosen:
                best = cand[0]
            else:
                best, best_score = None, -1e9
                for i in cand:
                    if self.keys[i] in chosen_keys:
                        continue  # equivalent to an already-picked suggestion
                    div = max(float(self.emb[i] @ self.emb[j]) for j in chosen)
                    score = mmr_lambda * float(sims[i]) - (1 - mmr_lambda) * div
                    if score > best_score:
                        best, best_score = i, score
                if best is None:
                    break  # every remaining candidate duplicates a chosen one
            chosen.append(best)
            chosen_keys.add(self.keys[best])
            cand.remove(best)
        return [
            {
                "display": self.displays[i],
                "phrase": self.phrases[i],
                "count": self.vocab[i]["count"],
                "sim": round(float(sims[i]), 4),
            }
            for i in chosen
        ]


# ===== French related searches — grounded in the ROME occupation taxonomy =====
# e5-small-v2 ranks French roles by morphology, not meaning (plombier->plongeur), so
# the English RoleSuggester can't serve French. This lane instead walks France
# Travail's ROME taxonomy: query --(appellation)--> ROME --(mobilite)--> related ROMEs,
# then displays a corpus-mined French role for each (so every pick has results). The
# bundle (fr_related.json) is built offline by build_fr_related.py.

_FR_WS = re.compile(r"\s+")


def _fr_fold(s: str) -> str:
    nfkd = unicodedata.normalize("NFKD", s.lower())
    base = "".join(c for c in nfkd if not unicodedata.combining(c))
    return _FR_WS.sub(" ", re.sub(r"[^a-z0-9]+", " ", base)).strip()


# Feminine occupational suffix -> masculine, applied per token to an already-FOLDED
# string (accents stripped) so a feminine free-text query or resume ("infirmière",
# "vendeuse", "technicienne", "directrice") matches the masculine canonical role
# vocabulary. Longest/most-specific suffix first; only fires when the stem stays >=3
# chars so a short word isn't mangled into a spurious root.
_FR_FEM = [
    ("trice", "teur"),  # directrice->directeur, animatrice->animateur, formatrice->formateur
    ("ienne", "ien"),  # technicienne->technicien, pharmacienne->pharmacien, gardienne->gardien
    ("iere", "ier"),  # (folded ière) infirmiere->infirmier, ouvriere->ouvrier, caissiere->caissier
    ("euse", "eur"),  # vendeuse->vendeur, serveuse->serveur, coiffeuse->coiffeur
    ("ante", "ant"),  # assistante->assistant, consultante->consultant
    ("ente", "ent"),  # agente->agent
    ("ere", "er"),  # (folded ère) boulangere->boulanger
]


def degender_fr(folded: str) -> str:
    """Map feminine French occupational tokens to their masculine form on an already
    accent-folded string, so feminine surface forms match a masculine role vocabulary."""
    out = []
    for tok in folded.split():
        for suf, rep in _FR_FEM:
            if len(tok) >= len(suf) + 3 and tok.endswith(suf):
                tok = tok[: -len(suf)] + rep
                break
        out.append(tok)
    return " ".join(out)


class FrRelatedSuggester:
    def __init__(self, bundle_path: str | None = None):
        path = bundle_path or os.path.join(HERE, "fr_related.json")
        with open(path) as f:
            b = json.load(f)
        self.label2rome: dict[str, str] = b["label2rome"]
        self.mobilite: dict[str, list[str]] = b["mobilite"]
        self.rome_roles: dict[str, list[dict]] = b["rome_roles"]
        self.dom2romes: dict[str, list[str]] = b["dom2romes"]
        self.rome_weight: dict[str, int] = b["rome_weight"]
        # head token of each corpus role -> {rome: count}: the empirical prior for
        # disambiguating a bare query ("chauffeur") to the ROME we actually staff.
        self.head_rome: dict[str, dict[str, int]] = {}
        for rome, roles in self.rome_roles.items():
            for d in roles:
                toks = _fr_fold(d["text"]).split()
                if toks:
                    self.head_rome.setdefault(toks[0], {})[rome] = (
                        self.head_rome.get(toks[0], {}).get(rome, 0) + d["n"]
                    )

    def _match(self, fq: str) -> str | None:
        """Resolve one folded phrase to a ROME: exact appellation, else the
        appellation(s) it heads, ranked by how much of that ROME we actually staff."""
        if not fq:
            return None
        if fq in self.label2rome:
            return self.label2rome[fq]
        cands: dict[str, int] = {}
        for k, rome in self.label2rome.items():
            if k == fq or k.startswith(fq + " "):
                cands[rome] = cands.get(rome, 0) + self.rome_weight.get(rome, 0)
        if not cands:
            return None
        corpus = self.head_rome.get(fq.split()[0], {})
        return max(cands, key=lambda r: (corpus.get(r, 0), self.rome_weight.get(r, 0), r))

    def _resolve(self, query: str) -> str | None:
        """Map a (possibly qualified) French query to a ROME, backing off trailing
        words: 'infirmier de nuit' -> 'infirmier de' -> 'infirmier' -> J1506."""
        toks = _fr_fold(query).split()
        for j in range(len(toks), 0, -1):
            rome = self._match(" ".join(toks[:j]))
            if rome:
                return rome
        # Feminine free-text ("vendeuse", "infirmière") won't match the degendered
        # masculine appellation table -- retry on the degendered form.
        dg = degender_fr(_fr_fold(query)).split()
        if dg != toks:
            for j in range(len(dg), 0, -1):
                rome = self._match(" ".join(dg[:j]))
                if rome:
                    return rome
        return None

    def suggest(self, query: str, k: int = DEFAULT_K) -> list[dict]:
        """Related French role searches for `query`: France-Travail-validated career
        moves (mobilite), filled with same-domaine peers, each shown as a corpus-mined
        role so it always returns results. Returns [{display, count}], best first."""
        rome = self._resolve(query)
        if not rome:
            return []
        qkey = _fr_fold(query)
        targets = list(self.mobilite.get(rome, []))
        for r in self.dom2romes.get(rome[:3], []):  # same-domaine fill
            if r != rome and r not in targets:
                targets.append(r)
        out: list[dict] = []
        seen = {qkey}
        for t in targets:
            for role in self.rome_roles.get(t, []):
                rk = _fr_fold(role["text"])
                # skip the query's own role and any sub/superstring of it (redundant)
                if rk in seen or qkey in rk or rk in qkey:
                    continue
                seen.add(rk)
                out.append({"display": role["text"], "phrase": role["text"], "count": role["n"]})
                break
            if len(out) >= k:
                break
        return out
