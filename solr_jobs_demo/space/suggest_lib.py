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


def _role_key(phrase: str) -> str:
    """Collapse a phrase to its level- and head-synonym-agnostic core so 'senior data
    engineer'/'data engineer' and 'software developer'/'software engineer' compare
    equal (level + synonym moves belong to the facet rail / are redundant, not here)."""
    p = _SENIORITY.sub(" ", phrase.lower())
    toks = _WS.sub(" ", p).strip().split()
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
        while cand and len(chosen) < k:
            if not chosen:
                best = cand[0]
            else:
                best, best_score = None, -1e9
                for i in cand:
                    div = max(float(self.emb[i] @ self.emb[j]) for j in chosen)
                    score = mmr_lambda * float(sims[i]) - (1 - mmr_lambda) * div
                    if score > best_score:
                        best, best_score = i, score
            chosen.append(best)
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
