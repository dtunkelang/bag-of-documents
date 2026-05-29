#!/usr/bin/env python3
"""Eval suggested searches on a handful of queries BEFORE wiring any UI.

For each query: print the top suggestions with cosine, and the raw band neighborhood
(what got cut as synonym above HIGH vs unrelated below LOW) so the band can be tuned
by eye. Good = NARROW/LATERAL role moves; bad = synonyms or level-only variants.
"""

from __future__ import annotations

import numpy as np
from sentence_transformers import SentenceTransformer
from suggest_lib import SIM_HIGH, SIM_LOW, RoleSuggester

QUERIES = [
    "software engineer",
    "senior software engineer",
    "data scientist",
    "product manager",
    "marketing manager",
    "registered nurse",
    "account executive",
    "mechanical engineer",
    "recruiter",
    "barista",
    "financial analyst",
    "ux designer",
]


def main() -> None:
    s = RoleSuggester()
    print(f"vocab: {len(s.phrases)} roles | band ({SIM_LOW}, {SIM_HIGH})\n")
    try:
        import torch

        device = "mps" if torch.backends.mps.is_available() else "cpu"
    except Exception:
        device = "cpu"
    model = SentenceTransformer("intfloat/e5-small-v2", device=device)

    for q in QUERIES:
        qv = model.encode([f"query: {q}"], normalize_embeddings=True)[0].astype(np.float32)
        sims = s.emb @ qv
        order = np.argsort(-sims)
        sugg = s.suggest(q, qv)

        print(f"=== {q!r} ===")
        print("  SUGGESTED:", " | ".join(f"{x['display']} ({x['sim']})" for x in sugg) or "(none)")
        above = [s.displays[i] for i in order if sims[i] >= SIM_HIGH][:6]
        print("  cut HIGH (synonym/self):", ", ".join(above) or "(none)")
        # nearest just-unrelated, to sanity-check the LOW floor
        below = [(s.displays[i], round(float(sims[i]), 3)) for i in order if sims[i] < SIM_LOW][:4]
        print("  just below LOW:", ", ".join(f"{d}({c})" for d, c in below))
        print()


if __name__ == "__main__":
    main()
