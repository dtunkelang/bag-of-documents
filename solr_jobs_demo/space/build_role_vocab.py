#!/usr/bin/env python3
"""Offline builder for query-context SUGGESTED SEARCHES (semantic role reformulation).

Mines a clean role vocabulary from the corpus `title_display` field, then embeds it
once with e5-small-v2 ("query:" prefix) so the live app can, given a query, suggest
NARROW (software engineer -> ML engineer) or LATERAL (-> data engineer) role moves.

Why offline + corpus-grounded (Option 2, no LLM): every suggestion is a real role
that has results, normalization happens once, and phrase<->phrase e5 cosine has the
dynamic range (synonym ~0.95 / sibling ~0.88 / unrelated ~0.7) the band heuristic needs
-- unlike the flat doc<->phrase signal.

Writes (next to this script):
  role_vocab.json      list of {"phrase","display","count"}, sorted by count desc
  role_vocab_emb.npy   float32 [N,384] L2-normalized e5-small-v2 query embeddings (row-aligned)
"""

from __future__ import annotations

import json
import os
import re
from collections import Counter

import numpy as np
import requests

SOLR = os.environ.get("SOLR", "http://localhost:8983")
CORE = os.environ.get("CORE", "jobs")
HERE = os.path.dirname(os.path.abspath(__file__))
DENSE_MODEL = "intfloat/e5-small-v2"

MIN_COUNT = 25  # a role must appear >= this many times -> suggestions always have results
PAGE = 20000

# Role-head nouns: a mined phrase is kept only if its last token is one of these.
# This is the junk filter -- titles full of locations / team names / codes get dropped.
ROLE_HEADS = {
    "engineer",
    "engineering",
    "developer",
    "programmer",
    "architect",
    "scientist",
    "analyst",
    "manager",
    "director",
    "lead",
    "head",
    "officer",
    "specialist",
    "coordinator",
    "administrator",
    "consultant",
    "advisor",
    "associate",
    "assistant",
    "representative",
    "rep",
    "executive",
    "agent",
    "designer",
    "researcher",
    "technician",
    "tech",
    "operator",
    "mechanic",
    "electrician",
    "plumber",
    "welder",
    "nurse",
    "physician",
    "psychiatrist",
    "therapist",
    "pharmacist",
    "dentist",
    "veterinarian",
    "hygienist",
    "paramedic",
    "counselor",
    "clinician",
    "teacher",
    "instructor",
    "professor",
    "tutor",
    "educator",
    "trainer",
    "accountant",
    "auditor",
    "controller",
    "bookkeeper",
    "underwriter",
    "actuary",
    "recruiter",
    "buyer",
    "planner",
    "estimator",
    "scheduler",
    "dispatcher",
    "supervisor",
    "foreman",
    "superintendent",
    "chef",
    "cook",
    "barista",
    "bartender",
    "server",
    "cashier",
    "clerk",
    "receptionist",
    "secretary",
    "paralegal",
    "attorney",
    "lawyer",
    "counsel",
    "writer",
    "editor",
    "copywriter",
    "producer",
    "strategist",
    "marketer",
    "salesperson",
    "seller",
    "broker",
    "trader",
    "banker",
    "teller",
    "driver",
    "pilot",
    "machinist",
    "fabricator",
    "installer",
    "inspector",
    "carpenter",
    "painter",
    "roofer",
    "laborer",
    "janitor",
    "custodian",
    "housekeeper",
    "guard",
    "phlebotomist",
    "sonographer",
    "radiologist",
    "anesthesiologist",
    "surgeon",
    "midwife",
    "optometrist",
    "podiatrist",
    "chiropractor",
    "dietitian",
    "nutritionist",
    "geologist",
    "biologist",
    "chemist",
    "physicist",
    "statistician",
    "economist",
    "cartographer",
    "surveyor",
    "draftsman",
    "toolmaker",
    "evangelist",
    "ambassador",
    "generalist",
    "partner",
    "principal",
    "fellow",
    "intern",
    "apprentice",
    "trainee",
    "owner",
    "founder",
    "president",
    "ceo",
    "cfo",
    "cto",
    "coo",
    "vp",
    "captain",
    "crew",
    "staff",
}

# Leading seniority / qualifier tokens to strip (so "senior software engineer" and
# "software engineer" collapse -- the level dimension belongs to the facet rail).
SENIORITY = re.compile(
    r"^(?:senior|sr\.?|junior|jr\.?|lead|principal|staff|chief|head|entry[- ]?level|"
    r"associate|assistant|mid[- ]?level|experienced|expert|master|apprentice|trainee|"
    r"intern|graduate|grad|global|regional|national|international|us|usa|uk|emea)\s+",
    re.I,
)
LEVEL_SUFFIX = re.compile(r"\s+(?:i{1,3}|iv|v|vi|[1-5]|level\s*[1-5])$", re.I)
LEADING_CODE = re.compile(r"^\s*[\(\[]\s*\d+\s*[\)\]]\s*")  # "(511) ", "[42] "
PAREN = re.compile(r"\([^)]*\)|\[[^\]]*\]")
NONWORD_EDGE = re.compile(r"^[^\w]+|[^\w]+$")
WS = re.compile(r"\s+")
# split on the first hard separator -> keep the head role phrase
SEP = re.compile(r"\s*[,/|:;–—]|\s+[-]\s+|\s+·\s+")


def normalize(title: str) -> str | None:
    """title_display -> canonical bare role phrase, or None if it isn't a clean role."""
    if not title:
        return None
    t = LEADING_CODE.sub("", title)
    t = SEP.split(t, 1)[0]  # head segment before first comma/dash/pipe/slash
    t = PAREN.sub(" ", t)
    t = t.replace("&amp;", "&")
    t = WS.sub(" ", t).strip()
    # strip stacked seniority prefixes ("senior staff engineer" -> "engineer"? no --
    # only peel ONE so "senior software engineer" -> "software engineer", not "engineer")
    t = SENIORITY.sub("", t, count=1).strip()
    t = LEVEL_SUFFIX.sub("", t).strip()
    t = NONWORD_EDGE.sub("", t)
    t = WS.sub(" ", t).strip().lower()
    if not t or len(t) < 4:
        return None
    toks = t.split()
    if not (2 <= len(toks) <= 5):  # bare "engineer" is too generic; >5 tokens = noise
        return None
    if toks[-1] not in ROLE_HEADS:
        return None
    if any(ch.isdigit() for ch in t):
        return None
    return t


_ACRONYM = {
    "it": "IT",
    "ai": "AI",
    "ml": "ML",
    "ux": "UX",
    "ui": "UI",
    "qa": "QA",
    "devops": "DevOps",
    "sre": "SRE",
    "ios": "iOS",
    "seo": "SEO",
    "sem": "SEM",
    "hr": "HR",
    "pr": "PR",
    "bi": "BI",
    "gtm": "GTM",
    "crm": "CRM",
    "erp": "ERP",
    "saas": "SaaS",
    "api": "API",
    "sql": "SQL",
    "gnc": "GNC",
    "rn": "RN",
    "cnc": "CNC",
    "hvac": "HVAC",
    "cdl": "CDL",
    "ceo": "CEO",
    "cfo": "CFO",
    "cto": "CTO",
    "coo": "COO",
    "vp": "VP",
    "svp": "SVP",
    "evp": "EVP",
    "ndt": "NDT",
    "scada": "SCADA",
    "kyc": "KYC",
    "aml": "AML",
    "cpa": "CPA",
}
_SMALL = {"of", "and", "or", "the", "for", "in", "to", "a", "an", "&"}


def display_form(phrase: str) -> str:
    out = []
    for w in phrase.split():
        if w in _ACRONYM:
            out.append(_ACRONYM[w])
        elif w == "net" or w == ".net":
            out.append(".NET")
        elif w in _SMALL:
            out.append(w)
        else:
            out.append(w.capitalize())
    return " ".join(out)


def fetch_titles() -> Counter:
    counts: Counter = Counter()
    start = 0
    total = None
    while True:
        r = requests.get(
            f"{SOLR}/solr/{CORE}/select",
            params={
                "q": "*:*",
                "rows": str(PAGE),
                "start": str(start),
                "fl": "title_display",
                "wt": "json",
            },
            timeout=60,
        )
        r.raise_for_status()
        resp = r.json()["response"]
        total = resp["numFound"] if total is None else total
        docs = resp["docs"]
        if not docs:
            break
        for d in docs:
            norm = normalize(d.get("title_display", ""))
            if norm:
                counts[norm] += 1
        start += PAGE
        print(f"  scanned {min(start, total)}/{total}", flush=True)
        if start >= total:
            break
    return counts


def main() -> None:
    print("fetching titles from Solr...", flush=True)
    counts = fetch_titles()
    vocab = [(p, c) for p, c in counts.items() if c >= MIN_COUNT]
    vocab.sort(key=lambda x: -x[1])
    print(
        f"{len(counts)} distinct role phrases -> {len(vocab)} with count >= {MIN_COUNT}", flush=True
    )

    phrases = [p for p, _ in vocab]
    print(f"loading {DENSE_MODEL}...", flush=True)
    from sentence_transformers import SentenceTransformer

    try:
        import torch

        device = "mps" if torch.backends.mps.is_available() else "cpu"
    except Exception:
        device = "cpu"
    model = SentenceTransformer(DENSE_MODEL, device=device)
    emb = model.encode(
        [f"query: {p}" for p in phrases],
        normalize_embeddings=True,
        batch_size=256,
        show_progress_bar=True,
    ).astype(np.float32)

    out_json = os.path.join(HERE, "role_vocab.json")
    out_npy = os.path.join(HERE, "role_vocab_emb.npy")
    with open(out_json, "w") as f:
        json.dump(
            [{"phrase": p, "display": display_form(p), "count": c} for p, c in vocab],
            f,
        )
    np.save(out_npy, emb)
    print(f"wrote {out_json} ({len(vocab)} roles) and {out_npy} {emb.shape}", flush=True)
    print("\nsample (top 25 by frequency):", flush=True)
    for p, c in vocab[:25]:
        print(f"  {c:6d}  {display_form(p)}")


if __name__ == "__main__":
    main()
