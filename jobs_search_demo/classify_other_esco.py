#!/usr/bin/env python3
"""Open-weight multilingual role_family rescue for the non-English 'other' residual.

The role_family heuristics are English-keyword based and never fire on French /
Swedish / German / Dutch / Spanish / Italian titles, so ~112k non-English docs
sit in 'other'. This classifier resolves each such title to an ESCO occupation
in its own language (lexical match over ESCO preferred + alt labels), then maps
the occupation's ISCO-08 code to a role_family via isco_role_family. No LLM, no
embeddings -- ESCO ships native labels in every EU language.

Match gate (precision-first):
  1. exact: normalized title == a normalized ESCO label
  2. contained: the longest ESCO label (>=MIN_LABEL_CHARS, whole-token) that is a
     token-substring of the normalized title
A label is only usable if every ISCO code it maps to agrees on the same
role_family (ambiguous labels are dropped). Assign only when the family != other.

Writes role_family_esco_overrides.json {doc_id: role_family}. Designed to be
called both as a CLI (--apply) and from the refresh stage.
"""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

from isco_role_family import role_family_for_isco

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
ESCO = HERE / ".esco_records.jsonl"
META = ROOT / "unified_jobs_daily" / "metadata.jsonl"
FACETS = HERE / "facets" / "facets.jsonl"
DEST = HERE / "role_family_esco_overrides.json"

LANGS = ("fr", "sv", "de", "nl", "es", "it")
MIN_LABEL_CHARS = 5  # a 1-token label must be at least this long to substring-match
COV_MIN = 0.5  # a contained label must cover >= this share of the title's content tokens
DOM_MIN = 0.60  # a label mapping to >1 ISCO is kept only if one family wins this share

# job-posting noise stripped before matching (gender tags, contract types, etc.)
NOISE = re.compile(
    r"\b(h/f|f/h|m/w/d|m/f/d|w/m/d|m/f|f/m|h/m|cdi|cdd|"
    r"temps plein|temps partiel|full[- ]?time|part[- ]?time|"
    r"heltid|deltid|vollzeit|teilzeit|voltijd|deeltijd|"
    r"jornada completa|tiempo completo|tempo pieno|tempo determinato)\b",
    re.I,
)

# spontaneous-application / evergreen wrappers carry a real role but add noise;
# their tokens are dropped before coverage scoring so the role still wins.
WRAPPER_TOKENS = frozenset(
    [
        "candidatura",
        "spontanea",
        "candidature",
        "spontanee",
        "spontane",
        "profil",
        "profile",
        "sokes",
        "soker",
        "recherche",
        "cerca",
        "busca",
        "gesucht",
        "gezocht",
        "stelle",
        "stellenangebot",
    ]
)

# generic occupation words that match many ESCO labels across families -> never a
# standalone match (normalized, accent-stripped). Kept multilingual + small.
STOPLABEL = frozenset(
    [
        "manager",
        "general",
        "generale",
        "general",
        "général",
        "assistant",
        "assistente",
        "asistente",
        "medewerker",
        "mitarbeiter",
        "employe",
        "empleado",
        "dipendente",
        "service",
        "services",
        "agent",
        "agente",
        "specialist",
        "specialiste",
        "especialista",
        "spezialist",
        "responsable",
        "responsabile",
        "coordinator",
        "coordinador",
        "coordinatore",
        "koordinator",
        "supervisor",
        "supervisore",
        "consultant",
        "consultante",
        "consulente",
        "operatore",
        "operator",
        "addetto",
        "operador",
        "tecnico",
        "technicien",
        "techniker",
        "tecnico",
        "worker",
        "arbetare",
        "profil",
        "profile",
    ]
)

# tokens that carry no occupational signal -> dropped before coverage scoring.
NOISE_TOKENS = frozenset(
    [
        "de",
        "la",
        "le",
        "les",
        "des",
        "du",
        "el",
        "en",
        "het",
        "the",
        "and",
        "und",
        "et",
        "e",
        "y",
        "i",
        "to",
        "till",
        "for",
    ]
)


def _load_jsonl(path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f]


def norm(s: str) -> str:
    """lowercase, strip accents, collapse non-alnum to single spaces."""
    s = NOISE.sub(" ", s or "")
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = re.sub(r"[^a-z0-9]+", " ", s.lower())
    return re.sub(r"\s+", " ", s).strip()


def label_variants(label: str) -> list[str]:
    """Split gendered/slash variants ('infirmier/infirmière') into separate labels."""
    parts = re.split(r"\s*[/|]\s*", label)
    return [p for p in parts if p.strip()]


def build_index(recs: list[dict]) -> dict[str, dict[str, str]]:
    """{lang: {normalized_label: role_family}}.

    A label that resolves to several ISCO codes is kept only if one family wins
    >= DOM_MIN of the votes (preferred labels weigh double); otherwise dropped as
    too ambiguous. STOPLABEL generic words are excluded entirely.
    """
    # lang -> norm_label -> Counter(role_family)
    raw: dict[str, dict[str, Counter]] = {lg: defaultdict(Counter) for lg in LANGS}
    for r in recs:
        fam = role_family_for_isco(r.get("isco"))
        if fam == "other":
            continue
        for lg in LANGS:
            pl = (r.get("preferredLabel") or {}).get(lg)
            alts = (r.get("altLabels") or {}).get(lg, []) or []
            for lab, weight in [(pl, 2)] + [(a, 1) for a in alts]:
                if not lab:
                    continue
                for v in label_variants(lab):
                    nv = norm(v)
                    if len(nv) >= 3 and nv not in STOPLABEL:
                        raw[lg][nv][fam] += weight
    index: dict[str, dict[str, str]] = {}
    for lg in LANGS:
        m: dict[str, str] = {}
        for lab, c in raw[lg].items():
            tot = sum(c.values())
            fam, n = c.most_common(1)[0]
            if n / tot >= DOM_MIN:
                m[lab] = fam
        index[lg] = m
    return index


def build_token_index(index: dict[str, dict[str, str]]):
    """Per lang: list of (label, ntokens, family) sorted longest-first for containment."""
    out = {}
    for lg, m in index.items():
        items = [(lab, lab.count(" ") + 1, fam) for lab, fam in m.items()]
        items.sort(key=lambda t: -len(t[0]))  # longest string first = most specific
        out[lg] = items
    return out


def match(title_norm: str, lg: str, index, tok_index) -> str | None:
    if not title_norm:
        return None
    # 1. exact normalized match -> highest precision
    fam = index[lg].get(title_norm)
    if fam:
        return fam
    # 2. contained label that explains >= COV_MIN of the title's content tokens.
    # Content tokens exclude noise/wrapper words and pure numbers; coverage guards
    # against generic words ("manager") matching inside an unrelated long title.
    content = [
        t
        for t in title_norm.split()
        if t not in NOISE_TOKENS and t not in WRAPPER_TOKENS and not t.isdigit()
    ]
    n_content = len(content) or 1
    padded = f" {title_norm} "
    best_fam, best_cov = None, 0.0
    for lab, ntok, fam in tok_index[lg]:
        if ntok == 1 and len(lab) < MIN_LABEL_CHARS:
            continue
        if f" {lab} " not in padded:
            continue
        cov = sum(1 for t in lab.split() if t in content) / n_content
        if cov > best_cov:
            best_fam, best_cov = fam, cov
            if best_cov >= 0.999:
                break  # full coverage, can't beat it
    if best_fam and best_cov >= COV_MIN:
        return best_fam
    return None


def rescue(metadata_path, heur_labels: list[str], recs: list[dict] | None = None) -> dict[str, str]:
    """Core: {doc_id: role_family} for non-English docs whose heuristic label is
    'other'. `heur_labels` is the heuristic role_family per line, positionally
    aligned to metadata_path (refresh passes its in-memory role_fams; the CLI reads
    facets.jsonl). Deterministic -> reproducible across refreshes on a fixed corpus."""
    if recs is None:
        recs = _load_jsonl(ESCO)
    index = build_index(recs)
    tok_index = build_token_index(index)
    out: dict[str, str] = {}
    with open(metadata_path) as f:
        for i, line in enumerate(f):
            if i >= len(heur_labels) or heur_labels[i] != "other":
                continue
            r = json.loads(line)
            lg = (r.get("lang") or "").strip()
            if lg not in LANGS:
                continue
            fam = match(norm(r.get("title") or ""), lg, index, tok_index)
            if fam and fam != "other":
                out[r["id"]] = fam
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="write the overrides file")
    ap.add_argument("--sample", type=int, default=0, help="print N sample assignments per lang")
    args = ap.parse_args()

    recs = _load_jsonl(ESCO)
    index = build_index(recs)
    print("ESCO label index: " + ", ".join(f"{lg}={len(index[lg])}" for lg in LANGS))

    # heuristic labels (facets.jsonl) aligned to metadata by line order -> same input
    # the refresh stage passes in-memory, so the committed file == refresh output.
    heur = [r["role_family"] for r in _load_jsonl(FACETS)]
    out = rescue(META, heur, recs)

    # reporting: per-language denominators + samples
    by_lang_total = Counter()
    lang_of: dict[str, str] = {}
    samples: dict[str, list] = defaultdict(list)
    with open(META) as f:
        for i, line in enumerate(f):
            if heur[i] != "other":
                continue
            r = json.loads(line)
            lg = (r.get("lang") or "").strip()
            if lg not in LANGS:
                continue
            by_lang_total[lg] += 1
            lang_of[r["id"]] = lg
            if r["id"] in out and len(samples[lg]) < args.sample:
                samples[lg].append(((r.get("title") or "")[:55], out[r["id"]]))

    rb = Counter(lang_of.get(did, "?") for did in out)
    fam_counts = Counter(out.values())
    total = sum(by_lang_total.values())
    print(f"\nnon-English 'other' docs: {total}")
    print(f"rescued: {len(out)} ({100 * len(out) / max(1, total):.1f}%)\n")
    print("by language (rescued / total):")
    for lg in LANGS:
        t = by_lang_total[lg]
        print(f"  {lg}: {rb[lg]:6d} / {t:6d}  {100 * rb[lg] / max(1, t):4.1f}%")
    print("\nby role_family:")
    for f, c in fam_counts.most_common():
        print(f"  {c:6d}  {f}")
    if args.sample:
        for lg in LANGS:
            print(f"\n--- {lg} samples ---")
            for title, fam in samples[lg]:
                print(f"  [{fam:28s}] {title}")

    if args.apply:
        # indent=0 + sort_keys: one key per line, stable order -> reviewable diffs
        # as the override set grows refresh to refresh (matches the emb-rescue file).
        with open(DEST, "w") as f:
            json.dump(out, f, indent=0, sort_keys=True, ensure_ascii=False)
        print(f"\nwrote {len(out)} overrides -> {DEST}")


if __name__ == "__main__":
    main()
