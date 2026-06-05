#!/usr/bin/env python3
"""Multilingual-embedding role_family rescue for the non-English 'other' residual.

The lexical ESCO rescue (classify_other_esco) resolves a non-English title to an
ESCO occupation by label-containment / head-noun match, then maps ISCO-08 ->
role_family. It misses titles that are *semantically* an ESCO occupation but share
no surface tokens with any label ('Ejecutivo de atencion al cliente', 'Analista de
Soporte Tecnico', 'Gerente Comercial') -- exactly the es/it residual, which is
dominated by Adzuna postings whose own category is 'Unknown' (no signal) so the
title is the only lever.

This classifier upgrades the match step to embeddings:
  * encode every ESCO label (preferred + alt) in the doc's language with
    multilingual-e5 and tag it with its occupation's ISCO->role_family;
  * encode the job title, take its k nearest ESCO labels, similarity-weighted-vote
    a family, and gate on (top-1 cosine >= SIMFLOOR) AND (vote-share >= CONF).
Coherence is a GATE, not a target: below the floor the doc stays 'other' (a wrong
family is worse than none). The encode is OFFLINE (match step only) -- not a
serving change; the served retrieval model is unchanged (e5-small-v2, English).

Mirrors classify_other_esco.rescue's signature so refresh.py can call it the same
way; writes role_family_esco_emb_overrides.json, applied by push_docs with the
LOWEST precedence (after authoritative ROME/JobTech/Adzuna and lexical ESCO) so it
only fills docs no stronger signal reached.
"""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path

import numpy as np
from isco_role_family import role_family_for_isco

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
ESCO = HERE / ".esco_records.jsonl"
META = ROOT / "unified_jobs_daily" / "metadata.jsonl"
FACETS = HERE / "facets" / "facets.jsonl"
DEST = HERE / "role_family_esco_emb_overrides.json"
CACHE = HERE / ".esco_emb_label_cache"  # {lang}.npz: vecs + parallel families/labels

MODEL = "intfloat/multilingual-e5-small"  # cached locally; multilingual match only
LANGS = ("fr", "sv", "de", "nl", "es", "it")

# Production operating point (precision-first; audited to the ~85% ROME bar).
# Both --apply and the refresh stage call rescue() with these defaults; changing
# them changes the live label set, so keep in sync with the committed overrides.
K = 10  # nearest ESCO labels voted per title
SIMFLOOR = 0.89  # min top-1 title<->label cosine (e5 cosines run high)
CONF = 0.65  # min similarity-weighted vote-share for the winning family

# evergreen / talent-pool wrappers carry no real role -> never classify. Multilingual.
JUNK = re.compile(
    r"\b(banco de talentos|bolsa de talento|candidatura spontanea|candidatura espontanea|"
    r"candidature spontanee|iniziativa spontanea|talent pool|talent community|"
    r"talent network|autocandidatura|talangpool|spontaneinitiativ|"
    r"open application|general application|future opportunit)\b",
    re.I,
)


def norm(s: str) -> str:
    """lowercase, strip accents, collapse non-alnum to single spaces."""
    s = unicodedata.normalize("NFKD", s or "")
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = re.sub(r"[^a-z0-9]+", " ", s.lower())
    return re.sub(r"\s+", " ", s).strip()


def _label_strings(recs, lang):
    """[(label_text, role_family)] for non-'other' ESCO occupations in `lang`.

    De-duplicated on (normalized label, family) so a label shared by several
    occupations of the same family is encoded once. Gendered/slash variants are
    split so each surface form is its own searchable label."""
    seen: set[tuple[str, str]] = set()
    out: list[tuple[str, str]] = []
    for r in recs:
        fam = role_family_for_isco(r.get("isco"))
        if fam == "other":
            continue
        pl = (r.get("preferredLabel") or {}).get(lang)
        alts = (r.get("altLabels") or {}).get(lang, []) or []
        for lab in [pl, *alts]:
            if not lab:
                continue
            for v in re.split(r"\s*[/|]\s*", lab):
                v = v.strip()
                key = (norm(v), fam)
                if v and len(norm(v)) >= 3 and key not in seen:
                    seen.add(key)
                    out.append((v, fam))
    return out


def _encoder():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(MODEL)


def _build_label_vecs(recs, lang, model, use_cache=True):
    """(L[n,d] normalized, families[n]) for `lang`; cached to disk per language."""
    cache_path = CACHE / f"{lang}.npz"
    pairs = _label_strings(recs, lang)
    labels = [p[0] for p in pairs]
    fams = np.array([p[1] for p in pairs])
    if use_cache and cache_path.exists():
        z = np.load(cache_path, allow_pickle=True)
        if list(z["labels"]) == labels:  # corpus of labels unchanged -> reuse vecs
            return z["vecs"].astype(np.float32), z["families"]
    texts = [f"passage: {lab}" for lab in labels]
    V = model.encode(texts, normalize_embeddings=True, batch_size=256, show_progress_bar=False)
    V = np.asarray(V, dtype=np.float32)
    if use_cache:
        CACHE.mkdir(exist_ok=True)
        np.savez(cache_path, vecs=V, families=fams, labels=np.array(labels, dtype=object))
    return V, fams


def _vote(sims_row, fam_row):
    """similarity-weighted family vote over a title's k neighbors -> (family, share)."""
    w: dict[str, float] = {}
    for f, s in zip(fam_row, sims_row):
        w[f] = w.get(f, 0.0) + float(s)
    tot = sum(w.values()) or 1.0
    best = max(w, key=w.get)
    return best, w[best] / tot


def rescue(metadata_path, heur_labels, recs=None, langs=LANGS, model=None, use_cache=True, audit=0):
    """{doc_id: role_family} for non-English docs whose heuristic label is 'other'.

    `heur_labels` is the heuristic role_family per line, positionally aligned to
    metadata_path (refresh passes its in-memory role_fams; the CLI reads
    facets.jsonl). Deterministic on a fixed corpus + model -> reproducible across
    refreshes. `audit` prints N sample assignments per language."""
    if recs is None:
        with open(ESCO) as f:
            recs = [json.loads(l) for l in f]
    if model is None:
        model = _encoder()

    # collect the residual titles per language
    todo: dict[str, list[tuple[str, str]]] = {lg: [] for lg in langs}  # lg -> [(id,title)]
    with open(metadata_path) as f:
        for i, line in enumerate(f):
            if i >= len(heur_labels) or heur_labels[i] != "other":
                continue
            r = json.loads(line)
            lg = (r.get("lang") or "").strip()
            if lg not in todo:
                continue
            title = (r.get("title") or "").strip()
            if title and not JUNK.search(title):
                todo[lg].append((str(r["id"]), title))

    out: dict[str, str] = {}
    samples: dict[str, list] = {}
    for lg in langs:
        items = todo[lg]
        if not items:
            continue
        L, fams = _build_label_vecs(recs, lg, model, use_cache=use_cache)
        titles = [f"query: {t}" for _, t in items]
        Q = np.asarray(
            model.encode(
                titles, normalize_embeddings=True, batch_size=256, show_progress_bar=False
            ),
            dtype=np.float32,
        )
        sims = Q @ L.T  # (n_titles, n_labels)
        k = min(K, L.shape[0])
        topk = np.argpartition(-sims, k - 1, axis=1)[:, :k]
        samp: list = []
        for row, (did, title) in enumerate(items):
            cols = topk[row]
            row_sims = sims[row, cols]
            order = np.argsort(-row_sims)
            cols, row_sims = cols[order], row_sims[order]
            top1 = float(row_sims[0])
            fam, share = _vote(row_sims, fams[cols])
            if fam != "other" and top1 >= SIMFLOOR and share >= CONF:
                out[did] = fam
                if len(samp) < audit:
                    samp.append((title[:50], fam, round(top1, 3), round(share, 2)))
        samples[lg] = samp
    if audit:
        for lg in langs:
            if samples.get(lg):
                print(f"\n--- {lg} samples (title -> family  cos  share) ---")
                for t, f, c, s in samples[lg]:
                    print(f"  [{f:28s} {c} {s}] {t}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="write the overrides file")
    ap.add_argument("--langs", default="es,it", help="comma list (default es,it)")
    ap.add_argument("--audit", type=int, default=0, help="print N samples per lang")
    ap.add_argument("--no-cache", action="store_true")
    args = ap.parse_args()
    langs = tuple(x.strip() for x in args.langs.split(",") if x.strip())

    with open(ESCO) as f:
        recs = [json.loads(l) for l in f]
    with open(FACETS) as f:
        heur = [json.loads(l)["role_family"] for l in f]
    out = rescue(META, heur, recs, langs=langs, use_cache=not args.no_cache, audit=args.audit)

    # per-language denominator (residual after the heuristic, before this rescue)
    lang_of: dict[str, str] = {}
    denom = Counter()
    with open(META) as f:
        for i, line in enumerate(f):
            if heur[i] != "other":
                continue
            r = json.loads(line)
            lg = (r.get("lang") or "").strip()
            if lg in langs:
                lang_of[str(r["id"])] = lg
                denom[lg] += 1
    rb = Counter(lang_of.get(d, "?") for d in out)
    print(f"\nresidual 'other' (langs {','.join(langs)}): {sum(denom.values())}")
    print(f"rescued by emb-ESCO: {len(out)}\n")
    for lg in langs:
        print(f"  {lg}: {rb[lg]:6d} / {denom[lg]:6d}  {100 * rb[lg] / max(1, denom[lg]):4.1f}%")
    print("\nby role_family:")
    for f, c in Counter(out.values()).most_common():
        print(f"  {c:6d}  {f}")

    if args.apply:
        with open(DEST, "w") as f:
            json.dump(out, f, indent=0, sort_keys=True, ensure_ascii=False)
        print(f"\nwrote {len(out)} overrides -> {DEST}")


if __name__ == "__main__":
    main()
