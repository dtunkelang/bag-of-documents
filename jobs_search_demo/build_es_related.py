#!/usr/bin/env python3
"""Build the grounded Spanish *related-searches* bundle (space/es_related.json).

Spanish related searches can't ride the e5-small-v2 role suggester (it ranks Spanish by
morphology, not meaning), and Spanish has no national mobilite graph the way the French
ROME lane uses. The grounded substitute is the ESCO occupation backbone (esco_backbone.py),
identical in mechanism to the German/Dutch lanes:

  query --(ESCO Spanish label)--> occupation --(shared-skill relatedness)--> related
        occupations --(corpus-mined Spanish role)--> a Spanish role that EXISTS in our
        inventory

ESCO has NO occupation->occupation mobility graph, so relatedness is derived from shared
essential/optional skills (weighted overlap coefficient — see esco_backbone.related). The
overlap coefficient is symmetric, so we only have to compute related() for the GROUNDED
occupations (the few hundred ESCO occupations we actually have Spanish postings for) and
invert it: a query occupation q gets the grounded occupations whose skill sets overlap it.
Every displayed suggestion is therefore a real ESCO-validated occupational neighbour AND a
role we have postings for, so it always returns results.

The emitted bundle is fully self-contained (it does NOT need .esco_records.jsonl at serve
time — only build time). Consumed by suggest_lib.EsRelatedSuggester. Re-run after a corpus
refresh (so display labels track live inventory) or an ESCO version bump.

Usage:
  ./run.sh start                 # local Solr must be up on :8983
  .venv/bin/python build_es_related.py
"""

import json
import os
import re
import sys
from collections import Counter, defaultdict

SOLR = "http://localhost:8983/solr/jobs"
MIN_COUNT = 5  # a cleaned title must recur this often to be a "real" corpus role
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "space", "es_related.json")

sys.path.insert(0, HERE)
from esco_backbone import backbone, fold  # noqa: E402
from mine_es_roles import _STOP_FOLDED, clean_title, fetch_titles  # noqa: E402

# Feminine Spanish occupational suffix -> masculine, applied to a FOLDED token, used only
# as a resolution fallback (ESCO carries most feminine alt labels directly, but corpus
# surfaces drift). Two productive patterns: '-ora' -> '-or' (programadora->programador,
# directora->director) BEFORE the general '-a' -> '-o' (cocinera->cocinero, enfermera->
# enfermero, médica->médico). '-ora' must precede '-a' so 'programadora' maps to
# 'programador', not 'programadoro'. Each only fires when the stem stays >=4 chars so short
# function words ("una", "para") aren't mangled.
_ES_FEM = [("ora", "or"), ("a", "o")]


def degender_es(folded: str) -> str:
    out = []
    for tok in folded.split():
        for suf, rep in _ES_FEM:
            if len(tok) >= len(suf) + 4 and tok.endswith(suf):
                tok = tok[: -len(suf)] + rep
                break
        out.append(tok)
    return " ".join(out)


def _resolve_uri(text: str, bb) -> str | None:
    """Folded-exact ESCO Spanish occupation for a corpus role label, with a degendered
    retry. Returns one URI (lexicographic min on collision) or None."""
    uris = bb.resolve(text, "es")
    if not uris:
        dg = degender_es(fold(text))
        if dg != fold(text):
            uris = bb.resolve(dg, "es")
    return min(uris) if uris else None


def main() -> None:
    bb = backbone()
    print(f"ESCO backbone: {len(bb.by_uri):,} occupations", file=sys.stderr)

    # ---- 1. corpus-mined Spanish role vocab (the grounded display labels) ----
    titles = fetch_titles()
    print(f"fetched {len(titles):,} Spanish titles", file=sys.stderr)
    counts: Counter[str] = Counter()
    surface: dict[str, Counter[str]] = {}
    for t in titles:
        c = clean_title(t)
        if not (2 <= len(c) <= 48) or len(c.split()) > 6:
            continue
        if not re.search(r"[a-zà-ÿ]", c) or re.search(r"\d", c):
            continue
        k = fold(c)
        if k in _STOP_FOLDED:
            continue
        counts[k] += 1
        surface.setdefault(k, Counter())[c] += 1

    # map each corpus role -> ESCO occupation, keep best (cased) surface + count
    uri_roles: dict[str, list[dict]] = defaultdict(list)
    uri_weight: Counter[str] = Counter()
    label2uri: dict[str, str] = {}
    mapped = unmapped = 0
    for k, n in counts.most_common():
        if n < MIN_COUNT:
            continue
        text = surface[k].most_common(1)[0][0]
        uri = _resolve_uri(text, bb)
        if not uri:
            unmapped += 1
            continue
        mapped += 1
        uri_roles[uri].append({"text": text, "n": n})
        uri_weight[uri] += n
        label2uri.setdefault(fold(text), uri)  # corpus surface -> its occupation
    for uri in uri_roles:
        uri_roles[uri].sort(key=lambda d: -d["n"])
    grounded = set(uri_roles)
    print(
        f"corpus roles: {mapped:,} mapped to {len(grounded):,} occupations, {unmapped:,} unmapped",
        file=sys.stderr,
    )

    # ---- 2. relatedness, computed only over grounded occupations and inverted ----
    # min_overlap 0.20, the thin-corpus-lane floor (same as the Dutch lane): Adzuna is the
    # SOLE Spanish source and heavily dedups, so the grounded set is small and the weakest
    # inverted links are incoherent cross-domain matches. 0.20 floors them out (empty >
    # wrong, coherence as a gate) while keeping coherent same-cluster links.
    reverse: dict[str, dict[str, float]] = defaultdict(dict)
    for g in grounded:
        for n, s in bb.related(g, top=24, min_overlap=0.20):
            if n == g:
                continue
            if s > reverse[n].get(g, 0.0):
                reverse[n][g] = s
    uri_related = {
        q: [g for g, _ in sorted(neigh.items(), key=lambda kv: -kv[1])]
        for q, neigh in reverse.items()
    }

    # ---- 3. label resolver: ESCO Spanish labels (preferred + alt) for every occupation a
    # query can land on (grounded or reachable). The Spanish preferred label is the dual-
    # gender "fontanero/fontanera" form, so we split it on '/' to index both surfaces.
    resolvable = set(uri_related) | grounded
    cand_labels: dict[str, set[str]] = defaultdict(set)
    for uri in resolvable:
        r = bb.by_uri.get(uri, {})
        for lab in (r.get("preferredLabel") or {}).get("es", "").split("/"):
            if fold(lab):
                cand_labels[fold(lab)].add(uri)
        for a in (r.get("altLabels") or {}).get("es", []) or []:
            for lab in a.split("/"):
                if fold(lab):
                    cand_labels[fold(lab)].add(uri)

    def _rank(u: str) -> tuple:
        return (u in uri_related, u in grounded, uri_weight.get(u, 0), u)

    for k, uris in cand_labels.items():
        if k in label2uri:  # a corpus surface already claimed this key
            continue
        label2uri[k] = max(uris, key=_rank)

    bundle = {
        "label2uri": label2uri,
        "uri_related": uri_related,
        "uri_roles": dict(uri_roles),
    }
    with open(OUT, "w") as f:
        json.dump(bundle, f, ensure_ascii=False)
    sz = os.path.getsize(OUT) / 1024
    print(
        f"wrote {OUT} ({sz:.0f} KB): {len(label2uri):,} query keys, "
        f"{len(uri_related):,} occupations with grounded neighbours",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
