#!/usr/bin/env python3
"""Build the grounded Swedish *related-searches* bundle (space/sv_related.json).

Swedish autocomplete shipped earlier (mine_sv_roles.py), but related searches did NOT:
SSYK (the Swedish occupation taxonomy) has no occupation->occupation mobility graph like
France's ROME, so the Swedish lane was deferred. ESCO closes that gap — it carries 100%
Swedish preferred-label coverage — so Swedish now rides the SAME ESCO skill-overlap backbone
as the German/Dutch/Spanish lanes:

  query --(ESCO Swedish label)--> occupation --(shared-skill relatedness)--> related
        occupations --(corpus-mined Swedish role)--> a Swedish role that EXISTS in our
        inventory

Unlike the Romance lanes (es/it) there is NO degender step: Swedish occupational nouns are
gender-neutral (the historical '-ska'/'-inna' feminine, e.g. 'sjuksköterska', is the standard
term, not a strippable variant). Relatedness is the weighted skill-overlap coefficient
(esco_backbone.related); every displayed suggestion is a real ESCO neighbour AND a role we
have postings for, so it always returns results.

The emitted bundle is self-contained (no .esco_records.jsonl at serve time). Consumed by
suggest_lib.SvRelatedSuggester. Re-run after a corpus refresh or an ESCO version bump.

Usage:
  ./run.sh start                 # local Solr must be up on :8983
  .venv/bin/python build_sv_related.py
"""

import json
import os
import re
import sys
from collections import Counter, defaultdict

SOLR = "http://localhost:8983/solr/jobs"
MIN_COUNT = 5  # a cleaned title must recur this often to be a "real" corpus role
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "space", "sv_related.json")

sys.path.insert(0, HERE)
from esco_backbone import backbone, fold  # noqa: E402
from mine_sv_roles import _STOP_FOLDED, clean_title, fetch_titles  # noqa: E402


def main() -> None:
    bb = backbone()
    print(f"ESCO backbone: {len(bb.by_uri):,} occupations", file=sys.stderr)

    # ---- 1. corpus-mined Swedish role vocab (the grounded display labels) ----
    titles = fetch_titles()
    print(f"fetched {len(titles):,} Swedish titles", file=sys.stderr)
    counts: Counter[str] = Counter()
    surface: dict[str, Counter[str]] = {}
    for t in titles:
        c = clean_title(t)
        if not (2 <= len(c) <= 48) or len(c.split()) > 6:
            continue
        if not re.search(r"[a-zà-ÿåäö]", c) or re.search(r"\d", c):
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
        uris = bb.resolve(text, "sv")
        if not uris:
            unmapped += 1
            continue
        uri = min(uris)
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
    # min_overlap 0.20, the thin-corpus-lane floor (same as de/nl/es): floors out incoherent
    # cross-domain links so an ungrounded role returns NOTHING rather than a wrong neighbour.
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

    # ---- 3. label resolver: ESCO Swedish labels (preferred + alt) for every occupation a
    # query can land on (grounded or reachable), so a typed Swedish role resolves even when
    # its exact surface isn't a corpus label. Corpus surfaces (step 1) take priority.
    resolvable = set(uri_related) | grounded
    cand_labels: dict[str, set[str]] = defaultdict(set)
    for uri in resolvable:
        r = bb.by_uri.get(uri, {})
        lab = (r.get("preferredLabel") or {}).get("sv", "")
        if fold(lab):
            cand_labels[fold(lab)].add(uri)
        for a in (r.get("altLabels") or {}).get("sv", []) or []:
            if fold(a):
                cand_labels[fold(a)].add(uri)

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
