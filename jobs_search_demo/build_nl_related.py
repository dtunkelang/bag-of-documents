#!/usr/bin/env python3
"""Build the grounded Dutch *related-searches* bundle (space/nl_related.json).

Dutch related searches can't ride the e5-small-v2 role suggester (it ranks Dutch by
morphology, not meaning), and Dutch has no national mobilite graph the way the French
ROME lane uses. The grounded substitute is the ESCO occupation backbone (esco_backbone.py):

  query --(ESCO Dutch label)--> occupation --(shared-skill relatedness)--> related
        occupations --(corpus-mined Dutch role)--> a Dutch role that EXISTS in our
        inventory

ESCO has NO occupation->occupation mobility graph, so relatedness is derived from shared
essential/optional skills (weighted overlap coefficient — see esco_backbone.related). The
overlap coefficient is symmetric, so we only have to compute related() for the GROUNDED
occupations (the few hundred ESCO occupations we actually have Dutch postings for) and
invert it: a query occupation q gets the grounded occupations whose skill sets overlap it.
Every displayed suggestion is therefore a real ESCO-validated occupational neighbour AND a
role we have postings for, so it always returns results.

The emitted bundle is fully self-contained (it does NOT need .esco_records.jsonl at serve
time — only build time): it carries the folded-label->occupation resolver, the
occupation->grounded-neighbours map, and the occupation->corpus-role display vocab.
Consumed by suggest_lib.NlRelatedSuggester. Re-run after a corpus refresh (so display
labels track live inventory) or an ESCO version bump.

Usage:
  ./run.sh start                 # local Solr must be up on :8983
  .venv/bin/python build_nl_related.py
"""

import json
import os
import re
import sys
from collections import Counter, defaultdict

SOLR = "http://localhost:8983/solr/jobs"
MIN_COUNT = 5  # a cleaned title must recur this often to be a "real" corpus role
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "space", "nl_related.json")

sys.path.insert(0, HERE)
from esco_backbone import backbone, fold  # noqa: E402
from mine_nl_roles import _STOP_FOLDED, clean_title, fetch_titles  # noqa: E402

# Feminine Dutch occupational suffix -> common form, applied to a FOLDED token, used only
# as a resolution fallback (ESCO carries most feminine alt labels directly, but corpus
# surfaces drift). The productive pattern is '-ster' -> '-er' (verkoopster->verkoper,
# schoonmaakster->schoonmaker); only fires when the stem stays >=4 chars so short words
# aren't mangled.
_NL_FEM = [("ster", "er")]


def degender_nl(folded: str) -> str:
    out = []
    for tok in folded.split():
        for suf, rep in _NL_FEM:
            if len(tok) >= len(suf) + 4 and tok.endswith(suf):
                tok = tok[: -len(suf)] + rep
                break
        out.append(tok)
    return " ".join(out)


def _resolve_uri(text: str, bb) -> str | None:
    """Folded-exact ESCO Dutch occupation for a corpus role label, with a degendered
    retry. Returns one URI (lexicographic min on collision) or None."""
    uris = bb.resolve(text, "nl")
    if not uris:
        dg = degender_nl(fold(text))
        if dg != fold(text):
            uris = bb.resolve(dg, "nl")
    return min(uris) if uris else None


def main() -> None:
    bb = backbone()
    print(f"ESCO backbone: {len(bb.by_uri):,} occupations", file=sys.stderr)

    # ---- 1. corpus-mined Dutch role vocab (the grounded display labels) ----
    titles = fetch_titles()
    print(f"fetched {len(titles):,} Dutch titles", file=sys.stderr)
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
    # overlap coefficient is symmetric -> related(g) gives, for each neighbour n, the
    # grounded occupation g that n is related to. Invert so a query occupation q (grounded
    # OR not) maps to the grounded occupations whose skills overlap it, best first.
    # min_overlap 0.20 (vs the German lane's 0.15): the Dutch grounded set is small and
    # skewed toward a logistics/warehouse cluster, so the weakest inverted links are
    # incoherent cross-domain matches (e.g. timmerman<->pakhuismedewerker share only 0.16
    # generic-skill overlap). Floor them out so an ungrounded trade returns NOTHING rather
    # than a wrong neighbour — coherence as a gate, not coverage at any cost. 0.20 (not 0.25)
    # keeps the coherent same-cluster links (heftruckchauffeur<->orderpicker at 0.21) while
    # still killing the cross-domain 0.16 noise.
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

    # ---- 3. label resolver: ESCO Dutch labels (preferred + alt) for every occupation a
    # query can land on (grounded or reachable), so a typed Dutch role resolves even when
    # its exact surface isn't a corpus label. Collision -> prefer a resolvable/grounded
    # occupation, then the most corpus-present one. Corpus surfaces (step 1) take priority.
    resolvable = set(uri_related) | grounded
    cand_labels: dict[str, set[str]] = defaultdict(set)
    for uri in resolvable:
        r = bb.by_uri.get(uri, {})
        for lab in (r.get("preferredLabel") or {}).get("nl", "").split("/"):
            if fold(lab):
                cand_labels[fold(lab)].add(uri)
        for a in (r.get("altLabels") or {}).get("nl", []) or []:
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
