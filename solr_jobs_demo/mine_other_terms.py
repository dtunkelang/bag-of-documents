#!/usr/bin/env python3
"""Mine candidate title n-grams to rescue docs from role_family:other.

Strategy A (search-mines-regex): the demo's e5-small dense lane surfaces docs
that are *semantically* about a target family (finance, sales, ...) even when
their titles don't contain the obvious keyword -- exactly the docs ROLE_PATTERNS
missed. We don't promote the docs (search is a soft, ranked signal that decays);
we mine the recurring *title phrases* and hand them back as regex candidates.

Pipeline per run:
  1. encode the family query with e5-small-v2 (same serving path as the demo)
  2. KNN over e5_vec, preFilter=role_family:other  -> top-N candidate docs
  3. tokenize their (geo/modifier-stripped) titles into 1-3 grams
  4. rank by LIFT = freq among candidates / freq among a random `other` baseline
     (distinctive to the finance-relevant slice of other, not generic to other)
  5. for each shortlisted n-gram, a corpus-wide mini dry-run: facet role_family
     over all titles matching it, so you see cross-family risk BEFORE writing
     the regex. n-grams that sit overwhelmingly in `other` are the safe rescues.

Output is a ranked shortlist to eyeball; accept the clean ones into
facets/heuristics.py ROLE_PATTERNS, then run reclassify_role_family.py --apply.

Usage:
  python mine_other_terms.py "finance and accounting" --family finance_accounting
  python mine_other_terms.py "sales account executive" --family sales --show 40

Pass --family to score precision correctly: a new rule routing the n-gram to the
target family only does damage when corpus titles matching it sit in a THIRD
family. Matches already in the target family are consistent (the rule agrees);
matches in 'other' are the rescue we want. Without --family the tool falls back
to a cruder 'share-in-other' signal.
"""

import argparse
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import requests

sys.path.insert(0, str(Path(__file__).parent / "facets"))
from heuristics import _strip_role_modifiers  # noqa: E402

SOLR = "http://localhost:8983"
CORE = "jobs"
DENSE_MODEL = "intfloat/e5-small-v2"
DENSE_QUERY_PREFIX = "query: "

_STOP = {
    "the",
    "and",
    "of",
    "for",
    "to",
    "in",
    "a",
    "an",
    "with",
    "at",
    "on",
    "or",
    "is",
    "as",
    "by",
    "ii",
    "iii",
    "iv",
    "i",
}
# Generic job-posting noise that survives geo/modifier stripping but carries no
# family signal. Kept short on purpose -- lift already suppresses most of these.
_GENERIC = {
    "remote",
    "hybrid",
    "full",
    "time",
    "part",
    "contract",
    "temporary",
    "manager",
    "specialist",
    "coordinator",
    "associate",
    "representative",
    "analyst",
    "lead",
    "supervisor",
    "director",
    "officer",
    "assistant",
}
_WORD = re.compile(r"[a-z0-9]+")


def _model():
    print(f"loading {DENSE_MODEL}...", file=sys.stderr, flush=True)
    from sentence_transformers import SentenceTransformer

    try:
        import torch

        dev = "mps" if torch.backends.mps.is_available() else "cpu"
    except Exception:
        dev = "cpu"
    return SentenceTransformer(DENSE_MODEL, device=dev)


def _dense_qv(model, query: str) -> list[float]:
    qv = model.encode(
        [DENSE_QUERY_PREFIX + query], normalize_embeddings=True, show_progress_bar=False
    )[0]
    return qv.astype(np.float32).tolist()


def _vec_str(vec: list[float]) -> str:
    return "[" + ",".join(f"{x:.6f}" for x in vec) + "]"


def _knn_other(qv: list[float], k: int) -> list[str]:
    """top-k titles from role_family:other by e5 cosine to the family query."""
    q = f"{{!knn f=e5_vec topK={k} preFilter='role_family:other'}}{_vec_str(qv)}"
    r = requests.get(
        f"{SOLR}/solr/{CORE}/select",
        params={"q": q, "fl": "title_display", "rows": k, "wt": "json"},
        timeout=60,
    )
    r.raise_for_status()
    return [d.get("title_display") or "" for d in r.json()["response"]["docs"]]


def _baseline_titles(n: int) -> list[str]:
    """random sample of role_family:other titles for the lift denominator."""
    r = requests.get(
        f"{SOLR}/solr/{CORE}/select",
        params={
            "q": "*:*",
            "fq": "role_family:other",
            "fl": "title_display",
            "rows": n,
            "sort": "random_1 asc",
            "wt": "json",
        },
        timeout=60,
    )
    # random_1 dynamic field may not exist; fall back to score sort over *:*
    if r.status_code != 200:
        r = requests.get(
            f"{SOLR}/solr/{CORE}/select",
            params={
                "q": "*:*",
                "fq": "role_family:other",
                "fl": "title_display",
                "rows": n,
                "wt": "json",
            },
            timeout=60,
        )
    r.raise_for_status()
    return [d.get("title_display") or "" for d in r.json()["response"]["docs"]]


def _ngrams(title: str, lo: int, hi: int):
    t = _strip_role_modifiers(title).lower()
    toks = [w for w in _WORD.findall(t) if w not in _STOP]
    for n in range(lo, hi + 1):
        for i in range(len(toks) - n + 1):
            gram = toks[i : i + n]
            if n == 1 and gram[0] in _GENERIC:
                continue
            yield " ".join(gram)


def _doc_freq(titles: list[str], lo: int, hi: int) -> Counter:
    """document frequency of each n-gram (count a gram once per title)."""
    c = Counter()
    for t in titles:
        for g in set(_ngrams(t, lo, hi)):
            c[g] += 1
    return c


def _role_dist(ngram: str) -> dict[str, int]:
    """corpus-wide role_family facet over titles matching the n-gram (phrase)."""
    r = requests.get(
        f"{SOLR}/solr/{CORE}/select",
        params=[
            ("q", f'title:"{ngram}"'),
            ("rows", "0"),
            ("facet", "true"),
            ("facet.field", "role_family"),
            ("facet.mincount", "1"),
            ("wt", "json"),
        ],
        timeout=30,
    )
    r.raise_for_status()
    d = r.json()
    counts = d["facet_counts"]["facet_fields"]["role_family"]
    return {counts[i]: counts[i + 1] for i in range(0, len(counts), 2)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("query", help="family query, e.g. 'finance and accounting'")
    ap.add_argument(
        "--family",
        default=None,
        help="target role_family id (e.g. finance_accounting). Enables target-aware risk scoring.",
    )
    ap.add_argument("--topn", type=int, default=1000, help="KNN candidate pool size")
    ap.add_argument("--baseline", type=int, default=4000, help="random other baseline size")
    ap.add_argument("--grams", default="1-3", help="n-gram range, e.g. 1-3")
    ap.add_argument(
        "--min-support", type=int, default=8, help="min candidate doc-freq to consider an n-gram"
    )
    ap.add_argument("--show", type=int, default=30, help="shortlist length")
    args = ap.parse_args()

    lo, hi = (
        (int(x) for x in args.grams.split("-"))
        if "-" in args.grams
        else (int(args.grams), int(args.grams))
    )

    model = _model()
    qv = _dense_qv(model, args.query)
    cand = _knn_other(qv, args.topn)
    base = _baseline_titles(args.baseline)
    print(f"candidates={len(cand)}  baseline={len(base)}", file=sys.stderr)

    cand_df = _doc_freq(cand, lo, hi)
    base_df = _doc_freq(base, lo, hi)
    nb = max(1, len(base))
    nc = max(1, len(cand))

    rows = []
    for g, cf in cand_df.items():
        if cf < args.min_support:
            continue
        cand_rate = cf / nc
        base_rate = base_df.get(g, 0) / nb
        # additive smoothing so a zero-baseline gram doesn't get infinite lift
        lift = (cand_rate + 1e-4) / (base_rate + 1e-4)
        rows.append((lift, cf, base_df.get(g, 0), g))
    rows.sort(reverse=True)

    print(f"\n# high-lift title n-grams in role_family:other near '{args.query}'")
    print(
        f"# (cand_df = # of top-{args.topn} candidates with the gram; "
        f"role_dist = corpus-wide role_family of ALL titles matching it)\n"
    )
    print(f"{'lift':>7}  {'cand':>5}  {'base':>5}  ngram")
    print("-" * 70)
    fam = args.family
    for lift, cf, bf, g in rows[: args.show]:
        dist = _role_dist(g)
        total = sum(dist.values()) or 1
        other = dist.get("other", 0)
        if fam:
            # a rule routing g -> fam only mis-moves titles that are in a THIRD
            # family. 'other' = rescued (good), fam = consistent (rule agrees).
            consistent = dist.get(fam, 0)
            risk = total - other - consistent
            risk_share = risk / total
            spill = (
                " ".join(
                    f"{k}:{v}"
                    for v, k in sorted(
                        ((v, k) for k, v in dist.items() if k not in (fam, "other")), reverse=True
                    )[:3]
                )
                or "-"
            )
            flag = "OK " if risk_share <= 0.10 else ("?? " if risk_share <= 0.25 else "XX ")
            print(
                f"{lift:7.1f}  {cf:5d}  {bf:5d}  {flag}{g!r}  "
                f"[rescue {other} + {fam} {consistent} | risk {risk}/{total} "
                f"{risk_share:.0%} -> {spill}]"
            )
        else:
            safe = other / total
            spill = (
                " ".join(
                    f"{k}:{v}"
                    for v, k in sorted(
                        ((v, k) for k, v in dist.items() if k != "other"), reverse=True
                    )[:3]
                )
                or "-"
            )
            flag = "OK " if safe >= 0.7 else ("?? " if safe >= 0.4 else "XX ")
            print(
                f"{lift:7.1f}  {cf:5d}  {bf:5d}  {flag}{g!r}  "
                f"[other {other}/{total} {safe:.0%} | spill {spill}]"
            )

    if fam:
        print(
            f"\nlegend (target={fam}): OK <=10% of corpus title-matches spill into a "
            f"THIRD family (safe) | ?? 10-25% (check) | XX >25% (cross-family, risky). "
            f"'rescue' = #other moved; '{fam}' = matches the rule agrees with."
        )
    else:
        print(
            "\nlegend: OK >=70% of corpus title-matches in 'other' | ?? 40-70% | "
            "XX <40%. Pass --family for target-aware risk scoring."
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
