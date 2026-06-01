#!/usr/bin/env python3
"""Confidence gate for slug -> industry labels.

slug_industry_labels_round2.csv mixes three kinds of labels: hand-labeled `seed`s,
deterministic `rule` hits, and nearest-seed PROPAGATION (`round2_hi`, `round2_med`,
`low_margin`). Propagation inherits the industry of the most-similar seed slug -- but
~57% of propagated labels anchor at cosine < 0.35, which is noise: you.com -> preply at
0.25 -> 'education_higher', NIST -> a Malaysian college at 0.23, etc. Loading those
verbatim filled buckets like education_higher with ~75% wrong members.

This gate keeps seeds + rules, drops the always-noisy tiers (round2_med, low_margin),
and requires any other propagated label to clear a similarity floor. Everything that
doesn't clear the gate is simply omitted, so a caller's `.get(slug, "unclassified")`
yields an honest gap instead of a confident guess.

Shared by push_docs.py (which APPLIES it) and qc_industry_labels.py (which AUDITS it),
so the rule that ships is the same rule the quality check measures. NOTE: a similarity
floor cannot catch a wrong *seed* (sim is 1.0 by definition) -- that's what the sampling
QC + the overrides layer are for.
"""

import csv
import os

TRUSTED_METHODS = {"seed", "rule"}  # hand-labeled / deterministic -> always keep
DROP_METHODS = {"round2_med", "low_margin"}  # too noisy at any similarity -> unclassified
# Every known-bad propagation (you.com, NIST, capitolis, flagright, safi) anchors <= 0.38;
# round2_hi p75 is only 0.388, so 0.50 prunes the noise while keeping ~75% doc coverage
# (the trusted seeds alone already cover ~69%). Env-tunable for A/B.
DEFAULT_SIM_FLOOR = float(os.environ.get("INDUSTRY_SIM_FLOOR", "0.50"))


def accept(method: str, top1_sim: float, floor: float = DEFAULT_SIM_FLOOR) -> bool:
    """Whether a label survives the confidence gate."""
    if method in TRUSTED_METHODS:
        return True
    if method in DROP_METHODS:
        return False
    return top1_sim >= floor  # round2_hi (or any other propagated tier)


def _to_float(x) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return 0.0


def load_slug_industry(csv_path: str, floor: float = DEFAULT_SIM_FLOOR) -> dict[str, str]:
    """slug -> industry with the confidence gate applied. Gated-out rows are omitted."""
    out: dict[str, str] = {}
    with open(csv_path) as f:
        for r in csv.DictReader(f):
            ind = (r.get("industry") or "").strip()
            if not ind or ind == "unclassified":
                continue
            if accept((r.get("method") or "").strip(), _to_float(r.get("top1_sim")), floor):
                out[r["slug"]] = ind
    return out


def load_overrides(csv_path: str) -> dict[str, str]:
    """Hand-curated slug -> industry corrections (LLM-tail self-labels + audited fixes).

    These are trusted ground truth that OVERRIDES the gated propagation. Crucially they
    are the only thing that can fix a wrong *seed*: a seed's self-similarity is 1.0, so
    the gate's floor structurally cannot prune it (see this module's docstring). Format:
    slug,industry,note -- the note column is documentation, not consumed here. Missing
    file -> empty dict, so callers degrade to gate-only labels.
    """
    out: dict[str, str] = {}
    if not os.path.exists(csv_path):
        return out
    with open(csv_path) as f:
        for r in csv.DictReader(f):
            ind = (r.get("industry") or "").strip()
            if ind and ind != "unclassified":
                out[r["slug"]] = ind
    return out
