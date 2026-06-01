#!/usr/bin/env python3
"""Recompute role_family over the LIVE Solr docs and atomic-update only the ones
that change.

Operates on Solr's ground truth (id + title_display + description) rather than
position-based facets.jsonl idx, because the persistent core accumulates docs
across delta refreshes and idx no longer aligns. Re-runs heuristics.
classify_role_family on each doc; pushes a `{"set": ...}` for role_family on the
docs whose family changed. Nothing else is touched (no vectors, no other facets).

Run after editing facets/heuristics.py to reflect the new taxonomy on the live
local index, then re-tar + redeploy the core.
"""

import sys
import time
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).parent / "facets"))
sys.path.insert(0, str(Path(__file__).parent))
from heuristics import classify_role_family  # noqa: E402
from push_docs import stable_id  # noqa: E402

SOLR = "http://localhost:8983"
CORE = "jobs"
PAGE = 2000
BATCH = 1000


def load_emb_overrides() -> dict[str, str]:
    """Model-derived role_family overrides, keyed by source doc id -> converted to
    the Solr stable_id so the live re-class can apply them. These aren't expressible
    as title heuristics, so without this they would revert to 'other' on every
    reclassify/refresh. Unions two sources (embedding/dept-agree from
    classify_other_emb.py, then LLM backfill from classify_other_llm.py); the
    embedding file is loaded first so it WINS on any id collision."""
    import json

    out: dict[str, str] = {}
    for name in ("role_family_emb_overrides.json", "role_family_llm_overrides.json"):
        p = Path(__file__).parent / name
        if not p.exists():
            continue
        for k, v in json.loads(p.read_text()).items():
            out.setdefault(str(stable_id(k)), v)
    return out


def iter_docs():
    cursor = "*"
    url = f"{SOLR}/solr/{CORE}/select"
    while True:
        r = requests.get(
            url,
            params={
                "q": "*:*",
                "fl": "id,title_display,description,role_family",
                "rows": PAGE,
                "sort": "id asc",
                "cursorMark": cursor,
                "wt": "json",
            },
            timeout=120,
        )
        r.raise_for_status()
        d = r.json()
        docs = d["response"]["docs"]
        yield from docs
        nxt = d.get("nextCursorMark")
        if not docs or nxt == cursor:
            break
        cursor = nxt


def flush(batch):
    if not batch:
        return
    r = requests.post(f"{SOLR}/solr/{CORE}/update", json=batch, timeout=180)
    r.raise_for_status()


def main(apply: bool) -> int:
    t0 = time.time()
    seen = 0
    changed = 0
    transitions: dict[str, int] = {}
    batch: list[dict] = []
    emb = load_emb_overrides()
    if emb:
        print(f"loaded {len(emb):,} embedding role_family overrides", flush=True)
    for doc in iter_docs():
        seen += 1
        title = doc.get("title_display") or ""
        desc = doc.get("description") or ""
        old = doc.get("role_family") or ""
        new = classify_role_family(title, desc)
        # embedding override only fills genuine 'other' (never overrides a heuristic hit)
        if new == "other":
            new = emb.get(str(doc["id"]), new)
        if new != old:
            changed += 1
            transitions[f"{old or '<empty>'} -> {new}"] = (
                transitions.get(f"{old or '<empty>'} -> {new}", 0) + 1
            )
            if apply:
                batch.append({"id": doc["id"], "role_family": {"set": new}})
                if len(batch) >= BATCH:
                    flush(batch)
                    batch = []
        if seen % 25000 == 0:
            print(f"  scanned {seen:,} changed {changed:,}", flush=True)
    if apply:
        flush(batch)
        print("committing...", flush=True)
        requests.get(
            f"{SOLR}/solr/{CORE}/update", params={"commit": "true"}, timeout=300
        ).raise_for_status()

    print(f"\nscanned {seen:,}  changed {changed:,} in {time.time() - t0:.1f}s")
    print("top transitions:")
    for k, v in sorted(transitions.items(), key=lambda x: -x[1])[:30]:
        print(f"  {v:6,}  {k}")
    # how many ended up ai_ml
    ai = sum(v for k, v in transitions.items() if k.endswith("-> ai_ml"))
    print(f"\n-> ai_ml total: {ai:,}")
    if not apply:
        print("\n(dry run — re-run with --apply to write to Solr)")
    return 0


if __name__ == "__main__":
    sys.exit(main(apply="--apply" in sys.argv))
