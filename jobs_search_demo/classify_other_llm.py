#!/usr/bin/env python3
"""LLM classification of the role_family:'other' residual (offline, one-time).

The heuristic + embedding-kNN + dept-agree rescues have driven 'other' down to
~25%; the remainder has domain-less titles (manager/director/lead/specialist)
where regex and the e5 vote both plateau. An LLM reading the full *description*
can disambiguate these. This runs OFFLINE over a fixed residual via the OpenAI
Batch API (a few dollars) -- it is NOT a serving path, so it does not touch the
no-live-encode policy.

Modes:
  --validate N   classify N labeled docs (stratified), report accuracy vs the
                 heuristic label + confidence calibration (precision proxy).
  --build-batch  write an OpenAI Batch API .jsonl for every 'other' doc.
  --parse FILE   read batch output, gate by confidence, write overrides.

Family taxonomy is the 32 role_family values, defined by FAM_QUERIES from
classify_other_emb (reused so the LLM maps onto the exact live taxonomy).
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from classify_other_emb import FAM_QUERIES

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "unified_jobs_daily"
MODEL = os.environ.get("LLM_CLASSIFY_MODEL", "gpt-4o-mini")
DESC_CHARS = 1500  # description chars sent to the model

FAMILIES = sorted(FAM_QUERIES)
SCHEMA = {
    "name": "role_family",
    "schema": {
        "type": "object",
        "properties": {
            "family": {"type": "string", "enum": [*FAMILIES, "other"]},
            "confidence": {"type": "number"},
        },
        "required": ["family", "confidence"],
        "additionalProperties": False,
    },
    "strict": True,
}

SYSTEM = (
    "You classify job postings into exactly one role_family. Use the description, "
    "not just the title. Pick the single best family from the list; if none fits "
    "(e.g. a generic 'general application', or a role with no clear home), return "
    '"other". Report confidence in [0,1] = your probability the label is correct.\n\n'
    "Families (name: cues):\n" + "\n".join(f"- {f}: {FAM_QUERIES[f]}" for f in FAMILIES)
)


def user_msg(title: str, desc: str) -> str:
    return f"Title: {title}\n\nDescription: {(desc or '')[:DESC_CHARS]}"


def _load():
    ids = json.load(open(DATA / "doc_ids.json"))
    lab = json.load(open(DATA / "role_labels.json"))
    y = [lab.get(i, "other") for i in ids]
    txt = {}
    with open(DATA / "metadata.jsonl") as f:
        for line in f:
            r = json.loads(line)
            txt[r["id"]] = ((r.get("title") or ""), (r.get("description") or ""))
    return ids, y, txt


def _client():
    from dotenv import load_dotenv
    from openai import OpenAI

    load_dotenv(override=True)
    return OpenAI()


def _classify_one(client, title, desc):
    r = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": user_msg(title, desc)},
        ],
        response_format={"type": "json_schema", "json_schema": SCHEMA},
        temperature=0,
    )
    return json.loads(r.choices[0].message.content)


def validate(n: int):
    ids, y, txt = _load()
    # stratified sample of LABELED docs (exclude 'other'); cap per family
    by_fam = defaultdict(list)
    for did, fam in zip(ids, y):
        if fam != "other":
            by_fam[fam].append(did)
    rng_order = sorted(by_fam)  # deterministic
    per = max(1, n // len(rng_order))
    sample = []
    for fam in rng_order:
        for did in by_fam[fam][:per]:
            sample.append((did, fam))
    client = _client()
    print(f"validating {len(sample)} labeled docs on {MODEL} (per-family cap {per})...")

    def work(item):
        did, gold = item
        t, d = txt[did]
        try:
            out = _classify_one(client, t, d)
            return gold, out["family"], float(out["confidence"])
        except Exception as e:
            return gold, f"ERR:{type(e).__name__}", 0.0

    rows = list(ThreadPoolExecutor(max_workers=12).map(work, sample))
    # accuracy among confident predictions (LLM != 'other')
    assigned = [(g, p, c) for g, p, c in rows if p != "other" and not p.startswith("ERR")]
    abstain = sum(1 for _, p, _ in rows if p == "other")
    errs = sum(1 for _, p, _ in rows if p.startswith("ERR"))
    acc = sum(g == p for g, p, _ in assigned) / max(1, len(assigned))
    print(f"\n  assigned {len(assigned)}  abstained(other) {abstain}  errors {errs}")
    print(f"  overall accuracy on assigned: {acc * 100:.1f}%")
    print(f"  {'conf>=':>7} {'coverage':>9} {'accuracy':>9}")
    for thr in (0.0, 0.5, 0.7, 0.8, 0.9):
        kept = [(g, p) for g, p, c in assigned if c >= thr]
        if not kept:
            continue
        a = sum(g == p for g, p in kept) / len(kept)
        print(f"  {thr:7.2f} {len(kept) / len(sample) * 100:8.1f}% {a * 100:8.1f}%")
    # per-family precision (gold == pred) at conf>=0.7
    print("\n  per-family accuracy (conf>=0.7):")
    tp, fp = Counter(), Counter()
    for g, p, c in assigned:
        if c >= 0.7:
            (tp if g == p else fp)[p] += 1
    for f in sorted(set(list(tp) + list(fp))):
        nn = tp[f] + fp[f]
        flag = "  <-- weak" if tp[f] / nn < 0.85 else ""
        print(f"    {tp[f] / nn * 100:5.1f}%  n={nn:4d}  {f}{flag}")


JUDGE_MODEL = os.environ.get("LLM_JUDGE_MODEL", "gpt-4.1")
JUDGE_SCHEMA = {
    "name": "verdict",
    "schema": {
        "type": "object",
        "properties": {
            "correct": {"type": "boolean"},
            "better_family": {"type": "string", "enum": [*FAMILIES, "other"]},
        },
        "required": ["correct", "better_family"],
        "additionalProperties": False,
    },
    "strict": True,
}
JUDGE_SYSTEM = (
    "You audit a role_family label proposed by another classifier. Read the title "
    "and description. Decide independently the single best family (or 'other' if no "
    "family is a clear home). Then judge whether the PROPOSED label is acceptable "
    "(correct=true if it equals your best choice or is a defensible match). Be "
    "skeptical: generic postings with no clear domain should be 'other'.\n\n"
    "Families (name: cues):\n" + "\n".join(f"- {f}: {FAM_QUERIES[f]}" for f in FAMILIES)
)


def audit(n: int):
    """Estimate TRUE precision of a single-signal LLM rescue on actual 'other' docs:
    classify with MODEL, then have JUDGE_MODEL (stronger, independent) verify each
    proposed label. Reports overall + per-predicted-family confirmation rate."""
    ids, y, txt = _load()
    others = [did for did, fam in zip(ids, y) if fam == "other"]
    sample = others[:: max(1, len(others) // n)][:n]  # deterministic spread
    client = _client()
    print(f"auditing {len(sample)} 'other' docs: classify={MODEL} judge={JUDGE_MODEL}")

    def classify(did):
        t, d = txt[did]
        try:
            o = _classify_one(client, t, d)
            return did, o["family"], float(o["confidence"])
        except Exception as e:
            return did, f"ERR:{type(e).__name__}", 0.0

    cls = list(ThreadPoolExecutor(max_workers=12).map(classify, sample))
    assigned = [(did, fam) for did, fam, _ in cls if fam != "other" and not fam.startswith("ERR")]
    abstain = sum(1 for _, fam, _ in cls if fam == "other")
    print(f"  classifier assigned {len(assigned)}, abstained 'other' {abstain}")

    def judge(item):
        did, fam = item
        t, d = txt[did]
        try:
            r = client.chat.completions.create(
                model=JUDGE_MODEL,
                messages=[
                    {"role": "system", "content": JUDGE_SYSTEM},
                    {"role": "user", "content": user_msg(t, d) + f"\n\nPROPOSED family: {fam}"},
                ],
                response_format={"type": "json_schema", "json_schema": JUDGE_SCHEMA},
                temperature=0,
            )
            v = json.loads(r.choices[0].message.content)
            return fam, bool(v["correct"]), v["better_family"]
        except Exception:
            return fam, False, "other"

    verdicts = list(ThreadPoolExecutor(max_workers=12).map(judge, assigned))
    ok = sum(c for _, c, _ in verdicts)
    print(
        f"\n  TRUE precision (judge-confirmed): {ok}/{len(verdicts)} = "
        f"{ok / max(1, len(verdicts)) * 100:.1f}%"
    )
    # where judge disagrees, what does it prefer?
    to_other = sum(1 for _, c, b in verdicts if not c and b == "other")
    to_fam = sum(1 for _, c, b in verdicts if not c and b != "other")
    print(f"  judge rejects -> 'other': {to_other}   -> different family: {to_fam}")
    # per-predicted-family confirmation rate
    tp, tot = Counter(), Counter()
    for fam, c, _ in verdicts:
        tot[fam] += 1
        if c:
            tp[fam] += 1
    print("\n  per-predicted-family confirmation (n>=5 actionable):")
    for f in sorted(tot, key=lambda x: -tot[x]):
        nn = tot[f]
        flag = "  <-- allowlist" if nn >= 5 and tp[f] / nn >= 0.90 else ""
        print(f"    {tp[f] / nn * 100:5.1f}%  n={nn:4d}  {f}{flag}")


# Per-family allowlist for the production rescue: only families the gpt-4.1 judge
# confirmed at >=90% (n>=5) in the audit are written as overrides. Set from a
# larger --audit run before --parse. The LLM is strongest on sales/ai_ml/
# skilled_trades/manufacturing -- exactly the grab-bag-department families that
# dept-agree had to exclude (it reads the description, not the org chart).
# Firm allowlist from the 600-doc audit (audit600.log): judge-confirmed >=90% at
# n>=8 (+retail 100% n=6). The 250-doc audit overstated operations_admin (95->72%)
# and ai_ml (100->64%) on small n -- the larger audit corrects that. High-volume
# sales (88%) and finance (86%) sit just under the 90% floor: candidates for an
# LLM-AND-kNN conjunction follow-up, NOT the pure-LLM allowlist.
LLM_ALLOW = frozenset(
    {
        "marketing",
        "project_program_management",
        "data_analytics",
        "design_ux",
        "manufacturing_production",
        "hr_people_ops",
        "legal",
        "product_management",
        "transportation_logistics",
        "retail",
    }
)

BATCH_ID_FILE = HERE / "llm_other_batch.id"
LLM_OVERRIDES = HERE / "role_family_llm_overrides.json"


def _already_rescued() -> set[str]:
    """doc ids already covered by the embedding/dept-agree overrides -- skip them
    so the LLM batch only spends on the genuine residual."""
    p = HERE / "role_family_emb_overrides.json"
    return set(json.loads(p.read_text())) if p.exists() else set()


SHARD_LINES = 20000  # OpenAI batch input file cap is 200MB; ~5.8KB/line -> shard


def build_batch(out_path: Path):
    """Write sharded JSONL (<=SHARD_LINES each) to stay under the 200MB batch cap.
    Returns the list of shard paths."""
    ids, y, txt = _load()
    skip = _already_rescued()
    others = [did for did, fam in zip(ids, y) if fam == "other" and did not in skip]
    shards = []
    for s in range(0, len(others), SHARD_LINES):
        chunk = others[s : s + SHARD_LINES]
        shard = out_path.with_suffix(f".{s // SHARD_LINES:03d}.jsonl")
        with open(shard, "w") as f:
            for did in chunk:
                t, d = txt[did]
                req = {
                    "custom_id": did,
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": {
                        "model": MODEL,
                        "messages": [
                            {"role": "system", "content": SYSTEM},
                            {"role": "user", "content": user_msg(t, d)},
                        ],
                        "response_format": {"type": "json_schema", "json_schema": SCHEMA},
                        "temperature": 0,
                    },
                }
                f.write(json.dumps(req) + "\n")
        shards.append(shard)
    print(
        f"wrote {len(others):,} requests across {len(shards)} shards "
        f"(skipped {len(skip):,} already rescued)"
    )
    return shards


def submit_batch(shards: list[Path]):
    client = _client()
    ids = []
    for shard in shards:
        up = client.files.create(file=open(shard, "rb"), purpose="batch")
        b = client.batches.create(
            input_file_id=up.id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
            metadata={"task": "role_family_other_rescue", "shard": shard.name},
        )
        ids.append(b.id)
        print(f"submitted {shard.name} -> batch {b.id} (status={b.status})")
    BATCH_ID_FILE.write_text("\n".join(ids) + "\n")
    print(f"saved {len(ids)} batch ids -> {BATCH_ID_FILE.name}")


def _batch_ids() -> list[str]:
    return [x for x in BATCH_ID_FILE.read_text().splitlines() if x.strip()]


def poll_batch():
    client = _client()
    statuses = []
    for bid in _batch_ids():
        b = client.batches.retrieve(bid)
        rc = b.request_counts
        statuses.append(b.status)
        print(f"  {bid}: status={b.status} completed={rc.completed}/{rc.total} failed={rc.failed}")
    done = all(s == "completed" for s in statuses)
    print(f"all completed: {done}")
    return statuses


BATCH_OUTPUT = HERE / "llm_other_batch_output.jsonl"


def _fetch_output() -> str:
    """All completed batch shards' raw JSONL, cached to disk so re-parses (e.g. the
    LLM-AND-kNN conjunction) don't re-download. Returns '' if any shard incomplete."""
    if BATCH_OUTPUT.exists():
        return BATCH_OUTPUT.read_text()
    client = _client()
    content = ""
    for bid in _batch_ids():
        b = client.batches.retrieve(bid)
        if b.status != "completed":
            print(f"batch {bid} not complete (status={b.status}); aborting")
            return ""
        content += client.files.content(b.output_file_id).text
    BATCH_OUTPUT.write_text(content)
    print(f"cached batch output -> {BATCH_OUTPUT.name}")
    return content


def llm_predictions() -> dict[str, str]:
    """{doc_id: llm_family} over all batch results (incl. 'other' abstentions)."""
    preds = {}
    for line in _fetch_output().splitlines():
        r = json.loads(line)
        out = json.loads(r["response"]["body"]["choices"][0]["message"]["content"])
        preds[r["custom_id"]] = out["family"]
    return preds


def parse_batch(apply: bool):
    """Download all completed batch shards, keep allowlisted family labels, write
    role_family_llm_overrides.json (separate from the embedding file so the nightly
    refresh that regenerates the embedding file never clobbers these)."""
    content = _fetch_output()
    if not content:
        return
    skip = _already_rescued()
    kept, assigned, abstain, dropped_fam, conflict = {}, 0, 0, 0, 0
    fam_counts = Counter()
    for line in content.splitlines():
        r = json.loads(line)
        did = r["custom_id"]
        body = r["response"]["body"]
        out = json.loads(body["choices"][0]["message"]["content"])
        fam = out["family"]
        if fam == "other":
            abstain += 1
            continue
        assigned += 1
        if did in skip:  # embedding/dept-agree already won this doc
            conflict += 1
            continue
        if fam not in LLM_ALLOW:
            dropped_fam += 1
            continue
        kept[did] = fam
        fam_counts[fam] += 1
    print(
        f"assigned {assigned}  abstain(other) {abstain}  off-allowlist {dropped_fam}  "
        f"already-rescued {conflict}  -> KEEP {len(kept)}"
    )
    for f, n in fam_counts.most_common():
        print(f"  {n:6d}  {f}")
    if apply:
        LLM_OVERRIDES.write_text(json.dumps(kept, indent=0, sort_keys=True))
        print(f"wrote {len(kept):,} -> {LLM_OVERRIDES.name}")
    else:
        print("\n(dry run -- re-run --parse --apply to write overrides)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--validate", type=int, metavar="N")
    ap.add_argument("--audit", type=int, metavar="N")
    ap.add_argument("--build-batch", action="store_true")
    ap.add_argument("--submit", action="store_true", help="build + upload + create batch")
    ap.add_argument("--poll", action="store_true")
    ap.add_argument("--parse", action="store_true")
    ap.add_argument("--apply", action="store_true", help="with --parse: write overrides")
    ap.add_argument("--out", default=str(HERE / "llm_other_batch.jsonl"))
    args = ap.parse_args()
    if args.validate:
        validate(args.validate)
    elif args.audit:
        audit(args.audit)
    elif args.submit:
        submit_batch(build_batch(Path(args.out)))
    elif args.build_batch:
        build_batch(Path(args.out))
    elif args.poll:
        poll_batch()
    elif args.parse:
        parse_batch(args.apply)
    else:
        ap.error("pick --validate N, --audit N, --build-batch, --submit, --poll, or --parse")
