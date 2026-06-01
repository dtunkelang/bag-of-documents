#!/usr/bin/env python3
"""Per-family LLM-AND-kNN conjunction audit sweep. Computes kNN once, then for
each candidate family judges a sample of the conjunction set with the gpt-4.1
judge to measure TRUE precision. Prints families clearing the 90% floor."""

import json
import sys
from concurrent.futures import ThreadPoolExecutor

from classify_other_llm import (
    JUDGE_MODEL,
    JUDGE_SCHEMA,
    JUDGE_SYSTEM,
    _client,
    _load,
    llm_predictions,
    user_msg,
)
from exp_llm_knn_conjunction import LLM_OVERRIDES, _already_rescued, _knn_other_preds

N = int(sys.argv[1]) if len(sys.argv) > 1 else 60
FLOOR = 0.90

llm = llm_predictions()
knn = _knn_other_preds()
skip = _already_rescued()
over = set(json.loads(LLM_OVERRIDES.read_text())) if LLM_OVERRIDES.exists() else set()

# conjunction set per family
by_fam: dict[str, list[str]] = {}
for did, lfam in llm.items():
    if lfam == "other" or did in skip or did in over:
        continue
    if knn.get(did) == lfam:
        by_fam.setdefault(lfam, []).append(did)

_, _, txt = _load()
client = _client()


def judge(did, fam):
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
        return bool(json.loads(r.choices[0].message.content)["correct"])
    except Exception:
        return False


cands = sorted(by_fam.items(), key=lambda kv: -len(kv[1]))
print(f"auditing {len(cands)} families, N<={N} each, floor={FLOOR:.0%}\n")
results = []
for fam, dids in cands:
    sample = dids[:: max(1, len(dids) // N)][:N]
    verdicts = list(ThreadPoolExecutor(max_workers=12).map(lambda x, f=fam: judge(x, f), sample))
    ok = sum(verdicts)
    prec = ok / max(1, len(verdicts))
    flag = "PASS" if prec >= FLOOR else "fail"
    results.append((fam, prec, len(dids)))
    print(f"  [{flag}] {prec * 100:5.1f}%  n={len(verdicts):3d}  pool={len(dids):5d}  {fam}")

passers = [f for f, p, n in results if p >= FLOOR]
total = sum(n for f, p, n in results if p >= FLOOR)
print(f"\nPASSERS ({len(passers)}): {passers}")
print(f"recoverable docs from passers: {total}")
