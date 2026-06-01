#!/usr/bin/env python3
"""LLM-AND-kNN conjunction rescue for the high-volume families held back from the
pure-LLM allowlist (sales 88%, finance_accounting 86% true precision -- under the
~90% floor). Hypothesis: requiring the embedding-kNN to INDEPENDENTLY agree with
the LLM lifts precision over the floor (same conjunction logic as dept-agree).

No new LLM classification spend: reuses the cached batch output (llm_predictions)
and the free e5 kNN. --audit N judges a sample of the conjunction set with gpt-4.1
to measure TRUE precision before --apply merges it into role_family_llm_overrides.json.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

import faiss
import numpy as np
from classify_other_emb import load as load_emb
from classify_other_llm import (
    JUDGE_MODEL,
    JUDGE_SCHEMA,
    JUDGE_SYSTEM,
    LLM_OVERRIDES,
    _already_rescued,
    _client,
    _load,
    llm_predictions,
    user_msg,
)

CONJ_FAMILIES = {"sales", "finance_accounting"}
K = 25


def _knn_other_preds():
    """{doc_id: knn_family} for every 'other' doc, similarity-weighted k-NN vote
    over the labeled docs (same core as classify_other_emb)."""
    ids, V, y = load_emb()
    is_other = y == "other"
    lab = np.where(~is_other)[0]
    oth = np.where(is_other)[0]
    index = faiss.IndexFlatIP(V.shape[1])
    index.add(V[lab])
    D, I = index.search(V[oth], K)
    fam = y[lab][I]
    preds = {}
    for row_f, row_d, oi in zip(fam, D, oth):
        w: dict[str, float] = {}
        for f, d in zip(row_f, row_d):
            w[f] = w.get(f, 0.0) + float(d)
        preds[ids[oi]] = max(w, key=w.get)
    return preds


def conjunction_set() -> dict[str, str]:
    """{doc_id: family} where LLM and kNN agree on a CONJ_FAMILY, excluding docs
    already rescued by the embedding/dept-agree or pure-LLM overrides."""
    llm = llm_predictions()
    knn = _knn_other_preds()
    skip = _already_rescued()
    over = set(json.loads(LLM_OVERRIDES.read_text())) if LLM_OVERRIDES.exists() else set()
    out = {}
    for did, lfam in llm.items():
        if lfam in CONJ_FAMILIES and knn.get(did) == lfam and did not in skip and did not in over:
            out[did] = lfam
    return out


def audit(n: int):
    cand = conjunction_set()
    print(f"conjunction candidates: {len(cand)}  ({Counter(cand.values())})")
    _, _, txt = _load()
    items = list(cand.items())[:: max(1, len(cand) // n)][:n]
    client = _client()

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
            return fam, bool(json.loads(r.choices[0].message.content)["correct"])
        except Exception:
            return fam, False

    verdicts = list(ThreadPoolExecutor(max_workers=12).map(judge, items))
    ok = sum(c for _, c in verdicts)
    print(
        f"\n  TRUE precision (judge-confirmed): {ok}/{len(verdicts)} = "
        f"{ok / max(1, len(verdicts)) * 100:.1f}%"
    )
    tp, tot = Counter(), Counter()
    for fam, c in verdicts:
        tot[fam] += 1
        tp[fam] += int(c)
    for f in sorted(tot):
        print(f"    {tp[f] / tot[f] * 100:5.1f}%  n={tot[f]:4d}  {f}")


def apply():
    cand = conjunction_set()
    over = json.loads(LLM_OVERRIDES.read_text()) if LLM_OVERRIDES.exists() else {}
    before = len(over)
    over.update(cand)  # conjunction docs are by construction not already present
    LLM_OVERRIDES.write_text(json.dumps(over, indent=0, sort_keys=True))
    print(
        f"merged {len(cand)} conjunction rescues ({Counter(cand.values())}); "
        f"{before} -> {len(over)} total in {LLM_OVERRIDES.name}"
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--audit", type=int, metavar="N")
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()
    if args.audit:
        audit(args.audit)
    elif args.apply:
        apply()
    else:
        ap.error("pick --audit N or --apply")
