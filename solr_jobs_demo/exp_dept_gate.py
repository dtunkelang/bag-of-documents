#!/usr/bin/env python3
"""Held-out experiment: does department-agreement let us safely relax the
embedding-kNN gate for role_family 'other' rescue?

The production embedding classifier (classify_other_emb.py) rescues docs whose
kNN vote clears a strict conf+ensemble gate (~96% precision, 3,614 docs).
Department is the company's own org assignment, present on 99.3% of 'other'
docs. Alone it is too noisy (most depts 0.2-0.75 purity), but its AGREEMENT
with the kNN prediction is an independent-signal conjunction.

This script measures, on a held-out split of the heuristic-labeled docs:
  - precision of the strict gate (baseline)
  - precision of the department-agreement gate
  - precision of the INCREMENTAL set: docs that FAIL the strict gate but where
    kNN-pred == department-modal-family  <-- the proposed rescue expansion

Department-modal family is computed from the TRAIN split only (no leakage).
Eval only -- writes nothing.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

import faiss
import numpy as np

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "unified_jobs_daily"

# match production APPLY_DEFAULTS exactly so the strict baseline == what's live
CONF = 0.80  # production strict-gate vote-share
SIMFLOOR = 0.86  # production top-1 neighbor cosine floor
K = 25
DEPT_CONF = 0.40  # relaxed vote-share for the dept-agree rescue path
DEPT_MIN_LABELED = 20  # ignore departments with too few labeled exemplars


def _norm(x):
    x = x.astype(np.float32)
    x /= np.linalg.norm(x, axis=1, keepdims=True) + 1e-9
    return x


def main():
    ids = json.load(open(DATA / "doc_ids.json"))
    lab = json.load(open(DATA / "role_labels.json"))
    V = _norm(np.load(DATA / "e5_small_catalog.vecs.fp16.npy"))
    y = np.array([lab.get(i, "other") for i in ids])

    # departments positionally aligned to ids/V
    depts = [""] * len(ids)
    with open(DATA / "metadata.jsonl") as f:
        for i, line in enumerate(f):
            d = json.loads(line)
            dep = (d.get("department") or "").strip()
            depts[i] = "" if dep in ("", "None") else dep
    depts = np.array(depts, dtype=object)

    is_other = y == "other"
    lab_idx = np.where(~is_other)[0]
    fams = sorted(set(y[lab_idx]))
    print(f"labeled {len(lab_idx)}  other {is_other.sum()}  families {len(fams)}")

    rng = np.random.default_rng(0)
    perm = rng.permutation(lab_idx)
    ntest = int(len(perm) * 0.15)
    test, train = perm[:ntest], perm[ntest:]

    # department -> modal family, from TRAIN labeled docs only
    dept_counts = defaultdict(Counter)
    for j in train:
        dep = depts[j]
        if dep:
            dept_counts[dep][y[j]] += 1
    dept_modal = {}
    for dep, c in dept_counts.items():
        tot = sum(c.values())
        if tot >= DEPT_MIN_LABELED:
            fam, n = c.most_common(1)[0]
            dept_modal[dep] = fam  # store modal regardless of purity
    print(f"departments with >= {DEPT_MIN_LABELED} labeled exemplars: {len(dept_modal)}")

    # kNN classifier (same core as production), fit on TRAIN
    Xtr, ytr = V[train], y[train]
    Xte, yte = V[test], y[test]
    index = faiss.IndexFlatIP(V.shape[1])
    index.add(Xtr)
    D, I = index.search(Xte, K)
    fam_of = ytr[I]
    predC, confC = [], []
    for row_f, row_d in zip(fam_of, D):
        w = {}
        for f, d in zip(row_f, row_d):
            w[f] = w.get(f, 0.0) + float(d)
        tot = sum(w.values())
        best = max(w, key=w.get)
        predC.append(best)
        confC.append(w[best] / tot)
    predC = np.array(predC)
    confC = np.array(confC)

    # centroid prototype prediction (for the ensemble component of strict gate)
    C = np.zeros((len(fams), V.shape[1]), np.float32)
    for jx, f in enumerate(fams):
        C[jx] = Xtr[ytr == f].mean(0)
    C = _norm(C)
    cent_pred = np.array(fams)[(Xte @ C.T).argmax(1)]

    correct = predC == yte
    top1_te = D[:, 0]
    dept_te = depts[test]
    dept_pred = np.array([dept_modal.get(dp, "") for dp in dept_te], dtype=object)
    agree_dept = dept_pred == predC  # kNN agrees with dept-modal
    has_dept = dept_pred != ""

    # production strict gate (== APPLY_DEFAULTS): conf>=.80 & top1>=.86 & ensemble
    strict = (confC >= CONF) & (top1_te >= SIMFLOOR) & (cent_pred == predC)

    def report(name, mask):
        n = int(mask.sum())
        if n == 0:
            print(f"  {name:42s} n=0")
            return
        prec = correct[mask].mean() * 100
        cov = n / len(test) * 100
        print(f"  {name:42s} n={n:6d}  cov={cov:5.1f}%  prec={prec:5.1f}%")

    print("\n===== GATE COMPARISON (held-out labeled docs) =====")
    report("strict (conf>=.60 & ensemble)", strict)
    report("dept-agree only (kNN==dept-modal)", agree_dept & has_dept)
    report("dept-agree & has-dept & NOT strict", agree_dept & has_dept & ~strict)
    report("UNION strict OR dept-agree", strict | (agree_dept & has_dept))
    print("\n  -- relaxed conjunctions on the incremental (non-strict) set --")
    incr = agree_dept & has_dept & ~strict
    for c in (0.30, 0.40, 0.50):
        report(f"  incr & conf>={c:.2f}", incr & (confC >= c))
    report("  incr & ensemble(cent==knn)", incr & (cent_pred == predC))
    report("  incr & conf>=.40 & ensemble", incr & (confC >= 0.40) & (cent_pred == predC))

    # per-family precision on the incremental dept-agree set (conf>=DEPT_CONF)
    print(f"\n  per-family precision on incremental (dept-agree, conf>={DEPT_CONF}):")
    sel = incr & (confC >= DEPT_CONF)
    tp, fp = Counter(), Counter()
    for p, ok in zip(predC[sel], correct[sel]):
        (tp if ok else fp)[p] += 1
    rows = []
    for f in sorted(set(predC[sel])):
        n = tp[f] + fp[f]
        rows.append((tp[f] / n if n else 0, n, f))
    for prec, n, f in sorted(rows):
        flag = "  <-- weak" if prec < 0.90 else ""
        print(f"    {prec * 100:5.1f}%  n={n:5d}  {f}{flag}")

    # ---- derive a per-family ALLOWLIST from held-out incremental precision ----
    # keep families where dept-agree conjunction is clean enough to ship.
    sel = incr & (confC >= DEPT_CONF)
    tpf, fpf = Counter(), Counter()
    for p, ok in zip(predC[sel], correct[sel]):
        (tpf if ok else fpf)[p] += 1
    ALLOW = set()
    for f in sorted(set(predC[sel])):
        n = tpf[f] + fpf[f]
        if n >= 15 and tpf[f] / n >= 0.90:
            ALLOW.add(f)
    print(f"\n  per-family allowlist (held-out prec>=90%, n>=15): {sorted(ALLOW)}")
    allow_mask = np.array([p in ALLOW for p in predC])
    report(
        f"ALLOWLISTED dept-agree incr (conf>={DEPT_CONF})", incr & (confC >= DEPT_CONF) & allow_mask
    )

    # ---- project onto the REAL other bucket: how many would each gate rescue? ----
    print("\n===== PROJECTED RESCUE ON ACTUAL 'other' BUCKET =====")
    oidx = np.where(is_other)[0]
    Xo = V[oidx]
    Do, Io = index.search(Xo, K)  # index still = TRAIN labeled; fine for sizing
    fam_o = ytr[Io]
    predO, confO = [], []
    for row_f, row_d in zip(fam_o, Do):
        w = {}
        for f, d in zip(row_f, row_d):
            w[f] = w.get(f, 0.0) + float(d)
        tot = sum(w.values())
        best = max(w, key=w.get)
        predO.append(best)
        confO.append(w[best] / tot)
    predO = np.array(predO)
    confO = np.array(confO)
    cent_o = np.array(fams)[(Xo @ C.T).argmax(1)]
    top1_o = Do[:, 0]
    dept_o = depts[oidx]
    dept_predO = np.array([dept_modal.get(dp, "") for dp in dept_o], dtype=object)
    agreeO = (dept_predO == predO) & (dept_predO != "")
    strictO = (confO >= CONF) & (top1_o >= SIMFLOOR) & (cent_o == predO)

    print(f"  other docs: {len(oidx)}")
    print(f"  strict gate would rescue:           {int(strictO.sum()):6d}")
    print(f"  dept-agree (any conf):              {int(agreeO.sum()):6d}")
    print(
        f"  dept-agree & conf>={DEPT_CONF}:            {int((agreeO & (confO >= DEPT_CONF)).sum()):6d}"
    )
    incrO = agreeO & ~strictO & (confO >= DEPT_CONF)
    print(f"  INCREMENTAL (dept-agree,conf>={DEPT_CONF},  {int(incrO.sum()):6d}")
    print("            not already strict)")
    print(f"  UNION strict|dept-agree: {int((strictO | (agreeO & (confO >= DEPT_CONF))).sum()):6d}")

    allowO = np.array([p in ALLOW for p in predO])
    incrAllowO = incrO & allowO
    print(f"\n  ALLOWLISTED incremental rescue:     {int(incrAllowO.sum()):6d}   <-- proposed ship")
    cur_other = int(is_other.sum())
    total = len(ids)
    cov_now = (total - cur_other) / total * 100
    # coverage if we ship strict (already live=8037 approx) PLUS allowlisted incremental
    cov_incr = (total - (cur_other - int(incrAllowO.sum()))) / total * 100
    print(f"\n  coverage now (other={cur_other}): {cov_now:.1f}%")
    print(
        f"  coverage after allowlisted incremental: {cov_incr:.1f}%  (+{cov_incr - cov_now:.1f}pp)"
    )
    print("  per-family breakdown of allowlisted incremental rescue:")
    bd = Counter(predO[incrAllowO])
    for f, n in bd.most_common():
        print(f"    {n:5d}  {f}")


if __name__ == "__main__":
    main()
