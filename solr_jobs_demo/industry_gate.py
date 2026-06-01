#!/usr/bin/env python3
"""Industry agreement-gate: rescue ensemble-DISAGREEMENT 'other' docs.

The embedding classifier (classify_other_emb.py) drops ~1.5k 'other' docs where
the kNN vote and the centroid prototype disagree on the family. Many of those are
genuinely ambiguous, but some are real misses where one signal is right. This adds
a THIRD independent signal -- the modal role_family of the doc's *industry* -- and
rescues a disagreement only when the high-confidence kNN pick is corroborated by a
high-purity industry-modal.

Two independent signals (embedding kNN + industry-modal) agreeing should be high
precision. This script first VALIDATES that on a held-out split before applying.

  --apply   write the rescued predictions to other_emb_industry_predictions.json

Mapping note: doc_ids/role_labels/metadata use source ids ("ashby:..."); the
industry file is keyed by the numeric Solr id = stable_id(source_id).
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import re
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "unified_jobs_daily"


def stable_id(doc_id: str) -> int:
    h = hashlib.blake2b(doc_id.encode("utf-8"), digest_size=7).digest()
    return int.from_bytes(h, "big") & ((1 << 52) - 1)


def _norm(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    x /= np.linalg.norm(x, axis=1, keepdims=True) + 1e-9
    return x


def knn_pred(index, Xq, ytr_ref, k):
    """similarity-weighted vote -> (pred, conf=vote-share, top1 cosine)."""
    D, idx = index.search(Xq, k)
    fam_of = ytr_ref[idx]
    pred, conf = [], []
    for row_f, row_d in zip(fam_of, D):
        w: dict[str, float] = {}
        for f, d in zip(row_f, row_d):
            w[f] = w.get(f, 0.0) + float(d)
        tot = sum(w.values())
        b = max(w, key=w.get)
        pred.append(b)
        conf.append(w[b] / tot)
    return np.array(pred), np.array(conf), D[:, 0]


def centroids(Xtr, ytr, fams, dim):
    C = np.zeros((len(fams), dim), np.float32)
    for j, f in enumerate(fams):
        C[j] = Xtr[ytr == f].mean(0)
    return _norm(C)


def industry_modal(ind_of, y_ref_mask_idx, y_ref, ids):
    """modal role_family + purity per industry, computed over the given ref docs.
    ind_of: dict source_id -> industry string. Returns {industry: (fam, purity, n)}.
    """
    by_ind: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    for i in y_ref_mask_idx:
        sid = ids[i]
        ind = ind_of.get(sid)
        if ind and ind != "unclassified":
            by_ind[ind][y_ref[i]] += 1
    out = {}
    for ind, c in by_ind.items():
        n = sum(c.values())
        fam, cnt = c.most_common(1)[0]
        out[ind] = (fam, cnt / n, n)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--k", type=int, default=25)
    ap.add_argument("--conf", type=float, default=0.85, help="min kNN vote-share")
    ap.add_argument("--simfloor", type=float, default=0.80, help="min top-1 cosine")
    ap.add_argument("--purity", type=float, default=0.55, help="min industry-modal purity")
    ap.add_argument("--minind", type=int, default=200, help="min docs in industry to trust modal")
    args = ap.parse_args()

    ids = json.load(open(DATA / "doc_ids.json"))
    lab = json.load(open(DATA / "role_labels.json"))
    ind_of = {}
    raw_ind = json.load(open(DATA / "solr_industry.json"))
    sid2src = {}
    for s in ids:
        sid2src[str(stable_id(s))] = s
    for k, v in raw_ind.items():
        src = sid2src.get(k)
        if src is not None:
            ind_of[src] = v

    V = _norm(np.load(DATA / "e5_small_catalog.vecs.fp16.npy"))
    y = np.array([lab.get(i, "other") for i in ids])
    is_other = y == "other"
    lab_idx = np.where(~is_other)[0]
    fams = sorted(set(y[lab_idx]))
    fam_arr = np.array(fams)

    import faiss

    # ---------- HELD-OUT VALIDATION ----------
    rng = np.random.default_rng(0)
    perm = rng.permutation(lab_idx)
    ntest = int(len(perm) * 0.15)
    test, train = perm[:ntest], perm[ntest:]

    idx_tr = faiss.IndexFlatIP(V.shape[1])
    idx_tr.add(V[train])
    kp, kc, kt = knn_pred(idx_tr, V[test], y[train], args.k)
    C = centroids(V[train], y[train], fams, V.shape[1])
    cp = fam_arr[(V[test] @ C.T).argmax(1)]

    imod_tr = industry_modal(ind_of, train, y, ids)

    disagree = (kp != cp) & (kc >= args.conf) & (kt >= args.simfloor)
    print(f"=== HELD-OUT (test n={len(test)}) ===")
    print(f"  disagreements clearing conf/simfloor gates: {disagree.sum()}")

    # baseline: if we trusted kNN on ALL disagreements (no industry gate)
    base = kp[disagree] == y[test][disagree]
    if base.size:
        print(
            f"  kNN-pred precision on raw disagreements: {base.mean() * 100:.1f}% (n={base.size})"
        )

    # industry-gate: rescue where industry-modal == kNN pred, on high-purity industry
    rescued, correct = [], []
    for i in np.where(disagree)[0]:
        sid = ids[test[i]]
        ind = ind_of.get(sid)
        m = imod_tr.get(ind) if ind else None
        if not m:
            continue
        fam, pur, n = m
        if pur < args.purity or n < args.minind:
            continue
        if fam == kp[i]:
            rescued.append(i)
            correct.append(kp[i] == y[test][i])
    correct = np.array(correct)
    print(
        f"  industry-gate (purity>={args.purity} minind>={args.minind}): "
        f"rescued {len(rescued)}  precision "
        f"{correct.mean() * 100:.1f}%"
        if correct.size
        else "  industry-gate: rescued 0"
    )
    if correct.size:
        pf = collections.Counter()
        nf = collections.Counter()
        for i, ok in zip(rescued, correct):
            (pf if ok else nf)[kp[i]] += 1
        print("  per-family (rescued held-out):")
        for f in sorted(set(kp[i] for i in rescued)):
            n = pf[f] + nf[f]
            print(f"    {pf[f] / n * 100:5.1f}%  n={n:4d}  {f}")

    if not args.apply:
        print("\n(eval only -- re-run with --apply to write predictions)")
        return

    # ---------- APPLY to real 'other' docs ----------
    oth = np.where(is_other)[0]
    idx_all = faiss.IndexFlatIP(V.shape[1])
    idx_all.add(V[lab_idx])
    kp, kc, kt = knn_pred(idx_all, V[oth], y[lab_idx], args.k)
    C = centroids(V[lab_idx], y[lab_idx], fams, V.shape[1])
    cp = fam_arr[(V[oth] @ C.T).argmax(1)]
    imod = industry_modal(ind_of, lab_idx, y, ids)

    # exclude docs already rescued by the base ensemble (those AGREE -> not here),
    # so this only adds NEW labels on the disagreement set.
    base_kept = set(json.load(open(DATA / "other_emb_predictions.json")))

    JUNK = re.compile(
        r"\b(talent network|talent community|talent pool|join our talent|"
        r"general application|open application|spontaneous application|"
        r"future opportunit|resume upload|upload your resume|don'?t see|"
        r"dream job|your own role|pitch your|test job|expression of interest|"
        r"general interest|other roles|future openings)\b",
        2,
    )
    VETO = re.compile(
        r"\b(board certified behavior analyst|bcba|medical instrument tech|"
        r"go[ -]?to[ -]?market)\b",
        2,
    )
    txt = {}
    with open(DATA / "metadata.jsonl") as f:
        for line in f:
            r = json.loads(line)
            txt[r["id"]] = (r.get("title") or "", r.get("description") or "")

    out = {}
    cnt = {"agree_in_base": 0, "junk": 0, "veto": 0}
    for i in range(len(oth)):
        if kp[i] == cp[i]:
            continue  # not a disagreement
        if kc[i] < args.conf or kt[i] < args.simfloor:
            continue
        did = ids[oth[i]]
        if did in base_kept:
            cnt["agree_in_base"] += 1
            continue
        ind = ind_of.get(did)
        m = imod.get(ind) if ind else None
        if not m:
            continue
        fam, pur, n = m
        if pur < args.purity or n < args.minind:
            continue
        if fam != kp[i]:
            continue
        t, _ = txt.get(did, ("", ""))
        if JUNK.search(t):
            cnt["junk"] += 1
            continue
        if VETO.search(t):
            cnt["veto"] += 1
            continue
        out[did] = kp[i]

    json.dump(out, open(DATA / "other_emb_industry_predictions.json", "w"))
    c = collections.Counter(out.values())
    print(f"\nAPPLY: rescued {len(out)} (dropped {cnt})")
    for f, n in c.most_common():
        print(f"  {n:6d} {f}")


if __name__ == "__main__":
    main()
