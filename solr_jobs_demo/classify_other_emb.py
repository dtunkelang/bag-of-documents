#!/usr/bin/env python3
"""Embedding-based classifier for the role_family:other residual.

Uses the cached e5-small-v2 catalog vectors (title + description, passage-side),
positionally aligned to doc_ids.json. Three candidate methods, bake-off on a
held-out split of the heuristic-labeled docs:

  A  text-query prototypes  -- cosine to e5 "query: <family description>"
  B  centroid prototypes    -- cosine to mean train vector per family
  C  knn vote               -- similarity-weighted vote of k nearest train docs

Reports argmax accuracy and the precision/coverage tradeoff (accuracy among the
most-confident X%). --apply writes the confident predictions for `other` docs.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "unified_jobs_daily"
EMBED_MODEL = "intfloat/e5-small-v2"

# Production operating point that reproduces role_family_emb_overrides.json
# (3,614 docs, ~96% manual-audited precision). Both the CLI --apply path and the
# refresh.py stage call classify_other() with these defaults; changing them
# changes the live label set, so keep them in sync with the committed overrides.
APPLY_DEFAULTS = dict(conf=0.80, simfloor=0.86, k=25, ensemble=True)

# ---- precision guards (shared by --apply CLI and the refresh stage) ----
# junk/evergreen/test postings have no real role -> never classify.
JUNK = re.compile(
    r"\b(talent network|talent community|talent pool|join our talent|"
    r"general application|open application|spontaneous application|"
    r"future opportunit|resume upload|upload your resume|don'?t see|"
    r"dream job|your own role|pitch your|test job|expression of interest|"
    r"general interest|other roles|future openings)\b",
    re.I,
)
# English-only (respects the exclude-multilingual policy): require a minimum
# density of high-frequency English function words in the description.
EN = set(
    "the and to of for with you we our your a an in on at is are be as that this "
    "will work team role job they have from or by".split()
)
# fitness/wellness studios cluster tightly near education_teaching but aren't
# teaching roles and have no home family -> leave in other.
FITNESS = re.compile(
    r"\b(jetset|pilates|yoga|barre|cycling|cycle bar|spin studio|"
    r"fitness studio|studio lead|studio manager|crossfit|workout)\b",
    re.I,
)
# cluster vetoes: titles e5 reliably misroutes -> keep in other (or relabel).
VETO = re.compile(
    r"\b(board certified behavior analyst|bcba|medical instrument tech|"
    r"go[ -]?to[ -]?market)\b",
    re.I,
)


def is_english(desc: str) -> bool:
    toks = desc.lower().split()[:120]
    if len(toks) < 8:
        return True  # too short to judge; don't penalize
    hits = sum(1 for t in toks if t.strip(".,;:()") in EN)
    return hits >= 4


def load_text(metadata_path: Path | str) -> dict[str, tuple[str, str]]:
    """{doc_id: (title, description)} for the precision-guard regexes."""
    txt: dict[str, tuple[str, str]] = {}
    with open(metadata_path) as f:
        for line in f:
            r = json.loads(line)
            txt[r["id"]] = ((r.get("title") or ""), (r.get("description") or ""))
    return txt


def classify_other(
    ids: list[str],
    V: np.ndarray,
    y: np.ndarray,
    txt: dict[str, tuple[str, str]],
    *,
    conf: float = 0.80,
    simfloor: float = 0.86,
    k: int = 25,
    ensemble: bool = True,
) -> tuple[dict[str, str], list, dict]:
    """Ensemble-gated e5 kNN classification of role_family:'other' docs.

    V must be L2-normalized and positionally aligned to `ids`; `y` holds the
    heuristic role_family per position ('other' for the residual). Builds a FAISS
    kNN over the labeled docs, similarity-weighted-votes a family for each 'other'
    doc, gates on vote-share (conf) + top-1 cosine (simfloor) + centroid agreement
    (ensemble) + the JUNK/English/FITNESS/VETO guards. Returns
    (predictions{id: family}, dropped[(reason, fam, title)], cnt{reason: n}).
    """
    import faiss

    is_other = y == "other"
    lab_idx = np.where(~is_other)[0]
    fams = sorted(set(y[lab_idx]))
    oth = np.where(is_other)[0]
    Xo = V[oth]
    Xall, yall = V[lab_idx], y[lab_idx]

    full = faiss.IndexFlatIP(V.shape[1])
    full.add(Xall)
    D, idx = full.search(Xo, k)
    famI = yall[idx]
    top1 = D[:, 0]
    pred, conf_arr = [], []
    for row_f, row_d in zip(famI, D):
        w: dict[str, float] = {}
        for f, d in zip(row_f, row_d):
            w[f] = w.get(f, 0.0) + float(d)
        tot = sum(w.values())
        b = max(w, key=w.get)
        pred.append(b)
        conf_arr.append(w[b] / tot)
    pred, conf_arr = np.array(pred), np.array(conf_arr)

    # ensemble second opinion: centroid prediction on the full labeled set
    Cfull = np.zeros((len(fams), V.shape[1]), np.float32)
    for j, f in enumerate(fams):
        Cfull[j] = Xall[yall == f].mean(0)
    Cfull = _norm(Cfull)
    cent_pred = np.array(fams)[(Xo @ Cfull.T).argmax(1)]
    agree = pred == cent_pred

    keep = (conf_arr >= conf) & (top1 >= simfloor)
    out: dict[str, str] = {}
    dropped: list = []  # (reason, family, title) for characterization
    cnt = {"junk": 0, "lang": 0, "fitness": 0, "veto": 0, "disagree": 0}
    for i in range(len(oth)):
        if not keep[i]:
            continue
        did = ids[oth[i]]
        t, d = txt.get(did, ("", ""))
        tt = (t or "").splitlines()[0][:55]
        if JUNK.search(t):
            cnt["junk"] += 1
            continue
        if not is_english(d):
            cnt["lang"] += 1
            continue
        if VETO.search(t):
            cnt["veto"] += 1
            dropped.append(("veto", pred[i], tt))
            continue
        if pred[i] == "education_teaching" and FITNESS.search(t + " " + d[:200]):
            cnt["fitness"] += 1
            continue
        if ensemble and not agree[i]:
            cnt["disagree"] += 1
            dropped.append(("disagree", f"{pred[i]}!={cent_pred[i]}", tt))
            continue
        out[did] = pred[i]
    return out, dropped, cnt


# Natural-language query per family (doc side was "passage: "; queries use "query: ").
# Extends mine_all_families.FAMILIES with the two it omits (finance, annotation).
FAM_QUERIES = {
    "software_engineering": "software engineer developer programmer backend frontend full stack",
    "sales": "sales account executive business development quota revenue closing deals",
    "marketing": "marketing brand content social media demand generation growth campaigns",
    "healthcare_clinical": "nurse physician clinical therapist medical doctor patient care",
    "skilled_trades_construction": "electrician plumber carpenter construction welder hvac technician",
    "customer_success_support": "customer success support help desk client services onboarding",
    "operations_admin": "operations administrative office manager executive assistant coordinator",
    "ai_ml": "machine learning artificial intelligence deep learning LLM model research engineer",
    "devops_sre_infra": "devops site reliability infrastructure platform cloud kubernetes",
    "product_management": "product manager product owner roadmap product strategy",
    "hr_people_ops": "human resources recruiter talent acquisition people operations",
    "project_program_management": "project manager program manager PMO scrum delivery",
    "security": "security cybersecurity information security SOC analyst threat",
    "creative_content": "content writer copywriter editor creative producer video",
    "design_ux": "designer UX UI product design graphic design researcher",
    "transportation_logistics": "logistics supply chain warehouse driver transportation fleet",
    "legal": "attorney lawyer legal counsel paralegal compliance contracts",
    "education_teaching": "teacher instructor professor tutor education faculty curriculum",
    "data_engineering": "data engineer ETL pipeline data platform warehouse",
    "data_science_ml": "data scientist machine learning statistics modeling experimentation",
    "consulting_strategy": "consultant strategy management consulting advisory transformation",
    "retail": "retail store associate sales floor merchandising cashier",
    "food_service_hospitality": "restaurant chef cook server hospitality hotel barista",
    "data_analytics": "data analyst business intelligence reporting dashboard BI insights",
    "manufacturing_production": "manufacturing production assembly machine operator plant quality",
    "healthcare_allied": "home health aide caregiver personal care direct support worker",
    "research_academic": "research scientist postdoc academic laboratory principal investigator",
    "nonprofit_social_services": "nonprofit social worker case manager community outreach program",
    "public_safety": "police firefighter security officer emergency dispatcher corrections",
    "healthcare_admin": "medical billing healthcare administration patient access medical records",
    "finance_accounting": "finance accounting accountant financial analyst controller audit tax",
    "ai_data_annotation": "ai trainer data annotation labeling freelance language specialist transcription rater",
}


def _norm(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    x /= np.linalg.norm(x, axis=1, keepdims=True) + 1e-9
    return x


def load():
    ids = json.load(open(DATA / "doc_ids.json"))
    lab = json.load(open(DATA / "role_labels.json"))
    V = np.load(DATA / "e5_small_catalog.vecs.fp16.npy")
    V = _norm(V)
    y = np.array([lab.get(i, "other") for i in ids])
    return ids, V, y


def encode_queries(fams):
    from sentence_transformers import SentenceTransformer

    m = SentenceTransformer(EMBED_MODEL)
    texts = [f"query: {FAM_QUERIES[f]}" for f in fams]
    Q = m.encode(texts, normalize_embeddings=True, batch_size=32)
    return np.asarray(Q, dtype=np.float32)


def calib_table(conf, correct, name):
    order = np.argsort(-conf)
    correct = correct[order]
    n = len(correct)
    print(f"\n  {name}: argmax acc {correct.mean() * 100:.1f}%  (n={n})")
    print(f"    {'accept top':>10} {'coverage':>9} {'precision':>10} {'conf>=':>8}")
    for cov in (1.0, 0.8, 0.6, 0.5, 0.4, 0.3, 0.2):
        k = int(n * cov)
        acc = correct[:k].mean() * 100
        thr = conf[order][k - 1]
        print(f"    {cov * 100:9.0f}% {k:9d} {acc:9.1f}% {thr:8.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    ap.add_argument(
        "--method",
        default="knn",
        choices=["A", "B", "C", "knn"],
        help="bake-off label only; --apply always uses the knn core",
    )
    ap.add_argument(
        "--conf", type=float, default=APPLY_DEFAULTS["conf"], help="min vote-share to assign"
    )
    ap.add_argument(
        "--simfloor",
        type=float,
        default=APPLY_DEFAULTS["simfloor"],
        help="min top-1 neighbor cosine",
    )
    ap.add_argument("--k", type=int, default=APPLY_DEFAULTS["k"])
    ap.add_argument(
        "--ensemble",
        action="store_true",
        default=APPLY_DEFAULTS["ensemble"],
        help="require knn==centroid agreement",
    )
    args = ap.parse_args()

    ids, V, y = load()
    is_other = y == "other"
    lab_idx = np.where(~is_other)[0]
    fams = sorted(set(y[lab_idx]))
    print(f"labeled {len(lab_idx)}  other {is_other.sum()}  families {len(fams)}")

    rng = np.random.default_rng(0)
    perm = rng.permutation(lab_idx)
    ntest = int(len(perm) * 0.15)
    test, train = perm[:ntest], perm[ntest:]
    Xtr, ytr = V[train], y[train]
    Xte, yte = V[test], y[test]

    import faiss

    # ---- A: text-query prototypes ----
    Q = encode_queries(fams)  # (F,384)
    simA = Xte @ Q.T
    predA = np.array(fams)[simA.argmax(1)]
    confA = simA.max(1)
    # ---- B: centroid prototypes ----
    C = np.zeros((len(fams), V.shape[1]), np.float32)
    for j, f in enumerate(fams):
        C[j] = Xtr[ytr == f].mean(0)
    C = _norm(C)
    simB = Xte @ C.T
    predB = np.array(fams)[simB.argmax(1)]
    confB = simB.max(1)
    # ---- C: knn vote ----
    index = faiss.IndexFlatIP(V.shape[1])
    index.add(Xtr)
    D, I = index.search(Xte, args.k)
    fam_of = ytr[I]  # (ntest,k) family strings
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

    print("\n===== HELD-OUT BAKE-OFF (test on heuristic labels) =====")
    calib_table(confA, (predA == yte), "A text-query prototypes")
    calib_table(confB, (predB == yte), "B centroid prototypes")
    calib_table(confC, (predC == yte), f"C knn vote (k={args.k})")

    # per-family precision for the winner (C), among predictions clearing the gate
    gate = (confC >= args.conf) & (D[:, 0] >= args.simfloor if args.simfloor > 0 else True)
    print(f"\n  per-family precision (C, conf>={args.conf} simfloor>={args.simfloor}):")
    import collections

    tp = collections.Counter()
    fp = collections.Counter()
    for p, t, g in zip(predC, yte, gate):
        if not g:
            continue
        (tp if p == t else fp)[p] += 1
    rows = []
    for f in sorted(set(predC[gate])):
        n = tp[f] + fp[f]
        rows.append((tp[f] / n if n else 0, n, f))
    for prec, n, f in sorted(rows):
        flag = "  <-- weak" if prec < 0.90 else ""
        print(f"    {prec * 100:5.1f}%  n={n:5d}  {f}{flag}")

    if not args.apply:
        print("\n(eval only — re-run with --apply --method --conf to write predictions)")
        return

    # ---- APPLY to other docs (production = knn core, shared with refresh stage) ----
    # Uses ALL labeled docs as reference, not just the 85% train split.
    txt = load_text(DATA / "metadata.jsonl")
    out, dropped, cnt = classify_other(
        ids, V, y, txt, conf=args.conf, simfloor=args.simfloor, k=args.k, ensemble=args.ensemble
    )
    json.dump(out, open(DATA / "other_emb_predictions.json", "w"))
    json.dump(dropped, open(DATA / "other_emb_dropped.json", "w"))
    import collections

    c = collections.Counter(out.values())
    print(
        f"\nAPPLY conf>={args.conf} simfloor>={args.simfloor} ensemble={args.ensemble}: "
        f"{len(out)} kept (dropped {cnt})"
    )
    for f, n in c.most_common():
        print(f"  {n:6d} {f}")


if __name__ == "__main__":
    main()
