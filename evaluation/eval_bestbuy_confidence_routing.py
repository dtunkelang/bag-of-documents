#!/usr/bin/env python3
"""Can a serve-time confidence signal tell us WHICH of base/BoD to trust per query?

`eval_bestbuy_base_vs_bod_divergence.py` established the setup this script
builds on: on the re-indexed BestBuy catalog, base MiniLM and the BoD
fine-tune agree on top-10 only 34% of the time (mean Jaccard 0.18). On the 166
divergent queries BoD wins categorical relevance 74% of the time and click
accuracy 80% of the time -- but it loses a real minority, with an identified
failure mode: multi-word natural-language queries whose head word collides with
a media/CD/DVD title ("balance board", "a skylit drive", "laptops for
teachers") get dragged into the music catalog by BoD's click-training prior.

Both models are tiny. Running BOTH at serve time is cheap. So the question is
whether an *unsupervised, ground-truth-free* per-query signal predicts which
model to serve -- classic query-performance-prediction (QPP) territory. The
candidate signals, computed per model per query from its own top-10:

  top1_sim        rank-1 cosine to the query (absolute confidence)
  mean_top10_sim  mean cosine of the top-10 (bulk confidence)
  gap12           top1 - top2 (the classic QPP "is there a clear winner" gap)
  gap1_mean       top1 - mean(top10)  (normalised version of the same idea)
  std_top10       spread of the top-10 scores
  coherence       mean PAIRWISE cosine among the top-10 result VECTORS -- not
                  their similarity to the query, but how tightly clustered the
                  results are with each other. A scattered result set is the
                  natural "the model is confused" tell.
  centroid_sim    mean cosine of each result to the top-10 centroid (the
                  monotone twin of coherence; reported for completeness)
  media_share     share of the top-10 whose title ends in a media suffix
                  ("- CD", "- DVD", ...). Not a QPP statistic -- it is a direct
                  serve-time probe of the KNOWN BoD failure mode, included so
                  the generic signals can be judged against a targeted one.

Plus two free query-side (pre-retrieval) features: token count and char length.

Cross-model comparison is scale-confounded -- the two encoders do not put their
cosines on the same scale (BoD's contrastive fine-tune inflates them). Every
"pick the model with higher X" rule is therefore evaluated twice: on raw values
and on per-model z-scores (standardised across the 250 queries within each
model), which removes the constant offset and asks the honest question -- is
THIS query unusually confident *for this model*?

Nothing here costs API budget. Win/loss labels, per-query junk@10, R@10, nDCG,
E@1 and hit@10 for both arms are read straight out of
evaluation/results/bestbuy_base_vs_bod_divergence.json (whose judge phase was
itself a $0 cache hit). Retrieval scores come from the cached
/tmp/bestbuy_reindex_output/retrieval_bestbuy.json; only the coherence features
touch the catalog vectors, and only for the 2,500 rows per model that are
actually in a top-10 (mmap gather, no full 1.27M-row pass).

Phases (cached to --work-dir, resumable):
  features  per-model per-query QPP + coherence features (mmap vector gather)
  analyze   correlations, BoD-wins-vs-losses comparison, routing payoff test

Usage:
  python evaluation/eval_bestbuy_confidence_routing.py --phase features
  python evaluation/eval_bestbuy_confidence_routing.py --phase analyze
  python evaluation/eval_bestbuy_confidence_routing.py --phase all
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np  # noqa: E402

K_EVAL = 10

DEFAULT_RETRIEVAL = "/tmp/bestbuy_reindex_output/retrieval_bestbuy.json"
DEFAULT_DIVERGENCE = "evaluation/results/bestbuy_base_vs_bod_divergence.json"
DEFAULT_OUT = "evaluation/results/bestbuy_confidence_routing.json"
NEW_VECS = {
    "base": "/tmp/bestbuy_reindex_output/artifacts/base_catalog.vecs.fp16.npy",
    "bod": "/tmp/bestbuy_reindex_output/artifacts/bod_catalog.vecs.fp16.npy",
}

ARMS = {"base": "base_new", "bod": "bod_new"}

# Titles in this catalog are "<name> - <format>"; these are the formats that
# make up the media/music/video shelf BoD's click prior over-fires on.
MEDIA_SUFFIXES = {
    "cd",
    "dvd",
    "blu-ray",
    "blu-ray disc",
    "4k ultra hd blu-ray",
    "vinyl lp",
    "lp",
    "sacd",
    "cassette",
    "vhs",
    "cd/dvd",
    "dvd/blu-ray",
    "super audio cd",
}

# Features computed per model per query. Order is the reporting order.
FEATURES = (
    "top1_sim",
    "mean_top10_sim",
    "gap12",
    "gap1_mean",
    "std_top10",
    "coherence",
    "centroid_sim",
    "media_share",
)
# Query-side features -- one value per query, not per model. They cannot drive a
# "pick the higher one" rule, only a "when the query looks like X, prefer base".
QUERY_FEATURES = ("n_tokens", "n_chars")


# --------------------------------------------------------------------------
# paths
# --------------------------------------------------------------------------
def work_paths(work_dir, tag):
    w = Path(work_dir)
    w.mkdir(parents=True, exist_ok=True)
    return {"root": w, "features": w / f"features_{tag}.json"}


# --------------------------------------------------------------------------
# phase: features
# --------------------------------------------------------------------------
def _is_media(title):
    if not title or " - " not in title:
        return False
    return title.rsplit(" - ", 1)[-1].strip().lower() in MEDIA_SUFFIXES


def _score_features(docs, k):
    """QPP statistics over the retrieval scores alone."""
    sims = np.asarray([d["sim"] for d in docs[:k]], dtype=np.float64)
    top1 = float(sims[0])
    top2 = float(sims[1]) if sims.size > 1 else float(sims[0])
    mean = float(sims.mean())
    return {
        "top1_sim": top1,
        "mean_top10_sim": mean,
        "gap12": top1 - top2,
        "gap1_mean": top1 - mean,
        "std_top10": float(sims.std(ddof=0)),
    }


def _coherence(vecs):
    """Mean pairwise cosine among the top-k result vectors + centroid version.

    Vectors are already L2-normalised in the artifact, but renormalise anyway --
    they are stored fp16 and the round-trip drifts the norm by ~2e-5.
    """
    v = vecs / np.clip(np.linalg.norm(vecs, axis=1, keepdims=True), 1e-8, None)
    n = v.shape[0]
    gram = v @ v.T
    off = (gram.sum() - np.trace(gram)) / max(n * (n - 1), 1)
    centroid = v.mean(axis=0)
    cn = float(np.linalg.norm(centroid))
    centroid_sim = float((v @ (centroid / max(cn, 1e-8))).mean())
    return float(off), centroid_sim


def phase_features(args, paths):
    with open(args.retrieval) as f:
        payload = json.load(f)
    holdout = [r for r in payload["rows"] if not r["is_manual"]]
    print(f"  {len(holdout)} holdout queries from {args.retrieval}", flush=True)

    vecs = {}
    for short, path in NEW_VECS.items():
        p = Path(path)
        if not p.exists():
            raise SystemExit(f"missing catalog vectors {p} -- re-run the re-index embed phase")
        vecs[short] = np.load(p, mmap_mode="r")
        print(f"  {short}: mmap {vecs[short].shape} {vecs[short].dtype}", flush=True)

    rows = []
    for r in holdout:
        rec = {
            "key": r["key"],
            "query": r["query"],
            "n_tokens": len(r["query"].split()),
            "n_chars": len(r["query"]),
        }
        for short, arm in ARMS.items():
            docs = r[arm][: args.top_k]
            feats = _score_features(docs, args.top_k)
            idx = [d["row"] for d in docs]
            v = np.asarray(vecs[short][idx], dtype=np.float32)
            coh, cent = _coherence(v)
            feats["coherence"] = coh
            feats["centroid_sim"] = cent
            feats["media_share"] = float(np.mean([_is_media(d.get("title_old", "")) for d in docs]))
            for name, val in feats.items():
                rec[f"{short}_{name}"] = val
        rows.append(rec)

    out = {
        "retrieval_source": args.retrieval,
        "base_model": payload["base_model"],
        "bod_model": payload["bod_model"],
        "catalog_size": payload["catalog_size"],
        "top_k": args.top_k,
        "n_queries": len(rows),
        "features": list(FEATURES),
        "query_features": list(QUERY_FEATURES),
        "coherence_definition": (
            "mean off-diagonal cosine among the top-10 result vectors "
            "(same encoder that retrieved them)"
        ),
        "rows": rows,
    }
    with open(paths["features"], "w") as f:
        json.dump(out, f, indent=2)
    for name in FEATURES:
        b = np.mean([x[f"base_{name}"] for x in rows])
        d = np.mean([x[f"bod_{name}"] for x in rows])
        print(f"    {name:16s} base {b: .4f}   bod {d: .4f}", flush=True)
    print(f"saved -> {paths['features']}", flush=True)
    return out


# --------------------------------------------------------------------------
# stats helpers (numpy only -- matching the thin-venv convention in this repo)
# --------------------------------------------------------------------------
def _paired_delta_ci(a, b, n_boot=2000, seed=0):
    """Paired bootstrap over queries on mean(a) - mean(b)."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    if a.size == 0:
        return float("nan"), (None, None)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, a.size, size=(n_boot, a.size))
    d = a[idx].mean(axis=1) - b[idx].mean(axis=1)
    return float(a.mean() - b.mean()), (
        float(np.percentile(d, 2.5)),
        float(np.percentile(d, 97.5)),
    )


def _unpaired_delta_ci(a, b, n_boot=2000, seed=0):
    """Bootstrap CI on mean(a) - mean(b) for two INDEPENDENT groups (wins vs losses)."""
    a = np.asarray([x for x in a if np.isfinite(x)], dtype=np.float64)
    b = np.asarray([x for x in b if np.isfinite(x)], dtype=np.float64)
    if a.size == 0 or b.size == 0:
        return float("nan"), (None, None)
    rng = np.random.default_rng(seed)
    da = a[rng.integers(0, a.size, size=(n_boot, a.size))].mean(axis=1)
    db = b[rng.integers(0, b.size, size=(n_boot, b.size))].mean(axis=1)
    d = da - db
    return float(a.mean() - b.mean()), (
        float(np.percentile(d, 2.5)),
        float(np.percentile(d, 97.5)),
    )


def _pearson(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.size < 3 or x.std() == 0 or y.std() == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _perm_p(x, y, n_perm=2000, seed=0):
    """Two-sided permutation p-value for |Pearson r| under label shuffling."""
    r = _pearson(x, y)
    if not np.isfinite(r):
        return float("nan")
    rng = np.random.default_rng(seed)
    y = np.asarray(y, dtype=np.float64)
    hits = sum(1 for _ in range(n_perm) if abs(_pearson(x, rng.permutation(y))) >= abs(r) - 1e-12)
    return (hits + 1) / (n_perm + 1)


def _auc(pos, neg):
    """AUC via the Mann-Whitney U identity, ties counted as 0.5."""
    pos = np.asarray(pos, dtype=np.float64)
    neg = np.asarray(neg, dtype=np.float64)
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    gt = (pos[:, None] > neg[None, :]).sum()
    eq = (pos[:, None] == neg[None, :]).sum()
    return float((gt + 0.5 * eq) / (pos.size * neg.size))


def _binom_p_two_sided(k, n, p=0.5, n_draw=20000, seed=0):
    """Monte-Carlo two-sided binomial test -- no scipy dependency."""
    if n == 0:
        return float("nan")
    rng = np.random.default_rng(seed)
    draws = rng.binomial(n, p, size=n_draw)
    obs = abs(k - n * p)
    return float((np.abs(draws - n * p) >= obs - 1e-12).mean())


# --------------------------------------------------------------------------
# phase: analyze
# --------------------------------------------------------------------------
def _load_join(args, paths):
    if not paths["features"].exists():
        raise SystemExit(f"missing {paths['features']} -- run --phase features first")
    with open(paths["features"]) as f:
        feats = json.load(f)
    with open(args.divergence) as f:
        div = json.load(f)
    by_key = {r["key"]: r for r in feats["rows"]}
    rows = []
    for pq in div["per_query"]:
        fr = by_key.get(pq["key"])
        if fr is None:
            continue
        rec = dict(fr)
        rec.update({k: v for k, v in pq.items() if k not in ("key", "query")})
        rows.append(rec)
    if len(rows) != len(div["per_query"]):
        raise SystemExit(
            f"join lost queries: {len(rows)} joined vs {len(div['per_query'])} labelled"
        )
    return feats, div, rows


def _zscore(rows, name, short):
    v = np.asarray([r[f"{short}_{name}"] for r in rows], dtype=np.float64)
    mu, sd = float(v.mean()), float(v.std(ddof=0))
    return mu, (sd if sd > 1e-12 else 1.0)


def _add_derived(rows):
    """Per-model z-scores (scale-free) + the bod-minus-base diffs both ways."""
    stats = {}
    for name in FEATURES:
        for short in ("base", "bod"):
            mu, sd = _zscore(rows, name, short)
            stats[f"{short}_{name}"] = {"mean": mu, "std": sd}
            for r in rows:
                r[f"z_{short}_{name}"] = (r[f"{short}_{name}"] - mu) / sd
    for r in rows:
        for name in FEATURES:
            r[f"diff_{name}"] = r[f"bod_{name}"] - r[f"base_{name}"]
            r[f"zdiff_{name}"] = r[f"z_bod_{name}"] - r[f"z_base_{name}"]
    return stats


def _decided(rows, label_field):
    """Queries with a real winner under this win definition (drop ties)."""
    return [r for r in rows if r[label_field] in ("bod", "base")]


def _rule_block(rows, feat_key, label_field, n_perm, seed):
    """'Pick whichever model scores higher on <feat>' -- how often is it right?"""
    dec = _decided(rows, label_field)
    x = np.asarray([r[feat_key] for r in dec], dtype=np.float64)
    y = np.asarray([1.0 if r[label_field] == "bod" else -1.0 for r in dec])
    pred_bod = x > 0
    correct = int(((pred_bod & (y > 0)) | (~pred_bod & (y < 0))).sum())
    n = len(dec)
    always_bod = float((y > 0).mean()) if n else float("nan")
    return {
        "feature": feat_key,
        "n_decided": n,
        "pick_higher_accuracy": correct / n if n else float("nan"),
        "always_bod_accuracy": always_bod,
        "lift_over_always_bod": (correct / n - always_bod) if n else float("nan"),
        "pearson_r_diff_vs_bodwin": _pearson(x, y),
        "perm_p": _perm_p(x, y, n_perm, seed),
        "auc_diff_ranks_bodwin": _auc(x[y > 0], x[y < 0]),
    }


def _within_model_block(rows, short, label_field, n_boot, seed):
    """Does the model's OWN confidence drop on the queries where it loses?"""
    wins = [r for r in rows if r[label_field] == short]
    losses = [r for r in rows if r[label_field] not in (short, "tie")]
    out = {"n_wins": len(wins), "n_losses": len(losses), "features": {}}
    for name in FEATURES:
        w = [r[f"{short}_{name}"] for r in wins]
        losing = [r[f"{short}_{name}"] for r in losses]
        delta, (lo, hi) = _unpaired_delta_ci(w, losing, n_boot, seed)
        out["features"][name] = {
            "mean_on_wins": float(np.mean(w)) if w else float("nan"),
            "mean_on_losses": float(np.mean(losing)) if losing else float("nan"),
            "delta_wins_minus_losses": delta,
            "ci95": [lo, hi],
            "auc_losses_below_wins": _auc(w, losing),
        }
    for name in QUERY_FEATURES:
        w = [r[name] for r in wins]
        losing = [r[name] for r in losses]
        delta, (lo, hi) = _unpaired_delta_ci(w, losing, n_boot, seed)
        out["features"][name] = {
            "mean_on_wins": float(np.mean(w)) if w else float("nan"),
            "mean_on_losses": float(np.mean(losing)) if losing else float("nan"),
            "delta_wins_minus_losses": delta,
            "ci95": [lo, hi],
            "auc_losses_below_wins": _auc(w, losing),
        }
    return out


# ---- routing evaluation ---------------------------------------------------
METRICS = (
    ("junk_rate", -1),  # lower is better
    ("hit", +1),
    ("recall", +1),
    ("ndcg", +1),
    ("e1", +1),
)


def _served(rows, choice):
    """choice: list of 'base'/'bod' per row -> per-metric per-query vectors."""
    out = {}
    for m, _ in METRICS:
        out[m] = np.asarray(
            [r[f"{c}_{m}"] for r, c in zip(rows, choice, strict=True)], dtype=np.float64
        )
    return out


def _metrics_block(rows, choice, base_choice, n_boot, seed):
    served = _served(rows, choice)
    ref = _served(rows, base_choice)
    out = {"n_routed_to_base": int(sum(1 for c in choice if c == "base"))}
    for m, sign in METRICS:
        delta, (lo, hi) = _paired_delta_ci(served[m], ref[m], n_boot, seed)
        out[m] = {
            "value": float(np.nanmean(served[m])),
            "reference": float(np.nanmean(ref[m])),
            "delta_vs_reference": delta,
            "ci95": [lo, hi],
            "better_is": "lower" if sign < 0 else "higher",
        }
    return out


def _threshold_rule(rows, feat_key, thr, direction):
    """Serve base when the feature is on the 'unconfident' side of thr."""
    if direction == "lt":
        return ["base" if r[feat_key] < thr else "bod" for r in rows]
    return ["base" if r[feat_key] > thr else "bod" for r in rows]


OBJECTIVES = ("combined", "junk", "hit")


def _rule_objective(rows, choice, objective="combined"):
    """In-fold selection criterion. Higher is better.

    'combined' weighs junk@10 down and hit@10 up equally; 'junk' is pure
    categorical relevance (which is where the coherence signal lives, so it
    gets its own run rather than being diluted by the click metric); 'hit' is
    pure click accuracy.
    """
    s = _served(rows, choice)
    if objective == "junk":
        return float(-np.nanmean(s["junk_rate"]))
    if objective == "hit":
        return float(np.nanmean(s["hit"]))
    return float(np.nanmean(s["hit"]) - np.nanmean(s["junk_rate"]))


def _candidate_rules(rows, feat_keys, n_thr):
    for fk in feat_keys:
        v = np.asarray([r[fk] for r in rows], dtype=np.float64)
        qs = np.unique(np.quantile(v, np.linspace(0.05, 0.95, n_thr)))
        for thr in qs:
            for direction in ("lt", "gt"):
                yield fk, float(thr), direction


def _cv_routing(rows, feat_keys, n_folds, n_thr, n_boot, seed, objective="combined"):
    """Honest payoff: pick the rule on train folds, apply it to the held-out fold.

    Selecting a feature AND a threshold on all 250 queries and then reporting
    the resulting lift would be pure in-sample overfitting -- with ~20 features
    x 2 directions x n_thr cut points, something always looks good. K-fold
    selection is the cheapest defensible answer.
    """
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(rows))
    folds = [order[i::n_folds] for i in range(n_folds)]
    choice = ["bod"] * len(rows)
    picked = []
    for f in range(n_folds):
        test_idx = set(int(i) for i in folds[f])
        train = [r for i, r in enumerate(rows) if i not in test_idx]
        best, best_obj = None, _rule_objective(train, ["bod"] * len(train), objective)
        for fk, thr, direction in _candidate_rules(train, feat_keys, n_thr):
            obj = _rule_objective(train, _threshold_rule(train, fk, thr, direction), objective)
            if obj > best_obj + 1e-12:
                best, best_obj = (fk, thr, direction), obj
        picked.append(
            {
                "fold": f,
                "rule": None
                if best is None
                else {"feature": best[0], "threshold": best[1], "direction": best[2]},
            }
        )
        if best is None:
            continue
        fk, thr, direction = best
        for i in sorted(test_idx):
            r = rows[i]
            hit = (r[fk] < thr) if direction == "lt" else (r[fk] > thr)
            if hit:
                choice[i] = "base"
    return {
        "objective": objective,
        "n_folds": n_folds,
        "selected_per_fold": picked,
        "metrics": _metrics_block(rows, choice, ["bod"] * len(rows), n_boot, seed),
    }


def _best_in_sample(rows, feat_keys, n_thr, n_boot, seed, objective="combined"):
    best, best_obj = None, _rule_objective(rows, ["bod"] * len(rows), objective)
    for fk, thr, direction in _candidate_rules(rows, feat_keys, n_thr):
        obj = _rule_objective(rows, _threshold_rule(rows, fk, thr, direction), objective)
        if obj > best_obj + 1e-12:
            best, best_obj = (fk, thr, direction), obj
    if best is None:
        return {
            "objective": objective,
            "rule": None,
            "note": "no threshold rule beat always-BoD even in sample",
        }
    fk, thr, direction = best
    choice = _threshold_rule(rows, fk, thr, direction)
    return {
        "objective": objective,
        "rule": {"feature": fk, "threshold": thr, "direction": direction},
        "note": "IN-SAMPLE optimum -- feature and threshold chosen on the same 250 queries",
        "metrics": _metrics_block(rows, choice, ["bod"] * len(rows), n_boot, seed),
    }


def _failure_mode_rule(rows, n_boot, seed, media_thr=0.5):
    """Hand-designed probe of the KNOWN BoD failure mode.

    Serve base when BoD's top-10 is majority media (CD/DVD/...) and base's is
    less media-heavy -- i.e. when BoD looks like it has fallen into the music
    catalog and base has not. Designed with hindsight from the divergence-run
    error analysis, so it is not an out-of-sample result either; it is here to
    put a number on the best *targeted* rule available.
    """
    choice = [
        "base"
        if (r["bod_media_share"] >= media_thr and r["base_media_share"] < r["bod_media_share"])
        else "bod"
        for r in rows
    ]
    return {
        "rule": (
            f"serve base when bod_media_share >= {media_thr} and base_media_share < bod_media_share"
        ),
        "note": "hand-designed from the known failure mode; hindsight-selected, not held out",
        "metrics": _metrics_block(rows, choice, ["bod"] * len(rows), n_boot, seed),
    }


def _pick_higher_routing(rows, feat_key, n_boot, seed):
    choice = ["bod" if r[feat_key] > 0 else "base" for r in rows]
    return {
        "rule": f"serve whichever model has the higher {feat_key.replace('zdiff_', '')}",
        "feature": feat_key,
        "metrics": _metrics_block(rows, choice, ["bod"] * len(rows), n_boot, seed),
    }


def _oracle(rows, label_field, n_boot, seed):
    choice = [r[label_field] if r[label_field] in ("base", "bod") else "bod" for r in rows]
    return {
        "rule": f"ORACLE -- serve the per-query {label_field} (upper bound, not achievable)",
        "metrics": _metrics_block(rows, choice, ["bod"] * len(rows), n_boot, seed),
    }


def phase_analyze(args, paths):
    feats, div, rows = _load_join(args, paths)
    zstats = _add_derived(rows)
    divergent = [r for r in rows if r["is_divergent"]]
    print(f"  {len(rows)} queries joined, {len(divergent)} divergent", flush=True)

    diff_keys = [f"diff_{n}" for n in FEATURES] + [f"zdiff_{n}" for n in FEATURES]

    correlations = {}
    for scope, scope_rows in (("divergent", divergent), ("all_250", rows)):
        correlations[scope] = {}
        for label_field in ("categorical_winner", "click_winner"):
            blocks = [
                _rule_block(scope_rows, fk, label_field, args.n_perm, args.seed) for fk in diff_keys
            ]
            blocks.sort(key=lambda b: -abs(b["pearson_r_diff_vs_bodwin"]))
            correlations[scope][label_field] = blocks

    within = {}
    for scope, scope_rows in (("divergent", divergent), ("all_250", rows)):
        within[scope] = {}
        for label_field in ("categorical_winner", "click_winner"):
            within[scope][label_field] = {
                short: _within_model_block(scope_rows, short, label_field, args.n_boot, args.seed)
                for short in ("bod", "base")
            }

    # ---- payoff. Routing is evaluated on all 250 (that is the serving
    # population); the rule may only look at serve-time features.
    feat_keys = diff_keys + [f"bod_{n}" for n in FEATURES] + list(QUERY_FEATURES)
    routing = {
        "population": "all 250 holdout queries (the real serving population)",
        "reference": "always serve BoD",
        "pick_higher_rules": {
            fk: _pick_higher_routing(rows, fk, args.n_boot, args.seed)
            for fk in [f"zdiff_{n}" for n in FEATURES]
        },
        "cv_selected_rule": _cv_routing(
            rows, feat_keys, args.n_folds, args.n_thresholds, args.n_boot, args.seed
        ),
        "cv_selected_rule_by_objective": {
            obj: _cv_routing(
                rows, feat_keys, args.n_folds, args.n_thresholds, args.n_boot, args.seed, obj
            )
            for obj in OBJECTIVES
        },
        "best_in_sample_rule": _best_in_sample(
            rows, feat_keys, args.n_thresholds, args.n_boot, args.seed
        ),
        "best_in_sample_rule_by_objective": {
            obj: _best_in_sample(rows, feat_keys, args.n_thresholds, args.n_boot, args.seed, obj)
            for obj in OBJECTIVES
        },
        "failure_mode_rule": _failure_mode_rule(rows, args.n_boot, args.seed),
        "oracle_categorical": _oracle(rows, "categorical_winner", args.n_boot, args.seed),
        "oracle_click": _oracle(rows, "click_winner", args.n_boot, args.seed),
    }

    # ---- headline verdict
    best_rule = max(
        correlations["divergent"]["categorical_winner"],
        key=lambda b: b["lift_over_always_bod"],
    )
    cv = routing["cv_selected_rule"]["metrics"]
    verdict = {
        "best_pick_higher_rule_divergent_categorical": best_rule,
        "cv_routing_beats_always_bod": bool(
            (cv["junk_rate"]["ci95"][1] is not None and cv["junk_rate"]["ci95"][1] < 0)
            or (cv["hit"]["ci95"][0] is not None and cv["hit"]["ci95"][0] > 0)
        ),
        "n_routed_to_base_under_cv_rule": cv["n_routed_to_base"],
        "best_ranking_signal_divergent_categorical": max(
            correlations["divergent"]["categorical_winner"],
            key=lambda b: abs(b["pearson_r_diff_vs_bodwin"]),
        ),
        "bod_self_awareness_coherence": within["divergent"]["categorical_winner"]["bod"][
            "features"
        ]["coherence"],
        "any_rule_beats_always_bod": {
            name: bool(
                (blk["metrics"]["junk_rate"]["ci95"][1] or 0) < 0
                or (blk["metrics"]["hit"]["ci95"][0] or 0) > 0
            )
            for name, blk in (
                [(f"cv_{o}", routing["cv_selected_rule_by_objective"][o]) for o in OBJECTIVES]
                + [
                    (f"in_sample_{o}", routing["best_in_sample_rule_by_objective"][o])
                    for o in OBJECTIVES
                    if routing["best_in_sample_rule_by_objective"][o].get("rule")
                ]
                + [("failure_mode_media", routing["failure_mode_rule"])]
                + [(f"pick_higher_{k}", v) for k, v in routing["pick_higher_rules"].items()]
            )
        },
        "oracle_headroom": {
            "categorical_junk_delta": routing["oracle_categorical"]["metrics"]["junk_rate"][
                "delta_vs_reference"
            ],
            "click_hit_delta": routing["oracle_click"]["metrics"]["hit"]["delta_vs_reference"],
            "note": (
                "even a perfect per-query oracle only moves junk@10 by ~3pt and hit@10 by "
                "~5.6pt, because BoD already wins or ties the large majority of queries -- "
                "the routing prize is small before any signal question is asked"
            ),
        },
    }

    out = {
        "experiment": "bestbuy_confidence_routing",
        "question": (
            "Is there a ground-truth-free, serve-time per-query signal (score-distribution "
            "QPP statistics, result-set coherence) that predicts whether base or BoD should "
            "be served for that query -- well enough to beat always-BoD?"
        ),
        "config": {
            "features_source": str(paths["features"]),
            "labels_source": args.divergence,
            "retrieval_source": feats["retrieval_source"],
            "base_model": feats["base_model"],
            "bod_model": feats["bod_model"],
            "catalog_size": feats["catalog_size"],
            "n_queries": len(rows),
            "n_divergent": len(divergent),
            "divergence_rule": div["config"]["divergence_rule"],
            "top_k": feats["top_k"],
            "features": list(FEATURES),
            "query_features": list(QUERY_FEATURES),
            "coherence_definition": feats["coherence_definition"],
            "n_boot": args.n_boot,
            "n_perm": args.n_perm,
            "n_folds": args.n_folds,
            "n_thresholds": args.n_thresholds,
            "seed": args.seed,
            "api_spend_usd": 0.0,
            "api_note": (
                "$0 -- win/loss labels, junk@10 and qrels metrics all reused from "
                "bestbuy_base_vs_bod_divergence.json; no new judge calls"
            ),
        },
        "feature_scale_stats": zstats,
        "feature_means": {
            scope_name: {
                f"{short}_{name}": float(np.mean([r[f"{short}_{name}"] for r in scope_rows]))
                for short in ("base", "bod")
                for name in FEATURES
            }
            for scope_name, scope_rows in (("all_250", rows), ("divergent", divergent))
        },
        "correlations": correlations,
        "within_model": within,
        "routing": routing,
        "verdict": verdict,
        "per_query": [
            {
                k: v
                for k, v in r.items()
                if not k.startswith(("z_", "base_brands", "bod_brands", "gold_titles"))
            }
            for r in rows
        ],
    }
    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    with open(outp, "w") as f:
        json.dump(out, f, indent=2)

    # ---- console report
    print("\n  == pick-the-higher-X rules, divergent set, categorical winner ==", flush=True)
    print(
        f"    {'feature':26s} {'n':>4s} {'acc':>7s} {'always-BoD':>11s} {'lift':>7s} "
        f"{'r':>7s} {'perm p':>7s}",
        flush=True,
    )
    for b in correlations["divergent"]["categorical_winner"]:
        print(
            f"    {b['feature']:26s} {b['n_decided']:4d} {b['pick_higher_accuracy']:7.3f} "
            f"{b['always_bod_accuracy']:11.3f} {b['lift_over_always_bod']:+7.3f} "
            f"{b['pearson_r_diff_vs_bodwin']:+7.3f} {b['perm_p']:7.4f}",
            flush=True,
        )

    print("\n  == BoD's own features: queries it wins vs loses (divergent, categorical) ==")
    wm = within["divergent"]["categorical_winner"]["bod"]
    print(f"    n_wins {wm['n_wins']}  n_losses {wm['n_losses']}")
    for name, blk in wm["features"].items():
        lo, hi = blk["ci95"]
        ci = "n/a" if lo is None else f"[{lo:+.4f}, {hi:+.4f}]"
        print(
            f"    {name:16s} win {blk['mean_on_wins']: .4f}  loss {blk['mean_on_losses']: .4f}  "
            f"d {blk['delta_wins_minus_losses']:+.4f} {ci}  auc {blk['auc_losses_below_wins']:.3f}"
        )

    print("\n  == routing payoff on all 250 (reference = always BoD) ==")

    def show(tag, blk):
        m = blk["metrics"]
        print(f"    {tag}  -> base on {m['n_routed_to_base']} queries")
        for metric, _ in METRICS:
            e = m[metric]
            lo, hi = e["ci95"]
            ci = "n/a" if lo is None else f"[{lo:+.4f}, {hi:+.4f}]"
            print(
                f"      {metric:10s} {e['value']:.4f} vs {e['reference']:.4f}  "
                f"delta {e['delta_vs_reference']:+.4f} {ci}"
            )

    for fk, blk in routing["pick_higher_rules"].items():
        show(f"pick-higher {fk}", blk)
    for obj in OBJECTIVES:
        show(
            f"CV-selected ({args.n_folds}-fold, obj={obj})",
            routing["cv_selected_rule_by_objective"][obj],
        )
    for obj in OBJECTIVES:
        blk = routing["best_in_sample_rule_by_objective"][obj]
        if blk.get("rule"):
            show(f"best in-sample obj={obj} (optimistic) {blk['rule']['feature']}", blk)
    show("failure-mode media rule", routing["failure_mode_rule"])
    show("ORACLE categorical", routing["oracle_categorical"])
    show("ORACLE click", routing["oracle_click"])

    print(f"\nsaved -> {outp}", flush=True)
    return out


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--phase", choices=("features", "analyze", "all"), default="all")
    ap.add_argument("--retrieval", default=DEFAULT_RETRIEVAL)
    ap.add_argument("--divergence", default=DEFAULT_DIVERGENCE)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--work-dir", default="/tmp/bestbuy_confidence_routing")
    ap.add_argument("--tag", default="bestbuy")
    ap.add_argument("--top-k", type=int, default=K_EVAL)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--n-perm", type=int, default=2000)
    ap.add_argument("--n-folds", type=int, default=5)
    ap.add_argument("--n-thresholds", type=int, default=19)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    paths = work_paths(args.work_dir, args.tag)
    if args.phase in ("features", "all"):
        print("[phase features]", flush=True)
        phase_features(args, paths)
    if args.phase in ("analyze", "all"):
        print("[phase analyze]", flush=True)
        phase_analyze(args, paths)


if __name__ == "__main__":
    main()
