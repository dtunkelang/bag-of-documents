#!/usr/bin/env python3
"""Push the tax-router investigation: BoD-side signals, disagreement features,
and a learned logistic-regression calibrator with train/test split.

Builds on probe_tax_router.py (which showed cheap base-side signals can't
identify tax queries). This script asks the harder question: with a richer
feature set and a learned model, can ANY router beat BoD-only?

Features (per query):
  base_top1, base_top1_minus_top2, base_mean_top10, base_top1_minus_topk
  bod_top1,  bod_top1_minus_top2,  bod_mean_top10,  bod_top1_minus_topk
  agree_top1   1 if base.top1 == bod.top1, else 0
  rank_overlap |base.top10 ∩ bod.top10| / 10  (Jaccard-ish over the 10s)
  cos_qq       cos(base_qv, bod_qv) — query-encoder agreement

Labels:
  delta = bod_R@10 - base_R@10  (continuous regression target)
  bod_wins = 1 if delta > 0 else 0  (binary classification target)

Routes: P(bod_wins) >= 0.5 → BoD; else base. Threshold tunable.

Train/test: 80/20 random split. Reports both in-sample (cheating ceiling)
and held-out router R@10. If even in-sample router can't beat BoD-only,
the tax is intrinsic; the routing approach is dead.
"""

import argparse
import json
import os
from collections import defaultdict

import numpy as np
import torch
from sentence_transformers import SentenceTransformer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="bestbuy_acm_data")
    ap.add_argument("--queries", default="holdout_queries.jsonl")
    ap.add_argument("--qrels", default="holdout_qrels.jsonl")
    ap.add_argument("--min-relevance", type=int, default=1)
    ap.add_argument("--base-model", default="all-MiniLM-L6-v2")
    ap.add_argument("--bod-model", default="query_model_bestbuy_bod")
    ap.add_argument("--base-vecs", default="base_catalog.vecs.fp16.npy")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    print("loading data...", flush=True)
    with open(os.path.join(args.data_dir, "titles.json")) as f:
        titles = json.load(f)
    with open(os.path.join(args.data_dir, "product_ids.json")) as f:
        pids = json.load(f)
    pid_to_idx = {p: i for i, p in enumerate(pids)}
    qids, queries = [], []
    with open(os.path.join(args.data_dir, args.queries)) as f:
        for line in f:
            d = json.loads(line)
            qids.append(d["query_id"])
            queries.append(d["query"])
    pos = defaultdict(set)
    field = None
    with open(os.path.join(args.data_dir, args.qrels)) as f:
        for line in f:
            r = json.loads(line)
            if field is None:
                field = "product_id" if "product_id" in r else "doc_id"
            if r[field] not in pid_to_idx:
                continue
            if r["relevance"] < args.min_relevance:
                continue
            pos[r["query_id"]].add(pid_to_idx[r[field]])
    print(f"  catalog={len(pids):,}  queries={len(queries):,}", flush=True)

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    base_pv = np.load(os.path.join(args.data_dir, args.base_vecs)).astype(np.float32)
    base = SentenceTransformer(args.base_model, device=device)
    bod = SentenceTransformer(args.bod_model, device=device)

    print("encoding catalog with BoD...", flush=True)
    bod_pv = bod.encode(
        titles, normalize_embeddings=True, batch_size=128, show_progress_bar=True
    ).astype(np.float32)
    print("encoding queries...", flush=True)
    base_qv = base.encode(
        queries, normalize_embeddings=True, batch_size=256, show_progress_bar=True
    ).astype(np.float32)
    bod_qv = bod.encode(
        queries, normalize_embeddings=True, batch_size=256, show_progress_bar=True
    ).astype(np.float32)
    del base, bod
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

    print("matmul + features...", flush=True)
    chunk = 1024
    feats = []  # per-query feature vectors
    bh_list = []
    dh_list = []
    cos_qq = (base_qv * bod_qv).sum(axis=1)  # query-encoder agreement
    k = args.k
    for start in range(0, len(qids), chunk):
        end = min(start + chunk, len(qids))
        bsim = base_qv[start:end] @ base_pv.T
        dsim = bod_qv[start:end] @ bod_pv.T
        b_topk_idx = np.argpartition(-bsim, k, axis=1)[:, :k]
        d_topk_idx = np.argpartition(-dsim, k, axis=1)[:, :k]
        for j, gi in enumerate(range(start, end)):
            g = pos.get(qids[gi], set())
            if not g:
                continue
            row_b = bsim[j]
            row_d = dsim[j]
            b_top = b_topk_idx[j]
            d_top = d_topk_idx[j]
            b_sorted = b_top[np.argsort(-row_b[b_top])]
            d_sorted = d_top[np.argsort(-row_d[d_top])]
            b_sims = row_b[b_sorted]
            d_sims = row_d[d_sorted]
            bh = len({int(x) for x in b_sorted} & g) / len(g)
            dh = len({int(x) for x in d_sorted} & g) / len(g)
            agree_top1 = 1.0 if int(b_sorted[0]) == int(d_sorted[0]) else 0.0
            overlap = len(set(int(x) for x in b_sorted) & set(int(x) for x in d_sorted)) / k
            feats.append(
                [
                    float(b_sims[0]),
                    float(b_sims[0] - b_sims[1]),
                    float(np.mean(b_sims)),
                    float(b_sims[0] - b_sims[-1]),
                    float(d_sims[0]),
                    float(d_sims[0] - d_sims[1]),
                    float(np.mean(d_sims)),
                    float(d_sims[0] - d_sims[-1]),
                    agree_top1,
                    overlap,
                    float(cos_qq[gi]),
                ]
            )
            bh_list.append(bh)
            dh_list.append(dh)
        del bsim, dsim
    X = np.array(feats, dtype=np.float32)
    bh = np.array(bh_list, dtype=np.float32)
    dh = np.array(dh_list, dtype=np.float32)
    delta = dh - bh
    bod_wins = (delta > 0).astype(np.float32)
    print(f"  {len(X):,} pos-bearing queries scored, feature dim {X.shape[1]}", flush=True)

    feature_names = [
        "base_top1",
        "base_top1_minus_top2",
        "base_mean_top10",
        "base_top1_minus_topk",
        "bod_top1",
        "bod_top1_minus_top2",
        "bod_mean_top10",
        "bod_top1_minus_topk",
        "agree_top1",
        "rank_overlap",
        "cos_qq",
    ]

    print("\n" + "=" * 78)
    print(f"v2 tax-router probe — BestBuy (n={len(X):,})")
    print("=" * 78)
    print(
        f"  base R@10: {bh.mean():.3f}   BoD R@10: {dh.mean():.3f}   "
        f"Δ_BoD−base: {delta.mean():+.3f}   bod_wins fraction: {bod_wins.mean():.3f}"
    )

    print("\nPearson correlation of each feature with Δ:")
    for i, name in enumerate(feature_names):
        r = float(np.corrcoef(X[:, i], delta)[0, 1])
        print(f"  {name:<24} {r:>+8.3f}")

    # Single-feature oracle router (ceiling check, in-sample).
    print("\nSingle-feature oracle threshold router (in-sample, route to BoD if feat <= τ):")
    print(
        f"  {'feature':<24} {'best τ':>8} {'route_bod%':>11} "
        f"{'router R@10':>12} {'gain vs BoD':>12}"
    )
    bod_only = float(dh.mean())
    for i, name in enumerate(feature_names):
        s = X[:, i]
        best_r = -1.0
        best_tau = None
        best_pct = None
        for tau in np.quantile(s, np.linspace(0.0, 1.0, 41)):
            mask_bod = s <= tau
            r = float(np.where(mask_bod, dh, bh).mean())
            if r > best_r:
                best_r = r
                best_tau = float(tau)
                best_pct = 100.0 * float(mask_bod.mean())
        gain = best_r - bod_only
        print(f"  {name:<24} {best_tau:>+8.3f} {best_pct:>10.1f}%  {best_r:>12.4f} {gain:>+12.4f}")

    # Learned router: logistic regression on all features, 80/20 split.
    print("\nLearned router — logistic regression on all features, 80/20 train/test:")
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(len(X))
    n_train = int(0.8 * len(X))
    tr, te = perm[:n_train], perm[n_train:]

    # Standardize features.
    mu = X[tr].mean(axis=0)
    sd = X[tr].std(axis=0) + 1e-8
    Xs = (X - mu) / sd

    # Logistic regression via gradient descent (no sklearn dep).
    Xb_tr = np.hstack([Xs[tr], np.ones((len(tr), 1))])
    y_tr = bod_wins[tr]
    w = np.zeros(Xb_tr.shape[1], dtype=np.float32)
    lr = 0.05
    for _ in range(2000):
        p = 1.0 / (1.0 + np.exp(-Xb_tr @ w))
        grad = Xb_tr.T @ (p - y_tr) / len(y_tr)
        w -= lr * grad

    Xb_te = np.hstack([Xs[te], np.ones((len(te), 1))])
    p_tr = 1.0 / (1.0 + np.exp(-Xb_tr @ w))
    p_te = 1.0 / (1.0 + np.exp(-Xb_te @ w))

    def router_R(p, idx, threshold=0.5):
        mask_bod = p >= threshold
        return float(np.where(mask_bod, dh[idx], bh[idx]).mean())

    # Sweep threshold on test.
    best = (-1.0, None, None)
    for tau in np.linspace(0.05, 0.95, 19):
        r = router_R(p_te, te, threshold=tau)
        if r > best[0]:
            mask_bod = p_te >= tau
            best = (r, float(tau), 100.0 * float(mask_bod.mean()))
    base_te = float(bh[te].mean())
    bod_te = float(dh[te].mean())
    print(
        f"  test (n={len(te):,})    base={base_te:.4f}  BoD={bod_te:.4f}  "
        f"router(best τ)={best[0]:.4f} (τ={best[1]:.2f}, route_bod={best[2]:.1f}%)"
    )
    print(f"  gain vs BoD-only on test: {best[0] - bod_te:+.4f}")

    # Also report at default τ=0.5 to see if it's any good without tuning.
    r05 = router_R(p_te, te, threshold=0.5)
    print(f"  router at τ=0.5 (no tuning): {r05:.4f}  gain vs BoD: {r05 - bod_te:+.4f}")

    # And the in-sample ceiling (cheating).
    r_tr_05 = router_R(p_tr, tr, threshold=0.5)
    print(
        f"  in-sample router at τ=0.5: {r_tr_05:.4f}  vs BoD on train: "
        f"{r_tr_05 - float(dh[tr].mean()):+.4f}  (cheating ceiling)"
    )

    # Learned weights for interpretability.
    print("\nLogistic regression weights (after standardization):")
    for name, ww in zip(feature_names, w[:-1]):
        print(f"  {name:<24} {ww:>+8.3f}")
    print(f"  {'(bias)':<24} {w[-1]:>+8.3f}")


if __name__ == "__main__":
    main()
