#!/usr/bin/env python3
"""Does enriching the *embedded* product text help dense retrieval on ESCI?

The whole ESCI pipeline (BM25s, tantivy, FAISS) is keyed on bare
`product_title` via `esci_us_data/titles.json`. The raw ESCI dataset also
ships `bullet_point` (91% populated), `brand` (96.7%) and `color` (70.3%).
`download/build_enriched_titles.py` concatenates those into
`titles_enriched.json` ("{title}. {brand}. {color}. {bullet}", capped at 400
chars) but nothing in the pipeline consumes it.

CHS_RESULTS.md Pattern 25 already refuted the *lexical* version of this
(richer text on the BM25 side). This script tests the *semantic* side, which
was queued and never run: encode the full 360,873-product catalog twice —
once from plain titles (OLD), once from enriched text (NEW) — and retrieve
top-K from the FULL catalog for all 22,458 ESCI test queries. No reduced
candidate pool: the BestBuy analogue (Pattern 30) showed closed-pool
prototypes do not generalize to genuine full-corpus retrieval.

The enriched text is an EMBEDDING-ONLY field. `esci_us_data/titles.json` is
never modified and remains the display text.

Metrics and their definitions are imported from
`evaluation/eval_mnrl_retriever.py` (R@10 over E+S, nDCG@10 with linear gains
E=1.0/S=0.1/C=0.01/I=0, E@1, E@3) so this result is directly comparable to
the headline numbers in space_demo/README.md.

Ground truth is ESCI's native graded human labels — no LLM judge, $0 API
spend.

Phases (each cached + resumable):
  enrich    verify/report esci_us_data/titles_enriched.json
  encode    product matrices (model x variant) + query matrices (model)
  retrieve  full-catalog brute-force top-K (model x variant)
  eval      metrics + paired bootstrap CIs -> results JSON
  inspect   spot-check queries whose top-10 changed most (optional)

Usage:
    .venv/bin/python evaluation/eval_esci_embedding_enrichment.py --phase all
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import time  # noqa: E402
from collections import defaultdict  # noqa: E402

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np  # noqa: E402

from evaluation.eval_mnrl_retriever import metrics_for  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(REPO, "esci_us_data")
WORK_DIR = os.path.join(REPO, "evaluation", "work", "esci_embedding_enrichment")
RESULTS_PATH = os.path.join(REPO, "evaluation", "results", "esci_embedding_enrichment.json")

K_EVAL = 10
K_RETRIEVE = 10

# Headline ESCI numbers these should be compared against (space_demo/README.md,
# evaluation/eval_mnrl_retriever.py docstring): base MiniLM retrieval R@10
# 15.60%, BoD-as-retriever (6M-MNRL alone) R@10 18.10%. Both are plain-title
# indexes, i.e. the OLD arm here should reproduce them.
HEADLINE = {"base": {"R@10": 0.1560}, "bod_6m_mnrl": {"R@10": 0.1810}}

MODELS = {
    # off-the-shelf base encoder (space_demo/app.py BASE_MODEL_NAME)
    "base": "sentence-transformers/all-MiniLM-L6-v2",
    # headline BoD fine-tuned retriever (space_demo/app.py RERANK_A_MODEL,
    # the "BoD-as-retriever (6M-MNRL alone)" demo mode)
    "bod_6m_mnrl": "dtunkelang/bag-of-documents-minilm-6m-mnrl",
}

VARIANTS = {
    "plain": "titles.json",  # OLD — what the pipeline ships today
    "enriched": "titles_enriched.json",  # NEW — title + brand + color + bullet
}


# ---------------------------------------------------------------- data


def load_eval_set():
    """ESCI test queries that have at least one E-or-S judgement.

    Mirrors evaluation/eval_mnrl_retriever.py's query filter exactly.
    """
    qrels = defaultdict(dict)
    with open(os.path.join(DATA_DIR, "test_qrels.jsonl")) as f:
        for line in f:
            r = json.loads(line)
            qrels[r["query_id"]][r["product_id"]] = r["relevance"]
    queries_all = {}
    with open(os.path.join(DATA_DIR, "test_queries.jsonl")) as f:
        for line in f:
            d = json.loads(line)
            queries_all[d["query_id"]] = d["query"]
    qids = [qid for qid in queries_all if qid in qrels and any(g >= 2 for g in qrels[qid].values())]
    return qids, [queries_all[q] for q in qids], dict(qrels)


def load_pids():
    with open(os.path.join(DATA_DIR, "product_ids.json")) as f:
        return json.load(f)


def load_texts(variant):
    path = os.path.join(DATA_DIR, VARIANTS[variant])
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------- phases


def phase_enrich(args):
    """Verify the enriched catalog exists, is aligned, and report coverage."""
    pids = load_pids()
    path = os.path.join(DATA_DIR, VARIANTS["enriched"])
    if not os.path.exists(path):
        raise SystemExit(
            f"{path} missing — run:\n"
            f"  .venv/bin/python download/build_enriched_titles.py --locale us"
        )
    plain = load_texts("plain")
    enriched = load_texts("enriched")
    if not (len(plain) == len(enriched) == len(pids)):
        raise SystemExit(
            f"length mismatch: pids={len(pids)} plain={len(plain)} enriched={len(enriched)}"
        )
    # The enriched text must never be empty where the plain title is not:
    # that is the exact bug the BestBuy reindex hit (fallback applied to the
    # embedding text but not the display text).
    empty_new = sum(1 for t in enriched if not t.strip())
    empty_old = sum(1 for t in plain if not t.strip())
    same = sum(1 for a, b in zip(plain, enriched) if a == b)
    lens_old = np.array([len(t) for t in plain])
    lens_new = np.array([len(t) for t in enriched])
    stats = {
        "n_products": len(pids),
        "empty_plain": empty_old,
        "empty_enriched": empty_new,
        "identical_to_plain": same,
        "identical_frac": same / len(pids),
        "median_chars_plain": float(np.median(lens_old)),
        "median_chars_enriched": float(np.median(lens_new)),
        "mean_chars_plain": float(lens_old.mean()),
        "mean_chars_enriched": float(lens_new.mean()),
    }
    print("enriched catalog:")
    for k, v in stats.items():
        print(f"  {k:<24s} {v}")
    if empty_new > empty_old:
        raise SystemExit("enriched text lost content relative to plain titles — refusing to run")
    _save_json(os.path.join(WORK_DIR, "enrich_stats.json"), stats)
    return stats


def _st_model(model_id, device):
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_id, device=device)


def _pick_device(arg):
    if arg != "auto":
        return arg
    import torch

    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def prod_path(tag, variant):
    return os.path.join(WORK_DIR, f"prod_{tag}_{variant}.fp16.npy")


def query_path(tag):
    return os.path.join(WORK_DIR, f"query_{tag}.fp32.npy")


def top_path(tag, variant):
    return os.path.join(WORK_DIR, f"top{K_RETRIEVE}_{tag}_{variant}.npy")


def phase_encode(args):
    device = _pick_device(args.device)
    print(f"device: {device}", flush=True)
    _, queries, _ = load_eval_set()
    for tag in args.models:
        model = None
        # queries
        qp = query_path(tag)
        if os.path.exists(qp) and not args.force:
            print(f"[skip] {os.path.basename(qp)}", flush=True)
        else:
            model = model or _st_model(MODELS[tag], device)
            t0 = time.time()
            qv = model.encode(
                queries,
                normalize_embeddings=True,
                batch_size=args.batch_size,
                show_progress_bar=False,
            )
            np.save(qp, np.asarray(qv, dtype=np.float32))
            print(f"[done] {os.path.basename(qp)} {qv.shape} {time.time() - t0:.0f}s", flush=True)
        # products
        for variant in args.variants:
            pp = prod_path(tag, variant)
            if os.path.exists(pp) and not args.force:
                print(f"[skip] {os.path.basename(pp)}", flush=True)
                continue
            texts = load_texts(variant)
            model = model or _st_model(MODELS[tag], device)
            t0 = time.time()
            pv = model.encode(
                texts,
                normalize_embeddings=True,
                batch_size=args.batch_size,
                show_progress_bar=True,
            )
            np.save(pp, np.asarray(pv, dtype=np.float16))
            print(f"[done] {os.path.basename(pp)} {pv.shape} {time.time() - t0:.0f}s", flush=True)
        del model


def brute_top_k(q_vecs, p_vecs, k, batch=256):
    """Exhaustive cosine top-k against the FULL catalog. p_vecs L2-normalized."""
    n = q_vecs.shape[0]
    out = np.zeros((n, k), dtype=np.int32)
    t0 = time.time()
    for start in range(0, n, batch):
        end = min(start + batch, n)
        sims = q_vecs[start:end] @ p_vecs.T
        part = np.argpartition(-sims, k, axis=1)[:, :k]
        rows = np.arange(end - start)[:, None]
        order = np.argsort(-sims[rows, part], axis=1)
        out[start:end] = part[rows, order]
        if (start // batch) % 20 == 0:
            print(f"    {end}/{n}  {time.time() - t0:.0f}s", flush=True)
    return out


def phase_retrieve(args):
    for tag in args.models:
        qv = np.load(query_path(tag)).astype(np.float32)
        for variant in args.variants:
            tp = top_path(tag, variant)
            if os.path.exists(tp) and not args.force:
                print(f"[skip] {os.path.basename(tp)}", flush=True)
                continue
            pv = np.load(prod_path(tag, variant)).astype(np.float32)
            print(f"[run ] {tag}/{variant}: {qv.shape[0]:,} queries x {pv.shape[0]:,} products")
            t0 = time.time()
            top = brute_top_k(qv, pv, K_RETRIEVE, batch=args.retrieve_batch)
            np.save(tp, top)
            print(f"[done] {os.path.basename(tp)} {time.time() - t0:.0f}s", flush=True)
            del pv


# ---------------------------------------------------------------- metrics

METRIC_KEYS = ["recall", "ndcg", "e_at_1", "e_at_3"]
METRIC_LABEL = {"recall": "R@10", "ndcg": "nDCG@10", "e_at_1": "E@1", "e_at_3": "E@3"}


def per_query_metrics(top_idx, pids, qids, qrels):
    """Returns {metric: (values, mask)} aligned to qids.

    mask marks queries where the metric is defined (E@1/E@3 need an Exact).
    """
    vals = {m: np.zeros(len(qids), dtype=np.float64) for m in METRIC_KEYS}
    mask = {m: np.zeros(len(qids), dtype=bool) for m in METRIC_KEYS}
    for i, qid in enumerate(qids):
        retrieved = [pids[j] for j in top_idx[i]]
        m = metrics_for(retrieved, qrels[qid], k_eval=K_EVAL)
        for key in METRIC_KEYS:
            if key in m:
                vals[key][i] = m[key]
                mask[key][i] = True
    return vals, mask


def paired_bootstrap(new_v, old_v, mask, n_boot=2000, seed=0):
    """Paired bootstrap over queries on the OLD->NEW delta."""
    idx = np.flatnonzero(mask)
    a, b = new_v[idx], old_v[idx]
    d = a - b
    rng = np.random.default_rng(seed)
    n = len(idx)
    boot = np.empty(n_boot)
    for i in range(n_boot):
        s = rng.integers(0, n, n)
        boot[i] = d[s].mean()
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return {
        "n": int(n),
        "old": float(b.mean()),
        "new": float(a.mean()),
        "delta": float(d.mean()),
        "delta_pp": float(d.mean() * 100),
        "ci95_pp": [float(lo * 100), float(hi * 100)],
        "significant": bool(lo > 0 or hi < 0),
        "p_two_sided": float(2 * min((boot <= 0).mean(), (boot >= 0).mean())),
    }


def phase_eval(args):
    pids = load_pids()
    qids, queries, qrels = load_eval_set()
    print(f"{len(qids):,} eval queries, {len(pids):,} products", flush=True)
    results = {
        "question": (
            "Does embedding enriched product text (title + brand + color + bullet, "
            "400-char cap) instead of the bare title improve FULL-CATALOG dense "
            "retrieval on ESCI-US?"
        ),
        "corpus": {"n_products": len(pids), "n_eval_queries": len(qids)},
        "k_eval": K_EVAL,
        "metric_defs": {
            "R@10": "recall of E+S products in top-10 (eval_mnrl_retriever.metrics_for)",
            "nDCG@10": "linear gains E=1.0 / S=0.1 / C=0.01 / I=0",
            "E@1": "top-1 is an Exact",
            "E@3": "Exacts in top-3 / min(3, #Exact)",
        },
        "ground_truth": "native ESCI graded labels (no LLM judge, $0 API spend)",
        "headline_reference": HEADLINE,
        "models": {},
    }
    for tag in args.models:
        per_variant, per_variant_mask = {}, {}
        for variant in args.variants:
            top = np.load(top_path(tag, variant))
            v, m = per_query_metrics(top, pids, qids, qrels)
            per_variant[variant] = v
            per_variant_mask[variant] = m
        entry = {"model_id": MODELS[tag], "arms": {}, "delta_enriched_vs_plain": {}}
        for variant in args.variants:
            entry["arms"][variant] = {
                METRIC_LABEL[k]: float(per_variant[variant][k][per_variant_mask[variant][k]].mean())
                for k in METRIC_KEYS
            }
            entry["arms"][variant]["n"] = {
                METRIC_LABEL[k]: int(per_variant_mask[variant][k].sum()) for k in METRIC_KEYS
            }
        if "plain" in per_variant and "enriched" in per_variant:
            # How much does enrichment actually churn the ranking? A near-zero
            # metric delta with high churn means "big reshuffle, no net gain",
            # which is a different finding from "the change was a no-op".
            old_top = np.load(top_path(tag, "plain"))
            new_top = np.load(top_path(tag, "enriched"))
            overlap = np.array(
                [
                    len(set(old_top[i].tolist()) & set(new_top[i].tolist())) / K_RETRIEVE
                    for i in range(len(qids))
                ]
            )
            entry["top10_churn"] = {
                "mean_overlap": float(overlap.mean()),
                "frac_queries_identical_set": float((overlap == 1.0).mean()),
            }
            for k in METRIC_KEYS:
                entry["delta_enriched_vs_plain"][METRIC_LABEL[k]] = paired_bootstrap(
                    per_variant["enriched"][k],
                    per_variant["plain"][k],
                    per_variant_mask["plain"][k],
                    n_boot=args.n_boot,
                    seed=args.seed,
                )
        results["models"][tag] = entry
        _print_model(tag, entry)
    _save_json(RESULTS_PATH, results)
    print(f"\nsaved {RESULTS_PATH}")
    return results


def _print_model(tag, entry):
    print(f"\n=== {tag}  ({entry['model_id']}) ===")
    print(
        f"  {'metric':<10s}{'plain (OLD)':>14s}{'enriched (NEW)':>16s}"
        f"{'delta pp':>11s}{'95% CI':>22s}"
    )
    for k in METRIC_KEYS:
        label = METRIC_LABEL[k]
        old = entry["arms"].get("plain", {}).get(label)
        new = entry["arms"].get("enriched", {}).get(label)
        d = entry["delta_enriched_vs_plain"].get(label)
        if old is None or new is None or d is None:
            continue
        ci = f"[{d['ci95_pp'][0]:+.2f}, {d['ci95_pp'][1]:+.2f}]"
        star = " *" if d["significant"] else ""
        print(
            f"  {label:<10s}{old * 100:>13.2f}%{new * 100:>15.2f}%"
            f"{d['delta_pp']:>+11.2f}{ci:>22s}{star}"
        )


def phase_inspect(args):
    """Spot-check queries whose top-10 changed most under enrichment."""
    pids = load_pids()
    plain_titles = load_texts("plain")
    enriched_titles = load_texts("enriched")
    qids, queries, qrels = load_eval_set()
    out = {}
    for tag in args.models:
        old = np.load(top_path(tag, "plain"))
        new = np.load(top_path(tag, "enriched"))
        v_old, m = per_query_metrics(old, pids, qids, qrels)
        v_new, _ = per_query_metrics(new, pids, qids, qrels)
        d = v_new["ndcg"] - v_old["ndcg"]
        order = np.argsort(d)
        picks = list(order[: args.n_inspect]) + list(order[-args.n_inspect :])
        rows = []
        for i in picks:
            qid = qids[i]
            g = qrels[qid]
            rows.append(
                {
                    "query": queries[i],
                    "query_id": qid,
                    "ndcg_plain": float(v_old["ndcg"][i]),
                    "ndcg_enriched": float(v_new["ndcg"][i]),
                    "delta": float(d[i]),
                    "top10_plain": [
                        {"pid": pids[j], "title": plain_titles[j], "grade": g.get(pids[j])}
                        for j in old[i]
                    ],
                    "top10_enriched": [
                        {
                            "pid": pids[j],
                            "title": plain_titles[j],
                            "embedded_text": enriched_titles[j][:200],
                            "grade": g.get(pids[j]),
                        }
                        for j in new[i]
                    ],
                }
            )
        # aggregate churn
        churn = np.array(
            [
                len(set(old[i].tolist()) & set(new[i].tolist())) / K_RETRIEVE
                for i in range(len(qids))
            ]
        )
        out[tag] = {
            "mean_top10_overlap": float(churn.mean()),
            "frac_queries_unchanged": float((churn == 1.0).mean()),
            "examples": rows,
        }
        print(f"{tag}: mean top-10 overlap plain vs enriched = {churn.mean():.3f}")
    path = os.path.join(WORK_DIR, "inspect.json")
    _save_json(path, out)
    print(f"saved {path}")


# ---------------------------------------------------------------- plumbing


def _save_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--phase",
        default="all",
        choices=["enrich", "encode", "retrieve", "eval", "inspect", "all"],
    )
    ap.add_argument("--models", nargs="+", default=list(MODELS), choices=list(MODELS))
    ap.add_argument("--variants", nargs="+", default=list(VARIANTS), choices=list(VARIANTS))
    ap.add_argument("--device", default="auto", choices=["auto", "mps", "cuda", "cpu"])
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--retrieve-batch", type=int, default=256)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-inspect", type=int, default=5)
    ap.add_argument("--force", action="store_true", help="recompute cached artifacts")
    args = ap.parse_args()

    os.makedirs(WORK_DIR, exist_ok=True)
    phases = ["enrich", "encode", "retrieve", "eval"] if args.phase == "all" else [args.phase]
    for p in phases:
        print(f"\n########## phase: {p} ##########", flush=True)
        t0 = time.time()
        {
            "enrich": phase_enrich,
            "encode": phase_encode,
            "retrieve": phase_retrieve,
            "eval": phase_eval,
            "inspect": phase_inspect,
        }[p](args)
        print(f"[phase {p}] {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
