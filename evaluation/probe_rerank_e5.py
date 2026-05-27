#!/usr/bin/env python3
"""Rerank-within-pool comparison: e5-large-v2 vs e5-base-v2 on the labeled probe set.

Cheap proxy (~10 min on MPS) to decide whether the ~15h full-catalog encode of
e5-large is worth it. Encodes only the ~3.1k unique candidate-pool docs and 102
probe queries; ranks candidates within each query's pool by cosine; computes
Hit@K against probe_labels.jsonl (strict label==2, lenient label>=1).

Usage:
  .venv/bin/python evaluation/probe_rerank_e5.py \\
      --candidates evaluation/results/probe_candidates_v2.jsonl \\
      --labels     evaluation/results/probe_labels.jsonl \\
      --output     evaluation/results/probe_rerank_e5.json
"""

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer


def load_candidates(path: Path):
    rows = []
    with open(path) as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def load_labels(path: Path) -> dict[tuple[str, str], int]:
    out = {}
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            out[(r["query_id"], r["doc_id"])] = r["label"]
    return out


def encode_with_prefix(model: SentenceTransformer, texts: list[str], prefix: str, batch_size: int):
    if prefix:
        texts = [prefix + t for t in texts]
    return model.encode(
        texts,
        normalize_embeddings=True,
        batch_size=batch_size,
        show_progress_bar=False,
    ).astype(np.float32)


def rerank_and_score(qmap, dmap, cands_by_q, labels, k_list=(1, 5, 10)):
    """Per-query rank pool by cosine; report strict/lenient Hit@K."""
    metrics = {f"h{k}_strict": 0 for k in k_list}
    metrics.update({f"h{k}_lenient": 0 for k in k_list})
    n_q = 0
    per_arch = defaultdict(
        lambda: {
            "n": 0,
            **{f"h{k}_strict": 0 for k in k_list},
            **{f"h{k}_lenient": 0 for k in k_list},
        }
    )
    for qid, items in cands_by_q.items():
        # items is list of {"doc_id": ..., "archetype": ...}
        arch = items[0]["archetype"]
        qv = qmap[qid]
        dvecs = np.stack([dmap[i["doc_id"]] for i in items], axis=0)
        scores = dvecs @ qv  # (n_pool,)
        order = np.argsort(-scores)
        ranked = [items[i]["doc_id"] for i in order]
        n_q += 1
        per_arch[arch]["n"] += 1
        for k in k_list:
            top = ranked[:k]
            top_labels = [labels.get((qid, d), 0) for d in top]
            strict_hit = any(label == 2 for label in top_labels)
            lenient_hit = any(label >= 1 for label in top_labels)
            metrics[f"h{k}_strict"] += int(strict_hit)
            metrics[f"h{k}_lenient"] += int(lenient_hit)
            per_arch[arch][f"h{k}_strict"] += int(strict_hit)
            per_arch[arch][f"h{k}_lenient"] += int(lenient_hit)
    return {
        "n": n_q,
        **{k: round(v / n_q, 4) for k, v in metrics.items()},
        "per_archetype": {
            arch: {
                "n": d["n"],
                **{k: round(d[k] / d["n"], 4) for k in d if k != "n"},
            }
            for arch, d in per_arch.items()
        },
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--batch-size-base", type=int, default=64)
    ap.add_argument("--batch-size-large", type=int, default=16)
    ap.add_argument(
        "--models",
        default="e5_base,e5_large",
        help="comma-separated list from {e5_small, e5_base, e5_large}",
    )
    args = ap.parse_args()

    model_registry = {
        "e5_small": ("intfloat/e5-small-v2", args.batch_size_base),
        "e5_base": ("intfloat/e5-base-v2", args.batch_size_base),
        "e5_large": ("intfloat/e5-large-v2", args.batch_size_large),
    }
    model_specs = []
    for name in [s.strip() for s in args.models.split(",") if s.strip()]:
        if name not in model_registry:
            raise SystemExit(f"unknown model {name!r}; pick from {sorted(model_registry)}")
        hf_id, bs = model_registry[name]
        model_specs.append((name, hf_id, bs))

    cands = load_candidates(Path(args.candidates))
    labels = load_labels(Path(args.labels))
    print(f"loaded {len(cands)} candidates, {len(labels)} labels", flush=True)

    # Build per-query candidate list (each item: doc_id, archetype, title)
    cands_by_q = defaultdict(list)
    for r in cands:
        cands_by_q[r["query_id"]].append(
            {
                "doc_id": r["doc_id"],
                "archetype": r.get("archetype", "?"),
                "title": r.get("title", ""),
            }
        )
    n_q = len(cands_by_q)
    print(f"unique queries: {n_q}", flush=True)

    # Unique queries with text (one query text per query_id)
    qid_to_text = {}
    for r in cands:
        if r["query_id"] not in qid_to_text:
            qid_to_text[r["query_id"]] = r["query"]
    queries = list(qid_to_text.keys())
    query_texts = [qid_to_text[q] for q in queries]

    # Unique docs
    docid_to_title = {}
    for r in cands:
        if r["doc_id"] not in docid_to_title:
            docid_to_title[r["doc_id"]] = r.get("title", "")
    doc_ids_list = list(docid_to_title.keys())
    doc_titles = [docid_to_title[d] for d in doc_ids_list]
    print(f"unique docs: {len(doc_ids_list)}", flush=True)

    out = {"models": {}}

    for model_name, hf_id, bs in model_specs:
        print(f"\n=== {model_name} ({hf_id}) ===", flush=True)
        t0 = time.time()
        m = SentenceTransformer(hf_id, device=args.device)
        print(
            f"  loaded in {time.time() - t0:.1f}s; dim={m.get_sentence_embedding_dimension()}",
            flush=True,
        )
        t0 = time.time()
        qv = encode_with_prefix(m, query_texts, "query: ", bs)
        print(f"  encoded {len(queries)} queries in {time.time() - t0:.1f}s", flush=True)
        t0 = time.time()
        dv = encode_with_prefix(m, doc_titles, "passage: ", bs)
        print(f"  encoded {len(doc_ids_list)} docs in {time.time() - t0:.1f}s", flush=True)

        qmap = dict(zip(queries, qv))
        dmap = dict(zip(doc_ids_list, dv))
        res = rerank_and_score(qmap, dmap, cands_by_q, labels)
        out["models"][model_name] = res
        print(
            f"  H@1_strict={res['h1_strict']:.4f}  H@5_strict={res['h5_strict']:.4f}  "
            f"H@10_strict={res['h10_strict']:.4f}",
            flush=True,
        )

        # Free MPS memory
        del m
        import torch

        if hasattr(torch, "mps") and torch.backends.mps.is_available():
            torch.mps.empty_cache()

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {args.output}", flush=True)

    # Side-by-side summary
    print("\n=== summary (rerank-within-pool Hit@K) ===", flush=True)
    print(
        f"  {'model':10s}  {'H@1_s':>7s}  {'H@5_s':>7s}  {'H@10_s':>7s}  "
        f"{'H@1_l':>7s}  {'H@5_l':>7s}  {'H@10_l':>7s}",
        flush=True,
    )
    for name, r in out["models"].items():
        print(
            f"  {name:10s}  "
            f"{r['h1_strict']:>7.4f}  {r['h5_strict']:>7.4f}  {r['h10_strict']:>7.4f}  "
            f"{r['h1_lenient']:>7.4f}  {r['h5_lenient']:>7.4f}  {r['h10_lenient']:>7.4f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
