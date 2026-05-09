#!/usr/bin/env python3
"""Re-encode every calibration corpus under BAAI/bge-base-en-v1.5 and
recompute bag stats. Used to test whether a stronger base encoder
captures the domain-encoder-fit signal that MiniLM-based `median_spec`
misses (Pattern 8a worst-residual: CQADup/mathematica at +6pp).

Outputs `logs/bge_base_bag_stats.tsv` with one row per corpus:
    corpus, n_bags_train, median_size, median_spec_bge,
    n_bags_minilm, median_spec_minilm, rescue_pp

Designed to run unattended overnight: encodes one corpus at a time
and discards vecs before moving to the next, so memory stays bounded
even on the 522K-doc Quora and 1.27M-doc BestBuy-full catalogs.
"""

import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# (label, data_dir, doc_id_field, measured_rescue_pp). Mirrors the 19-corpus
# calibration in probe_rescue_predictors.py. ESCI-Spanish kept for now even
# though bge-base is English-only — a clear "negative result" data point.
CORPORA = [
    ("BestBuy ACM", "bestbuy_acm_data", "product_id", 24.9),
    ("ESCI-Spanish", "esci_es_data", "product_id", 15.1),
    ("FiQA-2018", "fiqa_data", "doc_id", 13.0),
    ("SciFact", "scifact_data", "doc_id", 12.1),
    ("NFCorpus", "nfcorpus_data", "doc_id", 4.2),
    ("TREC-COVID", "trec_covid_data", "doc_id", 0.8),
    ("Quora", "quora_data", "doc_id", 14.0),
    ("SCIDOCS", "scidocs_data", "doc_id", 6.5),
    ("CQADup/programmers", "cqadupstack_programmers_data", "doc_id", 10.2),
    ("CQADup/gaming", "cqadupstack_gaming_data", "doc_id", 16.1),
    ("CQADup/tex", "cqadupstack_tex_data", "doc_id", 11.7),
    ("CQADup/gis", "cqadupstack_gis_data", "doc_id", 9.7),
    ("CQADup/mathematica", "cqadupstack_mathematica_data", "doc_id", 13.5),
    ("CQADup/physics", "cqadupstack_physics_data", "doc_id", 9.4),
    ("CQADup/stats", "cqadupstack_stats_data", "doc_id", 6.4),
    ("CQADup/unix", "cqadupstack_unix_data", "doc_id", 13.1),
    ("CQADup/webmasters", "cqadupstack_webmasters_data", "doc_id", 10.2),
    ("CQADup/android", "cqadupstack_android_data", "doc_id", 7.1),
    ("CQADup/english", "cqadupstack_english_data", "doc_id", 16.1),
    ("CQADup/wordpress", "cqadupstack_wordpress_data", "doc_id", 6.8),
]

ENCODER_NAME = "BAAI/bge-base-en-v1.5"


def compute_bag_stats(qrels_path, pid_to_idx, base_pv, doc_id_field, k_cap=50):
    """Returns (n_bags, median_size, median_spec) computed from qrels +
    already-encoded catalog. Skips singleton bags."""
    qrels = defaultdict(dict)
    with open(qrels_path) as f:
        for line in f:
            r = json.loads(line)
            field = (
                doc_id_field
                if doc_id_field in r
                else ("product_id" if "product_id" in r else "doc_id")
            )
            if r[field] not in pid_to_idx:
                continue
            qrels[r["query_id"]][r[field]] = r["relevance"]

    sizes, specs = [], []
    for _qid, doc_grades in qrels.items():
        idxs = [
            pid_to_idx[pid] for pid, rel in doc_grades.items() if pid in pid_to_idx and rel >= 1
        ][:k_cap]
        if len(idxs) < 2:
            continue
        bag_vecs = base_pv[idxs]
        centroid = bag_vecs.mean(axis=0)
        norm = float(np.linalg.norm(centroid))
        if norm < 1e-12:
            continue
        centroid /= norm
        spec = float(np.mean(bag_vecs @ centroid))
        sizes.append(len(idxs))
        specs.append(spec)
    if not sizes:
        return None
    return {
        "n_bags": len(sizes),
        "median_size": float(np.median(sizes)),
        "median_spec": float(np.median(specs)),
    }


def find_pids_file(data_dir):
    for name in ("doc_ids.json", "product_ids.json"):
        p = os.path.join(data_dir, name)
        if os.path.exists(p) and not os.path.islink(p):
            return p
    # symlinks ok
    for name in ("doc_ids.json", "product_ids.json"):
        p = os.path.join(data_dir, name)
        if os.path.exists(p):
            return p
    return None


def main():
    out_path = ROOT / "logs" / "bge_base_bag_stats.tsv"
    out_path.parent.mkdir(exist_ok=True)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"loading {ENCODER_NAME} on {device}...", flush=True)
    model = SentenceTransformer(ENCODER_NAME, device=device)

    headers = [
        "corpus",
        "n_bags",
        "median_size",
        "median_spec_bge",
        "rescue_pp",
        "data_dir",
    ]
    rows = []

    for label, data_dir, doc_id_field, rescue in CORPORA:
        catalog = os.path.join(data_dir, "titles.json")
        pids_path = find_pids_file(data_dir)
        # train_qrels.jsonl preferred (matches calibration); fall back to test.
        for q in ("train_qrels.jsonl", "test_qrels.jsonl"):
            qrels_path = os.path.join(data_dir, q)
            if os.path.exists(qrels_path):
                break
        if not os.path.exists(catalog) or not pids_path or not os.path.exists(qrels_path):
            print(f"[{label}] missing inputs — skipped", flush=True)
            continue

        print(f"\n[{label}] loading {catalog}...", flush=True)
        with open(catalog) as f:
            titles = json.load(f)
        with open(pids_path) as f:
            pids = json.load(f)
        pid_to_idx = {p: i for i, p in enumerate(pids)}
        print(f"[{label}] encoding {len(titles):,} docs under bge-base...", flush=True)
        pv = model.encode(
            titles,
            normalize_embeddings=True,
            batch_size=64,
            show_progress_bar=True,
        ).astype(np.float32)
        print(f"[{label}] computing bag stats from {qrels_path}...", flush=True)
        stats = compute_bag_stats(qrels_path, pid_to_idx, pv, doc_id_field)
        del pv
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        if stats is None:
            print(f"[{label}] no multi-positive bags — skipped", flush=True)
            continue
        row = {
            "corpus": label,
            "n_bags": stats["n_bags"],
            "median_size": stats["median_size"],
            "median_spec_bge": stats["median_spec"],
            "rescue_pp": rescue,
            "data_dir": data_dir,
        }
        rows.append(row)
        print(
            f"[{label}] n_bags={stats['n_bags']:,} median_size={stats['median_size']:.0f} "
            f"median_spec_bge={stats['median_spec']:.3f} rescue={rescue:.1f}pp",
            flush=True,
        )
        # Write incremental TSV after every corpus so a crash mid-run keeps
        # whatever finished.
        with open(out_path, "w") as f:
            f.write("\t".join(headers) + "\n")
            for r in rows:
                f.write("\t".join(str(r[h]) for h in headers) + "\n")

    print(f"\nfinal TSV at {out_path}", flush=True)


if __name__ == "__main__":
    main()
