#!/usr/bin/env python3
"""CLI: Run Cluster Hypothesis Score (CHS) on multiple corpora and tabulate.

Wraps `bagofdocs.cluster_hypothesis.compute_chs` with dataset adapters for
ESCI, NFCorpus, BestBuy ACM, and any BeIR/* corpus on HuggingFace Hub.

Quick start (run on a curated default set with the English MiniLM encoder):

    python evaluation/chs_corpus_compare.py

Use a multilingual encoder for fair English+Spanish comparison:

    python evaluation/chs_corpus_compare.py \\
        --encoder sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 \\
        --datasets esci_us_strict esci_us_relaxed esci_es_strict esci_es_relaxed

Add a BeIR dataset:

    python evaluation/chs_corpus_compare.py --datasets beir:scidocs beir:trec-covid

Dataset id formats:
    esci_us_<strict|relaxed>     reads esci_us_data/test_qrels.jsonl, etc.
    esci_es_<strict|relaxed>     reads esci_es_data/test_qrels.jsonl, etc.
    nfcorpus_strict              reads nfcorpus_data/test_qrels.jsonl
    bestbuy_acm                  reads bestbuy_acm_data/test_qrels.jsonl
                                 (build first with download/download_bestbuy_acm.py)
    beir:<name>                  downloads BeIR/<name> via HuggingFace
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bagofdocs.cluster_hypothesis import compute_chs, schs_verdict  # noqa: E402

SCRIPT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_local_jsonl(qrels_path, pids_path, titles_path, id_field):
    qrels = defaultdict(dict)
    with open(qrels_path) as f:
        for line in f:
            r = json.loads(line)
            qrels[r["query_id"]][r[id_field]] = r["relevance"]
    with open(pids_path) as f:
        pids = json.load(f)
    with open(titles_path) as f:
        titles = json.load(f)
    return dict(qrels), pids, titles


def load_beir(name, split="test"):
    """Load a BeIR/<name> dataset via HuggingFace.

    BEIR corpora ship qrels with score 1 (relevant) and sometimes 0
    (judged not-relevant), but most BEIR qrels are positives-only.
    Returns the same (qrels, pids, titles) tuple as load_local_jsonl.
    """
    from datasets import load_dataset

    print(f"  downloading BeIR/{name} corpus + qrels...", flush=True)
    corpus_ds = load_dataset(f"BeIR/{name}", "corpus", split="corpus")
    try:
        qrels_ds = load_dataset(f"BeIR/{name}-qrels", split=split)
    except Exception:
        qrels_ds = load_dataset(f"BeIR/{name}-qrels", split="test")

    pids = []
    titles = []
    seen = set()
    for row in corpus_ds:
        pid = row["_id"]
        if pid in seen:
            continue
        seen.add(pid)
        pids.append(pid)
        # Combine title + text where available; cap length to keep encoding fast.
        text = row.get("title", "") or ""
        if row.get("text"):
            text = (text + " " + row["text"]) if text else row["text"]
        titles.append(text[:1500])

    qrels = defaultdict(dict)
    for row in qrels_ds:
        qid = row["query-id"]
        pid = row["corpus-id"]
        qrels[qid][pid] = int(row["score"])
    return dict(qrels), pids, titles


def load_dataset_for(dataset_id):
    """Resolve a dataset id to (qrels, pids, titles, partition)."""
    if dataset_id.startswith("esci_us_"):
        partition = dataset_id.replace("esci_us_", "")
        q, p, t = load_local_jsonl(
            os.path.join(SCRIPT_DIR, "esci_us_data/test_qrels.jsonl"),
            os.path.join(SCRIPT_DIR, "esci_us_data/product_ids.json"),
            os.path.join(SCRIPT_DIR, "esci_us_data/titles.json"),
            "product_id",
        )
        return q, p, t, partition
    if dataset_id.startswith("esci_es_"):
        partition = dataset_id.replace("esci_es_", "")
        q, p, t = load_local_jsonl(
            os.path.join(SCRIPT_DIR, "esci_es_data/test_qrels.jsonl"),
            os.path.join(SCRIPT_DIR, "esci_es_data/product_ids.json"),
            os.path.join(SCRIPT_DIR, "esci_es_data/titles.json"),
            "product_id",
        )
        return q, p, t, partition
    if dataset_id.startswith("nfcorpus_"):
        partition = dataset_id.replace("nfcorpus_", "")
        q, p, t = load_local_jsonl(
            os.path.join(SCRIPT_DIR, "nfcorpus_data/test_qrels.jsonl"),
            os.path.join(SCRIPT_DIR, "nfcorpus_data/doc_ids.json"),
            os.path.join(SCRIPT_DIR, "nfcorpus_data/titles.json"),
            "doc_id",
        )
        return q, p, t, partition
    if dataset_id == "bestbuy_acm":
        q, p, t = load_local_jsonl(
            os.path.join(SCRIPT_DIR, "bestbuy_acm_data/test_qrels.jsonl"),
            os.path.join(SCRIPT_DIR, "bestbuy_acm_data/product_ids.json"),
            os.path.join(SCRIPT_DIR, "bestbuy_acm_data/titles.json"),
            "product_id",
        )
        return q, p, t, "strict"
    if dataset_id.startswith("beir:"):
        name = dataset_id.split(":", 1)[1]
        q, p, t = load_beir(name)
        return q, p, t, "strict"
    raise ValueError(f"unknown dataset id: {dataset_id!r}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--encoder", default="all-MiniLM-L6-v2")
    ap.add_argument(
        "--datasets",
        nargs="+",
        default=[
            "esci_us_strict",
            "esci_us_relaxed",
            "nfcorpus_strict",
            "beir:scidocs",
            "beir:trec-covid",
            "beir:climate-fever",
        ],
        help="One or more dataset ids; see module docstring for the format.",
    )
    ap.add_argument("--output", default="/tmp/chs_corpus_compare.jsonl")
    args = ap.parse_args()

    results = []
    for dataset_id in args.datasets:
        print(f"\n{'=' * 72}\nDATASET: {dataset_id}\n{'=' * 72}", flush=True)
        try:
            qrels, pids, titles, partition = load_dataset_for(dataset_id)
            res = compute_chs(qrels, pids, titles, args.encoder, partition)
            row = {
                "dataset": dataset_id,
                "partition": partition,
                "encoder": args.encoder,
                **res.to_dict(),
            }
            results.append(row)
            with open(args.output, "a") as f:
                f.write(json.dumps(row) + "\n")

            hchs_s = f"{res.hchs:.3f}" if res.has_explicit_negs else "n/a"
            si_s = f"{res.strong_inv_rate:.1%}" if res.has_explicit_negs else "n/a"
            print(
                f"  pos_bearing={res.n_pos_bearing:,}  "
                f"explicit_neg={res.n_explicit_neg:,}  "
                f"SCHS={res.schs:.3f}  HCHS={hchs_s}  strong_inv={si_s}",
                flush=True,
            )
            print(f"  Verdict: {schs_verdict(res.schs, res.n_pos_bearing)}", flush=True)
        except Exception as e:
            print(f"  FAILED: {e}", flush=True)
            import traceback

            traceback.print_exc()

    print(f"\n{'=' * 110}\nFINAL COMPARISON\n{'=' * 110}", flush=True)
    print(
        f"{'dataset':<25}  {'partition':<8}  {'n_pos':<8}  {'n_negq':<8}  "
        f"{'mean_intra':<10}  {'mean_neg':<10}  {'mean_rand':<10}  "
        f"{'SCHS':<6}  {'HCHS':<6}  {'strong_inv':<10}"
    )
    print("-" * 110)
    for r in results:
        hchs_s = f"{r['hchs']:.3f}" if r["has_explicit_negs"] else "n/a"
        si_s = f"{r['strong_inv_rate']:.1%}" if r["has_explicit_negs"] else "n/a"
        ineg_s = f"{r['mean_inter_neg']:.4f}" if r["has_explicit_negs"] else "n/a"
        print(
            f"{r['dataset']:<25}  {r['partition']:<8}  "
            f"{r['n_pos_bearing']:<8,}  {r['n_explicit_neg']:<8,}  "
            f"{r['mean_intra']:<10.4f}  {ineg_s:<10}  {r['mean_random']:<10.4f}  "
            f"{r['schs']:<6.3f}  {hchs_s:<6}  {si_s:<10}"
        )


if __name__ == "__main__":
    main()
