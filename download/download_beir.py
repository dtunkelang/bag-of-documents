#!/usr/bin/env python3
"""Generic BEIR corpus downloader via ir_datasets.

Replaces the per-corpus download_fiqa.py / download_scifact.py / download_nfcorpus.py
duplication. Takes a BEIR dataset id (e.g. arguana, trec-covid, fiqa, scifact,
nfcorpus) and writes the standard pipeline layout in `<dataset>_data/`.

Outputs:
    <dataset>_data/titles.json           — list of passage texts (title + body, capped 1500 chars)
    <dataset>_data/doc_ids.json          — parallel list of doc_ids
    <dataset>_data/product_ids.json      — symlink to doc_ids.json (pipeline compat)
    <dataset>_data/queries.jsonl         — train queries  (when train split exists)
    <dataset>_data/train_qrels.jsonl     — train qrels    (when train split exists)
    <dataset>_data/test_queries.jsonl    — test queries
    <dataset>_data/test_qrels.jsonl      — test qrels
"""

import argparse
import json
import os

import ir_datasets


def write_split(split_ds, out_dir, prefix):
    queries = list(split_ds.queries_iter())
    with open(f"{out_dir}/{prefix}_queries.jsonl", "w") as f:
        for q in queries:
            f.write(json.dumps({"query_id": q.query_id, "query": q.text}) + "\n")
    print(f"  {len(queries):,} {prefix} queries written")

    n = 0
    with open(f"{out_dir}/{prefix}_qrels.jsonl", "w") as f:
        for qr in split_ds.qrels_iter():
            if qr.relevance < 1:
                continue
            f.write(
                json.dumps(
                    {
                        "query_id": qr.query_id,
                        "product_id": qr.doc_id,
                        "relevance": qr.relevance,
                    }
                )
                + "\n"
            )
            n += 1
    print(f"  {n:,} {prefix} qrels written")
    return queries, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--dataset",
        required=True,
        help="BEIR dataset id (arguana, trec-covid, fiqa, scifact, nfcorpus, ...)",
    )
    ap.add_argument(
        "--out-dir",
        default=None,
        help="Output directory (default: <dataset>_data, with hyphens to underscores)",
    )
    args = ap.parse_args()

    out_dir = args.out_dir or f"{args.dataset.replace('-', '_')}_data"
    os.makedirs(out_dir, exist_ok=True)
    full_id = f"beir/{args.dataset}"

    print(f"Loading {full_id}...")
    docs_ds = ir_datasets.load(full_id)
    titles = []
    doc_ids = []
    for doc in docs_ds.docs_iter():
        title = doc.title if hasattr(doc, "title") and doc.title else ""
        text = doc.text or ""
        body = (title + " " + text).strip() if title else text
        titles.append(body[:1500])
        doc_ids.append(doc.doc_id)
    print(f"  {len(titles):,} documents")

    with open(f"{out_dir}/titles.json", "w") as f:
        json.dump(titles, f)
    with open(f"{out_dir}/doc_ids.json", "w") as f:
        json.dump(doc_ids, f)

    prod_ids_path = f"{out_dir}/product_ids.json"
    if os.path.lexists(prod_ids_path):
        os.remove(prod_ids_path)
    os.symlink("doc_ids.json", prod_ids_path)

    # Train split (some BEIR corpora don't have one).
    train_queries = None
    try:
        train_ds = ir_datasets.load(f"{full_id}/train")
        train_queries, _ = write_split(train_ds, out_dir, "train")
        with open(f"{out_dir}/queries.jsonl", "w") as f:
            for q in train_queries:
                f.write(json.dumps({"query_id": q.query_id, "query": q.text}) + "\n")
    except KeyError:
        print(f"  no train split for {full_id} — using base/test split for bag construction")

    # Test split. ArguAna has a single split with no /test suffix; fall back
    # to the base dataset id which carries queries+qrels.
    try:
        test_ds = ir_datasets.load(f"{full_id}/test")
    except KeyError:
        test_ds = ir_datasets.load(full_id)
        print(f"  {full_id} has no /test suffix; treating base id as test split")
    test_queries, _ = write_split(test_ds, out_dir, "test")

    if train_queries is None:
        # No train split — use test split for bags (typical of single-split BEIR datasets).
        with open(f"{out_dir}/queries.jsonl", "w") as f:
            for q in test_queries:
                f.write(json.dumps({"query_id": q.query_id, "query": q.text}) + "\n")
        with (
            open(f"{out_dir}/test_qrels.jsonl") as src,
            open(f"{out_dir}/train_qrels.jsonl", "w") as dst,
        ):
            dst.write(src.read())
        print("  copied test_qrels.jsonl -> train_qrels.jsonl (no separate train split exists)")

    avg_qlen = sum(len(q.text) for q in test_queries) / max(1, len(test_queries))
    avg_doclen = sum(len(t) for t in titles) / max(1, len(titles))
    print(f"\nAvg query length: {avg_qlen:.0f} chars")
    print(f"Avg doc length:   {avg_doclen:.0f} chars")


if __name__ == "__main__":
    main()
