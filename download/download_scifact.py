#!/usr/bin/env python3
"""Download BEIR SciFact via ir_datasets, write in the shape our pipeline expects.

Mirrors download_fiqa.py / download_nfcorpus.py. SciFact is scientific-claim
verification — a small corpus (~5K docs) with dense relevance judgments,
useful as a low-cost calibration point for the rescue-rate framework.

Outputs:
    scifact_data/titles.json           — list of passage texts
    scifact_data/doc_ids.json          — parallel list of doc_ids
    scifact_data/product_ids.json      — symlink to doc_ids.json (pipeline compat)
    scifact_data/queries.jsonl         — train queries
    scifact_data/train_qrels.jsonl     — train qrels (renamed doc_id -> product_id)
    scifact_data/test_queries.jsonl    — test queries
    scifact_data/test_qrels.jsonl      — test qrels
"""

import json
import os

import ir_datasets

OUT = "scifact_data"
os.makedirs(OUT, exist_ok=True)

print("Loading SciFact...")
docs_ds = ir_datasets.load("beir/scifact")
titles = []
doc_ids = []
for doc in docs_ds.docs_iter():
    text = (doc.title or "") + " " + (doc.text or "")
    titles.append(text.strip()[:1500])
    doc_ids.append(doc.doc_id)
print(f"  {len(titles):,} documents")
with open(f"{OUT}/titles.json", "w") as f:
    json.dump(titles, f)
with open(f"{OUT}/doc_ids.json", "w") as f:
    json.dump(doc_ids, f)

prod_ids_path = f"{OUT}/product_ids.json"
if os.path.lexists(prod_ids_path):
    os.remove(prod_ids_path)
os.symlink("doc_ids.json", prod_ids_path)

train_ds = ir_datasets.load("beir/scifact/train")
train_queries = list(train_ds.queries_iter())
with open(f"{OUT}/queries.jsonl", "w") as f:
    for q in train_queries:
        f.write(json.dumps({"query_id": q.query_id, "query": q.text}) + "\n")
print(f"  {len(train_queries):,} train queries written")

with open(f"{OUT}/train_qrels.jsonl", "w") as f:
    n = 0
    for qr in train_ds.qrels_iter():
        if qr.relevance < 1:
            continue
        f.write(
            json.dumps(
                {"query_id": qr.query_id, "product_id": qr.doc_id, "relevance": qr.relevance}
            )
            + "\n"
        )
        n += 1
print(f"  {n:,} train qrels written")

test_ds = ir_datasets.load("beir/scifact/test")
test_queries = list(test_ds.queries_iter())
with open(f"{OUT}/test_queries.jsonl", "w") as f:
    for q in test_queries:
        f.write(json.dumps({"query_id": q.query_id, "query": q.text}) + "\n")
print(f"  {len(test_queries):,} test queries written")

with open(f"{OUT}/test_qrels.jsonl", "w") as f:
    n = 0
    for qr in test_ds.qrels_iter():
        if qr.relevance < 1:
            continue
        f.write(
            json.dumps(
                {"query_id": qr.query_id, "product_id": qr.doc_id, "relevance": qr.relevance}
            )
            + "\n"
        )
        n += 1
print(f"  {n:,} test qrels written")

avg_qlen = sum(len(q.text) for q in train_queries) / max(1, len(train_queries))
avg_doclen = sum(len(t) for t in titles) / max(1, len(titles))
print(f"\nAvg query length: {avg_qlen:.0f} chars")
print(f"Avg doc length:   {avg_doclen:.0f} chars")
