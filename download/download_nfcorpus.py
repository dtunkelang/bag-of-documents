#!/usr/bin/env python3
"""Download BEIR NFCorpus via ir_datasets, write in the shape our pipeline expects.

Outputs:
    nfcorpus_data/titles.json          — list of passage texts (the corpus)
    nfcorpus_data/doc_ids.json         — parallel list of doc_ids
    nfcorpus_data/queries.jsonl        — train queries for bag construction
    nfcorpus_data/test_queries.jsonl   — test queries for eval
    nfcorpus_data/test_qrels.jsonl     — test qrels for eval
"""

import json
import os

import ir_datasets

OUT = "nfcorpus_data"
os.makedirs(OUT, exist_ok=True)

print("Loading NFCorpus...")
docs_ds = ir_datasets.load("beir/nfcorpus")
titles = []
doc_ids = []
for doc in docs_ds.docs_iter():
    # Use title + text to match BEIR convention (title is often informative for PubMed)
    body = (doc.title + ". " + doc.text).strip() if doc.title else doc.text
    titles.append(body)
    doc_ids.append(doc.doc_id)
print(f"  {len(titles):,} documents")
with open(f"{OUT}/titles.json", "w") as f:
    json.dump(titles, f)
with open(f"{OUT}/doc_ids.json", "w") as f:
    json.dump(doc_ids, f)

train_ds = ir_datasets.load("beir/nfcorpus/train")
train_queries = list(train_ds.queries_iter())
with open(f"{OUT}/queries.jsonl", "w") as f:
    for q in train_queries:
        f.write(json.dumps({"query_id": q.query_id, "query": q.text}) + "\n")
print(f"  {len(train_queries):,} train queries written")

test_ds = ir_datasets.load("beir/nfcorpus/test")
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
            json.dumps({"query_id": qr.query_id, "doc_id": qr.doc_id, "relevance": qr.relevance})
            + "\n"
        )
        n += 1
print(f"  {n:,} test qrels written")

# Basic stats
avg_qlen = sum(len(q.text) for q in train_queries) / len(train_queries)
avg_doclen = sum(len(t) for t in titles) / len(titles)
print(f"\nAvg query length: {avg_qlen:.0f} chars")
print(f"Avg doc length:   {avg_doclen:.0f} chars")
