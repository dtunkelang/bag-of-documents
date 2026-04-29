#!/usr/bin/env python3
"""Download and filter Amazon ESCI US (English) via HuggingFace tasksource/esci.

Writes (in esci_us_data/):
  titles.json          — unique English product titles
  product_ids.json     — parallel list of product_ids
  train_queries.jsonl  — English train queries
  train_qrels.jsonl    — graded 0-3
  test_queries.jsonl   — English test queries
  test_qrels.jsonl     — test qrels
"""

import json
import os
from collections import defaultdict

from datasets import load_dataset

OUT = "esci_us_data"
os.makedirs(OUT, exist_ok=True)

LABEL_GRADE = {"Exact": 3, "Substitute": 2, "Complement": 1, "Irrelevant": 0}

print("Loading ESCI dataset (tasksource/esci)...")
train = load_dataset("tasksource/esci", split="train")
test = load_dataset("tasksource/esci", split="test")


def collect(ds, split):
    products = {}
    queries = {}
    qrels = []
    for row in ds:
        if row["product_locale"] != "us":
            continue
        if row["product_id"] not in products:
            products[row["product_id"]] = row["product_title"] or ""
        if row["query_id"] not in queries:
            queries[row["query_id"]] = row["query"]
        qrels.append((row["query_id"], row["product_id"], LABEL_GRADE[row["esci_label"]]))
    print(f"  {split}: {len(products):,} products, {len(queries):,} queries, {len(qrels):,} qrels")
    return products, queries, qrels


train_products, train_queries, train_qrels = collect(train, "train")
test_products, test_queries, test_qrels = collect(test, "test")

all_products = {**train_products, **test_products}
print(f"\nTotal unique English products: {len(all_products):,}")

product_ids = list(all_products.keys())
titles = [all_products[pid] for pid in product_ids]
with open(f"{OUT}/titles.json", "w") as f:
    json.dump(titles, f)
with open(f"{OUT}/product_ids.json", "w") as f:
    json.dump(product_ids, f)

with open(f"{OUT}/train_queries.jsonl", "w") as f:
    for qid, qtext in train_queries.items():
        f.write(json.dumps({"query_id": qid, "query": qtext}) + "\n")
with open(f"{OUT}/train_qrels.jsonl", "w") as f:
    for qid, pid, grade in train_qrels:
        f.write(json.dumps({"query_id": qid, "product_id": pid, "relevance": grade}) + "\n")

with open(f"{OUT}/test_queries.jsonl", "w") as f:
    for qid, qtext in test_queries.items():
        f.write(json.dumps({"query_id": qid, "query": qtext}) + "\n")
with open(f"{OUT}/test_qrels.jsonl", "w") as f:
    for qid, pid, grade in test_qrels:
        f.write(json.dumps({"query_id": qid, "product_id": pid, "relevance": grade}) + "\n")

rels_per_query = defaultdict(int)
for qid, _pid, grade in train_qrels:
    if grade >= 2:
        rels_per_query[qid] += 1
mean_rels = sum(rels_per_query.values()) / len(rels_per_query) if rels_per_query else 0
print(
    f"\nTrain queries with >= 2 Exact/Substitute: {sum(1 for n in rels_per_query.values() if n >= 2):,}"
)
print(f"Mean Exact+Substitute per train query: {mean_rels:.1f}")
