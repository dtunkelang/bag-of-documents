#!/usr/bin/env python3
"""Push first 50 docs, then query Solr to verify BM25 + KNN work."""

import itertools
import sys

import requests

sys.path.insert(0, "/Users/dtunkelang/bagofdocs/solr_jobs_demo")
import numpy as np
from push_docs import CORE, SOLR, stream_docs

# Clear, push 50, commit
requests.post(
    f"{SOLR}/solr/{CORE}/update", json={"delete": {"query": "*:*"}}, params={"commit": "true"}
)
batch = list(itertools.islice(stream_docs(), 50))
print(f"pushing {len(batch)} docs...")
r = requests.post(
    f"{SOLR}/solr/{CORE}/update/json/docs", json=batch, params={"commit": "true"}, timeout=120
)
r.raise_for_status()
print("ok.")

# Probe 1: BM25 on 'engineer'
print("\nBM25 'engineer' top-5:")
r = requests.get(
    f"{SOLR}/solr/{CORE}/select", params={"q": "title:engineer", "rows": 5, "fl": "id,title,score"}
)
for d in r.json()["response"]["docs"]:
    print(f"  {d['id']:>4} {d['score']:.3f}  {d['title'][:80]}")

# Probe 2: KNN with first doc's bge vec — should rank itself #1.
bge = np.load(
    "/Users/dtunkelang/bagofdocs/space_demo_jobs/_stage/bge_catalog.vecs.fp16.npy", mmap_mode="r"
)
qv = bge[0].astype(np.float32).tolist()
qv_str = "[" + ",".join(f"{x:.6f}" for x in qv) + "]"
print("\nKNN bge_vec topK=5 with doc 0's vector (expect id=0 first):")
r = requests.get(
    f"{SOLR}/solr/{CORE}/select",
    params={"q": f"{{!knn f=bge_vec topK=5}}{qv_str}", "fl": "id,title,score"},
)
for d in r.json()["response"]["docs"]:
    print(f"  {d['id']:>4} {d['score']:.4f}  {d['title'][:80]}")
