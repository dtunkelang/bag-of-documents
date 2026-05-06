#!/usr/bin/env python3
"""Build CHS-ready files from the BestBuy ACM SF Hackathon dataset (2012).

This is a PREPARATION script, not a downloader. The Kaggle dataset is gated
behind a Kaggle login + competition agreement, so you have to download it
yourself from:

    https://www.kaggle.com/competitions/acm-sf-chapter-hackathon-big

Then unpack the archive into `acm-sf-chapter-hackathon-big/` so this script
can find:

    acm-sf-chapter-hackathon-big/train.csv                          (clickthrough log)
    acm-sf-chapter-hackathon-big/product_data/products/products_*.xml  (256 XML files)

Once that's in place, run:

    python download/prepare_bestbuy_acm.py

The script treats clickthrough as implicit positive labels: for each query,
the set of distinct SKUs that any user clicked (in train.csv) is the
positive set. Queries with >=2 distinct clicked SKUs become eligible
multi-positive bags.

Outputs (in bestbuy_acm_data/):
    product_ids.json      list of SKUs (as strings)
    titles.json           parallel list of <name> from product XML
    test_queries.jsonl    {"query_id": ..., "query": ...}
    test_qrels.jsonl      {"query_id": ..., "product_id": ..., "relevance": 1}
                          (positives only; relevance is binary 1)

These files are CHS-ready: pass to `evaluation/cluster_hypothesis_score.py`
or add `bestbuy_acm` to `evaluation/chs_corpus_compare.py --datasets`.
"""

import csv
import json
import os
import re
import sys
from collections import defaultdict

KAGGLE_URL = "https://www.kaggle.com/competitions/acm-sf-chapter-hackathon-big"
SOURCE_DIR = "acm-sf-chapter-hackathon-big"
TRAIN_CSV = os.path.join(SOURCE_DIR, "train.csv")
XML_DIR = os.path.join(SOURCE_DIR, "product_data", "products")


def check_prerequisites():
    """Verify the Kaggle data has been manually downloaded and unpacked."""
    missing = []
    if not os.path.exists(TRAIN_CSV):
        missing.append(TRAIN_CSV)
    if not os.path.isdir(XML_DIR):
        missing.append(XML_DIR + "/")
    elif not any(f.endswith(".xml") for f in os.listdir(XML_DIR)):
        missing.append(XML_DIR + "/products_*.xml")

    if missing:
        print(
            "ERROR: prerequisite files not found. This script does NOT download\n"
            "automatically — Kaggle requires an account login and competition\n"
            "agreement. Manual steps:\n"
            f"  1. Go to {KAGGLE_URL}\n"
            "  2. Accept the competition rules and download the data archive.\n"
            f"  3. Unpack it into ./{SOURCE_DIR}/ so the following exist:\n"
            f"       {TRAIN_CSV}\n"
            f"       {XML_DIR}/products_*.xml\n"
            "  4. Re-run this script.\n"
            "\nMissing:\n  " + "\n  ".join(missing),
            file=sys.stderr,
        )
        sys.exit(1)


def main():
    check_prerequisites()

    out_dir = "bestbuy_acm_data"
    os.makedirs(out_dir, exist_ok=True)

    # 1. Click signal: query -> set of clicked SKUs.
    print("parsing train.csv for click signal...", flush=True)
    clicks = defaultdict(set)
    with open(TRAIN_CSV) as f:
        reader = csv.DictReader(f)
        for r in reader:
            clicks[r["query"].strip().lower()].add(r["sku"])

    # 2. Multi-positive queries.
    multi = {q: skus for q, skus in clicks.items() if len(skus) >= 2}
    print(f"  unique queries: {len(clicks):,}; multi-positive (>=2): {len(multi):,}", flush=True)

    # 3. Collect SKUs we need to resolve to names.
    needed_skus = set()
    for skus in multi.values():
        needed_skus.update(skus)
    print(f"  unique SKUs in multi-positive queries: {len(needed_skus):,}", flush=True)

    # 4. Stream product XML files extracting <sku> + <name>.
    # Each XML file is a flat <products>...<product>...</product>...</products>
    # block; we split on </product> and regex out the fields we need.
    print("\nparsing product XML files for needed SKUs...", flush=True)
    sku_to_name = {}
    sku_re = re.compile(r"<sku>(\d+)</sku>")
    name_re = re.compile(r"<name>([^<]+)</name>")
    xml_files = sorted(f for f in os.listdir(XML_DIR) if f.endswith(".xml"))
    for fi, fname in enumerate(xml_files):
        with open(os.path.join(XML_DIR, fname), encoding="utf-8") as f:
            content = f.read()
        for block in content.split("</product>"):
            sku_m = sku_re.search(block)
            if not sku_m:
                continue
            sku = sku_m.group(1)
            if sku not in needed_skus:
                continue
            name_m = name_re.search(block)
            if name_m and sku not in sku_to_name:
                name = name_m.group(1)
                # Unescape minimal HTML entities common in BestBuy data.
                name = name.replace("&amp;", "&").replace("&quot;", '"')
                name = name.replace("&apos;", "'").replace("&lt;", "<").replace("&gt;", ">")
                sku_to_name[sku] = name.strip()
        if (fi + 1) % 25 == 0 or fi == len(xml_files) - 1:
            print(
                f"  scanned {fi + 1}/{len(xml_files)} files, "
                f"resolved {len(sku_to_name):,}/{len(needed_skus):,} SKUs",
                flush=True,
            )

    print(f"\nresolved {len(sku_to_name):,}/{len(needed_skus):,} SKUs to names", flush=True)

    # 5. Write CHS-ready files.
    print("writing CHS-ready files...", flush=True)
    pids = sorted(sku_to_name.keys())
    titles = [sku_to_name[p] for p in pids]
    with open(os.path.join(out_dir, "product_ids.json"), "w") as f:
        json.dump(pids, f)
    with open(os.path.join(out_dir, "titles.json"), "w") as f:
        json.dump(titles, f)

    # Stable hash-based query_ids so the file is deterministic across runs.
    def qid_for(query):
        return f"q{hash(query) & 0xFFFFFFFF:08x}"

    n_query = 0
    n_qrel = 0
    queries_path = os.path.join(out_dir, "test_queries.jsonl")
    qrels_path = os.path.join(out_dir, "test_qrels.jsonl")
    seen_q = set()
    with open(queries_path, "w") as fq_out, open(qrels_path, "w") as fr_out:
        for query, skus in multi.items():
            resolved = [s for s in skus if s in sku_to_name]
            if len(resolved) < 2:
                continue
            qid = qid_for(query)
            if qid in seen_q:
                continue
            seen_q.add(qid)
            fq_out.write(json.dumps({"query_id": qid, "query": query}) + "\n")
            n_query += 1
            for s in resolved:
                fr_out.write(json.dumps({"query_id": qid, "product_id": s, "relevance": 1}) + "\n")
                n_qrel += 1

    print(f"  wrote {n_query:,} test queries, {n_qrel:,} qrels rows", flush=True)
    print(f"\ndone. Files in {out_dir}/", flush=True)


if __name__ == "__main__":
    main()
