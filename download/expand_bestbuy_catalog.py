#!/usr/bin/env python3
"""Build the full BestBuy 2012 catalog from the Kaggle XML — no click filter.

`prepare_bestbuy_acm.py` only kept SKUs that appeared in queries with
≥2 distinct clicked SKUs (the multi-positive subset, ~53K of ~1.2M).
For demo purposes we want the full ~1M-product catalog so retrieval looks
realistic; this script extracts all SKU+name pairs from the 256 XMLs and
writes a parallel `bestbuy_acm_full/` directory that retains the same
queries / qrels as `bestbuy_acm_data/` but with a much larger catalog.

Bag training data is unchanged — the bags reference the same SKUs that
existed in the 53K subset; the new SKUs are just additional non-positive
documents that appear in the retrieval space.

Usage:
    python download/expand_bestbuy_catalog.py
"""

import json
import os
import re
import shutil
import sys
from pathlib import Path

XML_DIR = "acm-sf-chapter-hackathon-big/product_data/products"
SUBSET_DIR = "bestbuy_acm_data"
OUT_DIR = "bestbuy_acm_full"


def main():
    if not os.path.isdir(XML_DIR):
        print(f"ERROR: {XML_DIR} not found. Re-extract the Kaggle archive.", file=sys.stderr)
        sys.exit(1)
    if not os.path.isdir(SUBSET_DIR):
        print(f"ERROR: {SUBSET_DIR} not found — run prepare_bestbuy_acm.py first.", file=sys.stderr)
        sys.exit(1)

    Path(OUT_DIR).mkdir(exist_ok=True)

    sku_re = re.compile(r"<sku>(\d+)</sku>")
    name_re = re.compile(r"<name>([^<]+)</name>")

    sku_to_name = {}
    xml_files = sorted(f for f in os.listdir(XML_DIR) if f.endswith(".xml"))
    print(f"scanning {len(xml_files)} XML files...", flush=True)
    for fi, fname in enumerate(xml_files):
        with open(os.path.join(XML_DIR, fname), encoding="utf-8") as f:
            content = f.read()
        for block in content.split("</product>"):
            sku_m = sku_re.search(block)
            if not sku_m:
                continue
            sku = sku_m.group(1)
            if sku in sku_to_name:
                continue
            name_m = name_re.search(block)
            if not name_m:
                continue
            name = name_m.group(1)
            name = name.replace("&amp;", "&").replace("&quot;", '"')
            name = name.replace("&apos;", "'").replace("&lt;", "<").replace("&gt;", ">")
            sku_to_name[sku] = name.strip()
        if (fi + 1) % 25 == 0 or fi == len(xml_files) - 1:
            print(f"  {fi + 1}/{len(xml_files)} files, {len(sku_to_name):,} SKUs", flush=True)

    pids = sorted(sku_to_name.keys())
    titles = [sku_to_name[p] for p in pids]
    print(f"\nfull catalog: {len(pids):,} products", flush=True)

    with open(os.path.join(OUT_DIR, "product_ids.json"), "w") as f:
        json.dump(pids, f)
    with open(os.path.join(OUT_DIR, "titles.json"), "w") as f:
        json.dump(titles, f)

    # Copy queries + qrels from the subset (they still reference the same SKUs;
    # those SKUs are still in the full catalog by construction).
    for f in [
        "holdout_queries.jsonl",
        "holdout_qrels.jsonl",
        "test_queries.jsonl",
        "test_qrels.jsonl",
    ]:
        src = os.path.join(SUBSET_DIR, f)
        dst = os.path.join(OUT_DIR, f)
        if os.path.exists(src):
            shutil.copyfile(src, dst)
    print(f"  copied queries + qrels from {SUBSET_DIR} -> {OUT_DIR}")

    # Sanity check: do the 53K subset SKUs all appear in the full set?
    with open(os.path.join(SUBSET_DIR, "product_ids.json")) as f:
        subset_pids = set(json.load(f))
    full_set = set(pids)
    missing = subset_pids - full_set
    if missing:
        print(
            f"  WARNING: {len(missing)} subset SKUs missing from full catalog "
            f"(should be 0 if XML matches)"
        )
    else:
        print("  ✓ all 53K subset SKUs present in the expanded catalog")


if __name__ == "__main__":
    main()
