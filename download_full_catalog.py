#!/usr/bin/env python3
"""
Download all McAuley Lab Amazon product titles (33 categories),
deduplicate, and save a 20% random sample.

Downloads metadata files one at a time, extracts titles, then deletes
the compressed file to minimize disk usage.

Usage:
    python download_full_catalog.py
    python download_full_catalog.py --sample-pct 20  # default
    python download_full_catalog.py --sample-pct 100  # keep everything
"""

import argparse
import gzip
import json
import os
import random
import subprocess
from collections import Counter

CATEGORIES = [
    "All_Beauty",
    "Amazon_Fashion",
    "Appliances",
    "Arts_Crafts_and_Sewing",
    "Automotive",
    "Baby_Products",
    "Beauty_and_Personal_Care",
    "Books",
    "CDs_and_Vinyl",
    "Cell_Phones_and_Accessories",
    "Clothing_Shoes_and_Jewelry",
    "Digital_Music",
    "Electronics",
    "Gift_Cards",
    "Grocery_and_Gourmet_Food",
    "Handmade_Products",
    "Health_and_Household",
    "Health_and_Personal_Care",
    "Home_and_Kitchen",
    "Industrial_and_Scientific",
    "Kindle_Store",
    "Magazine_Subscriptions",
    "Movies_and_TV",
    "Musical_Instruments",
    "Office_Products",
    "Patio_Lawn_and_Garden",
    "Pet_Supplies",
    "Software",
    "Sports_and_Outdoors",
    "Subscription_Boxes",
    "Tools_and_Home_Improvement",
    "Toys_and_Games",
    "Video_Games",
]

BASE_URL = "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/meta_categories"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def download_category(category, output_dir):
    """Download and extract titles from one category. Returns list of (title, category)."""
    url = f"{BASE_URL}/meta_{category}.jsonl.gz"
    gz_path = os.path.join(output_dir, f"meta_{category}.jsonl.gz")

    print(f"  Downloading {category}...", flush=True)
    try:
        subprocess.run(
            [
                "curl",
                "-L",
                "-o",
                gz_path,
                "--connect-timeout",
                "30",
                "--retry",
                "3",
                "--retry-delay",
                "5",
                "-#",
                url,
            ],
            check=True,
            timeout=1800,
        )  # 30 min max per file
    except Exception as e:
        print(f"    FAILED: {e}")
        if os.path.exists(gz_path):
            os.remove(gz_path)
        return []

    titles = []
    with gzip.open(gz_path, "rt", encoding="utf-8") as f:
        for line in f:
            try:
                item = json.loads(line.strip())
                title = (item.get("title") or "").strip()
                if title and len(title) > 3:
                    titles.append((title, category))
            except json.JSONDecodeError:
                continue

    os.remove(gz_path)
    print(f"  {len(titles):,} products")
    return titles


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sample-pct", type=float, default=20, help="Percentage of products to keep (default: 20)"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default=os.path.join(SCRIPT_DIR, "full_catalog"))
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Track completed categories (lightweight checkpoint)
    done_path = os.path.join(args.output_dir, "done_categories.json")
    progress_path = os.path.join(args.output_dir, "progress.jsonl")
    done_cats = set()
    if os.path.exists(done_path):
        with open(done_path) as f:
            done_cats = set(json.load(f))
        print(f"Resumed: {len(done_cats)} categories already done")

    print(f"Downloading {len(CATEGORIES)} categories...")

    for cat in CATEGORIES:
        if cat in done_cats:
            print(f"  Skipping {cat} (already done)")
            continue
        titles = download_category(cat, args.output_dir)
        # Append to progress file
        with open(progress_path, "a") as f:
            for title, category in titles:
                json.dump({"title": title, "category": category}, f, ensure_ascii=False)
                f.write("\n")
        done_cats.add(cat)
        with open(done_path, "w") as f:
            json.dump(sorted(done_cats), f)

    # Dedup and sample by streaming through progress file
    print(f"\nDeduplicating and sampling from {progress_path}...")
    seen_titles = set()
    unique_count = 0
    sample_rate = args.sample_pct / 100.0
    random.seed(args.seed)

    output_path = os.path.join(args.output_dir, "titles_sampled.json")
    cat_counts = Counter()
    sampled_titles = []

    with open(progress_path) as f:
        for line in f:
            item = json.loads(line)
            title = item["title"]
            if title in seen_titles:
                continue
            seen_titles.add(title)
            unique_count += 1
            # Bernoulli sampling: include each title independently with probability sample_rate
            if random.random() < sample_rate:
                sampled_titles.append(title)
                cat_counts[item["category"]] += 1

    del seen_titles  # free memory

    print(f"Total unique products: {unique_count:,}")
    print(f"Sampled {args.sample_pct}%: {len(sampled_titles):,} products")

    with open(output_path, "w") as f:
        json.dump(sampled_titles, f)
    print(f"Saved to {output_path} ({os.path.getsize(output_path) / 1e6:.1f} MB)")

    # Clean up checkpoint files
    os.remove(progress_path)
    os.remove(done_path)
    print("Cleaned up checkpoint files")

    print(f"\nCategory distribution ({len(cat_counts)} categories):")
    for cat, count in cat_counts.most_common():
        print(f"  {count:>8,}  {cat}")


if __name__ == "__main__":
    main()
