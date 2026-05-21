#!/usr/bin/env python3
"""Build an enriched product-text catalog from the raw ESCI dataset.

Currently the pipeline uses title-only (median ~108 chars). ESCI ships
richer fields: bullet_point (91% pop, median 622 chars), description (53%
pop, median 799 chars), brand (96.7%), color (70.3%). This script
concatenates them into a single field and saves a parallel "titles_enriched.json"
aligned with the existing product_ids.json order.

Format: "{title} . {brand} {color}. {bullet_point}"
Bullet text truncated to keep total within ~400 chars so it fits the
512-token bi-encoder limit comfortably.

Usage:
    .venv/bin/python download/build_enriched_titles.py --locale us
"""

import argparse
import json
import os

from datasets import load_dataset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--locale", choices=["us", "es", "jp"], required=True)
    ap.add_argument("--max-chars", type=int, default=400)
    args = ap.parse_args()

    data_dir = f"esci_{args.locale}_data"
    if not os.path.isdir(data_dir):
        raise SystemExit(f"{data_dir} not found")
    with open(os.path.join(data_dir, "product_ids.json")) as f:
        product_ids = json.load(f)
    print(f"{len(product_ids):,} products in {data_dir}")

    pid_set = set(product_ids)
    fields = {}

    for split in ("train", "test"):
        print(f"loading {split}...", flush=True)
        ds = load_dataset("tasksource/esci", split=split)
        for row in ds:
            if row["product_locale"] != args.locale:
                continue
            pid = row["product_id"]
            if pid not in pid_set or pid in fields:
                continue
            fields[pid] = {
                "title": row.get("product_title", "") or "",
                "bullet": row.get("product_bullet_point", "") or "",
                "brand": row.get("product_brand", "") or "",
                "color": row.get("product_color", "") or "",
            }
        print(f"  fields collected for {len(fields):,} pids so far")

    print(f"\nbuilding enriched text (max {args.max_chars} chars)...")
    enriched = []
    coverage = {"bullet": 0, "brand": 0, "color": 0, "any_richer": 0}
    for pid in product_ids:
        f = fields.get(pid, {})
        title = f.get("title", "")
        brand = f.get("brand", "")
        color = f.get("color", "")
        bullet = f.get("bullet", "")
        parts = [title]
        if brand:
            parts.append(brand)
            coverage["brand"] += 1
        if color:
            parts.append(color)
            coverage["color"] += 1
        text = ". ".join(parts)
        if bullet:
            coverage["bullet"] += 1
            coverage["any_richer"] += 1
            text = (text + ". " + bullet)[: args.max_chars]
        elif brand or color:
            coverage["any_richer"] += 1
        enriched.append(text)

    out_path = os.path.join(data_dir, "titles_enriched.json")
    with open(out_path, "w") as f:
        json.dump(enriched, f)
    print(f"saved {out_path}")
    n = len(product_ids)
    for k, v in coverage.items():
        print(f"  {k:<12s} {v:,}/{n:,} ({v / n:.1%})")
    import statistics

    lens = [len(t) for t in enriched]
    print(
        f"  enriched-text lengths: median={statistics.median(lens):.0f}  mean={statistics.mean(lens):.0f}  max={max(lens)}"
    )


if __name__ == "__main__":
    main()
