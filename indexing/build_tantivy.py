#!/usr/bin/env python3
"""Build a tantivy index from titles.json with a configurable tokenizer.

Reads titles.json from the index directory and writes a new tantivy index to
<index_path>/tantivy_index_<suffix>/ (default suffix: _stem). Does not touch
FAISS, embeddings, or any existing tantivy index directory.

Once validated, swap into the canonical location with:
    mv <index_path>/tantivy_index <index_path>/tantivy_index_default
    mv <index_path>/tantivy_index_stem <index_path>/tantivy_index
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import json
import os
import time

import tantivy

from bagofdocs.utils import fmt_duration


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "index_path",
        nargs="?",
        default="combined_index_amazon",
        help="Index directory containing titles.json (default: combined_index_amazon)",
    )
    parser.add_argument(
        "--tokenizer",
        default="en_stem",
        help="Tantivy tokenizer name (default: en_stem)",
    )
    parser.add_argument(
        "--out-suffix",
        default="_stem",
        help="Suffix on output dir name, appended to 'tantivy_index' (default: _stem)",
    )
    args = parser.parse_args()

    titles_path = os.path.join(args.index_path, "titles.json")
    if not os.path.exists(titles_path):
        raise SystemExit(f"missing {titles_path}")

    out_dir = os.path.join(args.index_path, f"tantivy_index{args.out_suffix}")
    if os.path.exists(out_dir):
        raise SystemExit(f"refusing to overwrite existing {out_dir}")
    os.makedirs(out_dir)

    print(f"loading titles from {titles_path}...", flush=True)
    t0 = time.time()
    with open(titles_path) as f:
        titles = json.load(f)
    print(f"  loaded {len(titles):,} titles in {fmt_duration(time.time() - t0)}", flush=True)

    print(f"building tantivy index at {out_dir} (tokenizer={args.tokenizer})...", flush=True)
    schema_builder = tantivy.SchemaBuilder()
    schema_builder.add_text_field("title", stored=True, tokenizer_name=args.tokenizer)
    schema = schema_builder.build()

    idx = tantivy.Index(schema, path=out_dir)
    writer = idx.writer()

    t0 = time.time()
    for i, title in enumerate(titles):
        writer.add_document(tantivy.Document(title=title))
        if (i + 1) % 500_000 == 0:
            writer.commit()
            rate = (i + 1) / (time.time() - t0)
            print(f"  {i + 1:,} indexed ({rate:,.0f}/sec)", flush=True)
    writer.commit()
    print(f"  built in {fmt_duration(time.time() - t0)}", flush=True)

    # Sanity-check stemming
    idx.reload()
    s = idx.searcher()
    for q in ["eyes", "eye", "cards", "card", "tom AND ford AND eye"]:
        parsed = idx.parse_query(q, ["title"])
        n = s.search(parsed, 1).hits
        print(f"  verify {q!r}: {len(n)}+ hits")

    print(
        f"\ndone. validate by pointing kw_diagnose.py at {out_dir}, then atomic-swap when satisfied."
    )


if __name__ == "__main__":
    main()
