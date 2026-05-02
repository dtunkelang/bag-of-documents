#!/usr/bin/env python3
"""Build a catalog-vocabulary spell-correction dictionary from product titles.

Outputs combined_index_us_minilm/spell_vocab.json (~3 MB) — a dict mapping
each lowercase token (alphanumeric, freq >= 2) to its catalog frequency.
Loaded at demo startup to feed pyspellchecker as a custom dictionary.

Why catalog vocab specifically: it preserves brand names, model numbers,
and product-specific jargon that generic English dictionaries either miss
or "correct" incorrectly. Distance-2 lookups against this vocab fix typos
like 'moniter' → 'monitor' while leaving 'samsung' / 'k380' intact.

Usage:
    python indexing/build_spell_vocab.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import re  # noqa: E402
import time  # noqa: E402
from collections import Counter  # noqa: E402

SCRIPT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INDEX_DIR = os.path.join(SCRIPT_DIR, "combined_index_us_minilm")
TOK_RE = re.compile(r"[a-z0-9]+")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--titles", default=os.path.join(INDEX_DIR, "titles.json"))
    ap.add_argument("--output", default=os.path.join(INDEX_DIR, "spell_vocab.json"))
    ap.add_argument("--min-freq", type=int, default=2)
    args = ap.parse_args()

    with open(args.titles) as f:
        titles = json.load(f)
    print(f"  {len(titles):,} titles", flush=True)

    print("tokenizing + counting...", flush=True)
    t0 = time.time()
    counter = Counter()
    for t in titles:
        for tok in TOK_RE.findall(t.lower()):
            counter[tok] += 1
    print(f"  {len(counter):,} unique tokens, {time.time() - t0:.0f}s", flush=True)

    filtered = {tok: freq for tok, freq in counter.items() if freq >= args.min_freq}
    print(f"  {len(filtered):,} tokens with freq >= {args.min_freq}", flush=True)

    with open(args.output, "w") as f:
        json.dump(filtered, f)
    size_mb = os.path.getsize(args.output) / 1e6
    print(f"  saved to {args.output} ({size_mb:.1f} MB)", flush=True)


if __name__ == "__main__":
    main()
