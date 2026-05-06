#!/usr/bin/env python3
"""Per-query-bucket breakdown of BoD's +17.5pp BestBuy lift.

Buckets each holdout query along several axes, then reports base/BoD R@10
within each bucket so we can see whether the lift is uniform or concentrated.

Axes:
  - length        # of whitespace tokens in the query
  - difficulty    base R@10 hit count (0 / partial / full)
  - pattern       heuristic class: brand_abbrev, tokenization_variant,
                  joined_brand, category_phrase, brand_product, other

Results:
  diagnose_bestbuy_lift.tsv      tab-separated per-bucket means + counts
  stdout                         human-readable summary
"""

import json
import os
import re
from collections import Counter, defaultdict

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

DATA_DIR = "bestbuy_acm_data"
OUT_TSV = "evaluation/diagnose_bestbuy_lift.tsv"


# ----- Pattern heuristics -----------------------------------------------------
# All operate on a lowercased query string with tokens split by whitespace.

KNOWN_BRAND_ABBREVS = {
    "ati",
    "lg",
    "hp",
    "rca",
    "jvc",
    "lcd",
    "led",
    "dvd",
    "vcr",
    "psp",
    "wii",
    "gps",
    "usb",
    "hdmi",
    "ssd",
    "ccd",
    "amd",
    "nec",
    "cd",
    "uhd",
    "tv",
    "ps3",
    "ps2",
    "ipod",
    "ipad",
    "mp3",
    "mp4",
    "3ds",
    "ds",
}

# Tokens that often appear in spaced-tokenization queries (e.g., "i pad 2",
# "blu ray", "dr dre", "ear phones"). The pattern check is: short-leading-token
# OR multi-token query whose joined form matches a known compound.
SPACED_PREFIXES = {"i", "e", "u", "x"}
SPACED_COMPOUNDS = {
    "ipad",
    "ipod",
    "iphone",
    "imac",
    "blueray",
    "blu-ray",
    "earphones",
    "earbuds",
    "drdre",
    "kindle",
    "ereader",
    "xbox",
    "playstation",
}


def classify_query(q):
    """Return (length, pattern_class)."""
    toks = q.strip().split()
    n = len(toks)

    # 1. Single-token brand abbreviation (e.g., "ati", "wii", "dvd")
    if n == 1:
        if toks[0] in KNOWN_BRAND_ABBREVS:
            return n, "brand_abbrev"
        # Single short joined token — likely a brand or compound name
        if len(toks[0]) <= 12 and re.match(r"^[a-z][a-z0-9-]*$", toks[0]):
            return n, "joined_brand"
        return n, "other"

    # 2. Spaced tokenization (e.g., "i pad 2", "blu ray", "ear phones",
    #    "dr dre", "lap top"). Heuristic: any token of length 1 OR a
    #    consecutive 2-token pair that joins to a known compound.
    joined = "".join(toks)
    has_short = any(len(t) <= 1 for t in toks) or any(t in SPACED_PREFIXES for t in toks)
    pair_joined = ["".join(toks[i : i + 2]) for i in range(n - 1)]
    pair_match = any(p in SPACED_COMPOUNDS for p in pair_joined)
    if has_short or pair_match or joined in SPACED_COMPOUNDS:
        return n, "tokenization_variant"

    # 3. Brand + product (multi-word with at least one capital-style brand-like
    #    token at the start). Hard to detect on lowercase text; fall back to:
    #    if first token is a known brand abbrev OR a recognizable single brand.
    if toks[0] in KNOWN_BRAND_ABBREVS or (len(toks[0]) >= 4 and len(toks[0]) <= 10):
        # Conservative: check if any token is digit-bearing (model number) or
        # if last token is a generic noun (case, charger, cable, etc.)
        product_tail = {
            "case",
            "charger",
            "cable",
            "cover",
            "mount",
            "stand",
            "battery",
            "remote",
            "screen",
            "filter",
            "lens",
        }
        if any(re.search(r"\d", t) for t in toks) or toks[-1] in product_tail:
            return n, "brand_product"

    # 4. Category phrase (multi-word generic noun phrase: "stereo system",
    #    "dvd storage", "blu ray player"). Heuristic: 2+ tokens and no digits.
    if n >= 2 and not any(re.search(r"\d", t) for t in toks):
        return n, "category_phrase"

    return n, "other"


def main():
    print("loading data...", flush=True)
    with open(os.path.join(DATA_DIR, "product_ids.json")) as f:
        pids = json.load(f)
    with open(os.path.join(DATA_DIR, "titles.json")) as f:
        titles = json.load(f)
    pid_to_idx = {p: i for i, p in enumerate(pids)}

    qids, queries = [], []
    with open(os.path.join(DATA_DIR, "holdout_queries.jsonl")) as f:
        for line in f:
            d = json.loads(line)
            qids.append(d["query_id"])
            queries.append(d["query"])

    qrels = defaultdict(set)
    with open(os.path.join(DATA_DIR, "holdout_qrels.jsonl")) as f:
        for line in f:
            r = json.loads(line)
            qrels[r["query_id"]].add(r["product_id"])

    gold_idxs = []
    for qid in qids:
        gold_idxs.append({pid_to_idx[p] for p in qrels.get(qid, ()) if p in pid_to_idx})

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    base_pv = np.load(os.path.join(DATA_DIR, "base_catalog.vecs.fp16.npy")).astype(np.float32)
    base = SentenceTransformer("all-MiniLM-L6-v2", device=device)
    bod = SentenceTransformer("query_model_bestbuy_bod", device=device)

    print("encoding queries (base)...", flush=True)
    base_qv = base.encode(
        queries, normalize_embeddings=True, batch_size=256, show_progress_bar=True
    ).astype(np.float32)
    print("encoding catalog (BoD)...", flush=True)
    bod_pv = bod.encode(
        titles, normalize_embeddings=True, batch_size=128, show_progress_bar=True
    ).astype(np.float32)
    print("encoding queries (BoD)...", flush=True)
    bod_qv = bod.encode(
        queries, normalize_embeddings=True, batch_size=256, show_progress_bar=True
    ).astype(np.float32)

    # Per-query top-10 hits.
    per_q = []  # (qid, query, n_pos, base_hits, bod_hits, n_tok, pattern)
    chunk = 1024
    k = 10
    for start in range(0, len(qids), chunk):
        end = min(start + chunk, len(qids))
        bsim = base_qv[start:end] @ base_pv.T
        dsim = bod_qv[start:end] @ bod_pv.T
        btopk = np.argpartition(-bsim, k, axis=1)[:, :k]
        dtopk = np.argpartition(-dsim, k, axis=1)[:, :k]
        for j, gi in enumerate(range(start, end)):
            g = gold_idxs[gi]
            if not g:
                continue
            bh = len({int(x) for x in btopk[j]} & g)
            dh = len({int(x) for x in dtopk[j]} & g)
            n_tok, pat = classify_query(queries[gi])
            per_q.append((qids[gi], queries[gi], len(g), bh, dh, n_tok, pat))

    # ----- Aggregate -------------------------------------------------------
    def agg(rows):
        if not rows:
            return None
        n = len(rows)
        base_r = sum(r[3] / r[2] if r[2] else 0 for r in rows) / n
        bod_r = sum(r[4] / r[2] if r[2] else 0 for r in rows) / n
        # E@1: top-1 hit ratio approximated via "any hit in top-1" — we don't
        # have rank info here, but bh > 0 doesn't imply rank-1. Skip E@1 in
        # this script and report R@10 only.
        return n, base_r, bod_r, bod_r - base_r

    print("\n" + "=" * 78)
    print("BoD lift on BestBuy holdout — per-bucket R@10")
    print("=" * 78)

    # By query length.
    print("\n--- by query length (#tokens) ---")
    print(f"  {'len':>5} {'n':>6} {'base':>8} {'BoD':>8} {'Δ':>8}")
    by_len = defaultdict(list)
    for r in per_q:
        by_len[r[5]].append(r)
    for L in sorted(by_len):
        a = agg(by_len[L])
        print(f"  {L:>5} {a[0]:>6,} {a[1]:>8.3f} {a[2]:>8.3f} {a[3]:>+8.3f}")

    # By base-difficulty bucket.
    print("\n--- by base difficulty (base R@10 hits / n_pos) ---")
    by_diff = defaultdict(list)
    for r in per_q:
        ratio = r[3] / r[2] if r[2] else 0
        if ratio == 0:
            bucket = "0.0 (base misses entirely)"
        elif ratio < 0.5:
            bucket = "0.0-0.5"
        elif ratio < 1.0:
            bucket = "0.5-1.0"
        else:
            bucket = "1.0 (base perfect)"
        by_diff[bucket].append(r)
    print(f"  {'bucket':<28} {'n':>6} {'base':>8} {'BoD':>8} {'Δ':>8}")
    for k_ in [
        "0.0 (base misses entirely)",
        "0.0-0.5",
        "0.5-1.0",
        "1.0 (base perfect)",
    ]:
        if k_ in by_diff:
            a = agg(by_diff[k_])
            print(f"  {k_:<28} {a[0]:>6,} {a[1]:>8.3f} {a[2]:>8.3f} {a[3]:>+8.3f}")

    # By pattern class.
    print("\n--- by query pattern (heuristic) ---")
    by_pat = defaultdict(list)
    for r in per_q:
        by_pat[r[6]].append(r)
    print(f"  {'pattern':<22} {'n':>6} {'base':>8} {'BoD':>8} {'Δ':>8}")
    for p in sorted(by_pat, key=lambda x: -len(by_pat[x])):
        a = agg(by_pat[p])
        print(f"  {p:<22} {a[0]:>6,} {a[1]:>8.3f} {a[2]:>8.3f} {a[3]:>+8.3f}")

    # Pattern × length cross-tab (small).
    print("\n--- pattern share of top-100 biggest-lift queries ---")
    sorted_q = sorted(per_q, key=lambda r: -((r[4] - r[3]) / r[2] if r[2] else 0))[:100]
    pat_counts = Counter(r[6] for r in sorted_q)
    for p, c in pat_counts.most_common():
        print(f"  {p:<22} {c:>4}")

    # Save full per-query TSV.
    print(f"\nwriting {OUT_TSV}...", flush=True)
    with open(OUT_TSV, "w") as f:
        f.write("qid\tquery\tn_pos\tbase_hits\tbod_hits\tlen\tpattern\n")
        for r in per_q:
            f.write("\t".join(str(x) for x in r) + "\n")
    print(f"  wrote {len(per_q):,} rows", flush=True)


if __name__ == "__main__":
    main()
