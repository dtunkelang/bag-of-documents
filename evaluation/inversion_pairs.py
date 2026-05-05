#!/usr/bin/env python3
"""For each strong-inversion query, dump the (positive, negative) PAIR that
achieves pn_max plus the closest-pair-of-positives. Looking at the actual
product titles is what tells you whether the distinguishing axis is query-
independent (in the product text) or query-dependent (needs joint scoring).

Output a TSV (so it can be inspected line-by-line) sorted by gap descending,
plus a stratified sample by gap decile to a more readable text dump.
"""

import json
import os
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INDEX_DIR = os.path.join(SCRIPT_DIR, "combined_index_us_minilm")


def main():
    print("loading qrels + queries + product map...", flush=True)
    qrels = defaultdict(dict)
    with open(os.path.join(SCRIPT_DIR, "esci_us_data/test_qrels.jsonl")) as f:
        for line in f:
            r = json.loads(line)
            qrels[r["query_id"]][r["product_id"]] = r["relevance"]
    qrels = dict(qrels)

    queries_all = {}
    with open(os.path.join(SCRIPT_DIR, "esci_us_data/test_queries.jsonl")) as f:
        for line in f:
            d = json.loads(line)
            queries_all[d["query_id"]] = d["query"]

    with open(os.path.join(SCRIPT_DIR, "esci_us_data/product_ids.json")) as f:
        esci_pids = json.load(f)
    with open(os.path.join(SCRIPT_DIR, "esci_us_data/titles.json")) as f:
        esci_titles_arr = json.load(f)
    with open(os.path.join(INDEX_DIR, "titles.json")) as f:
        index_titles = json.load(f)

    title_to_pid = {t: p for p, t in zip(esci_pids, esci_titles_arr)}
    pid_to_pos = {}
    for i, t in enumerate(index_titles):
        p = title_to_pid.get(t)
        if p is not None:
            pid_to_pos[p] = i

    print("loading rerank_A.vecs.fp16.npy...", flush=True)
    pv = np.load(os.path.join(INDEX_DIR, "rerank_A.vecs.fp16.npy")).astype(np.float32)

    with open("/tmp/embedding_separation.json") as f:
        sep_data = json.load(f)

    # Index sep_data by qid for quick lookup of pn_max
    strong = [r for r in sep_data if r["pn_max"] > r["pp_max"]]
    strong.sort(key=lambda r: -r["gap"])
    print(f"strong inversions: {len(strong)}", flush=True)

    def pair_for_query(qid):
        qr = qrels[qid]
        pos_pids = [p for p, g in qr.items() if g == 3 and p in pid_to_pos]
        neg_pids = [p for p, g in qr.items() if g == 0 and p in pid_to_pos]
        if len(pos_pids) < 2 or len(neg_pids) < 1:
            return None
        pos_idx = [pid_to_pos[p] for p in pos_pids]
        neg_idx = [pid_to_pos[p] for p in neg_pids]
        pp = pv[pos_idx]
        nn = pv[neg_idx]
        pp_sims = pp @ pp.T
        np.fill_diagonal(pp_sims, -np.inf)
        pn_sims = pp @ nn.T
        pn_max_idx = np.unravel_index(np.argmax(pn_sims), pn_sims.shape)
        pos_for_pn = pos_pids[pn_max_idx[0]]
        neg_for_pn = neg_pids[pn_max_idx[1]]
        pn_max_val = float(pn_sims[pn_max_idx])
        pp_max_idx = np.unravel_index(np.argmax(pp_sims), pp_sims.shape)
        pos_a = pos_pids[pp_max_idx[0]]
        pos_b = pos_pids[pp_max_idx[1]]
        pp_max_val = float(pp_sims[pp_max_idx])
        pp_min_val = float(pp_sims[pp_sims > -np.inf].min())
        return {
            "qid": qid,
            "query": queries_all.get(qid, "?"),
            "pp_max": pp_max_val,
            "pp_min": pp_min_val,
            "pn_max": pn_max_val,
            "gap": pn_max_val - pp_min_val,
            "closest_pos_pair": (pos_a, pos_b),
            "pn_pair": (pos_for_pn, neg_for_pn),
            "pos_a_title": index_titles[pid_to_pos[pos_a]],
            "pos_b_title": index_titles[pid_to_pos[pos_b]],
            "pos_for_pn_title": index_titles[pid_to_pos[pos_for_pn]],
            "neg_for_pn_title": index_titles[pid_to_pos[neg_for_pn]],
        }

    # Build pairs for all strong inversions
    print("computing pairs for all strong inversions...", flush=True)
    out = []
    for r in strong:
        pair = pair_for_query(r["qid"])
        if pair is not None:
            out.append(pair)

    print(f"resolved pairs for {len(out)} queries", flush=True)

    # Save full data
    with open("/tmp/inversion_pairs.json", "w") as f:
        json.dump(out, f)

    # Stratified sample: 5 queries per gap decile (50 total)
    out.sort(key=lambda r: -r["gap"])
    n = len(out)
    sample_idx = []
    for d in range(10):
        start = d * n // 10
        end = (d + 1) * n // 10
        chunk = list(range(start, end))
        if not chunk:
            continue
        # Pick 5 evenly spread within the decile
        step = max(1, len(chunk) // 5)
        sample_idx.extend(chunk[::step][:5])

    sample = [out[i] for i in sample_idx]

    print(f"\n=== STRATIFIED SAMPLE ({len(sample)} queries across gap deciles) ===\n")
    for r in sample:
        print(
            f"qid={r['qid']:<10} gap={r['gap']:+.3f}  pn_max={r['pn_max']:.3f}  pp_max={r['pp_max']:.3f}"
        )
        print(f"  query: {r['query']!r}")
        print(f"  pos paired with neg: {r['pos_for_pn_title'][:120]}")
        print(f"  closest neg:         {r['neg_for_pn_title'][:120]}")
        print(f"  best-pos-pair (a):   {r['pos_a_title'][:120]}")
        print(f"  best-pos-pair (b):   {r['pos_b_title'][:120]}")
        print()


if __name__ == "__main__":
    main()
