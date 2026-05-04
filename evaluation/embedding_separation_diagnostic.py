#!/usr/bin/env python3
"""For each query, check whether there's a (positive, negative) pair with
higher product-product cosine than the closest (positive, positive) pair.

If yes, the embedding space can't linearly separate relevance for that query
even with a perfect query encoder — direct evidence that either the product
embeddings are confusable for this query, or a cross-encoder is needed.

For each query with >=2 positives and >=1 negative (in the catalog):
  pos_pos_max = max cosine between any two positives
  pos_neg_max = max cosine between any positive and any negative
  pos_pos_min = min cosine between any two positives
  pos_neg_min = min cosine between any positive and any negative

Aggregates:
  frac queries with pos_neg_max > pos_pos_min
    (some negative is closer to a positive than the worst positive pair)
  frac queries with pos_neg_max > pos_pos_max
    (some negative is closer to a positive than the BEST positive pair —
     unambiguous separability failure)

Encoders compared: rerank_A (6M-MNRL prod), rerank_B (qrels-hn), rerank_G
(ESCI-supervised). Two relevance partitions: E vs I (strict) and E+S vs I+C.
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
    print("loading qrels + product map...", flush=True)
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

    encoders = {
        "rerank_A_6M_MNRL": "rerank_A.vecs.fp16.npy",
        "rerank_B_qrels_hn": "rerank_B.vecs.fp16.npy",
        "rerank_G_esci_sup": "rerank_G.vecs.fp16.npy",
    }

    partitions = {
        "strict_E_vs_I": (lambda g: g == 3, lambda g: g == 0),
        "relaxed_ES_vs_IC": (lambda g: g >= 2, lambda g: g <= 1),
    }

    for enc_name, vec_file in encoders.items():
        path = os.path.join(INDEX_DIR, vec_file)
        print(f"\n=== {enc_name} ({vec_file}) ===", flush=True)
        pv = np.load(path).astype(np.float32)
        print(f"  pv shape: {pv.shape}", flush=True)

        for part_name, (is_pos, is_neg) in partitions.items():
            n_total = 0
            n_skipped = 0
            n_inverted_max = 0  # pos_neg_max > pos_pos_min
            n_inverted_strong = 0  # pos_neg_max > pos_pos_max
            gap_examples = []

            for qid, qr in qrels.items():
                pos_pids = [p for p, g in qr.items() if is_pos(g)]
                neg_pids = [p for p, g in qr.items() if is_neg(g)]
                pos_pos_idx = [pid_to_pos[p] for p in pos_pids if p in pid_to_pos]
                neg_pos_idx = [pid_to_pos[p] for p in neg_pids if p in pid_to_pos]
                if len(pos_pos_idx) < 2 or len(neg_pos_idx) < 1:
                    n_skipped += 1
                    continue
                n_total += 1

                pp = pv[pos_pos_idx]
                nn = pv[neg_pos_idx]

                pp_sims = pp @ pp.T
                pp_off = pp_sims[np.triu_indices(len(pp_sims), k=1)]
                pp_min = float(pp_off.min())
                pp_max = float(pp_off.max())

                pn_sims = pp @ nn.T
                pn_max = float(pn_sims.max())

                if pn_max > pp_min:
                    n_inverted_max += 1
                if pn_max > pp_max:
                    n_inverted_strong += 1

                gap = pn_max - pp_min
                gap_examples.append((gap, qid, pp_min, pp_max, pn_max))

            gap_examples.sort(reverse=True)
            print(
                f"  partition {part_name}: n_eligible={n_total} skipped={n_skipped}"
                f" pn_max>pp_min: {n_inverted_max} ({n_inverted_max / n_total:.1%})"
                f" pn_max>pp_max: {n_inverted_strong} ({n_inverted_strong / n_total:.1%})",
                flush=True,
            )
            print("    top-5 separability failures by gap (pn_max - pp_min):", flush=True)
            for gap, qid, pp_min, pp_max, pn_max in gap_examples[:5]:
                q = queries_all.get(qid, "?")
                print(
                    f"      qid={qid:<10} gap={gap:+.3f} pp=[{pp_min:.3f}, {pp_max:.3f}] "
                    f"pn_max={pn_max:.3f}  query={q!r}",
                    flush=True,
                )

            if part_name == "strict_E_vs_I" and enc_name == "rerank_A_6M_MNRL":
                # Save full per-query data for the strict partition under prod encoder
                out_data = []
                for gap, qid, pp_min, pp_max, pn_max in gap_examples:
                    out_data.append(
                        {
                            "qid": qid,
                            "query": queries_all.get(qid),
                            "pp_min": pp_min,
                            "pp_max": pp_max,
                            "pn_max": pn_max,
                            "gap": gap,
                        }
                    )
                out_path = "/tmp/embedding_separation.json"
                with open(out_path, "w") as f:
                    json.dump(out_data, f)
                print(f"\n  saved per-query data to {out_path}", flush=True)


if __name__ == "__main__":
    main()
