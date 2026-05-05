#!/usr/bin/env python3
"""Pre-flight diagnostic for title augmentation.

Downloads ESCI products with the augmented `product_text` field, encodes the
subset of products that appear in test_qrels with base MiniLM under two
text variants (title-only and product_text). Recomputes the separability
diagnostic and inversion rate.

If product_text reduces strong-inversion rate / increases mean pp_max under
the same encoder, that's evidence the corpus geometry becomes more cluster-
hypothesis-aligned with augmented text. Green-lights full catalog re-encode.

Computes only over the ~10K products that appear in test_qrels (not the full
1.2M catalog) to keep the probe fast.
"""

import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
from datasets import load_dataset
from sentence_transformers import SentenceTransformer

SCRIPT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    print("loading test_qrels to find relevant products...", flush=True)
    qrels = defaultdict(dict)
    with open(os.path.join(SCRIPT_DIR, "esci_us_data/test_qrels.jsonl")) as f:
        for line in f:
            r = json.loads(line)
            qrels[r["query_id"]][r["product_id"]] = r["relevance"]
    qrels = dict(qrels)

    pids_in_qrels = {pid for qrs in qrels.values() for pid in qrs}
    print(f"  {len(pids_in_qrels):,} unique products in test_qrels", flush=True)

    print("\nstreaming ESCI test split for product_text fields...", flush=True)
    t0 = time.time()
    title_by_pid = {}
    product_text_by_pid = {}
    seen = 0
    for split in ("test", "train"):
        ds = load_dataset("tasksource/esci", split=split, streaming=True)
        for row in ds:
            seen += 1
            if row["product_locale"] != "us":
                continue
            pid = row["product_id"]
            if pid in pids_in_qrels and pid not in title_by_pid:
                title_by_pid[pid] = row.get("product_title") or ""
                product_text_by_pid[pid] = row.get("product_text") or row.get("product_title") or ""
            if seen % 200_000 == 0:
                print(
                    f"    scanned {seen:,} rows, found {len(title_by_pid):,}/{len(pids_in_qrels):,} products "
                    f"({time.time() - t0:.0f}s)",
                    flush=True,
                )
            if len(title_by_pid) >= len(pids_in_qrels):
                break
        if len(title_by_pid) >= len(pids_in_qrels):
            break
    print(
        f"  found {len(title_by_pid):,}/{len(pids_in_qrels):,} products in {time.time() - t0:.0f}s",
        flush=True,
    )

    pids_resolved = sorted(title_by_pid.keys())
    titles = [title_by_pid[p] for p in pids_resolved]
    product_texts = [product_text_by_pid[p] for p in pids_resolved]
    pid_to_idx = {p: i for i, p in enumerate(pids_resolved)}

    # Truncate product_text aggressively to fit MiniLM seq budget (256 tokens roughly = 1000-1500 chars)
    MAX_CHARS = 1200
    product_texts = [(t[:MAX_CHARS] + "...") if len(t) > MAX_CHARS else t for t in product_texts]

    avg_title_len = sum(len(t) for t in titles) / max(1, len(titles))
    avg_text_len = sum(len(t) for t in product_texts) / max(1, len(product_texts))
    print(
        f"\ntext length: title avg {avg_title_len:.0f} chars, "
        f"product_text (truncated to {MAX_CHARS}) avg {avg_text_len:.0f} chars",
        flush=True,
    )

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"\nloading base MiniLM on {device}...", flush=True)
    model = SentenceTransformer("all-MiniLM-L6-v2", device=device)

    print(f"encoding {len(titles):,} titles...", flush=True)
    t0 = time.time()
    pv_title = model.encode(
        titles,
        normalize_embeddings=True,
        batch_size=128,
        show_progress_bar=True,
    ).astype(np.float32)
    print(f"  done in {time.time() - t0:.0f}s, shape={pv_title.shape}", flush=True)

    print(f"encoding {len(product_texts):,} augmented product_texts...", flush=True)
    t0 = time.time()
    pv_text = model.encode(
        product_texts,
        normalize_embeddings=True,
        batch_size=64,  # smaller batch since longer sequences
        show_progress_bar=True,
    ).astype(np.float32)
    print(f"  done in {time.time() - t0:.0f}s, shape={pv_text.shape}", flush=True)

    np.save("/tmp/preflight_pv_title.npy", pv_title.astype(np.float16))
    np.save("/tmp/preflight_pv_text.npy", pv_text.astype(np.float16))
    with open("/tmp/preflight_pids.json", "w") as f:
        json.dump(pids_resolved, f)

    # Separability diagnostic on both
    print("\ncomputing separability diagnostic on both encodings...", flush=True)
    partitions = {
        "strict_E_vs_I": (lambda g: g == 3, lambda g: g == 0),
        "relaxed_ES_vs_IC": (lambda g: g >= 2, lambda g: g <= 1),
    }

    for variant, pv in [("title_only", pv_title), ("product_text", pv_text)]:
        print(f"\n--- {variant} ---", flush=True)
        for part_name, (is_pos, is_neg) in partitions.items():
            n_total = 0
            n_skipped = 0
            n_inv_max = 0
            n_inv_strong = 0
            pp_min_list = []
            pp_max_list = []
            pn_max_list = []
            for _qid, qr in qrels.items():
                pos_pids = [p for p, g in qr.items() if is_pos(g)]
                neg_pids = [p for p, g in qr.items() if is_neg(g)]
                pos_idx = [pid_to_idx[p] for p in pos_pids if p in pid_to_idx]
                neg_idx = [pid_to_idx[p] for p in neg_pids if p in pid_to_idx]
                if len(pos_idx) < 2 or len(neg_idx) < 1:
                    n_skipped += 1
                    continue
                n_total += 1
                pp = pv[pos_idx]
                nn = pv[neg_idx]
                pp_sims = pp @ pp.T
                pp_off = pp_sims[np.triu_indices(len(pp_sims), k=1)]
                pp_min = float(pp_off.min())
                pp_max = float(pp_off.max())
                pn_max = float((pp @ nn.T).max())
                pp_min_list.append(pp_min)
                pp_max_list.append(pp_max)
                pn_max_list.append(pn_max)
                if pn_max > pp_min:
                    n_inv_max += 1
                if pn_max > pp_max:
                    n_inv_strong += 1
            print(
                f"  {part_name}: n_eligible={n_total} skipped={n_skipped} "
                f"pn>pp_min: {n_inv_max} ({n_inv_max / n_total:.1%}) "
                f"pn>pp_max: {n_inv_strong} ({n_inv_strong / n_total:.1%})",
                flush=True,
            )
            print(
                f"    means: pp_min={np.mean(pp_min_list):.3f}  "
                f"pp_max={np.mean(pp_max_list):.3f}  "
                f"pn_max={np.mean(pn_max_list):.3f}",
                flush=True,
            )


if __name__ == "__main__":
    main()
