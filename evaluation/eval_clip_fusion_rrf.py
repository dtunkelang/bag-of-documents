#!/usr/bin/env python3
"""ESCI-US image-fusion eval: RRF and z-score normalized fusion variants.

Naive weighted-sum fusion broke on score magnitude mismatch (eval_clip_fusion).
This adds:

  1. **RRF** (reciprocal rank fusion): rank-based, ignores score magnitudes.
     score(d) = sum_l 1 / (k + rank_l(d))   for lists l in {text, image}
  2. **Z-score fusion**: standardize each similarity per-query to mean 0,
     std 1 before weighted sum. Handles magnitude mismatch.

Both: image-bearing docs have a meaningful image rank/score; non-image docs
get a constant zero image-sim (and rank-tied at the end of the image list).
This is intentional — we want to test "when the image is available, can the
ranker use it?" without forcing a perfect-coverage assumption.
"""

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import open_clip
import torch
from sentence_transformers import SentenceTransformer

ESCI = Path("esci_us_data")
CLIP_MODEL = "ViT-B-32"
CLIP_PRETRAINED = "laion2b_s34b_b79k"
K = 10
TOP_M = 100  # take top-M from each ranker for RRF merge
MIN_RELEVANCE = 2
RRF_K = 60


def encode_queries(queries: list[str], device: str):
    print(f"encoding {len(queries):,} queries (MiniLM + CLIP)...", flush=True)
    text_enc = SentenceTransformer("all-MiniLM-L6-v2", device=device)
    text_qv = text_enc.encode(
        queries, normalize_embeddings=True, batch_size=256, show_progress_bar=False
    ).astype(np.float32)
    del text_enc
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

    clip_model, _, _ = open_clip.create_model_and_transforms(CLIP_MODEL, pretrained=CLIP_PRETRAINED)
    clip_model = clip_model.to(device).eval()
    tokenizer = open_clip.get_tokenizer(CLIP_MODEL)
    clip_qv_chunks = []
    BATCH = 256
    with torch.no_grad():
        for i in range(0, len(queries), BATCH):
            tokens = tokenizer(queries[i : i + BATCH]).to(device)
            v = clip_model.encode_text(tokens)
            v = v / v.norm(dim=-1, keepdim=True)
            clip_qv_chunks.append(v.cpu().float().numpy())
    clip_qv = np.concatenate(clip_qv_chunks, axis=0).astype(np.float32)
    del clip_model
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    return text_qv, clip_qv


def main():
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    with open(ESCI / "product_ids.json") as f:
        pids = json.load(f)
    pid_to_idx = {p: i for i, p in enumerate(pids)}

    queries_by_qid = {}
    with open(ESCI / "test_queries.jsonl") as f:
        for line in f:
            d = json.loads(line)
            queries_by_qid[d["query_id"]] = d["query"]
    pos = defaultdict(set)
    with open(ESCI / "test_qrels.jsonl") as f:
        for line in f:
            r = json.loads(line)
            if r["relevance"] < MIN_RELEVANCE or r["product_id"] not in pid_to_idx:
                continue
            pos[r["query_id"]].add(pid_to_idx[r["product_id"]])

    qids = sorted(queries_by_qid.keys())
    queries = [queries_by_qid[q] for q in qids]
    print(f"catalog={len(pids):,}  queries={len(qids):,}", flush=True)

    text_dv = np.load(ESCI / "base_catalog.vecs.fp16.npy").astype(np.float32)
    clip_image_vecs = np.load(ESCI / "clip_image_vecs.fp16.npy").astype(np.float32)
    clip_image_idx = np.load(ESCI / "clip_image_idx.npy").astype(np.int64)
    image_dv = np.zeros((len(pids), clip_image_vecs.shape[1]), dtype=np.float32)
    image_dv[clip_image_idx] = clip_image_vecs
    print(
        f"text_dv {text_dv.shape}  image_dv {image_dv.shape}  ({100 * len(clip_image_idx) / len(pids):.0f}% image-covered)",
        flush=True,
    )

    text_qv, clip_qv = encode_queries(queries, device)

    # Containers for per-query hits across methods/configs
    methods = {
        "text_only": [],
        "image_only": [],
        "rrf": [],
        "zscore_a0.5": [],
        "zscore_a0.7": [],
        "zscore_a0.8": [],
        "zscore_a0.9": [],
    }

    BATCH_Q = 200
    print(f"\nbatched retrieval (TOP_M={TOP_M} from each ranker for RRF):", flush=True)
    for q_start in range(0, len(qids), BATCH_Q):
        q_end = min(q_start + BATCH_Q, len(qids))
        sim_text = text_qv[q_start:q_end] @ text_dv.T  # (B, n_d)
        sim_image = clip_qv[q_start:q_end] @ image_dv.T  # (B, n_d)

        # Per-method top-K
        # 1. text only
        top_text = np.argpartition(-sim_text, K, axis=1)[:, :K]
        # 2. image only
        top_image = np.argpartition(-sim_image, K, axis=1)[:, :K]

        # 3. RRF: get top-M from each, merge by rank
        top_text_m_unsorted = np.argpartition(-sim_text, TOP_M, axis=1)[:, :TOP_M]
        # Sort the top-M by descending similarity
        rng = np.arange(q_end - q_start)[:, None]
        text_sort_order = np.argsort(-sim_text[rng, top_text_m_unsorted], axis=1)
        top_text_m = top_text_m_unsorted[rng, text_sort_order]
        top_image_m_unsorted = np.argpartition(-sim_image, TOP_M, axis=1)[:, :TOP_M]
        image_sort_order = np.argsort(-sim_image[rng, top_image_m_unsorted], axis=1)
        top_image_m = top_image_m_unsorted[rng, image_sort_order]

        # 4-7. z-score fusion at multiple alphas
        # Per-query: standardize each row to mean 0, std 1
        # (over the full doc list — same scale for both methods)
        z_text = (sim_text - sim_text.mean(axis=1, keepdims=True)) / (
            sim_text.std(axis=1, keepdims=True) + 1e-9
        )
        z_image = (sim_image - sim_image.mean(axis=1, keepdims=True)) / (
            sim_image.std(axis=1, keepdims=True) + 1e-9
        )

        for j in range(q_end - q_start):
            qid = qids[q_start + j]
            g = pos.get(qid, set())
            if not g:
                continue

            # text-only / image-only
            methods["text_only"].append((qid, len(g), len({int(x) for x in top_text[j]} & g)))
            methods["image_only"].append((qid, len(g), len({int(x) for x in top_image[j]} & g)))

            # RRF
            rrf_scores = defaultdict(float)
            for rank, d in enumerate(top_text_m[j]):
                rrf_scores[int(d)] += 1.0 / (RRF_K + rank)
            for rank, d in enumerate(top_image_m[j]):
                rrf_scores[int(d)] += 1.0 / (RRF_K + rank)
            top_rrf = sorted(rrf_scores.items(), key=lambda x: -x[1])[:K]
            rrf_set = {d for d, _ in top_rrf}
            methods["rrf"].append((qid, len(g), len(rrf_set & g)))

        # z-score fusion
        for alpha, name in [
            (0.5, "zscore_a0.5"),
            (0.7, "zscore_a0.7"),
            (0.8, "zscore_a0.8"),
            (0.9, "zscore_a0.9"),
        ]:
            fused = alpha * z_text + (1 - alpha) * z_image
            top = np.argpartition(-fused, K, axis=1)[:, :K]
            for j in range(q_end - q_start):
                qid = qids[q_start + j]
                g = pos.get(qid, set())
                if not g:
                    continue
                methods[name].append((qid, len(g), len({int(x) for x in top[j]} & g)))

        if (q_start // BATCH_Q) % 5 == 0:
            print(f"  processed {q_end:,}/{len(qids):,}", flush=True)

    # Summarize
    base = methods["text_only"]
    base_map = {qid: (g, h) for qid, g, h in base}
    base_r = np.mean([h / g for _, g, h in base])
    print(f"\nbase R@{K} (text only): {base_r:.4f}  (n={len(base):,})", flush=True)

    print(
        f"\n{'method':<15}  {'R@10':>7}  {'Δ vs text':>10}  {'miss rescue':>12}  {'mid Δ':>8}  {'perfect tax':>12}"
    )
    for name in ["image_only", "rrf", "zscore_a0.5", "zscore_a0.7", "zscore_a0.8", "zscore_a0.9"]:
        per_q = methods[name]
        r = np.mean([h / g for _, g, h in per_q])
        miss_d, mid_d, perf_d = [], [], []
        for qid, g, hit in per_q:
            bg, b_hit = base_map[qid]
            assert bg == g
            ratio = b_hit / g
            delta = (hit - b_hit) / g
            if ratio == 0:
                miss_d.append(delta)
            elif ratio == 1.0:
                perf_d.append(delta)
            else:
                mid_d.append(delta)
        print(
            f"{name:<15}  {r:>7.4f}  {r - base_r:+10.4f}  "
            f"{np.mean(miss_d) if miss_d else 0:+12.4f}  "
            f"{np.mean(mid_d) if mid_d else 0:+8.4f}  "
            f"{np.mean(perf_d) if perf_d else 0:+12.4f}"
        )


if __name__ == "__main__":
    main()
