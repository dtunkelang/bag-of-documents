#!/usr/bin/env python3
"""ESCI-US image-fusion evaluation: CLIP image cosine fused with text cosine.

Setup:
    text_sim(q, d)  = MiniLM(query) . MiniLM(doc)
    image_sim(q, d) = CLIP_text(query) . CLIP_image(doc)   for image-bearing d
    image_sim(q, d) = 0                                    otherwise
    fused(q, d)     = alpha * text_sim + (1 - alpha) * image_sim

Sweeps alpha. Reports per-bucket R@10 deltas relative to text-only.

Pre-reqs:
    - esci_us_data/base_catalog.vecs.fp16.npy  (MiniLM doc vecs)
    - esci_us_data/clip_image_vecs.fp16.npy + clip_image_idx.npy (CLIP image vecs)
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
MIN_RELEVANCE = 2


def main():
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    print("loading catalog + qrels + queries...", flush=True)
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
            if r["relevance"] < MIN_RELEVANCE:
                continue
            if r["product_id"] not in pid_to_idx:
                continue
            pos[r["query_id"]].add(pid_to_idx[r["product_id"]])

    qids = sorted(queries_by_qid.keys())
    queries = [queries_by_qid[q] for q in qids]
    n_pos_q = sum(1 for q in qids if pos.get(q))
    print(f"  catalog={len(pids):,}  queries={len(qids):,}  pos-bearing={n_pos_q:,}", flush=True)

    print(f"\nloading MiniLM doc vecs ({ESCI / 'base_catalog.vecs.fp16.npy'})...", flush=True)
    text_dv = np.load(ESCI / "base_catalog.vecs.fp16.npy").astype(np.float32)
    text_d = text_dv.shape[1]
    print(f"  shape={text_dv.shape}", flush=True)

    print("loading CLIP image vecs + index...", flush=True)
    clip_image_vecs = np.load(ESCI / "clip_image_vecs.fp16.npy").astype(np.float32)
    clip_image_idx = np.load(ESCI / "clip_image_idx.npy").astype(np.int64)
    clip_d = clip_image_vecs.shape[1]
    print(
        f"  shape={clip_image_vecs.shape}  (covers {len(clip_image_idx):,} / {len(pids):,} = "
        f"{100 * len(clip_image_idx) / len(pids):.0f}% of catalog)",
        flush=True,
    )

    # Build a full-catalog CLIP-image matrix with zero rows for non-image docs.
    # The non-image rows score 0 against any query → don't perturb the text component.
    print("building full-catalog CLIP image matrix (zero-pad for non-image docs)...", flush=True)
    image_dv = np.zeros((len(pids), clip_d), dtype=np.float32)
    image_dv[clip_image_idx] = clip_image_vecs

    print(f"\nencoding {len(qids):,} queries with MiniLM and CLIP text...", flush=True)
    text_enc = SentenceTransformer("all-MiniLM-L6-v2", device=device)
    text_qv = text_enc.encode(
        queries, normalize_embeddings=True, batch_size=256, show_progress_bar=False
    ).astype(np.float32)
    assert text_qv.shape[1] == text_d
    del text_enc
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

    clip_model, _, _ = open_clip.create_model_and_transforms(CLIP_MODEL, pretrained=CLIP_PRETRAINED)
    clip_model = clip_model.to(device).eval()
    tokenizer = open_clip.get_tokenizer(CLIP_MODEL)
    # Encode in batches to fit in MPS memory
    clip_qv_chunks = []
    BATCH = 256
    with torch.no_grad():
        for i in range(0, len(queries), BATCH):
            tokens = tokenizer(queries[i : i + BATCH]).to(device)
            v = clip_model.encode_text(tokens)
            v = v / v.norm(dim=-1, keepdim=True)
            clip_qv_chunks.append(v.cpu().float().numpy())
    clip_qv = np.concatenate(clip_qv_chunks, axis=0).astype(np.float32)
    print(f"  clip_qv shape={clip_qv.shape}", flush=True)
    del clip_model
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

    alphas = [1.0, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.5, 0.4, 0.3, 0.2]
    # For each alpha, accumulate (qid, n_gold, hits) per query.
    hits_by_alpha: dict[float, list[tuple]] = {a: [] for a in alphas}

    print(
        f"\nchunked retrieval: full sim matrix would be ~{22458 * len(pids) * 4 / 1e9:.0f} GB; "
        f"processing in batches...",
        flush=True,
    )
    BATCH_Q = 200  # per-batch sim matrix: 200 * 1.2M * 4 = 960MB
    for q_start in range(0, len(qids), BATCH_Q):
        q_end = min(q_start + BATCH_Q, len(qids))
        text_batch = text_qv[q_start:q_end]
        clip_batch = clip_qv[q_start:q_end]
        sim_text = text_batch @ text_dv.T  # (B, n_d)
        sim_image = clip_batch @ image_dv.T  # (B, n_d)
        for alpha in alphas:
            fused = alpha * sim_text + (1 - alpha) * sim_image
            top = np.argpartition(-fused, K, axis=1)[:, :K]
            for j in range(q_end - q_start):
                qid = qids[q_start + j]
                g = pos.get(qid, set())
                if not g:
                    continue
                hit = len({int(x) for x in top[j]} & g)
                hits_by_alpha[alpha].append((qid, len(g), hit))
        if (q_start // BATCH_Q) % 5 == 0:
            print(f"  processed {q_end:,}/{len(qids):,} queries", flush=True)

    # Base R@10 = alpha=1.0 (pure text)
    base_per_q = hits_by_alpha[1.0]
    base_r = np.mean([h / g for _, g, h in base_per_q])
    print(f"\nbase R@{K} (text only): {base_r:.4f}  (n_evaluated={len(base_per_q):,})", flush=True)

    print("\nfusion sweep (alpha = text weight, 1-alpha = image weight):")
    print(
        f"  {'alpha':>6}  {'R@10':>7}  {'Δ vs text':>10}  "
        f"{'miss rescue':>12}  {'mid Δ':>7}  {'perfect tax':>12}"
    )
    base_map = {qid: (g, h) for qid, g, h in base_per_q}
    for alpha in alphas:
        per_q = hits_by_alpha[alpha]
        r = np.mean([h / g for _, g, h in per_q])
        miss_d, mid_d, perfect_d = [], [], []
        for qid, g, f_hit in per_q:
            bg, b_hit = base_map[qid]
            assert bg == g
            ratio = b_hit / g
            delta = (f_hit - b_hit) / g
            if ratio == 0:
                miss_d.append(delta)
            elif ratio == 1.0:
                perfect_d.append(delta)
            else:
                mid_d.append(delta)
        print(
            f"  {alpha:>6.2f}  {r:>7.4f}  {r - base_r:+10.4f}  "
            f"{np.mean(miss_d) if miss_d else 0:+12.4f}  "
            f"{np.mean(mid_d) if mid_d else 0:+7.4f}  "
            f"{np.mean(perfect_d) if perfect_d else 0:+12.4f}"
        )


if __name__ == "__main__":
    main()
