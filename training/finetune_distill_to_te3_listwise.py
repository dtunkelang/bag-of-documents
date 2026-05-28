#!/usr/bin/env python3
"""Listwise distillation: bge-small + Dense(384->1024) student against te3-large.

Successor to finetune_distill_to_te3.py — the pointwise cosine variant
collapsed on retrieval (R@10 0.03) despite good pointwise approximation
(cos 0.79). Pointwise loss does not preserve listwise ranking; gold-vs-second
margins in 1024d are tighter than the 37deg angular noise the student inherits.

This script implements MarginMSE distillation against te3's similarity
function on a per-query candidate pool (te3 top-K from the unified-348k catalog).

The student only encodes queries; the catalog stays in te3 space. At inference,
cosine-retrieve student(q) against the existing te3_catalog.vecs.fp16.npy.

Inputs:
  --queries-file:   train_queries.jsonl  (just for query text)
  --query-vecs:     query_te3.vecs.fp16.npy  (n_train, 1024)
  --catalog-vecs:   te3_catalog.vecs.fp16.npy  (n_docs, 1024)
  --eval-queries-file / --eval-query-vecs: held-out R@10 sanity check
  --topk-cache:     path; (n_train, K) int32 doc-idx matrix, precomputed if absent
  --base:           sentence-transformers encoder (default bge-small-en-v1.5)

Loss (per query, per (pos, neg) triplet drawn from top-K):
  m_T = te3(q, d_pos) - te3(q, d_neg)            # teacher margin
  m_S = student(q) . te3(d_pos) - student(q) . te3(d_neg)
  L   = (m_S - m_T)^2

Usage:
  .venv/bin/python training/finetune_distill_to_te3_listwise.py \\
      --queries-file jobs_data/train_queries.jsonl \\
      --query-vecs jobs_data/train_queries_te3_1024.vecs.fp16.npy \\
      --catalog-vecs unified_jobs/te3_catalog.vecs.fp16.npy \\
      --topk-cache unified_jobs/train_topk_te3.npy \\
      --epochs 8 --batch-size 32 --triplets-per-query 4 --topk 50 \\
      --out query_model_jobs_te3distill_listwise
"""

import argparse
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
from sentence_transformers import SentenceTransformer, models


def cosine_sim_np(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-12)
    b = b / np.maximum(np.linalg.norm(b, axis=1, keepdims=True), 1e-12)
    return a @ b.T


def build_student(base: str, target_dim: int) -> SentenceTransformer:
    transformer = models.Transformer(base)
    pooling = models.Pooling(transformer.get_word_embedding_dimension(), pooling_mode="mean")
    src_dim = pooling.get_sentence_embedding_dimension()
    dense = models.Dense(
        in_features=src_dim,
        out_features=target_dim,
        activation_function=torch.nn.Identity(),
    )
    model = SentenceTransformer(modules=[transformer, pooling, dense])
    print(f"student: {base} ({src_dim}d) -> Dense -> {target_dim}d", flush=True)
    return model


def encode_with_grad(model: SentenceTransformer, texts: list[str], device: str) -> torch.Tensor:
    tokenized = model.tokenize(texts)
    moved = {}
    for k, v in tokenized.items():
        if hasattr(v, "to"):
            moved[k] = v.to(device)
        else:
            moved[k] = v
    out = model.forward(moved)
    return out["sentence_embedding"]


def precompute_topk(
    query_vecs: np.ndarray, catalog_vecs: np.ndarray, k: int, chunk: int = 128
) -> np.ndarray:
    """Per-query top-K doc indices into the catalog. Returns (n_q, k) int32."""
    n_q = query_vecs.shape[0]
    n_d = catalog_vecs.shape[0]
    q_norm = query_vecs / np.maximum(np.linalg.norm(query_vecs, axis=1, keepdims=True), 1e-12)
    d_norm = catalog_vecs / np.maximum(np.linalg.norm(catalog_vecs, axis=1, keepdims=True), 1e-12)
    out = np.zeros((n_q, k), dtype=np.int32)
    t0 = time.time()
    for i in range(0, n_q, chunk):
        sims = q_norm[i : i + chunk].astype(np.float32) @ d_norm.astype(np.float32).T
        # top-k along axis=1
        idx = np.argpartition(-sims, kth=k - 1, axis=1)[:, :k]
        # sort within top-k by score desc
        row_idx = np.arange(idx.shape[0])[:, None]
        sub_sims = sims[row_idx, idx]
        order = np.argsort(-sub_sims, axis=1)
        out[i : i + chunk] = idx[row_idx, order]
        if (i // chunk) % 10 == 0:
            elapsed = time.time() - t0
            print(f"  topk: {i + chunk:,}/{n_q:,}  ({elapsed:.1f}s)", flush=True)
    print(f"  topk done in {time.time() - t0:.1f}s ({n_q:,} queries x {n_d:,} docs)", flush=True)
    return out


def eval_r_at_10(
    student: SentenceTransformer,
    eval_queries: list[str],
    eval_query_vecs: np.ndarray,
    catalog_vecs_T: np.ndarray,
    device: str,
    bs: int = 256,
) -> tuple[float, float]:
    """R@10 and mean-rank-of-te3-top-1 doc, for sanity (proxy for te3 R@10)."""
    student.eval()
    with torch.no_grad():
        outs = []
        for i in range(0, len(eval_queries), bs):
            sub = eval_queries[i : i + bs]
            v = encode_with_grad(student, sub, device)
            outs.append(v.detach().cpu().numpy())
        s_vecs = np.concatenate(outs, axis=0).astype(np.float32)
    # normalize
    s_vecs = s_vecs / np.maximum(np.linalg.norm(s_vecs, axis=1, keepdims=True), 1e-12)
    te3_vecs = eval_query_vecs / np.maximum(
        np.linalg.norm(eval_query_vecs, axis=1, keepdims=True), 1e-12
    )
    # te3 top-1 doc per eval query = our "gold"
    te3_top1 = (te3_vecs.astype(np.float32) @ catalog_vecs_T).argmax(axis=1)
    student_sims = s_vecs @ catalog_vecs_T
    # rank of gold doc under student
    student_top10 = np.argpartition(-student_sims, kth=9, axis=1)[:, :10]
    hits = np.array([te3_top1[i] in student_top10[i] for i in range(len(eval_queries))])
    return float(hits.mean()), float(te3_top1.shape[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--queries-file", required=True)
    ap.add_argument("--query-vecs", required=True)
    ap.add_argument("--catalog-vecs", required=True)
    ap.add_argument("--eval-queries-file", default=None)
    ap.add_argument("--eval-query-vecs", default=None)
    ap.add_argument("--topk-cache", required=True)
    ap.add_argument("--topk", type=int, default=50)
    ap.add_argument("--triplets-per-query", type=int, default=4)
    ap.add_argument("--neg-min-rank", type=int, default=5)
    ap.add_argument("--base", default="BAAI/bge-small-en-v1.5")
    ap.add_argument("--target-dim", type=int, default=1024)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--val-fraction", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--out", required=True)
    ap.add_argument("--save-every-epoch", action="store_true")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Load data
    queries = []
    with open(args.queries_file) as f:
        for line in f:
            queries.append(json.loads(line)["query"])
    query_vecs = np.load(args.query_vecs).astype(np.float32)
    catalog_vecs = np.load(args.catalog_vecs).astype(np.float32)
    assert len(queries) == query_vecs.shape[0], (
        f"q text ({len(queries)}) != q vecs ({query_vecs.shape[0]})"
    )
    print(
        f"queries: {len(queries):,}  catalog: {catalog_vecs.shape[0]:,}  "
        f"dim: {catalog_vecs.shape[1]}",
        flush=True,
    )

    # Precompute or load top-K
    topk_cache = Path(args.topk_cache)
    if topk_cache.exists():
        topk = np.load(topk_cache)
        print(f"loaded top-{args.topk} cache: {topk.shape}", flush=True)
        if topk.shape != (len(queries), args.topk):
            raise SystemExit(f"cache shape {topk.shape} != ({len(queries)}, {args.topk})")
    else:
        print(f"precomputing te3 top-{args.topk} per query (one-shot)...", flush=True)
        topk = precompute_topk(query_vecs, catalog_vecs, args.topk)
        np.save(topk_cache, topk)
        print(f"saved {topk_cache}", flush=True)

    # Train/val split
    idx = list(range(len(queries)))
    rng.shuffle(idx)
    n_val = int(len(idx) * args.val_fraction)
    val_idx = sorted(idx[:n_val])
    train_idx = sorted(idx[n_val:])
    print(f"train: {len(train_idx):,}  val: {len(val_idx):,}", flush=True)

    # Build student
    student = build_student(args.base, args.target_dim)
    student.to(args.device)
    optim = torch.optim.AdamW(student.parameters(), lr=args.lr)

    # Precompute catalog norm and teacher sims for fast triplet sampling
    cat_norm = catalog_vecs / np.maximum(np.linalg.norm(catalog_vecs, axis=1, keepdims=True), 1e-12)
    q_norm = query_vecs / np.maximum(np.linalg.norm(query_vecs, axis=1, keepdims=True), 1e-12)

    # Pre-store teacher sims for each query's top-K (n_train, K)
    teacher_sims = np.zeros((len(queries), args.topk), dtype=np.float32)
    for i in range(len(queries)):
        teacher_sims[i] = q_norm[i] @ cat_norm[topk[i]].T
    print(
        f"teacher_sims: shape={teacher_sims.shape}  "
        f"mean={teacher_sims.mean():.3f} std={teacher_sims.std():.3f}",
        flush=True,
    )

    # Eval setup (optional)
    eval_queries = None
    eval_vecs = None
    catalog_T = cat_norm.T.copy()  # for fast student dot product (1024 x N)
    if args.eval_queries_file and args.eval_query_vecs:
        eval_queries = []
        with open(args.eval_queries_file) as f:
            for line in f:
                eval_queries.append(json.loads(line)["query"])
        eval_vecs = np.load(args.eval_query_vecs).astype(np.float32)
        assert len(eval_queries) == eval_vecs.shape[0]
        print(f"eval: {len(eval_queries):,} queries", flush=True)

    def do_eval(epoch_tag: str):
        if eval_queries is None:
            return
        r10, _ = eval_r_at_10(student, eval_queries, eval_vecs, catalog_T, args.device)
        print(f"  [{epoch_tag}] R@10 vs te3-top1: {r10:.4f}", flush=True)

    do_eval("baseline")

    # Training
    print(
        f"\ntraining {args.epochs} epochs  bs={args.batch_size}  "
        f"triplets/q={args.triplets_per_query}  topk={args.topk}",
        flush=True,
    )
    for epoch in range(1, args.epochs + 1):
        student.train()
        order = list(range(len(train_idx)))
        rng.shuffle(order)
        running_loss = 0.0
        n_batches = 0
        t0 = time.time()

        for i in range(0, len(order), args.batch_size):
            batch_pos = [train_idx[j] for j in order[i : i + args.batch_size]]
            texts = [queries[j] for j in batch_pos]

            # Sample triplets for each query
            pos_idx_per = []
            neg_idx_per = []
            for q_i in batch_pos:
                for _ in range(args.triplets_per_query):
                    pos_rank = rng.randint(0, args.neg_min_rank - 1)
                    neg_rank = rng.randint(args.neg_min_rank, args.topk - 1)
                    pos_idx_per.append(topk[q_i, pos_rank])
                    neg_idx_per.append(topk[q_i, neg_rank])

            pos_vecs = torch.tensor(
                cat_norm[pos_idx_per].reshape(
                    len(batch_pos), args.triplets_per_query, args.target_dim
                ),
                device=args.device,
                dtype=torch.float32,
            )
            neg_vecs = torch.tensor(
                cat_norm[neg_idx_per].reshape(
                    len(batch_pos), args.triplets_per_query, args.target_dim
                ),
                device=args.device,
                dtype=torch.float32,
            )

            # Teacher margins from precomputed teacher_sims
            t_pos = np.zeros((len(batch_pos), args.triplets_per_query), dtype=np.float32)
            t_neg = np.zeros((len(batch_pos), args.triplets_per_query), dtype=np.float32)
            for bi, q_i in enumerate(batch_pos):
                for ti in range(args.triplets_per_query):
                    sample_idx = bi * args.triplets_per_query + ti
                    pos_d = pos_idx_per[sample_idx]
                    neg_d = neg_idx_per[sample_idx]
                    # find rank in topk for sim lookup; faster to recompute from cat_norm:
                    t_pos[bi, ti] = q_norm[q_i] @ cat_norm[pos_d]
                    t_neg[bi, ti] = q_norm[q_i] @ cat_norm[neg_d]
            t_margin = torch.tensor(t_pos - t_neg, device=args.device, dtype=torch.float32)

            optim.zero_grad()
            student_vecs = encode_with_grad(student, texts, args.device)
            # L2-normalize the student vec for cosine-equivalent dot product
            s_norm = torch.nn.functional.normalize(student_vecs, dim=1).unsqueeze(1)
            # student dot doc vectors: (B, 1, D) x (B, T, D) -> (B, T)
            s_pos = (s_norm * pos_vecs).sum(dim=2)
            s_neg = (s_norm * neg_vecs).sum(dim=2)
            s_margin = s_pos - s_neg
            loss = torch.nn.functional.mse_loss(s_margin, t_margin)
            loss.backward()
            optim.step()

            running_loss += float(loss.item())
            n_batches += 1

        elapsed = time.time() - t0
        print(
            f"  epoch {epoch:2d}/{args.epochs}  train_loss={running_loss / n_batches:.5f}  "
            f"({elapsed:.1f}s)",
            flush=True,
        )
        do_eval(f"epoch {epoch}")
        if args.save_every_epoch:
            checkpoint = Path(args.out) / f"epoch_{epoch:02d}"
            checkpoint.mkdir(parents=True, exist_ok=True)
            student.save(str(checkpoint))
            print(f"  saved checkpoint {checkpoint}", flush=True)

    # Final save
    out_path = Path(args.out)
    out_path.mkdir(parents=True, exist_ok=True)
    student.save(str(out_path))
    print(f"\nsaved {out_path}", flush=True)


if __name__ == "__main__":
    main()
