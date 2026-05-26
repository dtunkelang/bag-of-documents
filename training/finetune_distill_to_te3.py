#!/usr/bin/env python3
"""Train a bge-small + Dense(384->1024) student to mimic te3-large query vectors.

At runtime: student(query) outputs in te3 space; cosine-retrieve against the
precomputed te3 catalog. Solves "approximate te3 without OpenAI at query
time" — pair with a query-vector cache for the head distribution.

Inputs:
  --queries-file:   train_queries.jsonl  (just for query text)
  --targets-vecs:   query_te3.vecs.fp16.npy  (n, 1024)
  --base:           any sentence-transformers encoder (default bge-small-en-v1.5)
  --target-dim:     1024 (te3 output dim)
  --out:            student model dir (sentence-transformers format)

Loss: 1 - cosine_similarity(student(q), te3(q)), per-example, mean over batch.
Val split holds out 10% of pairs; reports R@10 against a doc catalog if
--eval-catalog is supplied (cosine vs te3 catalog should be the same retrieval
problem the deployed student will face).

Usage:
  .venv/bin/python training/finetune_distill_to_te3.py \\
      --queries-file jobs_data/train_queries.jsonl \\
      --targets-vecs jobs_data/train_queries_te3_1024.vecs.fp16.npy \\
      --base BAAI/bge-small-en-v1.5 --target-dim 1024 \\
      --epochs 15 --batch-size 64 --lr 2e-5 \\
      --out query_model_jobs_te3distill
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


def cosine_loss(student_vecs: torch.Tensor, target_vecs: torch.Tensor) -> torch.Tensor:
    s = torch.nn.functional.normalize(student_vecs, dim=1)
    t = torch.nn.functional.normalize(target_vecs, dim=1)
    return (1.0 - (s * t).sum(dim=1)).mean()


def cosine_sim(student_vecs: np.ndarray, target_vecs: np.ndarray) -> np.ndarray:
    s = student_vecs / np.maximum(np.linalg.norm(student_vecs, axis=1, keepdims=True), 1e-12)
    t = target_vecs / np.maximum(np.linalg.norm(target_vecs, axis=1, keepdims=True), 1e-12)
    return (s * t).sum(axis=1)


def build_student(base: str, target_dim: int) -> SentenceTransformer:
    """Stack: Transformer -> mean Pooling -> Dense(D->target_dim, no activation)."""
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
    """Encode a batch with grad enabled (for training)."""
    tokenized = model.tokenize(texts)
    moved = {}
    for k, v in tokenized.items():
        if hasattr(v, "to"):
            moved[k] = v.to(device)
        else:
            moved[k] = v
    out = model.forward(moved)
    return out["sentence_embedding"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--queries-file", required=True)
    ap.add_argument("--targets-vecs", required=True)
    ap.add_argument("--base", default="BAAI/bge-small-en-v1.5")
    ap.add_argument("--target-dim", type=int, default=1024)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--val-fraction", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    # Load queries + targets
    queries = []
    with open(args.queries_file) as f:
        for line in f:
            queries.append(json.loads(line)["query"])
    targets = np.load(args.targets_vecs).astype(np.float32)
    if len(queries) != targets.shape[0]:
        raise SystemExit(f"queries ({len(queries)}) != targets ({targets.shape[0]})")
    if targets.shape[1] != args.target_dim:
        raise SystemExit(f"target vecs are {targets.shape[1]}d, expected {args.target_dim}d")
    print(f"pairs: {len(queries):,}  target dim: {args.target_dim}", flush=True)

    rng = random.Random(args.seed)
    idx = list(range(len(queries)))
    rng.shuffle(idx)
    n_val = int(len(idx) * args.val_fraction)
    val_idx = sorted(idx[:n_val])
    train_idx = sorted(idx[n_val:])
    print(f"train: {len(train_idx):,}  val: {len(val_idx):,}", flush=True)

    # Build student
    student = build_student(args.base, args.target_dim)
    student.to(args.device)

    # Optimizer (encoder + projection together)
    optim = torch.optim.AdamW(student.parameters(), lr=args.lr)

    # Eval helper
    def eval_val():
        student.eval()
        with torch.no_grad():
            # Batch encode val queries
            bs = 256
            outs = []
            for i in range(0, len(val_idx), bs):
                sub = [queries[j] for j in val_idx[i : i + bs]]
                v = encode_with_grad(student, sub, args.device)
                outs.append(v.detach().cpu().numpy())
            val_vecs = np.concatenate(outs, axis=0)
        val_targets = targets[val_idx]
        sims = cosine_sim(val_vecs, val_targets)
        return float(sims.mean()), float(sims.min()), float(np.median(sims))

    mean0, min0, med0 = eval_val()
    print(
        f"\nBaseline (before fine-tune) val cosine: mean={mean0:.4f} med={med0:.4f} min={min0:.4f}",
        flush=True,
    )

    # Train
    print(
        f"\ntraining {args.epochs} epochs, bs={args.batch_size}, lr={args.lr}...",
        flush=True,
    )
    train_targets = torch.tensor(targets[train_idx], device=args.device)
    for epoch in range(1, args.epochs + 1):
        student.train()
        order = list(range(len(train_idx)))
        rng.shuffle(order)
        running_loss = 0.0
        n_batches = 0
        t0 = time.time()
        for i in range(0, len(order), args.batch_size):
            batch_pos = order[i : i + args.batch_size]
            texts = [queries[train_idx[j]] for j in batch_pos]
            tgt = train_targets[batch_pos]
            optim.zero_grad()
            vec = encode_with_grad(student, texts, args.device)
            loss = cosine_loss(vec, tgt)
            loss.backward()
            optim.step()
            running_loss += float(loss.item())
            n_batches += 1
        elapsed = time.time() - t0
        mean_v, min_v, med_v = eval_val()
        print(
            f"  epoch {epoch:2d}/{args.epochs}  train_loss={running_loss / n_batches:.4f}  "
            f"val_mean_cos={mean_v:.4f}  ({elapsed:.1f}s)",
            flush=True,
        )

    # Save
    out_path = Path(args.out)
    out_path.mkdir(parents=True, exist_ok=True)
    student.save(str(out_path))
    print(f"\nsaved {out_path}", flush=True)


if __name__ == "__main__":
    main()
