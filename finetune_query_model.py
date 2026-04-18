#!/usr/bin/env python3
"""
Fine-tune a sentence transformer to predict bag-of-documents query vectors
from query text.

Loads query vectors from bags.jsonl, splits into train/val, fine-tunes
the base model (default: all-MiniLM-L6-v2) to map query text → bag centroid vector, and evaluates
by measuring nearest-neighbor recall on the held-out set.

Usage:
    python finetune_query_model.py bags.jsonl query_model/
    python finetune_query_model.py bags.jsonl query_model/ --epochs 20 --batch-size 64
    python finetune_query_model.py bags.jsonl query_model/ --loss mnrl
"""

import argparse
import json
import random
import sys
import time

import numpy as np
import torch
from sentence_transformers import (
    InputExample,
    SentenceTransformer,
    evaluation,
    losses,
)
from torch.utils.data import DataLoader

from utils import fmt_duration


def load_bags(bags_path):
    """Load bags from JSONL file. Returns list of (query_text, centroid_vector) pairs."""
    pairs = []
    with open(bags_path) as f:
        for line in f:
            bag = json.loads(line)
            if bag["num_results"] < 1:
                continue
            pairs.append(
                {
                    "query": bag["query"],
                    "vector": np.array(bag["query_vector"], dtype=np.float32),
                    "specificity": bag["specificity"],
                    "num_results": bag["num_results"],
                }
            )
    return pairs


def split_train_val(pairs, val_fraction=0.2, seed=42):
    """Split pairs into train and val sets."""
    rng = random.Random(seed)
    pairs = list(pairs)
    rng.shuffle(pairs)
    split = int(len(pairs) * (1 - val_fraction))
    return pairs[:split], pairs[split:]


def make_mse_examples(pairs):
    """Create InputExamples for MSE loss (query text → target vector)."""
    examples = []
    for p in pairs:
        examples.append(
            InputExample(
                texts=[p["query"]],
                label=p["vector"].tolist(),
            )
        )
    return examples


def make_mnrl_examples(pairs, seed=42):
    """Create InputExamples for Multiple Negatives Ranking Loss.

    For each query, pairs it with its nearest-neighbor query (by centroid
    similarity) as a positive, with in-batch negatives providing contrast.
    """
    # Build a matrix of all vectors for similarity computation
    vectors = np.stack([p["vector"] for p in pairs])
    # Normalize (should already be, but just in case)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    vectors = vectors / np.maximum(norms, 1e-8)

    # For each query, find its nearest neighbor as a positive pair
    # This trains the model to map similar queries to similar vectors
    sims = vectors @ vectors.T
    np.fill_diagonal(sims, -1)  # exclude self

    examples = []
    for i, p in enumerate(pairs):
        # Nearest neighbor as positive
        nn_idx = int(np.argmax(sims[i]))
        examples.append(
            InputExample(
                texts=[p["query"], pairs[nn_idx]["query"]],
                label=float(sims[i, nn_idx]),
            )
        )
    return examples


class MSEEmbeddingLoss(torch.nn.Module):
    """Loss that minimizes cosine distance between model output and target vector."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, sentence_features, labels):
        embeddings = self.model(sentence_features[0])["sentence_embedding"]
        # Normalize embeddings
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        # Target vectors — labels may arrive as a stacked tensor or list
        if isinstance(labels, torch.Tensor):
            targets = labels.to(embeddings.device).float()
        else:
            targets = torch.tensor(np.array(labels), dtype=torch.float32, device=embeddings.device)
        targets = torch.nn.functional.normalize(targets, p=2, dim=1)
        # Cosine similarity loss: minimize 1 - cos_sim
        cos_sim = torch.sum(embeddings * targets, dim=1)
        return (1 - cos_sim).mean()


class NeighborRecallEvaluator(evaluation.SentenceEvaluator):
    """Evaluate nearest-neighbor recall on a held-out set.

    For each val query, computes predicted nearest neighbors (from model
    embeddings) and ground-truth nearest neighbors (from bag centroids),
    then measures recall@k.
    """

    def __init__(self, val_pairs, all_pairs, k_values=(1, 5, 10)):
        super().__init__()
        self.val_pairs = val_pairs
        self.all_pairs = all_pairs
        self.k_values = k_values
        self.primary_metric = "recall@10"

        # Precompute ground-truth neighbor indices for val queries
        all_vectors = np.stack([p["vector"] for p in all_pairs])
        all_vectors = all_vectors / np.linalg.norm(all_vectors, axis=1, keepdims=True)
        self.all_vectors = all_vectors

        val_vectors = np.stack([p["vector"] for p in val_pairs])
        val_vectors = val_vectors / np.linalg.norm(val_vectors, axis=1, keepdims=True)

        # Ground truth: nearest neighbors by bag centroid similarity
        self.gt_sims = val_vectors @ all_vectors.T
        # Zero out self-matches using index lookup
        all_query_to_idx = {}
        for j, ap in enumerate(all_pairs):
            all_query_to_idx.setdefault(ap["query"], []).append(j)
        for i, vp in enumerate(val_pairs):
            for j in all_query_to_idx.get(vp["query"], []):
                self.gt_sims[i, j] = -1

        self.gt_rankings = np.argsort(-self.gt_sims, axis=1)

    def __call__(self, model, output_path=None, epoch=-1, steps=-1):
        # Encode val queries with the model
        val_texts = [p["query"] for p in self.val_pairs]
        all_texts = [p["query"] for p in self.all_pairs]

        val_embs = model.encode(val_texts, normalize_embeddings=True, show_progress_bar=False)
        all_embs = model.encode(all_texts, normalize_embeddings=True, show_progress_bar=False)

        # Predicted similarities
        pred_sims = val_embs @ all_embs.T
        # Zero out self-matches using precomputed index
        all_query_to_idx = {}
        for j, ap in enumerate(self.all_pairs):
            all_query_to_idx.setdefault(ap["query"], []).append(j)
        for i, vp in enumerate(self.val_pairs):
            for j in all_query_to_idx.get(vp["query"], []):
                pred_sims[i, j] = -1

        pred_rankings = np.argsort(-pred_sims, axis=1)

        # Compute recall@k
        results = {}
        for k in self.k_values:
            recalls = []
            for i in range(len(self.val_pairs)):
                gt_set = set(self.gt_rankings[i, :k].tolist())
                pred_set = set(pred_rankings[i, :k].tolist())
                recalls.append(len(gt_set & pred_set) / k)
            results[f"recall@{k}"] = np.mean(recalls)

        # Also compute mean cosine similarity between predicted and GT vectors
        # for val queries (how close is the predicted vector to the bag centroid?)
        cos_sims = []
        for i, vp in enumerate(self.val_pairs):
            gt_vec = vp["vector"]
            gt_vec = gt_vec / np.linalg.norm(gt_vec)
            pred_vec = val_embs[i]
            cos_sims.append(float(gt_vec @ pred_vec))
        results["mean_cos_sim"] = np.mean(cos_sims)

        # Print results
        parts = [f"Epoch {epoch}:" if epoch >= 0 else "Eval:"]
        for key, val in sorted(results.items()):
            parts.append(f"{key}={val:.4f}")
        print("  " + "  ".join(parts))

        # Return primary metric (higher is better)
        return results.get("recall@10", 0.0)


def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune sentence transformer on bag-of-documents query vectors"
    )
    parser.add_argument("bags_file", help="Bags JSONL file")
    parser.add_argument("output_dir", help="Directory to save fine-tuned model")
    parser.add_argument(
        "--epochs", type=int, default=15, help="Number of training epochs (default: 15)"
    )
    parser.add_argument(
        "--batch-size", type=int, default=32, help="Training batch size (default: 32)"
    )
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate (default: 2e-5)")
    parser.add_argument(
        "--val-fraction",
        type=float,
        default=0.2,
        help="Fraction of data for validation (default: 0.2)",
    )
    parser.add_argument(
        "--loss",
        choices=["mse", "mnrl"],
        default="mse",
        help="Loss function: mse (cosine to target vector) or "
        "mnrl (multiple negatives ranking) (default: mse)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument(
        "--min-results",
        type=int,
        default=2,
        help="Minimum relevant results to include query (default: 2)",
    )
    parser.add_argument(
        "--max-eval", type=int, default=5000, help="Max queries for evaluation (default: 5000)"
    )
    parser.add_argument(
        "--base-model",
        default="all-MiniLM-L6-v2",
        help="Base embedding model to fine-tune (default: all-MiniLM-L6-v2)",
    )
    args = parser.parse_args()

    # Load data
    print(f"Loading bags from {args.bags_file}...")
    all_pairs = load_bags(args.bags_file)
    print(f"Loaded {len(all_pairs)} bags")

    # Filter by minimum results
    all_pairs = [p for p in all_pairs if p["num_results"] >= args.min_results]
    print(f"After filtering (min {args.min_results} results): {len(all_pairs)}")

    if len(all_pairs) < 100:
        print("Too few bags to train. Need at least 100.", file=sys.stderr)
        sys.exit(1)

    # Split
    train_pairs, val_pairs = split_train_val(
        all_pairs, val_fraction=args.val_fraction, seed=args.seed
    )
    print(f"Train: {len(train_pairs)}, Val: {len(val_pairs)}")

    # Load model
    print(f"Loading model {args.base_model}...")
    model = SentenceTransformer(args.base_model)

    # Set device
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Using device: {device}")
    model.to(device)

    # Subsample for evaluation (full matrix is too large for 16GB)
    eval_val = val_pairs[: args.max_eval]
    eval_all = all_pairs[: args.max_eval * 5]
    print(f"Evaluation subset: {len(eval_val)} val, {len(eval_all)} total")

    # Evaluate baseline (before fine-tuning)
    print("\nBaseline (before fine-tuning):")
    evaluator = NeighborRecallEvaluator(eval_val, eval_all)
    evaluator(model)

    # Prepare training data and loss
    if args.loss == "mse":
        train_examples = make_mse_examples(train_pairs)
        train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=args.batch_size)
        train_loss = MSEEmbeddingLoss(model)
    elif args.loss == "mnrl":
        train_examples = make_mnrl_examples(train_pairs, seed=args.seed)
        train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=args.batch_size)
        train_loss = losses.MultipleNegativesRankingLoss(model)

    # Train
    warmup_steps = int(len(train_dataloader) * args.epochs * 0.1)
    total_steps = len(train_dataloader) * args.epochs

    print(
        f"\nTraining: {args.epochs} epochs, batch_size={args.batch_size}, "
        f"lr={args.lr}, loss={args.loss}"
    )
    print(f"Steps per epoch: {len(train_dataloader)}, warmup: {warmup_steps}, total: {total_steps}")

    start = time.time()
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=args.epochs,
        warmup_steps=warmup_steps,
        optimizer_params={"lr": args.lr},
        evaluator=evaluator,
        evaluation_steps=len(train_dataloader),  # evaluate each epoch
        output_path=args.output_dir,
        save_best_model=True,
        show_progress_bar=True,
    )
    elapsed = time.time() - start
    print(f"\nTraining completed in {fmt_duration(elapsed)}")

    # Save final model (may differ from best epoch saved during training)
    model.save(args.output_dir)

    # Final evaluation
    print("Final evaluation:")
    evaluator(model)

    # Print some example predictions using eval subset (already small enough)
    print("\nSample predictions (val set):")
    sample_pairs = val_pairs[:5]
    sample_texts = [p["query"] for p in sample_pairs]
    eval_texts = [p["query"] for p in eval_all]

    sample_embs = model.encode(sample_texts, normalize_embeddings=True)
    eval_embs = model.encode(eval_texts, normalize_embeddings=True)
    eval_vectors = np.stack([p["vector"] for p in eval_all])
    eval_vectors = eval_vectors / np.linalg.norm(eval_vectors, axis=1, keepdims=True)

    for i, vp in enumerate(sample_pairs):
        pred_sims = sample_embs[i] @ eval_embs.T
        pred_top = np.argsort(-pred_sims)[:5]

        gt_sims = vp["vector"] @ eval_vectors.T
        gt_top = np.argsort(-gt_sims)[:5]

        cos_to_gt = float(sample_embs[i] @ (vp["vector"] / np.linalg.norm(vp["vector"])))

        print(f"\n  Query: {vp['query']} (cos_to_centroid={cos_to_gt:.3f})")
        print(f"  GT neighbors:   {', '.join(eval_all[j]['query'] for j in gt_top[:5])}")
        print(f"  Pred neighbors: {', '.join(eval_all[j]['query'] for j in pred_top[:5])}")


if __name__ == "__main__":
    main()
