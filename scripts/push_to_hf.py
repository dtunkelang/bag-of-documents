#!/usr/bin/env python3
"""Push BoD-as-reranker artifacts to the HuggingFace dataset and Space.

Run after `hf auth login`. Token is read from the SDK's saved location.

Dataset target (huggingface.co/datasets/dtunkelang/bag-of-documents):
  - combined_index/rerank_A.vecs.fp16.npy
  - combined_index/rerank_B.vecs.fp16.npy
  - query_model_6m_mnrl/   (renamed from local query_model_us_full_6m_mnrl/)
  - query_model_hardneg/   (renamed from local query_model_us_qrels_mnrl_hardneg/)
  - evaluation/regime_queries.jsonl
  - README.md (regenerated with new artifact docs)

Space target (huggingface.co/spaces/dtunkelang/bag-of-documents-demo):
  - app.py (from space_demo/app.py)
  - README.md (from space_demo/README.md)
  - requirements.txt (from space_demo/requirements.txt)

Usage:
    python scripts/push_to_hf.py --dry-run    # preview only
    python scripts/push_to_hf.py              # push dataset + space
    python scripts/push_to_hf.py --dataset    # dataset only
    python scripts/push_to_hf.py --space      # space only
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import sys
from pathlib import Path

from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parent.parent
DATASET_REPO = "dtunkelang/bag-of-documents"
SPACE_REPO = "dtunkelang/bag-of-documents-demo"


# Dataset artifacts: (local_path, repo_path, size_check)
DATASET_FILES = [
    (
        ROOT / "combined_index_us_minilm" / "rerank_A.vecs.fp16.npy",
        "combined_index/rerank_A.vecs.fp16.npy",
        900_000_000,  # ~934 MB
    ),
    (
        ROOT / "combined_index_us_minilm" / "rerank_B.vecs.fp16.npy",
        "combined_index/rerank_B.vecs.fp16.npy",
        900_000_000,
    ),
    (
        ROOT / "eval" / "regime_queries.jsonl",
        "evaluation/regime_queries.jsonl",
        1_000,
    ),
]

DATASET_FOLDERS = [
    (
        ROOT / "query_model_us_full_6m_mnrl",
        "query_model_6m_mnrl",
    ),
    (
        ROOT / "query_model_us_qrels_mnrl_hardneg",
        "query_model_hardneg",
    ),
    (
        ROOT / "combined_index_us_minilm" / "bm25s_index",
        "combined_index/bm25s_index",
    ),
]

SPACE_FILES = [
    (ROOT / "space_demo" / "app.py", "app.py"),
    (ROOT / "space_demo" / "README.md", "README.md"),
    (ROOT / "space_demo" / "requirements.txt", "requirements.txt"),
]


def human(n):
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def push_dataset(api, dry_run):
    print(f"\n=== Pushing to dataset {DATASET_REPO} ===")
    for local, repo_path, _min_size in DATASET_FILES:
        if not local.exists():
            print(f"  MISSING: {local}", file=sys.stderr)
            continue
        size = local.stat().st_size
        print(f"  file: {local} → {repo_path}  ({human(size)})")
        if not dry_run:
            api.upload_file(
                path_or_fileobj=str(local),
                path_in_repo=repo_path,
                repo_id=DATASET_REPO,
                repo_type="dataset",
                commit_message=f"add {repo_path}",
            )

    for local, repo_path in DATASET_FOLDERS:
        if not local.is_dir():
            print(f"  MISSING DIR: {local}", file=sys.stderr)
            continue
        total = sum(p.stat().st_size for p in local.rglob("*") if p.is_file())
        nfiles = sum(1 for p in local.rglob("*") if p.is_file())
        print(f"  folder: {local} → {repo_path}/  ({nfiles} files, {human(total)})")
        if not dry_run:
            api.upload_folder(
                folder_path=str(local),
                path_in_repo=repo_path,
                repo_id=DATASET_REPO,
                repo_type="dataset",
                commit_message=f"add {repo_path}/",
                ignore_patterns=["*.pyc", "__pycache__/*"],
            )

    # README
    readme = ROOT / "space_demo" / "DATASET_README.md"
    if readme.exists():
        print(f"  file: {readme} → README.md")
        if not dry_run:
            api.upload_file(
                path_or_fileobj=str(readme),
                path_in_repo="README.md",
                repo_id=DATASET_REPO,
                repo_type="dataset",
                commit_message="docs: document rerank artifacts",
            )


def push_space(api, dry_run):
    print(f"\n=== Pushing to space {SPACE_REPO} ===")
    for local, repo_path in SPACE_FILES:
        if not local.exists():
            print(f"  MISSING: {local}", file=sys.stderr)
            continue
        size = local.stat().st_size
        print(f"  file: {local} → {repo_path}  ({human(size)})")
        if not dry_run:
            api.upload_file(
                path_or_fileobj=str(local),
                path_in_repo=repo_path,
                repo_id=SPACE_REPO,
                repo_type="space",
                commit_message=f"update {repo_path}",
            )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--dataset", action="store_true")
    parser.add_argument("--space", action="store_true")
    args = parser.parse_args()

    do_dataset = args.dataset or not (args.dataset or args.space)
    do_space = args.space or not (args.dataset or args.space)

    api = HfApi()
    me = api.whoami()
    print(f"Authenticated as: {me['name']}")
    if args.dry_run:
        print("(dry-run — no uploads will happen)")

    if do_dataset:
        push_dataset(api, args.dry_run)
    if do_space:
        push_space(api, args.dry_run)

    print("\nDone.")


if __name__ == "__main__":
    main()
