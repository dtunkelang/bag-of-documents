#!/usr/bin/env python3
"""Create and push the dtunkelang/bag-of-documents-jobs HF Space (Gradio SDK).

Idempotent: re-running re-uploads, which Spaces handles as a new commit (triggers a rebuild).
"""

import sys
from pathlib import Path

from huggingface_hub import HfApi, create_repo

REPO_ID = "dtunkelang/bag-of-documents-jobs"
HERE = Path(__file__).resolve().parent


def main() -> None:
    try:
        create_repo(REPO_ID, repo_type="space", space_sdk="gradio", exist_ok=True)
        print(f"space ready: {REPO_ID}", flush=True)
    except Exception as e:
        print(f"create_repo: {e}", flush=True)
        sys.exit(1)
    api = HfApi()
    api.upload_folder(
        folder_path=str(HERE),
        repo_id=REPO_ID,
        repo_type="space",
        allow_patterns=["app.py", "README.md", "requirements.txt"],
        commit_message="deploy unified jobs Gradio app",
    )
    print(f"pushed: https://huggingface.co/spaces/{REPO_ID}", flush=True)


if __name__ == "__main__":
    main()
