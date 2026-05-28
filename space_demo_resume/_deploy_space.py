#!/usr/bin/env python3
"""Create and push the dtunkelang/resume-job-match HF Space (Gradio SDK).

The Gradio SDK just runs `python app.py`; app.py is a pure FastAPI/uvicorn app
that binds 0.0.0.0:$PORT. Idempotent: re-running re-uploads as a new commit
(triggers a rebuild).
"""

import sys
from pathlib import Path

from huggingface_hub import HfApi, create_repo

REPO_ID = "dtunkelang/resume-job-match"
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
        allow_patterns=["app.py", "resume_match_lib.py", "README.md", "requirements.txt"],
        commit_message="deploy resume->job matching FastAPI app",
    )
    print(f"pushed: https://huggingface.co/spaces/{REPO_ID}", flush=True)


if __name__ == "__main__":
    main()
