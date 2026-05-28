#!/usr/bin/env python3
"""Stage the resume-match runtime artifacts into a flat dir + push to the companion
HF dataset (dtunkelang/resume-job-match).

Mirrors (hardlink-or-copy) six files the Space needs at runtime:
  - e5_base_catalog.vecs.fp16.npy  (job catalog vectors, ~510 MB)
  - metadata.jsonl                 (full postings, for description seeks, ~1.5 GB)
  - resume_vecs.fp16.npy           (6904 precomputed resume vectors)
  - resume_records.json            (per-resume parsed features)
  - job_features.pkl               (per-job parsed constraint features)
  - job_offsets.npy                (byte offsets into metadata.jsonl)

Run once. Idempotent — re-running re-mirrors + re-uploads; HF content-addresses
(LFS dedups), so repeated runs are cheap.
"""

import os
import shutil
import sys
from pathlib import Path

from huggingface_hub import HfApi, create_repo

REPO_ID = "dtunkelang/resume-job-match"
REPO_TYPE = "dataset"
ROOT = Path("/Users/dtunkelang/bagofdocs")
UJ = ROOT / "unified_jobs"
CACHE = UJ / "resume_match_cache"
STAGE = ROOT / "space_demo_resume" / "_stage"

# (source_path, staged_name)
ARTIFACTS = [
    (UJ / "e5_base_catalog.vecs.fp16.npy", "e5_base_catalog.vecs.fp16.npy"),
    (UJ / "metadata.jsonl", "metadata.jsonl"),
    (CACHE / "resume_vecs.fp16.npy", "resume_vecs.fp16.npy"),
    (CACHE / "resume_records.json", "resume_records.json"),
    (CACHE / "job_features.pkl", "job_features.pkl"),
    (CACHE / "job_offsets.npy", "job_offsets.npy"),
]


def stage() -> None:
    print("staging artifacts (hardlink-or-copy)...", flush=True)
    for src, name in ARTIFACTS:
        dst = STAGE / name
        if not src.exists():
            raise SystemExit(f"missing: {src}")
        if dst.exists():
            dst.unlink()
        try:
            os.link(src, dst)
            print(f"  hardlinked {name}", flush=True)
        except OSError:
            shutil.copy2(src, dst)
            print(f"  copied {name}", flush=True)


def write_readme() -> None:
    (STAGE / "README.md").write_text(
        "---\nlicense: cc-by-4.0\n---\n\n"
        "# Bag-of-Documents — Resume→Job Matching Artifacts\n\n"
        "Companion artifacts for the [resume→job matching Space]"
        "(https://huggingface.co/spaces/dtunkelang/resume-job-match).\n\n"
        "Contains: e5-base-v2 job catalog vectors + full postings metadata (JSONL) + "
        "byte offsets + precomputed resume vectors and parsed resume/job constraint "
        "features for 6,904 synthetic resumes against 347,900 job postings.\n\n"
        "Source: [github.com/dtunkelang/bag-of-documents]"
        "(https://github.com/dtunkelang/bag-of-documents)\n"
    )


def push() -> None:
    api = HfApi()
    try:
        create_repo(REPO_ID, repo_type=REPO_TYPE, exist_ok=True)
        print(f"repo ready: {REPO_ID}", flush=True)
    except Exception as e:
        print(f"create_repo: {e}", flush=True)
    print(f"upload_folder -> {REPO_ID} (~2.1 GB; LFS will chunk)...", flush=True)
    api.upload_folder(
        folder_path=str(STAGE),
        repo_id=REPO_ID,
        repo_type=REPO_TYPE,
        commit_message="upload resume->job matching artifacts",
    )
    print("upload done", flush=True)


def main() -> None:
    STAGE.mkdir(parents=True, exist_ok=True)
    stage()
    write_readme()
    push()


if __name__ == "__main__":
    if "--dry" in sys.argv:
        STAGE.mkdir(parents=True, exist_ok=True)
        stage()
        write_readme()
        print("dry run done; skip upload", flush=True)
    else:
        main()
