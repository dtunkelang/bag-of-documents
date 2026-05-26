#!/usr/bin/env python3
"""Stage the unified jobs artifacts into a flat dir + push to the companion HF dataset.

Steps:
 1. Pre-merge the 12 te3 query cache sources into 3 files
    (te3_queries.{vecs.fp16.npy,ids.json,sources.json}).
 2. Copy / hardlink the rest of unified_jobs/ artifacts into a staging dir.
 3. Create the HF dataset repo if missing.
 4. upload_folder to push.

Run once. Idempotent — re-running re-merges and re-uploads, which HF handles via
content-addressing (LFS dedups), so it's cheap on repeated runs.
"""

import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
from huggingface_hub import HfApi, create_repo

REPO_ID = "dtunkelang/bag-of-documents-jobs"
REPO_TYPE = "dataset"
ROOT = Path("/Users/dtunkelang/bagofdocs")
UJ = ROOT / "unified_jobs"
STAGE = ROOT / "space_demo_jobs" / "_stage"

# Query cache sources: (vec_path, ids_path, source_tag).
QUERY_CACHE_SOURCES: list[tuple[Path, Path, str]] = []
for stem in ("aug_titles", "aug_combos", "head_torso", "head_torso2"):
    QUERY_CACHE_SOURCES.append(
        (UJ / f"{stem}_te3_1024.vecs.fp16.npy", UJ / f"{stem}_te3_1024.ids.json", "aug")
    )
for sub in ("jobs_data", "jobs_data_linkedin", "jobs_data_jobstreet", "jobs_data_usajobs"):
    for split in ("train", "eval"):
        QUERY_CACHE_SOURCES.append(
            (
                ROOT / sub / f"{split}_queries_te3_1024.vecs.fp16.npy",
                ROOT / sub / f"{split}_queries_te3_1024.ids.json",
                "synth",
            )
        )

# Non-merged artifacts to mirror as-is into the dataset.
PASSTHROUGH = [
    "titles.json",
    "source_index.json",
    "metadata.jsonl",
    "bge_catalog.vecs.fp16.npy",
    "te3_catalog.vecs.fp16.npy",
    "te3_cache_canonical.json",
]


def merge_query_cache() -> None:
    print("merging te3 query cache...", flush=True)
    vecs_list = []
    ids_list: list[str] = []
    sources_list: list[str] = []
    for vec_p, id_p, tag in QUERY_CACHE_SOURCES:
        if not (vec_p.exists() and id_p.exists()):
            print(f"  SKIP missing: {vec_p.name}", flush=True)
            continue
        v = np.load(vec_p)
        with open(id_p) as f:
            ids = json.load(f)
        if v.shape[0] != len(ids):
            raise SystemExit(f"size mismatch in {vec_p}")
        vecs_list.append(v)
        ids_list.extend(ids)
        sources_list.extend([tag] * len(ids))
        print(f"  + {vec_p.name}: {v.shape[0]:,} rows ({tag})", flush=True)
    if not vecs_list:
        raise SystemExit("no query cache sources found")
    merged = np.concatenate(vecs_list, axis=0)
    print(f"merged: {merged.shape} {merged.dtype}", flush=True)
    np.save(STAGE / "te3_queries.vecs.fp16.npy", merged.astype(np.float16))
    # np.save adds .npy if not present; check + rename if so.
    extra = STAGE / "te3_queries.vecs.fp16.npy.npy"
    if extra.exists():
        extra.rename(STAGE / "te3_queries.vecs.fp16.npy")
    with open(STAGE / "te3_queries.ids.json", "w") as f:
        json.dump(ids_list, f)
    with open(STAGE / "te3_queries.sources.json", "w") as f:
        json.dump(sources_list, f)
    print(f"wrote: te3_queries.* ({merged.shape[0]:,} rows total)", flush=True)


def stage_passthrough() -> None:
    print("staging passthrough artifacts (hardlink-or-copy)...", flush=True)
    for name in PASSTHROUGH:
        src = UJ / name
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
    readme = STAGE / "README.md"
    readme.write_text(
        "---\nlicense: cc-by-4.0\n---\n\n"
        "# Bag-of-Documents — Unified Jobs Artifacts\n\n"
        "Companion artifacts for the [unified jobs search Space](https://huggingface.co/spaces/dtunkelang/bag-of-documents-jobs).\n\n"
        "Contains: titles + slim metadata + full metadata (JSONL) + bge-small + "
        "te3-large @ 1024 catalog vectors + pre-encoded te3 query cache (~196k "
        "popular queries) for 347,900 job postings across 4 corpora "
        "(Open-Apply, LinkedIn, JobStreet, USAJobs).\n\n"
        "Source: [github.com/dtunkelang/bag-of-documents](https://github.com/dtunkelang/bag-of-documents)\n"
    )


def push() -> None:
    api = HfApi()
    try:
        create_repo(REPO_ID, repo_type=REPO_TYPE, exist_ok=True)
        print(f"repo ready: {REPO_ID}", flush=True)
    except Exception as e:
        print(f"create_repo: {e}", flush=True)
    print(f"upload_folder -> {REPO_ID} (this is ~3.5 GB; LFS will chunk)...", flush=True)
    api.upload_folder(
        folder_path=str(STAGE),
        repo_id=REPO_ID,
        repo_type=REPO_TYPE,
        commit_message="upload unified jobs artifacts",
    )
    print("upload done", flush=True)


def main():
    STAGE.mkdir(parents=True, exist_ok=True)
    merge_query_cache()
    stage_passthrough()
    write_readme()
    push()


if __name__ == "__main__":
    if "--dry" in sys.argv:
        STAGE.mkdir(parents=True, exist_ok=True)
        merge_query_cache()
        stage_passthrough()
        write_readme()
        print("dry run done; skip upload", flush=True)
    else:
        main()
