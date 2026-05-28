#!/usr/bin/env python3
"""Precompute caches for the resume->job matching demo (demo_resume_match.py).

Builds four artifacts under CACHE_DIR (idempotent; --force to rebuild):
  resume_vecs.fp16.npy   (N_res, 768) e5-base-v2 vecs for every resume `text`
                         (encoded with the "query: " prefix to share the job space)
  resume_records.json    per-resume feature dicts (name, headline, loc, text, sen,
                         years, degree, creds, loc_tok) aligned to the vecs
  job_features.pkl       per-job feature dicts (title, sen, remote, loc, loc_tok,
                         years_req, degree_req, cred_gates, clearance, workauth)
  job_offsets.npy        int64 byte offset of each metadata.jsonl line (for detail seeks)

The job_features parse is the slow part (regex over 347.9k descriptions); cached so
demo boot is fast. Resume embeddings in the parquet are the WRONG models (nv/bge-m3/
ada/...), so we re-encode `text` with e5-base-v2 to share the catalog's vector space.
"""

import argparse
import glob
import json
import os
import pickle
import time

import numpy as np
import pandas as pd

import resume_match_lib as L

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESUME_GLOB = os.path.join(SCRIPT_DIR, "scratch/resume_synth/data/*.parquet")
META = os.path.join(SCRIPT_DIR, "unified_jobs/metadata.jsonl")
CACHE_DIR = os.path.join(SCRIPT_DIR, "unified_jobs/resume_match_cache")
MODEL = "intfloat/e5-base-v2"
QPREFIX = "query: "

RESUME_VECS = os.path.join(CACHE_DIR, "resume_vecs.fp16.npy")
RESUME_RECS = os.path.join(CACHE_DIR, "resume_records.json")
JOB_FEATS = os.path.join(CACHE_DIR, "job_features.pkl")
JOB_OFFSETS = os.path.join(CACHE_DIR, "job_offsets.npy")


def build_resumes(device):
    print("loading resume parquet(s)...", flush=True)
    files = sorted(glob.glob(RESUME_GLOB))
    if not files:
        raise SystemExit(f"no resume parquet matched {RESUME_GLOB}")
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    recs = []
    keep_idx = []
    for i, (_, row) in enumerate(df.iterrows()):
        f = L.resume_features(row)
        if not f["headline"] or not f["loc"]:
            continue
        f["rid"] = i
        recs.append(f)
        keep_idx.append(i)
    print(f"  {len(recs):,} resumes with headline+location (of {len(df):,})", flush=True)

    from sentence_transformers import SentenceTransformer

    print(f"loading {MODEL} on {device}...", flush=True)
    m = SentenceTransformer(MODEL, device=device)
    t0 = time.time()
    vecs = m.encode(
        [QPREFIX + r["text"] for r in recs],
        normalize_embeddings=True,
        batch_size=32,
        show_progress_bar=True,
    ).astype(np.float16)
    print(f"  encoded {vecs.shape} in {time.time() - t0:.1f}s", flush=True)

    np.save(RESUME_VECS, vecs)
    with open(RESUME_RECS, "w") as fh:
        json.dump(recs, fh)
    print(f"  wrote {RESUME_VECS} + {RESUME_RECS}", flush=True)


def build_jobs():
    print("parsing job features + byte offsets from metadata.jsonl...", flush=True)
    feats = []
    offsets = []
    t0 = time.time()
    with open(META, "rb") as fh:
        while True:
            off = fh.tell()
            line = fh.readline()
            if not line:
                break
            offsets.append(off)
            d = json.loads(line)
            feats.append(L.job_features(d))
    print(f"  parsed {len(feats):,} jobs in {time.time() - t0:.1f}s", flush=True)

    with open(JOB_FEATS, "wb") as fh:
        pickle.dump(feats, fh, protocol=pickle.HIGHEST_PROTOCOL)
    np.save(JOB_OFFSETS, np.asarray(offsets, dtype=np.int64))
    print(f"  wrote {JOB_FEATS} + {JOB_OFFSETS}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="mps", help="device for e5-base-v2 encode (mps/cpu)")
    ap.add_argument("--force", action="store_true", help="rebuild even if caches exist")
    args = ap.parse_args()
    os.makedirs(CACHE_DIR, exist_ok=True)

    have_res = os.path.exists(RESUME_VECS) and os.path.exists(RESUME_RECS)
    have_job = os.path.exists(JOB_FEATS) and os.path.exists(JOB_OFFSETS)

    if args.force or not have_res:
        build_resumes(args.device)
    else:
        print(f"resume caches present, skipping (use --force): {RESUME_VECS}", flush=True)

    if args.force or not have_job:
        build_jobs()
    else:
        print(f"job caches present, skipping (use --force): {JOB_FEATS}", flush=True)

    print("done.", flush=True)


if __name__ == "__main__":
    main()
