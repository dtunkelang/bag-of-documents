#!/usr/bin/env bash
# Wait for the bge-base catalog to land, then run Rocchio sweeps on
# bge-base, bge-small, and te3. Safe to launch separately from the main chain
# (in case bash buffered the chain script before the Rocchio step was added).
set -euo pipefail
cd /Users/dtunkelang/bagofdocs

ROOT=/Users/dtunkelang/bagofdocs
PY=$ROOT/.venv/bin/python
DATA=$ROOT/unified_jobs
RESULTS=$ROOT/evaluation/results
TARGET=$DATA/bge_base_catalog.vecs.fp16.npy
PROGRESS=$DATA/bge_base_catalog.progress.json

ts() { date '+%Y-%m-%d %H:%M:%S'; }

echo "[$(ts)] rocchio_watcher waiting for bge-base catalog at $TARGET"

# Wait until catalog file exists AND progress.json shows all 7 chunks done.
# This guards against firing while the memmap is half-written.
until [ -f "$TARGET" ] && \
      [ -f "$PROGRESS" ] && \
      $PY -c "import json,sys; sys.exit(0 if len(json.load(open('$PROGRESS')))>=7 else 1)" 2>/dev/null; do
  sleep 60
done

echo "[$(ts)] catalog ready"

# Wait for the main chain to finish before running MPS-dependent Rocchio
# (live query encoding would conflict with the chain's e5/gte encoding on MPS).
# Pass the chain PID via $MAIN_CHAIN_PID env or fall back to PID file.
CHAIN_PID="${MAIN_CHAIN_PID:-$(cat /Users/dtunkelang/bagofdocs/logs/jobs_base_chain.pid 2>/dev/null || echo 0)}"
if [ "$CHAIN_PID" != "0" ]; then
  echo "[$(ts)] waiting for main chain PID=$CHAIN_PID to exit before MPS Rocchio runs..."
  while kill -0 "$CHAIN_PID" 2>/dev/null; do
    sleep 300
  done
  echo "[$(ts)] main chain done; starting Rocchio sweeps"
else
  echo "[$(ts)] no chain PID found; running Rocchio immediately"
fi

echo "[$(ts)] Rocchio sweep: te3 (preenc queries, no MPS)..."
PYTHONUNBUFFERED=1 $PY -u evaluation/eval_rocchio.py \
  --data-dir $DATA \
  --queries-file eval_queries_unified.jsonl \
  --name te3_large_1024 \
  --catalog-vecs te3_catalog.vecs.fp16.npy \
  --query-vec-dirs $ROOT/jobs_data,$ROOT/jobs_data_linkedin,$ROOT/jobs_data_usajobs,$ROOT/jobs_data_jobstreet \
  --sweep \
  --output $RESULTS/rocchio_te3.json

echo "[$(ts)] Rocchio sweep: bge-small..."
PYTHONUNBUFFERED=1 $PY -u evaluation/eval_rocchio.py \
  --data-dir $DATA \
  --queries-file eval_queries_unified.jsonl \
  --name bge_small \
  --catalog-vecs bge_catalog.vecs.fp16.npy \
  --query-encoder BAAI/bge-small-en-v1.5 \
  --sweep \
  --output $RESULTS/rocchio_bge_small.json

echo "[$(ts)] Rocchio sweep: bge-base..."
PYTHONUNBUFFERED=1 $PY -u evaluation/eval_rocchio.py \
  --data-dir $DATA \
  --queries-file eval_queries_unified.jsonl \
  --name bge_base \
  --catalog-vecs bge_base_catalog.vecs.fp16.npy \
  --query-encoder BAAI/bge-base-en-v1.5 \
  --sweep \
  --output $RESULTS/rocchio_bge_base.json

# e5-base + gte-base (only run if their catalogs landed)
if [ -f $DATA/e5_base_catalog.vecs.fp16.npy ]; then
  echo "[$(ts)] Rocchio sweep: e5-base..."
  PYTHONUNBUFFERED=1 $PY -u evaluation/eval_rocchio.py \
    --data-dir $DATA \
    --queries-file eval_queries_unified.jsonl \
    --name e5_base \
    --catalog-vecs e5_base_catalog.vecs.fp16.npy \
    --query-encoder intfloat/e5-base-v2 \
    --query-prefix "query: " \
    --sweep \
    --output $RESULTS/rocchio_e5_base.json
fi

if [ -f $DATA/gte_base_catalog.vecs.fp16.npy ]; then
  echo "[$(ts)] Rocchio sweep: gte-base..."
  PYTHONUNBUFFERED=1 $PY -u evaluation/eval_rocchio.py \
    --data-dir $DATA \
    --queries-file eval_queries_unified.jsonl \
    --name gte_base \
    --catalog-vecs gte_base_catalog.vecs.fp16.npy \
    --query-encoder Alibaba-NLP/gte-base-en-v1.5 \
    --trust-remote-code \
    --sweep \
    --output $RESULTS/rocchio_gte_base.json
fi

echo "[$(ts)] rocchio_watcher done"
