#!/usr/bin/env bash
# Train BoD with bge-small-en-v1.5 as the base on jobs_data/.
# Three steps: build bge bags -> finetune -> encode catalog.
# Idempotent (each step skipped if output exists).
set -euo pipefail
cd /Users/dtunkelang/bagofdocs

ROOT=/Users/dtunkelang/bagofdocs
PY=$ROOT/.venv/bin/python
DATA=$ROOT/jobs_data
MODEL=$ROOT/query_model_jobs_bge_bod
BASE=BAAI/bge-small-en-v1.5

ts() { date '+%H:%M:%S'; }

echo "[$(ts)] === bge-BoD chain start ==="

# Step 1: bge bags
if [ ! -f $DATA/bags_bge.jsonl ]; then
  echo "[$(ts)] STEP 1: building bge bags..."
  PYTHONUNBUFFERED=1 $PY -u download/build_jobs_bags.py \
    --data-dir $DATA \
    --queries-file train_queries.jsonl \
    --encoder $BASE \
    --catalog $DATA/bge_small_en_catalog.vecs.fp16.npy \
    --k 20 \
    --output $DATA/bags_bge.jsonl
  echo "[$(ts)] bags_bge done"
else
  echo "[$(ts)] STEP 1 skipped (bags_bge exists)"
fi

# Step 2: finetune bge with cos loss
if [ ! -f $MODEL/model.safetensors ]; then
  echo "[$(ts)] STEP 2: finetuning bge with cos loss..."
  PYTHONUNBUFFERED=1 $PY -u training/finetune_query_model.py \
    $DATA/bags_bge.jsonl \
    $MODEL/ \
    --base-model $BASE \
    --epochs 15 \
    --batch-size 64 \
    --lr 2e-5 \
    --loss cos
  echo "[$(ts)] BoD trained"
else
  echo "[$(ts)] STEP 2 skipped (model exists)"
fi

# Step 3: encode bge-BoD catalog
if [ ! -f $DATA/jobs_bge_bod_catalog.vecs.fp16.npy ]; then
  echo "[$(ts)] STEP 3: encoding bge-BoD catalog..."
  PYTHONUNBUFFERED=1 $PY -u download/encode_st_catalog.py \
    --data-dir $DATA \
    --model $MODEL \
    --out-name jobs_bge_bod_catalog \
    --batch-size 64 \
    --device mps
  echo "[$(ts)] jobs_bge_bod_catalog done"
else
  echo "[$(ts)] STEP 3 skipped (catalog exists)"
fi

echo "[$(ts)] === bge-BoD chain complete ==="
