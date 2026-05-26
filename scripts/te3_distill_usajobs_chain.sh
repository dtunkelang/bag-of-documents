#!/usr/bin/env bash
# te3-distillation: te3 picks bag docs, bge-small learns to land near their bge centroid.
# Idempotent at step granularity.
set -euo pipefail
cd /Users/dtunkelang/bagofdocs

ROOT=/Users/dtunkelang/bagofdocs
PY=$ROOT/.venv/bin/python
DATA=$ROOT/jobs_data_usajobs
MODEL=$ROOT/query_model_usajobs_te3bod
BASE=BAAI/bge-small-en-v1.5

ts() { date '+%H:%M:%S'; }

echo "[$(ts)] === te3-distill usajobs start ==="

if [ ! -f $DATA/bags_te3.jsonl ]; then
  echo "[$(ts)] STEP 1: te3 bag build..."
  PYTHONUNBUFFERED=1 $PY -u download/build_jobs_bags_te3.py \
    --data-dir $DATA \
    --queries-file train_queries.jsonl \
    --te3-catalog $DATA/te3_large_1024.vecs.fp16.npy \
    --bge-catalog $DATA/bge_small_en_catalog.vecs.fp16.npy \
    --te3-model text-embedding-3-large \
    --te3-dim 1024 \
    --k 20 \
    --output $DATA/bags_te3.jsonl
  echo "[$(ts)] bags_te3 done"
else
  echo "[$(ts)] STEP 1 skipped"
fi

if [ ! -f $MODEL/model.safetensors ]; then
  echo "[$(ts)] STEP 2: finetune bge with te3-bags (cos loss)..."
  PYTHONUNBUFFERED=1 $PY -u training/finetune_query_model.py \
    $DATA/bags_te3.jsonl \
    $MODEL/ \
    --base-model $BASE \
    --epochs 15 \
    --batch-size 64 \
    --lr 2e-5 \
    --loss cos
  echo "[$(ts)] model trained"
else
  echo "[$(ts)] STEP 2 skipped"
fi

if [ ! -f $DATA/te3bod_catalog.vecs.fp16.npy ]; then
  echo "[$(ts)] STEP 3: encode te3bod catalog..."
  PYTHONUNBUFFERED=1 $PY -u download/encode_st_catalog.py \
    --data-dir $DATA \
    --model $MODEL \
    --out-name te3bod_catalog \
    --batch-size 64 \
    --device mps
  echo "[$(ts)] catalog done"
else
  echo "[$(ts)] STEP 3 skipped"
fi

echo "[$(ts)] STEP 4: eval..."
$PY -u evaluation/eval_jobs_retrievers.py \
  --data-dir $DATA \
  --k 10 \
  --device cpu \
  --output scratch/eval_4_corpora/jobs_data_usajobs_te3bod.json \
  --retriever 'name=bm25:bm25' \
  --retriever 'name=bge:st:bge_small_en_catalog.vecs.fp16.npy:BAAI/bge-small-en-v1.5' \
  --retriever 'name=te3:openai:te3_large_1024.vecs.fp16.npy:text-embedding-3-large:1024' \
  --retriever "name=te3bod:st:te3bod_catalog.vecs.fp16.npy:$MODEL" \
  | tail -20

echo "[$(ts)] === te3-distill usajobs complete ==="
