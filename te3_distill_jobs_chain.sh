#!/usr/bin/env bash
# Train a bge-small + 384->1024 projection student on jobs_data to mimic te3-large
# query vectors. Eval against the te3 catalog (no API at query time).
set -euo pipefail
cd /Users/dtunkelang/bagofdocs

ROOT=/Users/dtunkelang/bagofdocs
PY=$ROOT/.venv/bin/python
DATA=$ROOT/jobs_data
MODEL=$ROOT/query_model_jobs_te3distill
BASE=BAAI/bge-small-en-v1.5

ts() { date '+%H:%M:%S'; }

echo "[$(ts)] === te3-distill jobs_data start ==="

# Step 1: te3-encode train queries
if [ ! -f $DATA/train_queries_te3_1024.vecs.fp16.npy ]; then
  echo "[$(ts)] STEP 1: te3-encode train queries..."
  PYTHONUNBUFFERED=1 $PY -u download/encode_queries_te3.py \
    --queries-file $DATA/train_queries.jsonl \
    --model text-embedding-3-large --dim 1024 \
    --batch-size 512 \
    --out $DATA/train_queries_te3_1024
else
  echo "[$(ts)] STEP 1 skipped (train queries already te3-encoded)"
fi

# Step 2: train student
if [ ! -f $MODEL/model.safetensors ] && [ ! -f $MODEL/pytorch_model.bin ]; then
  echo "[$(ts)] STEP 2: train bge-small + projection student..."
  PYTHONUNBUFFERED=1 $PY -u training/finetune_distill_to_te3.py \
    --queries-file $DATA/train_queries.jsonl \
    --targets-vecs $DATA/train_queries_te3_1024.vecs.fp16.npy \
    --base $BASE --target-dim 1024 \
    --epochs 15 --batch-size 64 --lr 2e-5 \
    --device mps \
    --out $MODEL
else
  echo "[$(ts)] STEP 2 skipped (student model exists)"
fi

# Step 3: eval — student vs te3 catalog vs raw bge / bm25 / te3
echo "[$(ts)] STEP 3: eval student against te3 catalog..."
$PY -u evaluation/eval_jobs_retrievers.py \
  --data-dir $DATA \
  --k 10 \
  --device cpu \
  --output scratch/eval_4_corpora/jobs_data_te3distill.json \
  --retriever 'name=bm25:bm25' \
  --retriever 'name=bge:st:bge_small_en_catalog.vecs.fp16.npy:BAAI/bge-small-en-v1.5' \
  --retriever 'name=te3:openai:te3_large_1024.vecs.fp16.npy:text-embedding-3-large:1024' \
  --retriever "name=te3distill:st:te3_large_1024.vecs.fp16.npy:$MODEL" \
  | tail -25

echo "[$(ts)] === te3-distill jobs_data complete ==="
