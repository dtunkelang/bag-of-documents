#!/usr/bin/env bash
# Plan A chain: scale jobs corpus from 100K to ~192K.
# Idempotent: each step is skipped if its output already exists.
# Launch: nohup caffeinate -di scripts/jobs_a_chain.sh > logs/jobs_a_chain.log 2>&1 &

set -euo pipefail
cd /Users/dtunkelang/bagofdocs

ROOT=/Users/dtunkelang/bagofdocs
PY=$ROOT/.venv/bin/python
DATA=$ROOT/jobs_data_244k
MODEL=$ROOT/query_model_jobs_bod_244k

ts() { date '+%H:%M:%S'; }

echo "[$(ts)] === Plan A chain starting ==="
echo "[$(ts)] target dir: $DATA"

# Step 1: copy reusable query files
if [ ! -f $DATA/train_queries.jsonl ]; then
  cp $ROOT/jobs_data/train_queries.jsonl $DATA/
  cp $ROOT/jobs_data/eval_queries.jsonl $DATA/
  echo "[$(ts)] copied train/eval queries"
fi

# Step 2: encode base_minilm catalog (~25 min)
if [ ! -f $DATA/base_catalog.vecs.fp16.npy ]; then
  echo "[$(ts)] STEP 2: encoding base_minilm catalog..."
  PYTHONUNBUFFERED=1 $PY -u download/encode_st_catalog.py \
    --data-dir $DATA \
    --model sentence-transformers/all-MiniLM-L6-v2 \
    --out-name base_catalog \
    --batch-size 64 \
    --device mps
  echo "[$(ts)] base_catalog done"
else
  echo "[$(ts)] STEP 2 skipped (base_catalog exists)"
fi

# Step 3: build bags from new base_catalog + reused train queries
if [ ! -f $DATA/bags.jsonl ]; then
  echo "[$(ts)] STEP 3: building bags..."
  PYTHONUNBUFFERED=1 $PY -u download/build_jobs_bags.py \
    --data-dir $DATA \
    --queries-file train_queries.jsonl \
    --encoder sentence-transformers/all-MiniLM-L6-v2 \
    --catalog $DATA/base_catalog.vecs.fp16.npy \
    --k 20 \
    --output $DATA/bags.jsonl
  echo "[$(ts)] bags done"
else
  echo "[$(ts)] STEP 3 skipped (bags exist)"
fi

# Step 4: retrain BoD (~10 min)
if [ ! -f $MODEL/model.safetensors ]; then
  echo "[$(ts)] STEP 4: retraining BoD..."
  PYTHONUNBUFFERED=1 $PY -u training/finetune_query_model.py \
    $DATA/bags.jsonl \
    $MODEL/ \
    --epochs 15 \
    --batch-size 64 \
    --lr 2e-5 \
    --loss cos
  echo "[$(ts)] BoD trained"
else
  echo "[$(ts)] STEP 4 skipped (BoD model exists)"
fi

# Step 5: encode BoD catalog (~25 min)
if [ ! -f $DATA/jobs_bod_catalog.vecs.fp16.npy ]; then
  echo "[$(ts)] STEP 5: encoding BoD catalog..."
  PYTHONUNBUFFERED=1 $PY -u download/encode_st_catalog.py \
    --data-dir $DATA \
    --model $MODEL \
    --out-name jobs_bod_catalog \
    --batch-size 64 \
    --device mps
  echo "[$(ts)] jobs_bod_catalog done"
else
  echo "[$(ts)] STEP 5 skipped (BoD catalog exists)"
fi

# Step 6: encode me5_small catalog (~2h)
if [ ! -f $DATA/me5_small_catalog.vecs.fp16.npy ]; then
  echo "[$(ts)] STEP 6: encoding me5_small catalog..."
  PYTHONUNBUFFERED=1 $PY -u download/encode_st_catalog.py \
    --data-dir $DATA \
    --model intfloat/multilingual-e5-small \
    --out-name me5_small_catalog \
    --batch-size 32 \
    --device mps \
    --doc-prefix "passage: "
  echo "[$(ts)] me5_small_catalog done"
else
  echo "[$(ts)] STEP 6 skipped (me5 catalog exists)"
fi

# Step 7: encode bge-small-en catalog (~10 min MPS / ~30 min CPU)
# Replaced nomic after 2026-05-22 model shootout: bge-small-en is new English default.
if [ ! -f $DATA/bge_small_en_catalog.vecs.fp16.npy ]; then
  echo "[$(ts)] STEP 7: encoding bge-small-en catalog..."
  PYTHONUNBUFFERED=1 $PY -u download/encode_st_catalog.py \
    --data-dir $DATA \
    --model BAAI/bge-small-en-v1.5 \
    --out-name bge_small_en_catalog \
    --batch-size 64 \
    --device mps
  echo "[$(ts)] bge_small_en_catalog done"
else
  echo "[$(ts)] STEP 7 skipped (bge_small_en catalog exists)"
fi

# Step 8: verify all outputs
echo "[$(ts)] STEP 8: verifying artifacts..."
for f in $DATA/titles.json $DATA/doc_ids.json \
         $DATA/base_catalog.vecs.fp16.npy \
         $DATA/jobs_bod_catalog.vecs.fp16.npy \
         $DATA/me5_small_catalog.vecs.fp16.npy \
         $DATA/bge_small_en_catalog.vecs.fp16.npy \
         $MODEL/model.safetensors; do
  if [ ! -e $f ]; then
    echo "[$(ts)] FAIL: missing $f" >&2
    exit 1
  fi
done
echo "[$(ts)] all artifacts present"

# Step 9: atomic swap
echo "[$(ts)] STEP 9: swapping dirs..."
if [ -d jobs_data_100k_v1 ]; then
  echo "[$(ts)] WARN: jobs_data_100k_v1 already exists, skipping swap" >&2
else
  mv jobs_data jobs_data_100k_v1
  mv jobs_data_244k jobs_data
  mv query_model_jobs_bod query_model_jobs_bod_100k_v1
  mv query_model_jobs_bod_244k query_model_jobs_bod
  echo "[$(ts)] swap complete; 100K artifacts preserved as _100k_v1"
fi

# Step 10: restart demo
echo "[$(ts)] STEP 10: restarting demo..."
DEMO_PID=$(cat /tmp/jobs_demo.pid 2>/dev/null || true)
if [ -n "$DEMO_PID" ] && kill -0 $DEMO_PID 2>/dev/null; then
  kill $DEMO_PID
  sleep 2
fi
nohup $PY demo_jobs.py --port 7861 > $ROOT/logs/jobs_demo.log 2>&1 &
echo $! > /tmp/jobs_demo.pid
disown
echo "[$(ts)] new demo PID $(cat /tmp/jobs_demo.pid)"

echo "[$(ts)] === Plan A chain complete ==="
