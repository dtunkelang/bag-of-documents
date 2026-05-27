#!/usr/bin/env bash
# Encode three base-sized local retrievers on unified_jobs (348k docs) and
# eval them against te3-large/bge-small/bge-bod/bm25.
#
# Goal: test whether a larger local base closes the ~20pp R@10 gap to te3
# before we commit to te3-distillation.
#
# Each model takes roughly 9h on MPS; total chain ~27h.
# After each encode the eval runs incrementally so we get signal early.
set -euo pipefail
cd /Users/dtunkelang/bagofdocs

ROOT=/Users/dtunkelang/bagofdocs
PY=$ROOT/.venv/bin/python
DATA=$ROOT/unified_jobs
RESULTS=$ROOT/evaluation/results

ts() { date '+%Y-%m-%d %H:%M:%S'; }

run_eval() {
  local tag=$1
  shift
  echo "[$(ts)] EVAL: $tag"
  PYTHONUNBUFFERED=1 $PY -u evaluation/eval_jobs_retrievers.py \
    --data-dir $DATA \
    --queries-file eval_queries_unified.jsonl \
    --k 10 \
    --device mps \
    --output $RESULTS/jobs_unified_${tag}.json \
    --breakdown-key source_corpus \
    --retriever name=bm25:bm25 \
    --retriever name=te3_large_1024:preenc:te3_catalog.vecs.fp16.npy:$ROOT/jobs_data,$ROOT/jobs_data_linkedin,$ROOT/jobs_data_usajobs,$ROOT/jobs_data_jobstreet \
    --retriever name=bge_small:st:bge_catalog.vecs.fp16.npy:BAAI/bge-small-en-v1.5 \
    --retriever name=bge_bod:st:jobs_bge_bod_catalog.vecs.fp16.npy:$ROOT/query_model_jobs_bge_bod \
    "$@"
}

echo "[$(ts)] === base-model shootout chain start ==="

# Step 1: bge-base-en-v1.5 (no prefix needed for v1.5)
if [ ! -f $DATA/bge_base_catalog.vecs.fp16.npy ]; then
  echo "[$(ts)] STEP 1a: encoding bge-base-en-v1.5..."
  PYTHONUNBUFFERED=1 $PY -u download/encode_st_catalog.py \
    --data-dir $DATA \
    --model BAAI/bge-base-en-v1.5 \
    --out-name bge_base_catalog \
    --batch-size 32 \
    --device mps
else
  echo "[$(ts)] STEP 1a skipped (bge-base catalog exists)"
fi

echo "[$(ts)] STEP 1b: eval with bge-base..."
run_eval base_5ret \
  --retriever name=bge_base:st:bge_base_catalog.vecs.fp16.npy:BAAI/bge-base-en-v1.5

# Step 2: e5-base-v2 (needs "passage: " on docs, "query: " on queries)
if [ ! -f $DATA/e5_base_catalog.vecs.fp16.npy ]; then
  echo "[$(ts)] STEP 2a: encoding e5-base-v2 (with passage: prefix)..."
  PYTHONUNBUFFERED=1 $PY -u download/encode_st_catalog.py \
    --data-dir $DATA \
    --model intfloat/e5-base-v2 \
    --out-name e5_base_catalog \
    --doc-prefix "passage: " \
    --batch-size 32 \
    --device mps
else
  echo "[$(ts)] STEP 2a skipped (e5-base catalog exists)"
fi

echo "[$(ts)] STEP 2b: eval with bge-base + e5-base..."
run_eval base_6ret \
  --retriever name=bge_base:st:bge_base_catalog.vecs.fp16.npy:BAAI/bge-base-en-v1.5 \
  --retriever name=e5_base:st:e5_base_catalog.vecs.fp16.npy:intfloat/e5-base-v2 \
  --query-prefix "e5_base=query: "

# Step 3: gte-base-en-v1.5 (custom NewModel arch, needs trust_remote_code)
if [ ! -f $DATA/gte_base_catalog.vecs.fp16.npy ]; then
  echo "[$(ts)] STEP 3a: encoding gte-base-en-v1.5..."
  PYTHONUNBUFFERED=1 $PY -u download/encode_st_catalog.py \
    --data-dir $DATA \
    --model Alibaba-NLP/gte-base-en-v1.5 \
    --out-name gte_base_catalog \
    --batch-size 32 \
    --device mps \
    --trust-remote-code
else
  echo "[$(ts)] STEP 3a skipped (gte-base catalog exists)"
fi

echo "[$(ts)] STEP 3b: full eval with all 7 retrievers..."
run_eval base_7ret \
  --retriever name=bge_base:st:bge_base_catalog.vecs.fp16.npy:BAAI/bge-base-en-v1.5 \
  --retriever name=e5_base:st:e5_base_catalog.vecs.fp16.npy:intfloat/e5-base-v2 \
  --retriever name=gte_base:st:gte_base_catalog.vecs.fp16.npy:Alibaba-NLP/gte-base-en-v1.5 \
  --query-prefix "e5_base=query: " \
  --trust-remote-code

echo "[$(ts)] === base-model shootout chain complete ==="
