#!/bin/bash
# Master pipeline script. Safe to run after a crash/reboot.
# Skips already-completed steps. Can be re-run idempotently.
set -e
cd "$(dirname "$0")"

VENV=.venv/bin/python
QUERIES=queries.jsonl
BAGS=bags.jsonl
MODEL_DIR=retrieval_model
CE_MODEL=models/esci-cross-encoder

echo "[$(date)] Pipeline starting..."
command -v caffeinate > /dev/null && caffeinate -i -w $$ &

# Preflight checks
echo "[$(date)] Running preflight checks..."
$VENV preflight.py --quick
echo "[$(date)] Preflight passed."

# Step 1: Build indexes (FAISS + tantivy) if needed
if [ ! -f combined_index/index.faiss ]; then
    echo "[$(date)] Building FAISS + tantivy indexes..."
    $VENV -u build_index.py --model ${MODEL_DIR:-all-MiniLM-L6-v2}
    echo "[$(date)] Indexes built."
else
    echo "[$(date)] Indexes already exist, skipping build."
fi

# Step 2: Compute bags
echo "[$(date)] Computing bags..."
$VENV -u compute_bags.py $QUERIES $BAGS \
    --model ${MODEL_DIR:-all-MiniLM-L6-v2} --ce-rerank $CE_MODEL --ce-threshold 0.3
echo "[$(date)] Bags complete."

# Step 3: Eval current model
if [ -d "$MODEL_DIR" ]; then
    echo "[$(date)] Evaluating current model ($MODEL_DIR)..."
    $VENV eval_finetuned.py $MODEL_DIR/ --base
fi

# Step 4: Fine-tune new model (to temp dir, then swap)
echo "[$(date)] Fine-tuning on $BAGS..."
rm -rf query_model_tmp
$VENV finetune_query_model.py $BAGS query_model_tmp

# Step 5: Eval new model
echo "[$(date)] Evaluating new model..."
$VENV eval_finetuned.py query_model_tmp/ --base

# Step 6: Install new model
echo "[$(date)] Installing new model..."
rm -rf query_model
mv query_model_tmp query_model
rm -f retrieval_model
ln -s query_model retrieval_model
echo "[$(date)] Model updated."

echo "[$(date)] Pipeline done."
