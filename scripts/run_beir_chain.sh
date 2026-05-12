#!/bin/bash

# Always run from project root (script lives in scripts/ after the May 2026 reorg).
cd "$(dirname "$0")/.."
# Generic BEIR chain: download -> readiness -> bags -> hardnegs -> train -> diagnose.
#
# Usage:
#   ./run_beir_chain.sh <dataset>
#
# Example:
#   ./run_beir_chain.sh arguana
#   ./run_beir_chain.sh trec-covid

set -u
if [ $# -lt 1 ]; then
    echo "usage: $0 <dataset>"
    exit 2
fi
DATASET="$1"
# Fold both '/' (subset paths like cqadupstack/programmers) and '-' for safe dir/label.
SAFE="${DATASET//\//_}"
SAFE="${SAFE//-/_}"
DDIR="${SAFE}_data"
LABEL="${SAFE}"
STATUS=logs/${LABEL}_chain_status.log
PY=.venv/bin/python

mkdir -p logs
log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$STATUS"; }

log "=== ${LABEL} chain start ==="

# --- 1. Download ----------------------------------------------------------
log "phase 1: download ${DATASET}"
$PY download/download_beir.py --dataset "${DATASET}" --out-dir "${DDIR}" \
    > logs/${LABEL}_download.log 2>&1
tail -8 logs/${LABEL}_download.log | tee -a "$STATUS"

# --- 2. Readiness report (predict before training) ------------------------
log "phase 2: readiness report (PREDICTION)"
TRAIN_QRELS_ARG=""
if [ -f "${DDIR}/train_qrels.jsonl" ]; then
    TRAIN_QRELS_ARG="--train-qrels ${DDIR}/train_qrels.jsonl"
fi
$PY evaluation/bod_readiness_report.py \
    --catalog "${DDIR}/titles.json" \
    --product-ids "${DDIR}/product_ids.json" \
    --qrels "${DDIR}/test_qrels.jsonl" --min-relevance 1 \
    ${TRAIN_QRELS_ARG} \
    --queries "${DDIR}/test_queries.jsonl" \
    --encoder all-MiniLM-L6-v2 \
    --vecs-cache "${DDIR}/base_catalog.vecs.fp16.npy" \
    --label "${LABEL}" \
    > logs/${LABEL}_readiness.log 2>&1
tail -25 logs/${LABEL}_readiness.log | LC_ALL=C tr '\r' '\n' | grep -E "SCHS:|n_pos|verdict|R@10|base-blind|base-perfect|pessimistic|realistic|optimistic|VERDICT|reason:" | tee -a "$STATUS"

# --- 3. Build bags --------------------------------------------------------
log "phase 3: build bags from train qrels"
$PY training/bags_from_qrels.py \
    --titles "${DDIR}/titles.json" \
    --product-ids "${DDIR}/product_ids.json" \
    --queries "${DDIR}/queries.jsonl" \
    --qrels "${DDIR}/train_qrels.jsonl" \
    --output "${DDIR}/bags.jsonl" \
    --model all-MiniLM-L6-v2 \
    --min-relevance 1 --k 50 \
    > logs/${LABEL}_bags.log 2>&1
tail -3 logs/${LABEL}_bags.log | tee -a "$STATUS"

# --- 4. Hardnegs ----------------------------------------------------------
log "phase 4: add random hardnegs"
$PY download/add_random_hardnegs.py --data-dir "${DDIR}" \
    > logs/${LABEL}_hardnegs.log 2>&1
tail -3 logs/${LABEL}_hardnegs.log | tee -a "$STATUS"

# --- 5. Train BoD ---------------------------------------------------------
log "phase 5: train BoD (3 epochs, MiniLM-L6)"
caffeinate -i $PY training/finetune_with_hardnegs.py \
    "${DDIR}/bags_with_hardnegs.jsonl" \
    "query_model_${LABEL}_bod" \
    --epochs 3 --batch-size 32 --triplets-per-bag 5 \
    > logs/${LABEL}_train.log 2>&1
log "phase 5 done"

# --- 6. Diagnose ----------------------------------------------------------
log "phase 6: diagnose lift (ACTUAL)"
$PY evaluation/diagnose_lift.py \
    --catalog "${DDIR}/titles.json" \
    --product-ids "${DDIR}/product_ids.json" \
    --qrels "${DDIR}/test_qrels.jsonl" --min-relevance 1 \
    --queries "${DDIR}/test_queries.jsonl" \
    --base-vecs "${DDIR}/base_catalog.vecs.fp16.npy" \
    --base-model all-MiniLM-L6-v2 \
    --bod-model "query_model_${LABEL}_bod" \
    --label "${LABEL}" \
    > logs/${LABEL}_diagnostic.log 2>&1
log "phase 6 done; diagnostic table:"
tail -20 logs/${LABEL}_diagnostic.log | LC_ALL=C tr '\r' '\n' | grep -E "bucket|0\.0|0\.5|1\.0|overall" | tee -a "$STATUS"

log "=== ${LABEL} chain complete ==="
