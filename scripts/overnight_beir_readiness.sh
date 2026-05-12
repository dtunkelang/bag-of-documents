#!/bin/bash
# Overnight readiness-only sweep over 3 untouched large BEIR datasets.
# Produces SCHS + base-difficulty + predicted-lift verdicts. No training.
#
# Order: smallest-download first, so smaller ones finish even if the biggest
# (Climate-FEVER, ~1.8GB, previously OOMed) fails.

set -u
PY=.venv/bin/python
mkdir -p logs
SUMMARY=logs/overnight_beir_readiness_summary.log

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$SUMMARY"; }

log "=== overnight BEIR readiness sweep start ==="

run_one() {
    local DATASET="$1"
    local SAFE="${DATASET//\//_}"
    SAFE="${SAFE//-/_}"
    local DDIR="${SAFE}_data"
    local LABEL="${SAFE}"

    log "--- ${DATASET} ---"

    if [ ! -f "${DDIR}/titles.json" ]; then
        log "downloading ${DATASET}..."
        $PY download/download_beir.py --dataset "${DATASET}" --out-dir "${DDIR}" \
            > "logs/${LABEL}_download.log" 2>&1 || {
            log "  download FAILED — see logs/${LABEL}_download.log"
            return 1
        }
        log "  downloaded"
    else
        log "  already downloaded at ${DDIR}/"
    fi

    local TRAIN_ARG=""
    if [ -f "${DDIR}/train_qrels.jsonl" ]; then
        TRAIN_ARG="--train-qrels ${DDIR}/train_qrels.jsonl"
    fi

    log "running readiness on ${DATASET} (chunk=64 for big-corpora matmul memory)..."
    $PY evaluation/bod_readiness_report.py \
        --catalog "${DDIR}/titles.json" \
        --product-ids "${DDIR}/product_ids.json" \
        --qrels "${DDIR}/test_qrels.jsonl" --min-relevance 1 \
        ${TRAIN_ARG} \
        --queries "${DDIR}/test_queries.jsonl" \
        --encoder all-MiniLM-L6-v2 \
        --vecs-cache "${DDIR}/base_catalog.vecs.fp16.npy" \
        --label "${LABEL}" \
        --chunk 64 \
        > "logs/${LABEL}_readiness.log" 2>&1 || {
        log "  readiness FAILED — see logs/${LABEL}_readiness.log"
        return 1
    }

    # Pull headline numbers into the summary.
    log "  results:"
    grep -E "SCHS:|overall R@10:|base-blind:|base-perfect:|rescue~|VERDICT|recommendation:" \
        "logs/${LABEL}_readiness.log" | LC_ALL=C tr '\r' '\n' | head -10 | tee -a "$SUMMARY"

    return 0
}

run_one "dbpedia-entity"
run_one "hotpotqa"
run_one "climate-fever"

log "=== overnight BEIR readiness sweep complete ==="
