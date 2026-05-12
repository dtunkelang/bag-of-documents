#!/bin/bash

# Always run from project root (script lives in scripts/ after the May 2026 reorg).
cd "$(dirname "$0")/.."
# Overnight HyDE chain: for each corpus, generate hypothetical passages,
# evaluate, run BoD diagnose, RRF ensemble, and HyDE-vs-BoD overlap.
# All output flushes to logs/overnight_hyde_chain_status.log.
set -u
PY=.venv/bin/python
mkdir -p logs
STATUS=logs/overnight_hyde_chain_status.log
log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$STATUS"; }

log "=== overnight HyDE chain start ==="

run_one() {
    local LABEL="$1"
    local DDIR="$2"
    local BOD_MODEL="$3"

    log "--- $LABEL ---"

    # Resolve doc-id file name (some corpora use doc_ids, some product_ids)
    local PID_FILE="$DDIR/doc_ids.json"
    [ -f "$PID_FILE" ] || PID_FILE="$DDIR/product_ids.json"

    log "  phase A: HyDE generation + eval"
    $PY evaluation/eval_hyde.py \
        --catalog "$DDIR/titles.json" \
        --product-ids "$PID_FILE" \
        --qrels "$DDIR/test_qrels.jsonl" --min-relevance 1 \
        --queries "$DDIR/test_queries.jsonl" \
        --base-model all-MiniLM-L6-v2 \
        --base-vecs "$DDIR/base_catalog.vecs.fp16.npy" \
        --llm-model llama3.1:8b-instruct-q4_K_M \
        --label "$LABEL" \
        > "logs/${LABEL}_hyde.log" 2>&1
    log "  HyDE done; result:"
    tail -15 "logs/${LABEL}_hyde.log" | LC_ALL=C tr '\r' '\n' | grep -E "bucket|0\.0|0\.5|1\.0|overall|HyDE vs" | tee -a "$STATUS"

    log "  phase B: BoD diagnose"
    $PY evaluation/diagnose_lift.py \
        --catalog "$DDIR/titles.json" \
        --product-ids "$PID_FILE" \
        --qrels "$DDIR/test_qrels.jsonl" --min-relevance 1 \
        --queries "$DDIR/test_queries.jsonl" \
        --base-model all-MiniLM-L6-v2 \
        --bod-model "$BOD_MODEL" \
        --base-vecs "$DDIR/base_catalog.vecs.fp16.npy" \
        --label "$LABEL" \
        > "logs/${LABEL}_bod_diag.log" 2>&1
    log "  BoD done; bucket table:"
    tail -15 "logs/${LABEL}_bod_diag.log" | LC_ALL=C tr '\r' '\n' | grep -E "bucket|0\.0|0\.5|1\.0|overall" | tee -a "$STATUS"

    log "  phase C: RRF ensemble"
    $PY evaluation/eval_rrf_ensemble.py \
        --catalog "$DDIR/titles.json" \
        --product-ids "$PID_FILE" \
        --qrels "$DDIR/test_qrels.jsonl" --min-relevance 1 \
        --queries "$DDIR/test_queries.jsonl" \
        --base-model all-MiniLM-L6-v2 \
        --base-vecs "$DDIR/base_catalog.vecs.fp16.npy" \
        --bod-model "$BOD_MODEL" \
        --hyde-passages "$DDIR/hyde_passages_llama3.1_8b-instruct-q4_K_M.jsonl" \
        --label "$LABEL" \
        > "logs/${LABEL}_rrf.log" 2>&1
    log "  RRF done; ensemble:"
    tail -25 "logs/${LABEL}_rrf.log" | LC_ALL=C tr '\r' '\n' | grep -E "R@10=|rescues" | tee -a "$STATUS"

    log "  phase D: HyDE-vs-BoD overlap"
    $PY evaluation/diagnose_hyde_vs_bod.py \
        --bod-per-query "$DDIR/bod_per_query_${LABEL}.jsonl" \
        --hyde-per-query "$DDIR/hyde_per_query_${LABEL}.jsonl" \
        --queries "$DDIR/test_queries.jsonl" \
        --label "$LABEL" \
        > "logs/${LABEL}_overlap.log" 2>&1
    log "  overlap done; contingency:"
    grep -E "SUMMARY|HyDE-rescues|HyDE-misses|BoD-rescues|BoD-misses" "logs/${LABEL}_overlap.log" | tee -a "$STATUS"
}

run_one nfcorpus nfcorpus_data query_model_nfcorpus_bod
run_one programmers cqadupstack_programmers_data query_model_cqadupstack_programmers_bod
run_one english cqadupstack_english_data query_model_cqadupstack_english_bod

log "=== overnight HyDE chain complete ==="
