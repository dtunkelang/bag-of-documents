#!/usr/bin/env bash
# H1: English e5-*-v2 drop-in on ESCI-US, reporting E@1 (head-of-list) + R@10.
# Tests whether the jobs head-of-list inversion (e5 wins H@1) replicates on ecommerce.
# Waits for the running jobs CE-rerank job to finish first to avoid MPS contention.
set -euo pipefail

cd /Users/dtunkelang/bagofdocs

WAIT_PID=62862                      # eval_ce_rerank_jobs.py (BGE reranker on jobs)
DATA=esci_us_data
LOG_DIR=evaluation/logs
mkdir -p "$LOG_DIR" evaluation/results

echo "[$(date)] waiting for jobs CE-rerank PID $WAIT_PID to finish..."
while kill -0 "$WAIT_PID" 2>/dev/null; do
  sleep 60
done
echo "[$(date)] PID $WAIT_PID done — starting e5 drop-in runs."

run_e5 () {
  local model="$1" out_name="$2" log="$3"
  echo "[$(date)] === $model -> $out_name ==="
  python3 evaluation/eval_alt_encoder.py \
    --data-dir "$DATA" \
    --model "$model" \
    --out-name "$out_name" \
    --query-prefix "query: " \
    --doc-prefix "passage: " \
    --k 10 \
    --min-relevance 1 \
    --exact-relevance 3 \
    --batch-size 64 \
    2>&1 | tee "$log"
}

# e5-small-v2 first (33M, fast) — quick signal before the slower base run.
run_e5 intfloat/e5-small-v2 e5_small_v2_catalog "$LOG_DIR/e5_small_v2_escius.log"
run_e5 intfloat/e5-base-v2  e5_base_v2_catalog  "$LOG_DIR/e5_base_v2_escius.log"

echo "[$(date)] H1 e5 drop-in chain complete. Grep 'E@1\\|R@10' in $LOG_DIR/e5_*_escius.log"
