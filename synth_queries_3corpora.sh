#!/usr/bin/env bash
# Synthesize train+eval queries for the 3 new jobs corpora.
# Serial across corpora (shares OpenAI rate limit). Idempotent at corpus level.
set -euo pipefail
cd /Users/dtunkelang/bagofdocs

ROOT=/Users/dtunkelang/bagofdocs
PY=$ROOT/.venv/bin/python

ts() { date '+%H:%M:%S'; }

for CORPUS in jobs_data_usajobs jobs_data_jobstreet jobs_data_linkedin; do
  if [ -f $ROOT/$CORPUS/train_queries.jsonl ] && [ -f $ROOT/$CORPUS/eval_queries.jsonl ]; then
    echo "[$(ts)] SKIP $CORPUS (queries exist)"
    continue
  fi
  echo "[$(ts)] === synth $CORPUS ==="
  PYTHONUNBUFFERED=1 $PY -u download/synthesize_jobs_queries.py \
    --data-dir $CORPUS \
    --n-distilled-train 10000 \
    --n-distilled-eval 900 \
    --n-head 200 \
    --seed 42 \
    --concurrency 24
  echo "[$(ts)] === $CORPUS done ==="
done

echo "[$(ts)] === all 3 corpora done ==="
