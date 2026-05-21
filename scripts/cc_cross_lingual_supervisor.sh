#!/bin/bash
# Chain supervisor for CC cross-lingual experiment.
# Waits for ES results.json, then launches JP, then exits.
set -e
cd "$(dirname "$0")/.."

VENV=.venv/bin/python

ES_RESULTS=esci_es_data/cc_eval/results.json
JP_RESULTS=esci_jp_data/cc_eval/results.json

echo "[$(date)] supervisor up, waiting for ${ES_RESULTS}..."
while [ ! -f "${ES_RESULTS}" ]; do
    sleep 60
done

echo "[$(date)] ES complete. Launching JP run..."
caffeinate -di ${VENV} evaluation/eval_cc_cross_lingual.py \
    --data-dir esci_jp_data \
    --bod-model query_model_esci_jp_me5_small_lora_bod \
    --top-k 100 \
    > logs/cc_jp_run.log 2>&1

if [ -f "${JP_RESULTS}" ]; then
    echo "[$(date)] JP complete. Chain done."
else
    echo "[$(date)] WARNING: JP run exited but ${JP_RESULTS} not present."
fi
