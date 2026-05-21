#!/bin/bash
# Two-step chain for Task #9 (nomic-embed-text-v1.5 Pattern 20 candidate):
#   1. Eval the merged nomic+LoRA-BoD model on BestBuy 1K   (~5 min)
#   2. Drop-in encode + eval nomic on ESCI-US 22k          (~14 hours)
# Sequential to avoid MPS contention. Each step writes its own log.
set -e
cd "$(dirname "$0")/.."

VENV=.venv/bin/python

echo "[$(date)] step 1: eval nomic+LoRA-BoD on BestBuy"
$VENV evaluation/eval_alt_encoder.py \
    --data-dir bestbuy_acm_data \
    --queries test_queries_1k.jsonl --qrels test_qrels_1k.jsonl \
    --model query_model_bestbuy_nomic_lora_bod \
    --query-prefix 'search_query: ' --doc-prefix 'search_document: ' \
    --max-seq-length 256 --batch-size 16 \
    --out-name nomic_bod_catalog \
    --baseline-per-query bod_per_query_bestbuy_1k.jsonl \
    > logs/nomic_bestbuy_bod_eval.log 2>&1

echo "[$(date)] step 1 done. R@10:"
grep "R@10:" logs/nomic_bestbuy_bod_eval.log

echo "[$(date)] step 2: drop-in encode + eval nomic on ESCI-US"
$VENV evaluation/eval_alt_encoder.py \
    --data-dir esci_us_data \
    --queries test_queries.jsonl --qrels test_qrels.jsonl \
    --model nomic-ai/nomic-embed-text-v1.5 \
    --query-prefix 'search_query: ' --doc-prefix 'search_document: ' \
    --max-seq-length 256 --batch-size 16 \
    --out-name nomic_catalog \
    > logs/nomic_esci_us.log 2>&1

echo "[$(date)] step 2 done. R@10:"
grep "R@10:" logs/nomic_esci_us.log

echo "[$(date)] chain complete."
