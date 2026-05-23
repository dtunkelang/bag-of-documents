#!/usr/bin/env bash
# Run eval on the 3 new corpora (linkedin/usajobs/jobstreet) while B still encodes.
# Uses --device cpu to avoid MPS contention with the bge-BoD catalog encode.
set -euo pipefail
cd /Users/dtunkelang/bagofdocs

ROOT=/Users/dtunkelang/bagofdocs
PY=$ROOT/.venv/bin/python
OUT_DIR=$ROOT/scratch/eval_4_corpora
mkdir -p $OUT_DIR

ts() { date '+%H:%M:%S'; }

run_eval() {
  local corpus="$1"
  shift
  echo "[$(ts)] === eval $corpus ==="
  $PY -u evaluation/eval_jobs_retrievers.py \
    --data-dir "$corpus" \
    --k 10 \
    --device cpu \
    --output "$OUT_DIR/${corpus}.json" \
    "$@"
  echo "[$(ts)] === $corpus done ==="
}

# usajobs (6k docs)
run_eval jobs_data_usajobs \
  --retriever 'name=bm25:bm25' \
  --retriever 'name=base:st:base_catalog.vecs.fp16.npy:sentence-transformers/all-MiniLM-L6-v2' \
  --retriever 'name=bge:st:bge_small_en_catalog.vecs.fp16.npy:BAAI/bge-small-en-v1.5' \
  --retriever 'name=me5:st:me5_small_catalog.vecs.fp16.npy:intfloat/multilingual-e5-small' \
  --retriever 'name=te3:openai:te3_large_1024.vecs.fp16.npy:text-embedding-3-large:1024'

# jobstreet (53k docs)
run_eval jobs_data_jobstreet \
  --retriever 'name=bm25:bm25' \
  --retriever 'name=base:st:base_catalog.vecs.fp16.npy:sentence-transformers/all-MiniLM-L6-v2' \
  --retriever 'name=bge:st:bge_small_en_catalog.vecs.fp16.npy:BAAI/bge-small-en-v1.5' \
  --retriever 'name=me5:st:me5_small_catalog.vecs.fp16.npy:intfloat/multilingual-e5-small' \
  --retriever 'name=te3:openai:openai_te3large_1024.vecs.fp16.npy:text-embedding-3-large:1024'

# linkedin (96k docs) - no base_minilm
run_eval jobs_data_linkedin \
  --retriever 'name=bm25:bm25' \
  --retriever 'name=bge:st:bge_small_en_catalog.vecs.fp16.npy:BAAI/bge-small-en-v1.5' \
  --retriever 'name=me5:st:me5_small_catalog.vecs.fp16.npy:intfloat/multilingual-e5-small' \
  --retriever 'name=te3:openai:te3_large_1024.vecs.fp16.npy:text-embedding-3-large:1024'

echo "[$(ts)] === 3-corpus eval complete ==="
