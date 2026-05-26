#!/usr/bin/env bash
# Run the 4-corpus retriever comparison and write results.
# Run AFTER bge_bod_chain.sh and synth_queries_3corpora.sh both complete.
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
  local args=("$@")
  echo "[$(ts)] === eval $corpus ==="
  $PY -u evaluation/eval_jobs_retrievers.py \
    --data-dir "$corpus" \
    --k 10 \
    --output "$OUT_DIR/${corpus}.json" \
    "${args[@]}" \
    2>&1 | tail -40
  echo "[$(ts)] === $corpus done ==="
}

# jobs_data/ : 244K, has every model (incl bge-BoD)
run_eval jobs_data \
  --retriever 'name=bm25:bm25' \
  --retriever 'name=base:st:base_catalog.vecs.fp16.npy:sentence-transformers/all-MiniLM-L6-v2' \
  --retriever 'name=bge:st:bge_small_en_catalog.vecs.fp16.npy:BAAI/bge-small-en-v1.5' \
  --retriever 'name=me5:st:me5_small_catalog.vecs.fp16.npy:intfloat/multilingual-e5-small' \
  --retriever "name=bod_minilm:st:jobs_bod_catalog.vecs.fp16.npy:$ROOT/query_model_jobs_bod" \
  --retriever "name=bge_bod:st:jobs_bge_bod_catalog.vecs.fp16.npy:$ROOT/query_model_jobs_bge_bod"

# jobs_data_usajobs/ : has bge/me5/te3/base
run_eval jobs_data_usajobs \
  --retriever 'name=bm25:bm25' \
  --retriever 'name=base:st:base_catalog.vecs.fp16.npy:sentence-transformers/all-MiniLM-L6-v2' \
  --retriever 'name=bge:st:bge_small_en_catalog.vecs.fp16.npy:BAAI/bge-small-en-v1.5' \
  --retriever 'name=me5:st:me5_small_catalog.vecs.fp16.npy:intfloat/multilingual-e5-small' \
  --retriever 'name=te3:openai:te3_large_1024.vecs.fp16.npy:text-embedding-3-large:1024'

# jobs_data_jobstreet/ : same lineup, te3 file is openai_te3large_1024.vecs.fp16.npy
run_eval jobs_data_jobstreet \
  --retriever 'name=bm25:bm25' \
  --retriever 'name=base:st:base_catalog.vecs.fp16.npy:sentence-transformers/all-MiniLM-L6-v2' \
  --retriever 'name=bge:st:bge_small_en_catalog.vecs.fp16.npy:BAAI/bge-small-en-v1.5' \
  --retriever 'name=me5:st:me5_small_catalog.vecs.fp16.npy:intfloat/multilingual-e5-small' \
  --retriever 'name=te3:openai:openai_te3large_1024.vecs.fp16.npy:text-embedding-3-large:1024'

# jobs_data_linkedin/ : no base_catalog (skipped overnight); has bge/me5/te3
run_eval jobs_data_linkedin \
  --retriever 'name=bm25:bm25' \
  --retriever 'name=bge:st:bge_small_en_catalog.vecs.fp16.npy:BAAI/bge-small-en-v1.5' \
  --retriever 'name=me5:st:me5_small_catalog.vecs.fp16.npy:intfloat/multilingual-e5-small' \
  --retriever 'name=te3:openai:te3_large_1024.vecs.fp16.npy:text-embedding-3-large:1024'

echo "[$(ts)] === 4-corpus eval complete ==="
echo "results in $OUT_DIR/"
