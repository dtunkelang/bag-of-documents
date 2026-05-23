#!/usr/bin/env bash
# Overnight queue: chain recovery + full catalog encoding for all 4 corpora.
# Tolerant of single-step failures; logs everything; runs under caffeinate -di.
#
# Pre-conditions (running when this script starts):
#   - PID 15195: 244k bge-small-en encode (MPS, caffeinated separately)
#   - PID 14717: LinkedIn 96k bge-small-en encode (CPU, caffeinated separately)
#
# Launch (note: don't double-caffeinate — wrap THIS script):
#   nohup caffeinate -di ./overnight_queue.sh > logs/overnight_$(date +%Y%m%d_%H%M).log 2>&1 &
#   disown

set -uo pipefail   # no -e; we want to continue on per-step failures
cd /Users/dtunkelang/bagofdocs

PY=.venv/bin/python
ts() { date '+%H:%M:%S'; }
ok() { echo "[$(ts)] ✓ $*"; }
fail() { echo "[$(ts)] ✗ FAIL: $*" >&2; }
section() { echo; echo "[$(ts)] === $* ==="; }

# Load OPENAI_API_KEY etc. for API encodes
set -a && source .env && set +a

wait_for_pid() {
  local pid=$1; local label=$2
  echo "[$(ts)] waiting for PID $pid ($label)..."
  while kill -0 "$pid" 2>/dev/null; do sleep 30; done
  echo "[$(ts)] PID $pid ($label) finished"
}

encode_st() {
  local data_dir=$1; local model=$2; local out_name=$3; local device=$4; local prefix=${5:-}
  local args=(--data-dir "$data_dir" --model "$model" --out-name "$out_name" --batch-size 64 --device "$device")
  [ -n "$prefix" ] && args+=(--doc-prefix "$prefix")
  $PY -u download/encode_st_catalog.py "${args[@]}"
}

encode_openai() {
  local data_dir=$1; local out_name=$2
  $PY -u download/encode_openai_embeddings.py --data-dir "$data_dir" --model text-embedding-3-large --dim 1024 --out-name "$out_name"
}

# =================================================================
section "PHASE 1: wait for 244k bge encode, then run chain swap"
# =================================================================
wait_for_pid 15195 "244k bge encode"

if [ ! -f jobs_data_244k/bge_small_en_catalog.vecs.fp16.npy ]; then
  fail "244k bge catalog missing after wait — skipping chain swap"
else
  ok "244k bge catalog present"
  echo "[$(ts)] re-launching jobs_a_chain.sh (steps 2-7 will skip; 8-10 run)..."
  bash ./jobs_a_chain.sh && ok "chain swap + demo restart complete" || fail "chain swap"
fi

# After this point, the corpus formerly at jobs_data_244k/ lives at jobs_data/.
# Other corpora (linkedin, usajobs, jobstreet) are unaffected by the swap.

# =================================================================
section "PHASE 2: wait for LinkedIn bge CPU encode"
# =================================================================
wait_for_pid 14717 "LinkedIn 96k bge CPU encode"
[ -f jobs_data_linkedin/bge_small_en_catalog.vecs.fp16.npy ] && ok "LinkedIn bge present" || fail "LinkedIn bge missing"

# =================================================================
section "PHASE 3: encode LinkedIn me5_small + te3-large"
# =================================================================
echo "[$(ts)] LinkedIn me5_small (MPS)..."
encode_st jobs_data_linkedin intfloat/multilingual-e5-small me5_small_catalog mps "passage: " && \
  ok "LinkedIn me5_small done" || fail "LinkedIn me5_small"

echo "[$(ts)] LinkedIn te3-large @ 1024 (API)..."
encode_openai jobs_data_linkedin te3_large_1024 && ok "LinkedIn te3-large done" || fail "LinkedIn te3-large"

# =================================================================
section "PHASE 4: encode usajobs with full lineup (bge + me5 + te3)"
# =================================================================
# Already have base_minilm. Add the 3 new ones for parity with linkedin/jobstreet.
echo "[$(ts)] usajobs bge-small-en (MPS)..."
encode_st jobs_data_usajobs BAAI/bge-small-en-v1.5 bge_small_en_catalog mps && \
  ok "usajobs bge done" || fail "usajobs bge"

echo "[$(ts)] usajobs me5_small (MPS)..."
encode_st jobs_data_usajobs intfloat/multilingual-e5-small me5_small_catalog mps "passage: " && \
  ok "usajobs me5_small done" || fail "usajobs me5_small"

echo "[$(ts)] usajobs te3-large @ 1024 (API)..."
encode_openai jobs_data_usajobs te3_large_1024 && ok "usajobs te3 done" || fail "usajobs te3"

# =================================================================
section "PHASE 5: encode jobstreet bge-small-en (already has mini + me5 + te3)"
# =================================================================
echo "[$(ts)] jobstreet bge-small-en (MPS)..."
encode_st jobs_data_jobstreet BAAI/bge-small-en-v1.5 bge_small_en_catalog mps && \
  ok "jobstreet bge done" || fail "jobstreet bge"

# =================================================================
section "PHASE 6: summary"
# =================================================================
echo "[$(ts)] catalog inventory by corpus:"
for d in jobs_data jobs_data_linkedin jobs_data_usajobs jobs_data_jobstreet; do
  if [ -d "$d" ]; then
    echo
    echo "  $d/"
    ls -lh "$d"/*.vecs.fp16.npy 2>/dev/null | awk '{print "    " $5, $9}'
  fi
done

echo
echo "[$(ts)] === overnight queue complete ==="
