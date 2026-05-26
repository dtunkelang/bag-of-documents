#!/bin/bash
# Unattended ingest chain for the movies/TV demo.
# Order: MovieLens (small, fast, reliable) → Wikidata bridge (long, fragile).
# Each phase is checkpointed/skippable; chain continues if a phase fails.

set -u
cd "$(dirname "$0")"
LOG="movies_ingest_chain.log"

ts() { date '+%Y-%m-%d %H:%M:%S'; }
say() { echo "[$(ts)] $*" | tee -a "$LOG"; }

say "=== chain start (pid $$) ==="
say "disk: $(df -h /Users/dtunkelang | tail -1)"

say "--- Phase 5a: fetch MovieLens 25M ---"
.venv/bin/python download/fetch_movielens.py --out-dir movies_data/movielens 2>&1 | tee -a "$LOG"
rc=${PIPESTATUS[0]}
say "Phase 5a exit=$rc"

if [ -f movies_data/movielens/ml-25m/ratings.csv ]; then
  say "--- Phase 5b: build MovieLens co-rated bags ---"
  .venv/bin/python download/build_movielens_corated_bags.py \
    --ml-dir movies_data/movielens/ml-25m \
    --imdb-titles movies_data/titles_imdb.jsonl \
    --out-path movies_data/movielens_corated_bags.jsonl 2>&1 | tee -a "$LOG"
  rc=${PIPESTATUS[0]}
  say "Phase 5b exit=$rc"
else
  say "SKIP Phase 5b: ratings.csv missing"
fi

say "--- Phase 2: Wikidata SPARQL bridge ---"
.venv/bin/python download/build_wikidata_bridge.py \
  --imdb-titles movies_data/titles_imdb.jsonl \
  --out-path movies_data/wikidata_bridge.jsonl \
  --chunk-size 200 --rate-per-sec 1.0 2>&1 | tee -a "$LOG"
rc=${PIPESTATUS[0]}
say "Phase 2 exit=$rc"

say "disk: $(df -h /Users/dtunkelang | tail -1)"
say "=== chain done ==="
