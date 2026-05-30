#!/usr/bin/env bash
# Scheduled jobs-demo refresh (driven by launchd; see launchd/com.dtunkelang.jobs-refresh.plist).
#
#   Mon-Sat -> incremental --delta (add new + delete closed postings, no wipe);
#              the stage-6 Xet upload dedups the tar to ~MB/secs.
#   Sunday  -> full rebuild (no --delta): reconciles in-place content edits and
#              compacts Lucene segment bloat. One-time ~full upload that day.
#
# Both pull OpenApply from the maintainer's daily HF snapshot (--openapply-source hf,
# ~2min, freshness <=1 day). Swap to `crawl` for a same-day deep refresh by hand.
set -euo pipefail

ROOT=/Users/dtunkelang/bagofdocs
cd "$ROOT"
export JAVA_HOME=/opt/homebrew/opt/openjdk@21
PY="$ROOT/.venv/bin/python"

# Optional secrets (USAJOBS_EMAIL / USAJOBS_API_KEY). Absent -> USAJobs reuses its
# existing raw parquet (it won't refresh, but the run won't fail). HF auth comes
# from the stored huggingface_hub token, so no token needed here.
ENV_FILE="$HOME/.config/jobs-refresh.env"
# shellcheck disable=SC1090
[ -f "$ENV_FILE" ] && . "$ENV_FILE"

if [ "$(date +%u)" = "7" ]; then
  echo "[$(date)] WEEKLY full rebuild (no --delta)"
  MODE=()
else
  echo "[$(date)] DAILY incremental (--delta)"
  MODE=(--delta)
fi

# -di: keep system AND display awake (display sleep throttles MPS during stage-2 encode).
exec caffeinate -di "$PY" solr_jobs_demo/refresh.py \
  --from-stage 0 --to-stage 7 --no-dry-run \
  --openapply-source hf --out-dir "$ROOT/unified_jobs_daily" \
  "${MODE[@]}"
