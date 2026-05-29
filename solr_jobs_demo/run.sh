#!/usr/bin/env bash
# Launch Solr (if not already running) and the FastAPI shim.
# The shim is space/app.py — the SAME app deployed to the HF Space
# (RRF(BM25 + e5-small) + profile-match lane + suggested searches +
# personalized search) — so local dev mirrors production. It loads e5-small
# and downloads the suggestion corpus from HF on startup, so first start is slow.
# Usage:
#   ./run.sh            # start both
#   ./run.sh stop       # stop both
#   ./run.sh status     # show what's running

set -euo pipefail

export JAVA_HOME=${JAVA_HOME:-/opt/homebrew/opt/openjdk@21}
SOLR_BIN=/opt/homebrew/opt/solr/bin/solr
SOLR_HOME=/Users/dtunkelang/bagofdocs/solr_jobs_demo/solr_home
SOLR_PORT=8983
SHIM_PORT=7864
SHIM_LOG=/Users/dtunkelang/bagofdocs/solr_jobs_demo/solr_shim.log
SHIM_PID_FILE=/Users/dtunkelang/bagofdocs/solr_jobs_demo/solr_shim.pid
PY=/Users/dtunkelang/bagofdocs/.venv/bin/python3
HERE=/Users/dtunkelang/bagofdocs/solr_jobs_demo
SPACE_DIR=/Users/dtunkelang/bagofdocs/solr_jobs_demo/space

cmd=${1:-start}

solr_up() {
  curl -sS -o /dev/null -w '%{http_code}' "http://localhost:${SOLR_PORT}/solr/admin/info/system" 2>/dev/null | grep -q 200
}

shim_up() {
  curl -sS -o /dev/null -w '%{http_code}' "http://127.0.0.1:${SHIM_PORT}/api/search?q=foo" 2>/dev/null | grep -q 200
}

start_solr() {
  if solr_up; then echo "Solr already up on ${SOLR_PORT}"; return; fi
  echo "starting Solr..."
  "$SOLR_BIN" start --user-managed --solr-home "$SOLR_HOME" -p "$SOLR_PORT"
  for _ in $(seq 1 30); do solr_up && { echo "  Solr ready"; return; }; sleep 1; done
  echo "Solr failed to come up"; exit 1
}

start_shim() {
  if shim_up; then echo "shim already up on ${SHIM_PORT}"; return; fi
  if [ -f "$SHIM_PID_FILE" ] && kill -0 "$(cat "$SHIM_PID_FILE")" 2>/dev/null; then
    echo "shim PID $(cat "$SHIM_PID_FILE") alive but not responding; killing..."
    kill "$(cat "$SHIM_PID_FILE")" || true; sleep 2
  fi
  echo "starting shim (space/app.py — loads e5-small + suggestion corpus, ~15-20s)..."
  # Run from space/ so resume_match_lib resolves and cwd matches the deployed Space.
  ( cd "$SPACE_DIR" && SHIM_PORT="$SHIM_PORT" SOLR="http://localhost:${SOLR_PORT}" \
      nohup "$PY" app.py > "$SHIM_LOG" 2>&1 & echo $! > "$SHIM_PID_FILE" )
  for _ in $(seq 1 90); do shim_up && { echo "  shim ready at http://127.0.0.1:${SHIM_PORT}/"; return; }; sleep 1; done
  echo "shim failed to come up; tail of log:"; tail -30 "$SHIM_LOG"; exit 1
}

stop_shim() {
  if [ -f "$SHIM_PID_FILE" ]; then
    pid=$(cat "$SHIM_PID_FILE")
    if kill -0 "$pid" 2>/dev/null; then kill "$pid"; echo "stopped shim (pid $pid)"; fi
    rm -f "$SHIM_PID_FILE"
  fi
}

stop_solr() {
  if solr_up; then "$SOLR_BIN" stop -p "$SOLR_PORT"; fi
}

case "$cmd" in
  start) start_solr; start_shim ;;
  stop) stop_shim; stop_solr ;;
  status)
    solr_up && echo "Solr: up (port ${SOLR_PORT})" || echo "Solr: down"
    shim_up && echo "Shim: up (http://127.0.0.1:${SHIM_PORT}/)" || echo "Shim: down"
    ;;
  *) echo "usage: $0 [start|stop|status]"; exit 2 ;;
esac
