#!/usr/bin/env bash
# Launch Solr (movies core) on port 8984 to avoid collision with jobs (8983),
# plus the FastAPI demo shim on port 7865.
# Usage:
#   ./run.sh            # start both
#   ./run.sh stop       # stop both
#   ./run.sh status     # show state

set -euo pipefail

export JAVA_HOME=/opt/homebrew/opt/openjdk@21
SOLR_BIN=/opt/homebrew/opt/solr/bin/solr
SOLR_HOME=/Users/dtunkelang/bagofdocs/solr_movies_demo/solr_home
SOLR_PORT=${SOLR_PORT:-8984}
SHIM_PORT=${SHIM_PORT:-7865}
HERE=/Users/dtunkelang/bagofdocs/solr_movies_demo
SHIM_LOG=$HERE/solr_shim.log
SHIM_PID_FILE=$HERE/solr_shim.pid
PY=/Users/dtunkelang/bagofdocs/.venv/bin/python3

cmd=${1:-start}

solr_up() {
  curl -sS -o /dev/null -w '%{http_code}' "http://localhost:${SOLR_PORT}/solr/admin/info/system" 2>/dev/null | grep -q 200
}

shim_up() {
  curl -sS -o /dev/null -w '%{http_code}' "http://127.0.0.1:${SHIM_PORT}/api/search?q=foo" 2>/dev/null | grep -q 200
}

start_solr() {
  if solr_up; then echo "Solr already up on ${SOLR_PORT}"; return; fi
  echo "starting Solr (movies, port ${SOLR_PORT})..."
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
  echo "starting shim (port ${SHIM_PORT})..."
  PORT=$SHIM_PORT nohup "$PY" "$HERE/app.py" > "$SHIM_LOG" 2>&1 &
  echo $! > "$SHIM_PID_FILE"
  for _ in $(seq 1 60); do shim_up && { echo "  shim ready at http://127.0.0.1:${SHIM_PORT}/"; return; }; sleep 1; done
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
