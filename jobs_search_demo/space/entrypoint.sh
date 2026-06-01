#!/usr/bin/env bash
# Cold-start bootstrap: hydrate Solr index from the companion HF dataset, start
# Solr in the background, then exec the FastAPI shim in the foreground.
set -euo pipefail

SOLR_HOME=${SOLR_HOME:-/tmp/solr_home}
SHIM_PORT=${SHIM_PORT:-7860}
DATASET_REPO=${DATASET_REPO:-dtunkelang/jobs-demo}

mkdir -p "$SOLR_HOME"

# Fresh solr_home needs a solr.xml.
if [ ! -f "$SOLR_HOME/solr.xml" ]; then
  cp /opt/solr/server/solr/solr.xml "$SOLR_HOME/solr.xml"
fi

# Always re-hydrate the 'jobs' core from the dataset. The previous
# `if [ ! -d $SOLR_HOME/jobs/data ]` guard skipped the download when the
# data dir existed, which would silently serve a stale index if HF ever
# changed /tmp from ephemeral to persistent across restarts.
rm -rf "$SOLR_HOME/jobs"
echo "[entrypoint] downloading solr index from $DATASET_REPO ..."
TARBALL=$(python3 - <<PY
from huggingface_hub import hf_hub_download
print(hf_hub_download(
    repo_id="${DATASET_REPO}",
    repo_type="dataset",
    filename="solr_index/solr_jobs_core.tar",
))
PY
)
echo "[entrypoint] downloaded to $TARBALL"
echo "[entrypoint] extracting to $SOLR_HOME ..."
tar -xf "$TARBALL" -C "$SOLR_HOME"
echo "[entrypoint] extraction done."

# Free ~4 GB: drop the tarball + its blob to keep /tmp from filling up.
BLOB=$(readlink -f "$TARBALL" || echo "$TARBALL")
rm -f "$TARBALL" "$BLOB"
echo "[entrypoint] removed cached tarball."

echo "[entrypoint] starting Solr on 8983 ..."
# No --force needed — we're already running as the solr user.
# Solr defaults to background; my shim runs in foreground next.
solr start --user-managed --solr-home "$SOLR_HOME" -p 8983

# Wait for Solr admin endpoint to come up.
for i in $(seq 1 60); do
  if curl -sS -o /dev/null -w '%{http_code}' "http://localhost:8983/solr/admin/info/system" 2>/dev/null | grep -q 200; then
    echo "[entrypoint] Solr up."
    break
  fi
  if [ "$i" -eq 60 ]; then
    echo "[entrypoint] Solr failed to come up; tail of solr log:" >&2
    tail -50 "$SOLR_HOME"/../*solr*.log 2>/dev/null || true
    exit 1
  fi
  sleep 1
done

# Ensure the 'jobs' core is loaded. The deploy tar is self-contained
# (jobs/core.properties + conf inside it), so Solr auto-discovers the core at
# startup — but discovery can lag a beat behind the admin API coming up. Poll
# STATUS briefly; only fall back to an explicit Core-admin CREATE if the core
# never registers. (A CREATE when the core is already defined returns a harmless
# 400 "another core is already defined there"; tolerate it instead of treating
# it as fatal — this is what produced the noisy startup error.)
JOBS_LOADED=0
for _ in $(seq 1 30); do
  JOBS_STATUS=$(curl -sS "http://localhost:8983/solr/admin/cores?action=STATUS&core=jobs&wt=json" || true)
  if echo "$JOBS_STATUS" | grep -q '"instanceDir"'; then
    JOBS_LOADED=1
    echo "[entrypoint] jobs core auto-discovered."
    break
  fi
  sleep 1
done
if [ "$JOBS_LOADED" -eq 0 ]; then
  echo "[entrypoint] core not auto-discovered; invoking Core admin CREATE ..."
  CREATE_RESP=$(curl -sS -X POST "http://localhost:8983/solr/admin/cores?action=CREATE&name=jobs&instanceDir=$SOLR_HOME/jobs" || true)
  if echo "$CREATE_RESP" | grep -q '"status":0'; then
    echo "[entrypoint] core created."
  elif echo "$CREATE_RESP" | grep -qiE "already defined|already exists"; then
    echo "[entrypoint] core already defined (ok)."
  else
    echo "[entrypoint] core CREATE failed: $CREATE_RESP" >&2
    tail -50 /var/solr/logs/solr.log 2>/dev/null || tail -50 "$SOLR_HOME"/../*solr*.log 2>/dev/null || true
    exit 1
  fi
fi

# Confirm doc count is plausible.
COUNT=$(curl -sS "http://localhost:8983/solr/jobs/select?q=*:*&rows=0&wt=json" \
        | python3 -c "import sys, json; print(json.load(sys.stdin)['response']['numFound'])" 2>/dev/null || echo "?")
echo "[entrypoint] jobs core numDocs=$COUNT"

echo "[entrypoint] starting FastAPI shim on $SHIM_PORT ..."
exec python3 /opt/app/app.py
