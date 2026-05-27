#!/usr/bin/env bash
# Cold-start bootstrap: hydrate Solr index from the companion HF dataset, start
# Solr in the background, then exec the FastAPI shim in the foreground.
set -euo pipefail

SOLR_HOME=${SOLR_HOME:-/tmp/solr_home}
SHIM_PORT=${SHIM_PORT:-7860}
DATASET_REPO=${DATASET_REPO:-dtunkelang/movies-demo}
CORE_NAME=${CORE_NAME:-movies}
TARBALL_NAME=${TARBALL_NAME:-solr_index/solr_movies_core.tar}

mkdir -p "$SOLR_HOME"

# Fresh solr_home needs a solr.xml.
if [ ! -f "$SOLR_HOME/solr.xml" ]; then
  cp /opt/solr/server/solr/solr.xml "$SOLR_HOME/solr.xml"
fi

# Hydrate the core if it's not already on disk.
if [ ! -d "$SOLR_HOME/$CORE_NAME/data" ]; then
  echo "[entrypoint] downloading solr index from $DATASET_REPO ..."
  TARBALL=$(python3 - <<PY
from huggingface_hub import hf_hub_download
print(hf_hub_download(
    repo_id="${DATASET_REPO}",
    repo_type="dataset",
    filename="${TARBALL_NAME}",
))
PY
)
  echo "[entrypoint] downloaded to $TARBALL"
  echo "[entrypoint] extracting to $SOLR_HOME ..."
  tar -xf "$TARBALL" -C "$SOLR_HOME"
  echo "[entrypoint] extraction done."

  # Free disk: drop the tarball + its blob to keep /tmp from filling up.
  BLOB=$(readlink -f "$TARBALL" || echo "$TARBALL")
  rm -f "$TARBALL" "$BLOB"
  echo "[entrypoint] removed cached tarball."
fi

echo "[entrypoint] starting Solr on 8983 ..."
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

# Ensure the core is loaded. Auto-discovery doesn't always pick up cores from
# a freshly-extracted tarball, so we explicitly invoke the Core admin API.
STATUS=$(curl -sS "http://localhost:8983/solr/admin/cores?action=STATUS&core=${CORE_NAME}&wt=json" || true)
if echo "$STATUS" | grep -q "\"name\":\"${CORE_NAME}\"" && echo "$STATUS" | grep -q '"instanceDir"'; then
  echo "[entrypoint] ${CORE_NAME} core already loaded."
else
  echo "[entrypoint] loading ${CORE_NAME} core via Core admin ..."
  curl -sS -X POST "http://localhost:8983/solr/admin/cores?action=CREATE&name=${CORE_NAME}&instanceDir=$SOLR_HOME/${CORE_NAME}" || {
    echo "[entrypoint] core CREATE failed; tail of solr log:" >&2
    tail -50 /var/solr/logs/solr.log 2>/dev/null || tail -50 "$SOLR_HOME"/../*solr*.log 2>/dev/null || true
    exit 1
  }
  echo
fi

# Confirm doc count is plausible.
COUNT=$(curl -sS "http://localhost:8983/solr/${CORE_NAME}/select?q=*:*&rows=0&wt=json" \
        | python3 -c "import sys, json; print(json.load(sys.stdin)['response']['numFound'])" 2>/dev/null || echo "?")
echo "[entrypoint] ${CORE_NAME} core numDocs=$COUNT"

echo "[entrypoint] starting FastAPI shim on $SHIM_PORT ..."
exec python3 /opt/app/app.py
