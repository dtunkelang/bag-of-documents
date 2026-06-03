#!/usr/bin/env bash
set -euo pipefail
SOLR=${SOLR:-http://localhost:8983}
CORE=${JOBS_CORE:-jobs}
curl -sS -X POST -H 'Content-Type: application/json' \
  "$SOLR/solr/$CORE/schema" --data-binary @- <<'JSON'
{
  "add-field": [
    {"name": "role_family",             "type": "string", "indexed": true, "stored": true, "multiValued": false},
    {"name": "seniority",               "type": "string", "indexed": true, "stored": true, "multiValued": false},
    {"name": "remote_mode",             "type": "string", "indexed": true, "stored": true, "multiValued": false},
    {"name": "location_country",        "type": "string", "indexed": true, "stored": true, "multiValued": false},
    {"name": "location_state",          "type": "string", "indexed": true, "stored": true, "multiValued": false},
    {"name": "location_city",           "type": "string", "indexed": true, "stored": true, "multiValued": false},
    {"name": "posted_bucket",           "type": "string", "indexed": true, "stored": true, "multiValued": false},
    {"name": "salary_band_usd_annual",  "type": "string", "indexed": true, "stored": true, "multiValued": false},
    {"name": "tech_stack",              "type": "string", "indexed": true, "stored": true, "multiValued": true},
    {"name": "lang",                    "type": "string", "indexed": true, "stored": true, "multiValued": false},
    {"name": "rome_code",               "type": "string", "indexed": true, "stored": true, "multiValued": false}
  ]
}
JSON
echo
echo "schema updated."
