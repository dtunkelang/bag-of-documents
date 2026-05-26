#!/usr/bin/env bash
set -euo pipefail
SOLR=${SOLR:-http://localhost:8983}

# 1. Disable schemaless field-guessing — we want our schema to be authoritative.
curl -sS -X POST -H 'Content-Type: application/json' \
  "$SOLR/solr/jobs/config" \
  --data-binary '{"set-user-property": {"update.autoCreateFields":"false"}}'
echo

# 2. Field types: BM25 (k1=0.9, b=0.4) on title; dense vectors for bge + te3.
curl -sS -X POST -H 'Content-Type: application/json' \
  "$SOLR/solr/jobs/schema" --data-binary @- <<'JSON'
{
  "add-field-type": [
    {
      "name": "text_en_bm25",
      "class": "solr.TextField",
      "positionIncrementGap": "100",
      "similarity": {
        "class": "solr.BM25SimilarityFactory",
        "k1": 0.9,
        "b": 0.4
      },
      "analyzer": {
        "tokenizer": {"class": "solr.StandardTokenizerFactory"},
        "filters": [
          {"class": "solr.LowerCaseFilterFactory"},
          {"class": "solr.StopFilterFactory", "ignoreCase": "true", "words": "lang/stopwords_en.txt"},
          {"class": "solr.SnowballPorterFilterFactory", "language": "English"}
        ]
      }
    },
    {
      "name": "knn_vector_384",
      "class": "solr.DenseVectorField",
      "vectorDimension": 384,
      "similarityFunction": "cosine"
    },
    {
      "name": "knn_vector_1024",
      "class": "solr.DenseVectorField",
      "vectorDimension": 1024,
      "similarityFunction": "cosine"
    }
  ]
}
JSON
echo

# 3. Fields.
curl -sS -X POST -H 'Content-Type: application/json' \
  "$SOLR/solr/jobs/schema" --data-binary @- <<'JSON'
{
  "add-field": [
    {"name": "title",             "type": "text_en_bm25", "indexed": true,  "stored": true,  "multiValued": false},
    {"name": "title_display",     "type": "string",       "indexed": false, "stored": true,  "multiValued": false},
    {"name": "employer",          "type": "string",       "indexed": false, "stored": true,  "multiValued": false},
    {"name": "locations",         "type": "string",       "indexed": false, "stored": true,  "multiValued": true},
    {"name": "employment_type",   "type": "string",       "indexed": false, "stored": true,  "multiValued": false},
    {"name": "salary_min",        "type": "pdouble",      "indexed": false, "stored": true,  "multiValued": false},
    {"name": "salary_max",        "type": "pdouble",      "indexed": false, "stored": true,  "multiValued": false},
    {"name": "salary_currency",   "type": "string",       "indexed": false, "stored": true,  "multiValued": false},
    {"name": "department",        "type": "string",       "indexed": false, "stored": true,  "multiValued": false},
    {"name": "posted_at",         "type": "string",       "indexed": false, "stored": true,  "multiValued": false},
    {"name": "source_corpus",     "type": "string",       "indexed": true,  "stored": true,  "multiValued": false},
    {"name": "description",       "type": "string",       "indexed": false, "stored": true,  "multiValued": false},
    {"name": "bge_vec",           "type": "knn_vector_384",  "indexed": true, "stored": false, "multiValued": false},
    {"name": "te3_vec",           "type": "knn_vector_1024", "indexed": true, "stored": false, "multiValued": false}
  ]
}
JSON
echo
echo "schema configured."
