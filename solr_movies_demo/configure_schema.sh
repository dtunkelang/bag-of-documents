#!/usr/bin/env bash
set -euo pipefail
SOLR=${SOLR:-http://localhost:8984}
CORE=${CORE:-movies}

# 1. Disable schemaless field-guessing — we own the schema.
curl -sS -X POST -H 'Content-Type: application/json' \
  "$SOLR/solr/$CORE/config" \
  --data-binary '{"set-user-property": {"update.autoCreateFields":"false"}}'
echo

# 2. Field types: BM25 text + (deferred) 384-dim dense vector slot for bge.
curl -sS -X POST -H 'Content-Type: application/json' \
  "$SOLR/solr/$CORE/schema" --data-binary @- <<'JSON'
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
    }
  ]
}
JSON
echo

# 3. Fields.
#    Searchable text: title, original_title, lead, plot, cast_names, director_names, genres.
#    Stored display: title_display, year, type, rating, votes, runtime, is_adult, enwiki_title.
#    Facets / filters: type, genres, year_bucket, has_lead, has_plot, has_bag, decade.
#    Bag: corated_bag (mv strings, tconst-only).
#    Vector slot: bge_vec (open-weight encoder, populated in a later phase).
curl -sS -X POST -H 'Content-Type: application/json' \
  "$SOLR/solr/$CORE/schema" --data-binary @- <<'JSON'
{
  "add-field": [
    {"name": "title",           "type": "text_en_bm25", "indexed": true,  "stored": true,  "multiValued": false},
    {"name": "title_display",   "type": "string",       "indexed": false, "stored": true,  "multiValued": false},
    {"name": "original_title",  "type": "text_en_bm25", "indexed": true,  "stored": true,  "multiValued": false},
    {"name": "lead",            "type": "text_en_bm25", "indexed": true,  "stored": true,  "multiValued": false},
    {"name": "plot",            "type": "text_en_bm25", "indexed": true,  "stored": true,  "multiValued": false},
    {"name": "cast_names",      "type": "text_en_bm25", "indexed": true,  "stored": true,  "multiValued": true},
    {"name": "director_names",  "type": "text_en_bm25", "indexed": true,  "stored": true,  "multiValued": true},
    {"name": "genres",          "type": "string",       "indexed": true,  "stored": true,  "multiValued": true},
    {"name": "year",            "type": "pint",         "indexed": true,  "stored": true,  "multiValued": false},
    {"name": "decade",          "type": "string",       "indexed": true,  "stored": true,  "multiValued": false},
    {"name": "type",            "type": "string",       "indexed": true,  "stored": true,  "multiValued": false},
    {"name": "runtime",         "type": "pint",         "indexed": true,  "stored": true,  "multiValued": false},
    {"name": "is_adult",        "type": "boolean",      "indexed": true,  "stored": true,  "multiValued": false},
    {"name": "rating",          "type": "pdouble",      "indexed": true,  "stored": true,  "multiValued": false},
    {"name": "votes",           "type": "pint",         "indexed": true,  "stored": true,  "multiValued": false},
    {"name": "enwiki_title",    "type": "string",       "indexed": false, "stored": true,  "multiValued": false},
    {"name": "has_lead",        "type": "boolean",      "indexed": true,  "stored": true,  "multiValued": false},
    {"name": "has_plot",        "type": "boolean",      "indexed": true,  "stored": true,  "multiValued": false},
    {"name": "has_bag",         "type": "boolean",      "indexed": true,  "stored": true,  "multiValued": false},
    {"name": "corated_bag",     "type": "string",       "indexed": true,  "stored": true,  "multiValued": true},
    {"name": "bge_vec",         "type": "knn_vector_384", "indexed": true, "stored": false, "multiValued": false}
  ]
}
JSON
echo
echo "schema configured."
