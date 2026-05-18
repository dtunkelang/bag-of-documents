#!/bin/bash
# End-to-end ESCI-US demo: download → index → bag → fine-tune.
# Idempotent: each step skips if its output already exists.
# Uses a public cross-encoder for relevance filtering (no local CE setup
# required). After this completes, run `python demo.py` to try the model.
set -e
cd "$(dirname "$0")/.."

VENV=.venv/bin/python
[ -x "$VENV" ] || VENV=python

# 1. Download ESCI-US catalog + train/test queries + qrels
if [ ! -f esci_us_data/titles.json ]; then
    echo "[1/5] Downloading ESCI-US catalog + queries..."
    $VENV download/download_esci_us.py
else
    echo "[1/5] ESCI-US data already present; skipping download."
fi

# 2. Link ESCI-US catalog into combined_index/ (where build_index.py looks)
mkdir -p combined_index
if [ ! -L combined_index/titles.json ] && [ ! -f combined_index/titles.json ]; then
    echo "[2/5] Linking ESCI-US titles.json into combined_index/..."
    ln -sf ../esci_us_data/titles.json combined_index/titles.json
else
    echo "[2/5] combined_index/titles.json already present; skipping link."
fi

# 3. Build FAISS + tantivy indexes
if [ ! -f combined_index/index.faiss ]; then
    echo "[3/5] Building FAISS + tantivy indexes..."
    $VENV indexing/build_index.py
else
    echo "[3/5] combined_index/index.faiss exists; skipping index build."
fi

# 4. Aggregate queries -> bags, filtered by a public cross-encoder
if [ ! -f bags.jsonl ]; then
    echo "[4/5] Computing bags (CE-filtered)..."
    $VENV training/compute_bags.py \
        esci_us_data/test_queries.jsonl bags.jsonl \
        --ce-rerank cross-encoder/ms-marco-MiniLM-L-12-v2 \
        --ce-threshold 0.3
else
    echo "[4/5] bags.jsonl exists; skipping bag computation."
fi

# 5. Fine-tune the BoD query encoder with MNRL loss
if [ ! -d query_model ]; then
    echo "[5/5] Fine-tuning BoD query encoder..."
    $VENV training/finetune_query_model.py bags.jsonl query_model/ --loss mnrl
else
    echo "[5/5] query_model/ exists; skipping fine-tune."
fi

echo "Done. Try: python demo.py"
