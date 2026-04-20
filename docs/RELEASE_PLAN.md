# Release Plan

## Three-part release

### 1. HuggingFace Dataset (`dtunkelang/bag-of-documents`)

Contents:
- Product titles + categories + brands (~5M Amazon products, parquet)
- Pre-computed fine-tuned MiniLM embeddings (parquet or numpy shards)
- Bags with centroids, specificity, product titles (parquet)
- ESCI evaluation results
- Metadata: category distribution, product counts, schema docs

Size: ~10-15GB compressed. Use parquet for columnar access.

Dataset card: `DATASET_CARD.md` (drafted)

### 2. HuggingFace Space (`dtunkelang/bag-of-documents-demo`)

Stack:
- Gradio (natively supported on Spaces)
- Loads FAISS index + fine-tuned MiniLM at startup
- Search modes: fine-tuned vs base model comparison, bag search
- Free tier: 16GB RAM CPU-only

### 3. GitHub Repository (`dtunkelang/bag-of-documents`)

Include:
- `compute_bags.py` — bag computation (hybrid retrieval + CE scoring)
- `build_index.py` — Build FAISS HNSW + tantivy indexes with validation
- `download_catalog.py` — Download and sample McAuley Lab product catalog
- `finetune_query_model.py` — fine-tune on bag centroids
- `eval_model.py` — ESCI evaluation
- `demo.py` — web demo (fine-tuned vs base, optional bag search)
- `preflight.py` — pre-run validation
- `query_index.py` — CLI for querying the index
- `download_catalog.py` — download and sample McAuley catalog
- `run_pipeline.sh` — pipeline orchestration
- `tests/` — unit tests
- `README.md`, `LICENSE` (MIT), `requirements.txt`, `pyproject.toml`
- `SCALING.md` — guide for scaling to the full 30M product catalog
- `SUMMARY.md`, `DATASET_CARD.md`, `BLOG_DRAFT.md`

Exclude:
- Raw data files (hosted on HuggingFace Hub)
- `.venv/`, `__pycache__/`, `.pytest_cache/`
- Legacy scripts: `relevance_scorer.py`, `rule_filter.py`, `finetune_specificity.py`, `finetune_retrieval_model.py`, `finetune_vector_model.py`
- Legacy data: `product_attributes.db`, `brand_vocabulary.json`, `category_accessory_rates.json`, `product_categories.jsonl`
- Intermediate files: `gold_*.csv`, `llm_eval_sample.jsonl`, `eval_log.txt`

## Pre-release checklist

- [ ] Fine-tuned model validated (eval numbers in overnight.log)
- [ ] Broad catalog experiment (20%, 6M products) validated
- [ ] Demo launches and both modes work (default + bag search)
- [ ] All tests pass (`pytest tests/`)
- [ ] Lint clean (`ruff check && ruff format --check`)
- [ ] README has clear setup instructions
- [ ] Dataset card complete
- [ ] No API keys, credentials, or personal paths in code
- [ ] All scripts have accurate docstrings
- [ ] Requirements.txt is complete and minimal
- [ ] Legacy files deleted (see cleanup queue in memory)
- [ ] Version numbers removed from directory names (bags/, query_model/)
- [ ] Blog post finalized
