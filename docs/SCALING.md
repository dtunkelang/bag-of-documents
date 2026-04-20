# Scaling to the Full McAuley Catalog (~30M Products)

## Overview

The current pipeline runs on 6M products (20% sample of all 33 Amazon categories) and fits on a 16GB laptop. Scaling to the full 30M-product McAuley catalog would produce the largest English-language bag-of-documents dataset for product search.

The pipeline itself is category-agnostic — no heuristics or category-specific logic. The main challenges are compute and storage.

## What changes

### 1. Embed all 30M products

One-time cost. MiniLM (all-MiniLM-L6-v2) encoding at ~1000 titles/sec on CPU:
- **CPU**: ~13 hours
- **GPU**: ~1-2 hours

Output: 30M × 384-dim × 4 bytes = **~44GB** raw vectors.

### 2. FAISS index type

HNSW (current) stores full vectors in memory. 30M × 384-dim = ~44GB — too large for most machines.

Options:
- **IVF + PQ**: compressed index, ~5-10GB in memory. Requires tuning `nlist` and `nprobe`. Some recall loss.
- **IVF + HNSW coarse quantizer + PQ**: better recall than plain IVF, still compressed.
- **HNSW on a large-memory instance**: 128GB+ RAM. Simplest, most accurate, but expensive.

`build_index.py` builds both FAISS HNSW and tantivy indexes. For larger-than-memory indexes, would need to support IVF/PQ variants.

**Note**: PQ was previously observed to corrupt ranking order. If using PQ, validate retrieval quality carefully.

### 3. Tantivy keyword index

Scales to 30M titles with no code changes. Index size will grow proportionally (~10-20GB on disk).

### 4. Bag computation

No code changes needed. With 75K ESCI queries (up from 18K in-scope today):
- At ~1.8s/query (CE scoring all candidates): ~37 hours
- Candidate pool may be larger with 30M products, which would increase CE scoring time

### 5. Fine-tuning and evaluation

No changes. `finetune_query_model.py` and `eval_model.py` work on (query, centroid) pairs regardless of catalog size.

## What stays the same

- `compute_bags.py` — hybrid retrieval → CE score all → threshold → centroid
- `finetune_query_model.py` — MSE loss on (query text → bag centroid)
- `eval_model.py` — ESCI benchmark evaluation
- `demo.py` — works with any FAISS index type (has HNSW/non-HNSW code paths)
- CE threshold (0.3), batch size (32), candidate multiplier (4×)

## Resource requirements

| Resource | Current (6M, 20%) | Full catalog (30M) |
|----------|-------------------|-------------------|
| Products | 6M | 30M |
| ESCI query coverage | ~35K (47%) | ~60-70K (80-90%) |
| Raw vectors | ~9GB | ~44GB |
| FAISS index (HNSW) | ~10GB | ~55GB (needs 64GB+ RAM) |
| FAISS index (IVF+PQ) | — | ~5-10GB (fits 16-32GB RAM) |
| Tantivy index | ~2GB | ~10GB |
| Disk total | ~25GB | ~100-120GB |
| Bag computation | ~19h (MPS) | ~19h (same queries, more candidates) |
| Embedding products | ~1.5h | ~8h CPU / ~1.5h GPU |

## Hardware options

### Mac Studio (recommended for iterative work)

A Mac Studio with M4 Max (128GB unified memory) fits the full HNSW index (~90GB) with room for models and OS. The unified memory architecture is ideal — FAISS, embedding models, and the CE all share the same memory pool.

| Config | RAM | Price (approx) | Notes |
|--------|-----|----------------|-------|
| M4 Max, 128GB | 128GB | ~$3,200 | Fits HNSW index, some headroom |
| M3 Ultra, 192GB | 192GB | ~$5,000+ | Comfortable headroom |

Disk: 1TB minimum (configurable to 8TB). Neural Engine can accelerate MiniLM inference.

The one-time hardware cost pays for itself quickly if you're iterating (recompute bags → fine-tune → rebuild index → repeat). No hourly charges, no data transfer latency.

### Cloud (AWS, optimized for cost)

Use the right instance for each stage, spot pricing throughout, and a shared EBS volume to avoid data transfers. The pipeline is resumable (skips existing bags), so spot interruptions are safe.

| Stage | Instance | Duration | Spot cost |
|-------|----------|----------|-----------|
| 1. Embed 30M products | `g5.xlarge` (A10G GPU, 16GB) | ~2h | ~$2 |
| 2. Build FAISS HNSW + tantivy | `r6i.4xlarge` (128GB RAM) | ~1-2h | ~$3 |
| 3. Compute bags (CE scoring) | `r6i.4xlarge` (128GB RAM) | ~40h | ~$15 |
| 4. Fine-tune MiniLM | `g5.xlarge` (GPU) | ~1-2h | ~$1 |
| 5. Eval against ESCI | `g5.xlarge` (GPU) | ~30min | ~$0.25 |
| **Total** | | **~45h elapsed** | **~$20-25** |

Plus ~$40/month for a 500GB gp3 EBS volume shared across instances.

Tips:
- **Spot instances throughout** — pipeline is resumable, interruptions just mean restarting
- **Single EBS volume** (500GB, gp3) mounted across instances — avoids repeated S3 transfers between stages
- **Terminate instances between stages** — don't pay for 128GB RAM during GPU work or vice versa
- **Cheapest region** for spot availability (usually `us-east-2` or `us-west-2`)

Better for one-off runs; worse for iteration (latency of spinning up/down instances).

## Recommended approach

1. Get a machine with 128GB+ RAM (Mac Studio or cloud)
2. Embed all 30M products with MiniLM
3. Build HNSW index (simplest with 128GB+ RAM)
4. Run bag computation with all 75K ESCI US queries + existing 99K cleaned queries
5. Fine-tune and evaluate
6. If using cloud, results (model + compressed index) can be transferred to laptop for demo
