#!/usr/bin/env python3
"""
Build FAISS + tantivy indexes by encoding all titles with the current model.

Encodes titles in batches with validation, builds HNSW index and tantivy
full-text index.

Usage:
    python build_index.py
    python build_index.py --model query_model/
"""

import argparse
import json
import os
import shutil
import sys
import time

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from utils import fmt_duration

INDEX_DIR = "combined_index"
EMBED_MODEL_ID = "all-MiniLM-L6-v2"
BATCH_SIZE = 512
HNSW_M = 32
HNSW_EF_CONSTRUCTION = 200


def main():
    parser = argparse.ArgumentParser(description="Build FAISS + tantivy indexes")
    parser.add_argument(
        "--model", default=EMBED_MODEL_ID, help=f"Model to encode with (default: {EMBED_MODEL_ID})"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="GPU batch size for sentence-transformer encoding (default: 128). "
        "Lower (e.g., 64) on memory-constrained machines if you see swap pressure.",
    )
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    index_path = os.path.join(script_dir, INDEX_DIR)

    # Load titles
    print("Loading titles...")
    with open(os.path.join(index_path, "titles.json")) as f:
        titles = json.load(f)
    print(f"  {len(titles):,} titles")

    # Load model
    model_id = args.model
    print(f"Loading model {model_id}...")
    model = SentenceTransformer(model_id)
    dim = model.get_sentence_embedding_dimension()
    embeddings_size_gb = len(titles) * dim * 4 / 1e9
    print(f"  Embedding dimension: {dim}, estimated size: {embeddings_size_gb:.1f} GB")

    # --- EARLY VALIDATION (first 1000) ---
    print("\n=== EARLY VALIDATION (first 1000 titles) ===")
    sample = titles[:1000]
    sample_vecs = model.encode(
        sample, normalize_embeddings=True, batch_size=256, show_progress_bar=False
    )
    sample_vecs = np.array(sample_vecs, dtype=np.float32)

    # Self-search: each title should be nearest to itself
    # Build a small flat index
    flat_index = faiss.IndexFlatIP(dim)
    flat_index.add(sample_vecs)
    D, I = flat_index.search(sample_vecs[:20], 5)

    hits = sum(1 for i in range(20) if I[i][0] == i)
    print(f"  Self-retrieval: {hits}/20 titles are their own nearest neighbor")
    if hits < 18:
        print("  WARNING: Self-retrieval below 90% — model may be broken!")
        sys.exit(1)

    # Cross-check: similar titles should have high cosine
    pairs = [
        ("Apple iPhone 12 Pro Max 128GB", "Apple iPhone 12 Pro 128GB"),
        ("Samsung Galaxy S21 Ultra", "Samsung Galaxy S21 Plus"),
        ("USB-C Charger Cable", "USB Type C Charging Cable"),
    ]
    print("  Cross-check (similar titles should have high cosine):")
    for a, b in pairs:
        va = model.encode(a, normalize_embeddings=True)
        vb = model.encode(b, normalize_embeddings=True)
        cos = float(va @ vb)
        status = "OK" if cos > 0.7 else "LOW"
        print(f"    [{status}] {cos:.3f}: '{a}' vs '{b}'")

    # Check norms
    norms = np.linalg.norm(sample_vecs, axis=1)
    print(f"  Vector norms: min={norms.min():.4f} max={norms.max():.4f} mean={norms.mean():.4f}")

    # Check cosine distribution of first 100 vs query
    q = model.encode("iphone case", normalize_embeddings=True)
    cosines = sample_vecs @ q
    print(f"  'iphone case' vs first 1000: max={cosines.max():.3f} mean={cosines.mean():.3f}")

    print("  Validation PASSED\n")

    # --- FULL ENCODING ---
    embeddings_path = os.path.join(index_path, "embeddings.npy")
    progress_path = os.path.join(index_path, "embed_progress.txt")
    n = len(titles)

    # Check for resume
    start_idx = 0
    if os.path.exists(embeddings_path) and os.path.exists(progress_path):
        with open(progress_path) as pf:
            start_idx = int(pf.read().strip())
        if start_idx >= n:
            print(f"Embeddings already complete ({start_idx:,}/{n:,})")
        else:
            print(f"Resuming from {start_idx:,}/{n:,}")

    if start_idx < n:
        if start_idx == 0:
            print(f"Pre-allocating {embeddings_size_gb:.1f} GB embeddings file...", flush=True)
            fp = np.lib.format.open_memmap(
                embeddings_path, mode="w+", dtype=np.float32, shape=(n, dim)
            )
        else:
            fp = np.lib.format.open_memmap(
                embeddings_path, mode="r+", dtype=np.float32, shape=(n, dim)
            )

        CHECKPOINT_INTERVAL = 500_000  # validate every 500K titles

        print(f"Encoding {n - start_idx:,} titles (batch_size={BATCH_SIZE})...", flush=True)
        encode_start = time.time()
        processed = 0
        next_checkpoint = CHECKPOINT_INTERVAL

        for i in range(start_idx, n, BATCH_SIZE):
            batch_end = min(i + BATCH_SIZE, n)
            batch = titles[i:batch_end]

            # Inner batch_size controls model forward pass; outer BATCH_SIZE controls I/O chunking
            vecs = model.encode(
                batch,
                normalize_embeddings=True,
                batch_size=args.batch_size,
                show_progress_bar=False,
            )

            # Per-batch sanity: check for NaN or zero vectors
            batch_norms = np.linalg.norm(vecs, axis=1)
            if np.any(np.isnan(batch_norms)) or np.any(batch_norms < 0.5):
                bad = int(np.sum(np.isnan(batch_norms) | (batch_norms < 0.5)))
                print(
                    f"\n  ABORT: {bad} bad vectors in batch "
                    f"[{i:,}:{batch_end:,}] — model may be corrupted",
                    flush=True,
                )
                sys.exit(1)

            fp[i:batch_end] = vecs
            processed += len(batch)

            # Progress every ~25K
            if processed % (BATCH_SIZE * 50) < BATCH_SIZE or batch_end == n:
                elapsed = time.time() - encode_start
                rate = processed / elapsed
                remaining = (n - start_idx - processed) / rate
                print(
                    f"  [{start_idx + processed:,}/{n:,}] "
                    f"{rate:.0f}/sec, "
                    f"elapsed {fmt_duration(elapsed)}, "
                    f"ETA {fmt_duration(remaining)}",
                    flush=True,
                )
                fp.flush()
                # Save progress for resume
                with open(progress_path, "w") as pf:
                    pf.write(str(start_idx + processed))

            # Periodic checkpoint every 500K
            total_done = start_idx + processed
            if total_done >= next_checkpoint:
                next_checkpoint = total_done + CHECKPOINT_INTERVAL
                print(f"\n  --- CHECKPOINT at {total_done:,} ---", flush=True)

                # 1. Re-encode a stored title and verify consistency
                check_idx = total_done // 2  # mid-point of what's done
                fresh = model.encode(titles[check_idx], normalize_embeddings=True)
                stored = np.array(fp[check_idx])
                cos = float(fresh @ stored)
                ok = cos > 0.99
                print(
                    f"  Consistency [{check_idx}]: cos={cos:.4f} "
                    f"{'OK' if ok else 'DRIFT DETECTED!'}",
                    flush=True,
                )
                if not ok:
                    print(
                        "  ABORT: Encoding drift detected — "
                        "stored vector doesn't match re-encoding",
                        flush=True,
                    )
                    sys.exit(1)

                # 2. Norm check on recent batch
                recent = np.array(fp[total_done - 1000 : total_done])
                norms = np.linalg.norm(recent, axis=1)
                print(
                    f"  Recent norms: min={norms.min():.4f} "
                    f"max={norms.max():.4f} mean={norms.mean():.4f}",
                    flush=True,
                )

                # 3. Disk space check
                free_gb = shutil.disk_usage(index_path).free / 1e9
                print(f"  Disk free: {free_gb:.1f} GB", flush=True)
                if free_gb < 5:
                    print(
                        f"  ABORT: Only {free_gb:.1f} GB free — need ~10 GB for HNSW build",
                        flush=True,
                    )
                    sys.exit(1)

                print("  --- checkpoint OK ---\n", flush=True)

        del fp
        # Mark complete
        with open(progress_path, "w") as pf:
            pf.write(str(n))
        total_encode = time.time() - encode_start
        print(f"Encoding complete: {fmt_duration(total_encode)}", flush=True)

    # --- VALIDATE FULL EMBEDDINGS ---
    print("\n=== POST-ENCODE VALIDATION ===")
    embeddings = np.load(embeddings_path, mmap_mode="r")
    print(f"  Shape: {embeddings.shape}")
    norms = np.linalg.norm(embeddings[:10000], axis=1)
    print(f"  Norms (first 10K): min={norms.min():.4f} max={norms.max():.4f}")

    # Spot check: encode a few titles and compare to stored
    for idx in [0, len(titles) // 2, len(titles) - 1]:
        fresh = model.encode(titles[idx], normalize_embeddings=True)
        stored = np.array(embeddings[idx])
        cos = float(fresh @ stored)
        print(f"  Spot check [{idx}]: cos={cos:.4f} ({'OK' if cos > 0.99 else 'MISMATCH!'})")

    # --- BUILD HNSW INDEX ---
    print(f"\n=== BUILDING HNSW INDEX (M={HNSW_M}, ef_construction={HNSW_EF_CONSTRUCTION}) ===")

    # Pre-build disk check: HNSW index will be ~9GB
    free_gb = shutil.disk_usage(index_path).free / 1e9
    est_index_gb = len(titles) * dim * 4 * 1.2 / 1e9  # ~1.2x overhead for HNSW graph
    print(f"  Disk free: {free_gb:.1f} GB, estimated index size: {est_index_gb:.1f} GB", flush=True)
    if free_gb < est_index_gb + 2:
        print("  ABORT: Not enough disk space for index build", flush=True)
        sys.exit(1)

    print("  Note: HNSW build is single-threaded and cannot be checkpointed.", flush=True)

    print("Loading embeddings into memory...", flush=True)
    data = np.array(embeddings[:], dtype=np.float32)
    del embeddings

    print(f"Building index over {data.shape[0]:,} vectors...", flush=True)
    build_start = time.time()
    index = faiss.IndexHNSWFlat(dim, HNSW_M)
    index.hnsw.efConstruction = HNSW_EF_CONSTRUCTION
    index.add(data)
    build_time = time.time() - build_start
    print(f"Index built in {fmt_duration(build_time)}", flush=True)

    # Save new index alongside old
    new_index_path = os.path.join(index_path, "index.faiss.new")
    print(f"Saving index to {new_index_path}...")
    faiss.write_index(index, new_index_path)
    new_size = os.path.getsize(new_index_path) / 1e9
    print(f"  New index: {new_size:.1f} GB")

    # --- FINAL VALIDATION ---
    print("\n=== FINAL VALIDATION ===")
    index.hnsw.efSearch = 128

    # Self-retrieval check
    test_indices = [0, 1000, len(titles) // 2, len(titles) - 1]
    for idx in test_indices:
        q = data[idx : idx + 1]
        D, I = index.search(q, 5)
        is_self = I[0][0] == idx
        cos_top = 1 - D[0][0] / 2
        print(
            f"  [{idx}] self={'YES' if is_self else 'NO'} "
            f"top_cos={cos_top:.4f} '{titles[idx][:60]}'"
        )

    # Query test
    q_vec = model.encode("iphone case", normalize_embeddings=True)
    q_vec = np.array(q_vec, dtype=np.float32).reshape(1, -1)
    faiss.normalize_L2(q_vec)
    D, I = index.search(q_vec, 10)
    print("\n  Query 'iphone case' top 10:")
    for rank, (dist, idx) in enumerate(zip(D[0], I[0])):
        cos = 1 - dist / 2
        print(f"    {rank + 1}. [{cos:.3f}] {titles[idx][:80]}")

    # Install new index
    old_index_path = os.path.join(index_path, "index.faiss")
    if os.path.exists(old_index_path):
        backup_path = os.path.join(index_path, "index.faiss.old")
        print("\nSwapping index files...")
        os.rename(old_index_path, backup_path)
        os.rename(new_index_path, old_index_path)
        print(f"  Old index backed up to {backup_path}")
    else:
        os.rename(new_index_path, old_index_path)
    print("  FAISS index is now active")

    # --- BUILD TANTIVY INDEX ---
    print("\n=== BUILDING TANTIVY INDEX ===")
    import tantivy

    tantivy_path = os.path.join(index_path, "tantivy_index")
    os.makedirs(tantivy_path, exist_ok=True)
    schema_builder = tantivy.SchemaBuilder()
    schema_builder.add_text_field("title", stored=True)
    schema = schema_builder.build()

    tv_index = tantivy.Index(schema, path=tantivy_path)
    writer = tv_index.writer()
    tv_start = time.time()
    for i, title in enumerate(titles):
        writer.add_document(tantivy.Document(title=title))
        if (i + 1) % 500_000 == 0:
            writer.commit()
            rate = (i + 1) / (time.time() - tv_start)
            print(f"  {i + 1:,} indexed ({rate:,.0f}/sec)", flush=True)
    writer.commit()
    print(f"  Tantivy index built in {fmt_duration(time.time() - tv_start)}")

    # Verify tantivy
    tv_index.reload()
    searcher = tv_index.searcher()
    parsed = tv_index.parse_query("keyboard", ["title"])
    results = searcher.search(parsed, limit=5)
    print(f"  Verification: 'keyboard' returned {results.count} results")

    # Report disk usage
    print("\nDisk usage:")
    print(f"  embeddings.npy: {os.path.getsize(embeddings_path) / 1e9:.1f} GB")
    print(f"  index.faiss: {os.path.getsize(old_index_path) / 1e9:.1f} GB")
    # Clean up progress file
    if os.path.exists(progress_path):
        os.remove(progress_path)

    print("\nTo free space, delete embeddings after confirming index works:")
    print(f"  rm {embeddings_path}")
    print("Done!")


if __name__ == "__main__":
    main()
