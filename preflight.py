#!/usr/bin/env python3
"""
Preflight checks before any long pipeline run.

Validates:
1. Disk space sufficient
1b. Memory headroom for encoding (warning only)
2. No competing ML processes
3. Required files exist
4. FAISS index consistency (stored vectors match current model)
5. Tantivy index loads and returns results
6. Smoke test: CE-based bag computation on diverse queries

Usage:
    python preflight.py              # full check
    python preflight.py --quick      # skip smoke test (index + disk only)
"""

import argparse
import json
import os
import re
import subprocess
import sys

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INDEX_DIR = os.path.join(SCRIPT_DIR, "combined_index")
DEFAULT_MODEL = "all-MiniLM-L6-v2"

# Diverse test queries covering different categories and edge cases
SMOKE_QUERIES = [
    "iphone case",
    "nvidia rtx 3080",
    "samsung galaxy s21",
    "wireless keyboard",
    "yoga mat",
    "ring light",
    "usb c cable",
    "hp laptop 16gb ram",
    "electric guitar strings",
    "watercolor paint set",
]

FAIL = False


def check(ok, msg):
    global FAIL
    status = "OK" if ok else "FAIL"
    print(f"  [{status}] {msg}")
    if not ok:
        FAIL = True
    return ok


def main():
    global FAIL
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="Skip smoke test")
    parser.add_argument(
        "--ce-threshold", type=float, default=0.3, help="CE threshold for smoke test (default: 0.3)"
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Embedding model for consistency check (default: {DEFAULT_MODEL})",
    )
    args = parser.parse_args()

    print("=== PREFLIGHT CHECKS ===\n")

    # 1. Disk space
    print("1. Disk space")
    try:
        import shutil

        free_gb = shutil.disk_usage(SCRIPT_DIR).free // (1024**3)
        check(free_gb >= 10, f"{free_gb} GB available (need >= 10 GB)")
    except Exception:
        check(False, "Could not determine disk space")

    # 1b. Memory headroom for encoding (warning only, not a hard fail)
    print("\n1b. Memory headroom for encoding")
    try:
        titles_probe = os.path.join(INDEX_DIR, "titles.json")
        if os.path.exists(titles_probe):
            with open(titles_probe) as f:
                n_titles = len(json.load(f))
            # Assume 384-dim float32 (matches all-MiniLM-L6-v2 default)
            embeddings_gb = n_titles * 384 * 4 / (1024**3)
            total_ram_gb = (os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")) / (1024**3)
            pct = embeddings_gb / total_ram_gb * 100
            if pct > 25:
                print(
                    f"  [WARN] Embeddings file will be ~{embeddings_gb:.1f} GB "
                    f"({pct:.0f}% of {total_ram_gb:.0f} GB RAM). "
                    f"Close memory-heavy apps (Chrome, Slack, etc.) before running "
                    f"or lower --batch-size to avoid swap thrashing."
                )
            else:
                check(
                    True,
                    f"Embeddings ~{embeddings_gb:.1f} GB on {total_ram_gb:.0f} GB RAM "
                    f"({pct:.0f}%)",
                )
        else:
            print("  [SKIP] titles.json not yet written")
    except Exception as e:
        print(f"  [SKIP] Memory check failed: {e}")

    # 2. Competing processes
    print("\n2. Competing ML processes")
    result = subprocess.run(["ps", "aux"], capture_output=True, text=True)
    ml_procs = [
        l
        for l in result.stdout.split("\n")
        if any(k in l for k in ["compute_bags", "build_index", "finetune", "eval_model", "demo.py"])
        and "preflight" not in l
        and "grep" not in l
    ]
    if ml_procs:
        for p in ml_procs:
            parts = p.split()
            pid, cmd = parts[1], " ".join(parts[10:])[:60]
            print(f"  [WARN] Running: PID {pid} — {cmd}")
        check(False, f"{len(ml_procs)} ML process(es) running — risk of OOM on 16GB")
    else:
        check(True, "No competing ML processes")

    # 3. Required files exist
    print("\n3. Required files")
    required = [
        ("combined_index/index.faiss", "FAISS index"),
        ("combined_index/titles.json", "Title list"),
        ("combined_index/tantivy_index", "Tantivy index dir"),
        ("models/esci-cross-encoder", "Cross-encoder model"),
    ]
    for path, desc in required:
        full = os.path.join(SCRIPT_DIR, path)
        exists = os.path.exists(full)
        size = ""
        if exists and os.path.isfile(full):
            size = f" ({os.path.getsize(full) / 1e9:.1f} GB)"
        check(exists, f"{desc}: {path}{size}")

    # 4. Load model and verify index consistency
    print("\n4. Index-model consistency")
    model = SentenceTransformer(args.model)

    titles_path = os.path.join(INDEX_DIR, "titles.json")
    with open(titles_path) as f:
        titles = json.load(f)
    check(len(titles) > 0, f"{len(titles):,} titles loaded")

    index_path = os.path.join(INDEX_DIR, "index.faiss")
    index = faiss.read_index(index_path)
    check(
        index.ntotal == len(titles),
        f"Index size ({index.ntotal:,}) matches titles ({len(titles):,})",
    )

    # Encode sample titles and compare to stored vectors
    sample_indices = [0, len(titles) // 2, len(titles) - 1]
    try:
        for idx in sample_indices:
            stored = index.reconstruct(idx)
            stored = stored / np.linalg.norm(stored)
            fresh = model.encode(titles[idx], normalize_embeddings=True)
            cos = float(stored @ fresh)
            check(
                cos > 0.95,
                f"Vector consistency [{idx}]: cosine={cos:.4f} ('{titles[idx][:50]}...')",
            )
    except RuntimeError:
        print("  [SKIP] Index does not support vector reconstruction")

    # 5. Tantivy index
    print("\n5. Tantivy index")
    import tantivy

    schema_builder = tantivy.SchemaBuilder()
    schema_builder.add_text_field("title", stored=True)
    schema = schema_builder.build()
    tv_index = tantivy.Index(schema, path=os.path.join(INDEX_DIR, "tantivy_index"))
    tv_index.reload()
    tv_searcher = tv_index.searcher()
    parsed = tv_index.parse_query("iphone AND case", ["title"])
    results = tv_searcher.search(parsed, limit=10)
    check(results.count >= 10, f"'iphone AND case' returned {results.count} results")

    # 6. Smoke test: CE-based bag computation
    if not args.quick:
        print("\n6. Smoke test (hybrid retrieval + CE scoring)")
        ce_path = os.path.join(SCRIPT_DIR, "models", "esci-cross-encoder")
        if not os.path.exists(ce_path):
            check(False, "Cross-encoder not found, skipping smoke test")
        else:
            from sentence_transformers import CrossEncoder

            ce_model = CrossEncoder(ce_path)

            index.hnsw.efSearch = 64
            for query in SMOKE_QUERIES:
                ql = query.lower()
                words = [w for w in re.findall(r"[a-z0-9]+", ql) if len(w) > 1]

                # Keyword retrieval
                seen = set()
                candidates = []
                if words:
                    try:
                        parsed = tv_index.parse_query(" AND ".join(words), ["title"])
                        results = tv_searcher.search(parsed, limit=200)
                        for _, addr in results.hits:
                            title = tv_searcher.doc(addr)["title"][0]
                            if title not in seen:
                                seen.add(title)
                                candidates.append(title)
                    except Exception:
                        pass

                # FAISS retrieval
                q_vec = model.encode(query, normalize_embeddings=True)
                q_vec = np.array(q_vec, dtype=np.float32).reshape(1, -1)
                faiss.normalize_L2(q_vec)
                D, I = index.search(q_vec, 200)
                for idx in I[0]:
                    if idx >= 0:
                        title = titles[idx]
                        if title not in seen:
                            seen.add(title)
                            candidates.append(title)

                # CE score and threshold
                if candidates:
                    pairs = [(query, t) for t in candidates]
                    scores = ce_model.predict(pairs)
                    passing_scores = [s for s in scores if s >= args.ce_threshold]
                    n_pass = len(passing_scores)
                    mean_ce = np.mean(passing_scores) if passing_scores else 0
                else:
                    n_pass = 0
                    mean_ce = 0

                check(
                    n_pass >= 3,
                    f"'{query}': {n_pass}/{len(candidates)} pass CE (mean={mean_ce:.3f})",
                )

            # Centroid consistency check
            print("\n7. Centroid-to-member consistency")
            test_query = "iphone case"
            ql = test_query.lower()
            words = [w for w in re.findall(r"[a-z0-9]+", ql) if len(w) > 1]
            parsed = tv_index.parse_query(" AND ".join(words), ["title"])
            results = tv_searcher.search(parsed, limit=50)
            members = [tv_searcher.doc(addr)["title"][0] for _, addr in results.hits][:20]

            if members:
                vecs = model.encode(members, normalize_embeddings=True)
                centroid = vecs.mean(axis=0)
                centroid = centroid / np.linalg.norm(centroid)
                cq = np.array(centroid, dtype=np.float32).reshape(1, -1)
                faiss.normalize_L2(cq)
                D, I = index.search(cq, 50)
                retrieved = set(titles[i] for i in I[0] if i >= 0)
                overlap = len(set(members) & retrieved)
                check(
                    overlap >= 3,
                    f"'{test_query}': {overlap}/{len(members)} bag members "
                    f"in top-50 centroid re-retrieval",
                )
            else:
                check(False, f"'{test_query}': no bag members found")

    # Summary
    print(f"\n{'=' * 40}")
    if FAIL:
        print("PREFLIGHT FAILED — fix issues before starting pipeline")
        sys.exit(1)
    else:
        print("ALL CHECKS PASSED — safe to proceed")
        sys.exit(0)


if __name__ == "__main__":
    main()
