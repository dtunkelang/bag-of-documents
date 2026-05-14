#!/usr/bin/env python3
"""Doc2Query evaluation — generate K synthetic queries per doc via local LLM,
append to doc text, re-encode catalog, evaluate retrieval lift.

Doc-side dual to HyDE: HyDE puts LLM on the query side (generate doc -> encode);
Doc2Query puts LLM on the doc side (generate queries -> augment doc text).

Tiered modes (run cheapest first; abort early if no signal):
  --sample-n N        Tier 1 — generate for N random gold docs, print outputs
                      alongside the actual gold queries from qrels. Eyeball:
                      do the LLM's queries look anything like real ones? No
                      retrieval scoring done. Cost: N * ~3 sec.
  --oracle-only       Tier 2 — only expand docs that appear in qrels. This is
                      the BEST CASE for Doc2Query (every expanded doc is one
                      we want to retrieve). If even this oracle lifts R@k by
                      < 1pp, the full version has no chance.
  (default)           Tier 3 — expand entire catalog.

Generated queries cached to <data_dir>/doc2query_<llm>_k<K>.jsonl keyed by
doc_id so failed runs resume.

Usage:
    # Tier 1 — eyeball gate
    python evaluation/eval_doc2query.py \\
        --catalog scifact_data/titles.json \\
        --product-ids scifact_data/doc_ids.json \\
        --qrels scifact_data/test_qrels.jsonl \\
        --queries scifact_data/test_queries.jsonl \\
        --base-model all-MiniLM-L6-v2 \\
        --base-vecs scifact_data/base_catalog.vecs.fp16.npy \\
        --label scifact-d2q --sample-n 20

    # Tier 2 — oracle upper bound
    python evaluation/eval_doc2query.py ... --label scifact-d2q --oracle-only

    # Tier 3 — full corpus
    python evaluation/eval_doc2query.py ... --label scifact-d2q
"""

import argparse
import json
import os
import random
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from sentence_transformers import SentenceTransformer  # noqa: E402

try:
    import requests  # noqa: E402
except ImportError:
    print("ERROR: `requests` not installed. `pip install requests`.", file=sys.stderr)
    sys.exit(1)

OLLAMA_DEFAULT_URL = "http://localhost:11434/api/generate"
DOC2QUERY_PROMPT = (
    "Generate {k} distinct, realistic search queries a user might type to find "
    "the passage below. Output ONLY the queries, one per line. No preamble. "
    "No commentary. No markdown. No numbering. No quotes.\n\n"
    "Passage: {passage}\n\nQueries:"
)

_PREAMBLE_PATTERNS = (
    "here are",
    "here is",
    "below are",
    "search quer",
    "i'll provide",
    "i will provide",
    "i can generate",
    "the following",
    "this query",
    "this passage",
)


def parse_queries(text, k):
    """Parse query list from LLM output. Strips numbering/bullets/quotes,
    drops preamble / meta-commentary lines, rejects too-short / too-long."""
    out = []
    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue
        line = re.sub(r"^\d+[\.\)\-:]\s*", "", line)
        line = re.sub(r"^[-*`]+\s*", "", line)
        line = line.strip("`*\"' ")
        if not line:
            continue
        low = line.lower()
        if any(p in low for p in _PREAMBLE_PATTERNS):
            continue
        if line.endswith(":"):
            continue
        if len(line) < 10 or len(line) > 200:
            continue
        out.append(line)
    return out[:k]


def generate_queries(session, url, model, passage, k, max_tokens=240, temperature=0.7):
    payload = {
        "model": model,
        "prompt": DOC2QUERY_PROMPT.format(k=k, passage=passage[:1500]),
        "stream": False,
        "options": {"temperature": temperature, "num_predict": max_tokens},
    }
    r = session.post(url, json=payload, timeout=180)
    r.raise_for_status()
    raw = r.json().get("response", "").strip()
    return parse_queries(raw, k)


def load_or_generate(args, doc_ids_to_expand, titles, pid_to_idx):
    """Generate K queries per doc for the given doc_id subset. Cache as
    JSONL keyed by doc_id."""
    cache_safe = args.llm_model.replace(":", "_").replace("/", "_")
    cache_path = os.path.join(
        os.path.dirname(args.queries) or ".",
        f"doc2query_{cache_safe}_k{args.k_queries}.jsonl",
    )
    cached = {}
    if os.path.exists(cache_path):
        with open(cache_path) as f:
            for line in f:
                d = json.loads(line)
                cached[d["doc_id"]] = d["queries"]
        print(f"loaded {len(cached):,} cached doc expansions from {cache_path}", flush=True)

    needed = [d for d in doc_ids_to_expand if d not in cached or len(cached[d]) == 0]
    print(
        f"  to generate: {len(needed):,}  (already cached: {len(doc_ids_to_expand) - len(needed):,})",
        flush=True,
    )

    if not needed:
        return cached, cache_path

    session = requests.Session()
    t0 = time.time()
    with open(cache_path, "a") as cache_handle:
        for n_done, doc_id in enumerate(needed, 1):
            idx = pid_to_idx[doc_id]
            passage = titles[idx]
            try:
                queries = generate_queries(
                    session,
                    args.ollama_url,
                    args.llm_model,
                    passage,
                    k=args.k_queries,
                    temperature=args.temperature,
                )
            except Exception as e:
                print(f"  [{doc_id}] LLM error: {e}", flush=True)
                queries = []
            cached[doc_id] = queries
            cache_handle.write(json.dumps({"doc_id": doc_id, "queries": queries}) + "\n")
            cache_handle.flush()
            if n_done % 25 == 0 or n_done == len(needed):
                elapsed = time.time() - t0
                rate = n_done / elapsed if elapsed > 0 else 0.0
                avg_n = np.mean([len(q) for q in cached.values()]) if cached else 0.0
                eta_sec = (len(needed) - n_done) / rate if rate > 0 else 0
                print(
                    f"  generated {n_done:,}/{len(needed):,}  "
                    f"({rate:.2f} docs/s; avg queries/doc={avg_n:.1f}; "
                    f"ETA {eta_sec / 3600:.1f}h)",
                    flush=True,
                )
    return cached, cache_path


def score_buckets(per_q, k_label, label):
    n = len(per_q)
    base_r = np.mean([h / g for _, g, h, _ in per_q])
    d2q_r = np.mean([h / g for _, g, _, h in per_q])
    print(f"\nDoc2Query vs base — {label}  (R@{k_label})")
    print("  bucket                              n     base      D2Q        Δ")
    buckets = [
        ("0.0 (base misses entirely)", lambda b, g: b == 0),
        ("0.0-0.5", lambda b, g: 0 < b / g <= 0.5),
        ("0.5-1.0", lambda b, g: 0.5 < b / g < 1.0),
        ("1.0 (base perfect)", lambda b, g: b == g),
    ]
    for blabel, pred in buckets:
        rows = [(g, b, h) for _, g, b, h in per_q if pred(b, g)]
        if not rows:
            continue
        bn = len(rows)
        br = np.mean([b / g for g, b, _ in rows])
        dr = np.mean([h / g for g, _, h in rows])
        pct = bn / n * 100
        print(f"  {blabel:<30}  {bn:>5} ({pct:>4.1f}%)  {br:>7.3f}  {dr:>7.3f}  {dr - br:+7.3f}")
    print(f"  overall (n={n:,}): base={base_r:.3f}  D2Q={d2q_r:.3f}  Δ={d2q_r - base_r:+.3f}")
    return base_r, d2q_r


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--catalog", required=True)
    ap.add_argument("--product-ids", required=True)
    ap.add_argument("--qrels", required=True)
    ap.add_argument("--queries", required=True)
    ap.add_argument("--min-relevance", type=int, default=1)
    ap.add_argument("--base-model", required=True)
    ap.add_argument("--base-vecs", default=None)
    ap.add_argument("--llm-model", default="llama3.1:8b-instruct-q4_K_M")
    ap.add_argument("--ollama-url", default=OLLAMA_DEFAULT_URL)
    ap.add_argument("--label", default="corpus")
    ap.add_argument("--k", type=int, default=10, help="R@k metric")
    ap.add_argument("--k-queries", type=int, default=5, help="queries to generate per doc")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument(
        "--oracle-only",
        action="store_true",
        help="Only expand docs in qrels — best-case upper bound on Doc2Query lift.",
    )
    ap.add_argument(
        "--sample-n",
        type=int,
        default=0,
        help="Tier 1: generate for N random gold docs and print outputs (no scoring).",
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--augmentation-mode",
        choices=["append", "prepend", "vec_avg"],
        default="vec_avg",
        help=(
            "How to merge generated queries into the doc rep. "
            "append: queries after passage text (loses queries when passage > "
            "encoder context). prepend: queries before passage (loses end of "
            "passage). vec_avg: encode queries separately, average with "
            "passage vec — preserves both."
        ),
    )
    args = ap.parse_args()

    random.seed(args.seed)

    print("loading data...", flush=True)
    with open(args.catalog) as f:
        titles = json.load(f)
    with open(args.product_ids) as f:
        pids = json.load(f)
    pid_to_idx = {p: i for i, p in enumerate(pids)}

    queries_by_qid = {}
    with open(args.queries) as f:
        for line in f:
            d = json.loads(line)
            queries_by_qid[d["query_id"]] = d["query"]

    pos = defaultdict(set)
    gold_doc_ids = set()
    qid_to_gold_docs = defaultdict(set)
    qrel_field = None
    with open(args.qrels) as f:
        for line in f:
            r = json.loads(line)
            if qrel_field is None:
                qrel_field = "product_id" if "product_id" in r else "doc_id"
            doc_id = r[qrel_field]
            if doc_id not in pid_to_idx:
                continue
            if r["relevance"] < args.min_relevance:
                continue
            pos[r["query_id"]].add(pid_to_idx[doc_id])
            gold_doc_ids.add(doc_id)
            qid_to_gold_docs[r["query_id"]].add(doc_id)

    print(
        f"  catalog={len(pids):,}  queries={len(queries_by_qid):,}  "
        f"gold-docs={len(gold_doc_ids):,}",
        flush=True,
    )

    if args.sample_n > 0:
        candidates = sorted(gold_doc_ids)
        random.shuffle(candidates)
        doc_ids_to_expand = candidates[: args.sample_n]
        print(f"\nTier 1 — generating for {len(doc_ids_to_expand)} random gold docs.", flush=True)
    elif args.oracle_only:
        doc_ids_to_expand = sorted(gold_doc_ids)
        print(
            f"\nTier 2 (oracle) — expanding {len(doc_ids_to_expand):,} gold docs only.", flush=True
        )
    else:
        doc_ids_to_expand = list(pids)
        print(f"\nTier 3 (full) — expanding all {len(doc_ids_to_expand):,} docs.", flush=True)

    print(
        f"\ngenerating Doc2Query expansions via {args.llm_model} "
        f"(K={args.k_queries}, T={args.temperature})...",
        flush=True,
    )
    cached, cache_path = load_or_generate(args, doc_ids_to_expand, titles, pid_to_idx)
    print(f"  cache at {cache_path}", flush=True)

    if args.sample_n > 0:
        print(f"\n=== Tier 1 eyeball: {len(doc_ids_to_expand)} gold docs ===\n")
        for doc_id in doc_ids_to_expand:
            idx = pid_to_idx[doc_id]
            passage = titles[idx][:400].replace("\n", " ")
            gold_qids_here = [qid for qid, gd in qid_to_gold_docs.items() if doc_id in gd]
            gold_text = [queries_by_qid[q] for q in gold_qids_here if q in queries_by_qid]
            print(f"--- doc_id={doc_id} ---")
            print(f"PASSAGE: {passage}")
            print(f"GOLD QUERIES ({len(gold_text)}):")
            for gq in gold_text[:5]:
                print(f"  * {gq}")
            print(f"GENERATED ({len(cached.get(doc_id, []))}):")
            for q in cached.get(doc_id, []):
                print(f"  * {q}")
            print()
        return

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    if args.base_vecs and os.path.exists(args.base_vecs):
        print(f"\nloading cached base vecs {args.base_vecs}...", flush=True)
        base_pv = np.load(args.base_vecs).astype(np.float32)
    else:
        print(f"\nencoding catalog with {args.base_model} on {device}...", flush=True)
        m = SentenceTransformer(args.base_model, device=device)
        base_pv = m.encode(
            titles, normalize_embeddings=True, batch_size=128, show_progress_bar=True
        ).astype(np.float32)
        del m
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

    # Restrict augmentation to docs the user asked us to expand. The cache
    # may contain leftover expansions from a prior full-corpus run; in
    # oracle / sample modes we must ignore those.
    requested_set = set(doc_ids_to_expand)
    expand_doc_ids = [
        d
        for d in doc_ids_to_expand
        if d in requested_set and d in cached and cached[d] and d in pid_to_idx
    ]
    expand_indices = [pid_to_idx[d] for d in expand_doc_ids]
    print(
        f"\naugmentation-mode={args.augmentation_mode}; "
        f"re-encoding {len(expand_doc_ids):,} docs "
        f"({100 * len(expand_doc_ids) / len(pids):.1f}% of catalog) on {device}...",
        flush=True,
    )
    enc = SentenceTransformer(args.base_model, device=device)

    if args.augmentation_mode == "vec_avg":
        # Encode each generated query separately and average with the doc vec.
        # Preserves the doc's text fully; queries enter the rep via vector
        # space rather than being truncated out of the encoder context.
        flat_queries = []
        per_doc_counts = []
        for doc_id in expand_doc_ids:
            qs = cached[doc_id]
            per_doc_counts.append(len(qs))
            flat_queries.extend(qs)
        print(f"  encoding {len(flat_queries):,} generated queries individually...", flush=True)
        q_vecs = enc.encode(
            flat_queries,
            normalize_embeddings=True,
            batch_size=256,
            show_progress_bar=True,
        ).astype(np.float32)
        # Average per-doc query vectors with the existing passage vector
        aug_vecs = np.zeros((len(expand_doc_ids), q_vecs.shape[1]), dtype=np.float32)
        cur = 0
        for i, n_q in enumerate(per_doc_counts):
            doc_idx = expand_indices[i]
            doc_vec = base_pv[doc_idx]
            mean_q = q_vecs[cur : cur + n_q].mean(axis=0)
            combined = doc_vec + mean_q  # equal weight to passage and query-mean
            norm = np.linalg.norm(combined)
            aug_vecs[i] = combined / norm if norm > 1e-12 else doc_vec
            cur += n_q
    else:
        augmented_texts = []
        for doc_id in expand_doc_ids:
            qs = cached[doc_id]
            doc_idx = pid_to_idx[doc_id]
            joined = " ".join(qs)
            if args.augmentation_mode == "prepend":
                augmented_texts.append(joined + " " + titles[doc_idx])
            else:  # append
                augmented_texts.append(titles[doc_idx] + " " + joined)
        aug_vecs = enc.encode(
            augmented_texts,
            normalize_embeddings=True,
            batch_size=128,
            show_progress_bar=True,
        ).astype(np.float32)

    del enc
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

    d2q_pv = base_pv.copy()
    for i, doc_idx in enumerate(expand_indices):
        d2q_pv[doc_idx] = aug_vecs[i]

    print("\nencoding base queries...", flush=True)
    qenc = SentenceTransformer(args.base_model, device=device)
    qids = sorted(queries_by_qid)
    queries_text = [queries_by_qid[q] for q in qids]
    base_qv = qenc.encode(
        queries_text,
        normalize_embeddings=True,
        batch_size=256,
        show_progress_bar=True,
    ).astype(np.float32)
    del qenc

    print("\nretrieving + scoring...", flush=True)
    sim_base = base_qv @ base_pv.T
    sim_d2q = base_qv @ d2q_pv.T
    base_top = np.argpartition(-sim_base, args.k, axis=1)[:, : args.k]
    d2q_top = np.argpartition(-sim_d2q, args.k, axis=1)[:, : args.k]

    per_q = []
    for j, qid in enumerate(qids):
        g = pos.get(qid, set())
        if not g:
            continue
        b_hit = len({int(x) for x in base_top[j]} & g)
        d_hit = len({int(x) for x in d2q_top[j]} & g)
        per_q.append((qid, len(g), b_hit, d_hit))

    score_buckets(per_q, args.k, args.label)

    out_path = os.path.join(
        os.path.dirname(args.queries) or ".",
        f"doc2query_per_query_{args.label}.jsonl",
    )
    with open(out_path, "w") as f:
        for qid, g, b, d in per_q:
            f.write(json.dumps({"query_id": qid, "n_gold": g, "base_hit": b, "d2q_hit": d}) + "\n")
    print(f"\nper-query results at {out_path}", flush=True)


if __name__ == "__main__":
    main()
