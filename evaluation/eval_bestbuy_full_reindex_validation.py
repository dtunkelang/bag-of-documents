#!/usr/bin/env python3
"""Does catalog-field enrichment survive FULL-CORPUS retrieval? (the real test)

Context. `eval_bestbuy_llm_judge_junkrate.py` measured that 25.8% (base MiniLM)
/ 14.1% (BoD MiniLM) of the BestBuy demo's top-10 for 250 sampled holdout
queries are the *wrong product TYPE* ("apple tablet" -> keyboards). The demo's
index embeds nothing but the bare product name, discarding `manufacturer`,
`categoryPath` and `class`, which the raw Kaggle catalog XML carries at ~100%
population. `eval_bestbuy_category_enrichment_prototype.py` showed that
re-encoding from

    "{name} - {manufacturer} - {categoryPath leaf} - {class}"

cuts junk *within a closed reranking pool*. That test could not add or remove
anything from the top-10 -- R@10 could only stay flat or fall, and junk could
only improve by reordering products that were already there. It was a
necessary-not-sufficient signal.

This script removes that limitation. It rebuilds the ENTIRE 1,274,801-product
index from enriched text with both encoders, then runs genuine top-10 retrieval
over the full catalog under OLD (currently-shipped, name-only) and NEW
(enriched) vectors, for the same 250-query seed-0 holdout sample every other
eval this session used. Nothing is uploaded: all artifacts stay local.

Two text fields are built per product and kept deliberately distinct:

    display title  -> html.unescape(name) only. Goes in the new titles.json.
                      Fixes the shipped titles' raw `&#xAE;` / `&#x2122;`
                      entities without dumping taxonomy jargon into the Space's
                      result table, which renders titles[i] directly.
    embedding text -> enriched_text() from the prototype, verbatim, so what is
                      indexed here is exactly what was already validated.

SKU alignment. The raw archive has 1,275,077 unique SKUs; the shipped catalog
has 1,274,801. The new index is built over EXACTLY the shipped product_ids.json
set, in EXACTLY its order, so the only thing that changes between OLD and NEW
is the embedding text -- not catalog coverage, not row order. Any shipped SKU
missing from the raw XML falls back to its current titles.json name for both
display and embedding, and is counted in the results JSON.

Judge-label reuse. A (query, product) category judgment does not depend on
which vectors retrieved the product, so every pair already judged by the
junk-rate or BM25-gate runs reuses its cached label, and the judge prompt is
always built from the OLD shipped title -- the same string the cache was
produced from. That keeps the label a pure function of (query, product) and
keeps OLD-vs-NEW unconfounded by prompt text. Only genuinely new candidates
that NEW-vector retrieval surfaces cost money.

Phases (cached to --work-dir, resumable):

    --phase catalog   stream the raw product XML tarball, pull name /
                      manufacturer / categoryPath-leaf / class for all
                      1,274,801 shipped SKUs -> fields jsonl, new titles.json,
                      embedding texts
    --phase embed     re-encode all 1.27M embedding texts with base + BoD into
                      fp16 memmaps matching the shipped (N, 384) float16 shape
    --phase verify    shape / dtype / norm / order checks, OLD vs NEW
    --phase retrieve  encode the 250-query sample, exact top-10 over the FULL
                      catalog under OLD and NEW vectors, both models
    --phase estimate  project the cost of judging the NEW candidates -- free
    --phase judge     gpt-4o-mini category yes/no over only the unjudged pairs
    --phase eval      junk-rate@10 / R@10 / E@1, OLD vs NEW, both models,
                      paired bootstrap CIs -> results JSON

Usage:
    python evaluation/eval_bestbuy_full_reindex_validation.py --phase catalog
    python evaluation/eval_bestbuy_full_reindex_validation.py --phase embed
    python evaluation/eval_bestbuy_full_reindex_validation.py --phase verify
    python evaluation/eval_bestbuy_full_reindex_validation.py --phase retrieve
    python evaluation/eval_bestbuy_full_reindex_validation.py --phase estimate
    python evaluation/eval_bestbuy_full_reindex_validation.py --phase judge
    python evaluation/eval_bestbuy_full_reindex_validation.py --phase eval
"""

import argparse
import asyncio
import html
import json
import math
import os
import random
import sys
import tarfile
import time
import xml.etree.ElementTree as ET
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np  # noqa: E402
from dotenv import load_dotenv  # noqa: E402

from evaluation.eval_bestbuy_category_enrichment_prototype import (  # noqa: E402
    _leaf_category,
    enriched_text,
)
from evaluation.eval_bestbuy_llm_judge_junkrate import (  # noqa: E402
    CATEGORY_PROMPT,
    DATASET_FILES,
    DATASET_REPO,
    Usage,
    _bootstrap_ci,
    _chat,
    _load_judged,
    _score_from_logprobs,
    _topk_over_catalog,
    estimate_cost,
    make_client,
    pair_key,
    record_spend,
)
from evaluation.eval_bestbuy_llm_judge_rerank import (  # noqa: E402
    load_corpus,
    load_split,
    per_query_metrics,
)

load_dotenv(override=True)

K_EVAL = 10
EMBED_DIM = 384
CATALOG_SIZE = 1_274_801  # the shipped product_ids.json; asserted, not assumed

# Hard stop: refuse to start the judge phase if the projection exceeds this.
COST_CEILING_USD = 3.0

# Cached judge labels from the two prior runs on this same 250-query sample.
PRIOR_JUDGE_CACHES = (
    "/tmp/bestbuy_junkrate/junk_judge_bestbuy.jsonl",
    "/tmp/bestbuy_bm25_gate/gate_judge_bestbuy.jsonl",
)

DEFAULT_MANUAL_QUERIES = ("apple tablet", "nokia phone")

MODELS = ("base", "bod")
CONDITIONS = ("old", "new")


# --------------------------------------------------------------------------
# paths
# --------------------------------------------------------------------------
def work_paths(work_dir, tag):
    w = Path(work_dir)
    arts = w / "artifacts"
    w.mkdir(parents=True, exist_ok=True)
    arts.mkdir(parents=True, exist_ok=True)
    return {
        "root": w,
        "artifacts": arts,
        "fields": w / f"catalog_fields_{tag}.jsonl",
        "embed_texts": w / f"embed_texts_{tag}.json",
        "catalog_meta": w / f"catalog_meta_{tag}.json",
        "titles": arts / "titles.json",
        "product_ids": arts / "product_ids.json",
        "vecs": {
            "base": arts / "base_catalog.vecs.fp16.npy",
            "bod": arts / "bod_catalog.vecs.fp16.npy",
        },
        "embed_progress": {
            "base": w / f"embed_progress_base_{tag}.json",
            "bod": w / f"embed_progress_bod_{tag}.json",
        },
        "verify": w / f"verify_{tag}.json",
        "retrieval": w / f"retrieval_{tag}.json",
        "judge": w / f"reindex_judge_{tag}.jsonl",
        "judge_meta": w / f"reindex_judge_meta_{tag}.json",
    }


def resolve_data_dir(args):
    """Local --data-dir if given, else snapshot_download the shipped dataset."""
    if args.data_dir:
        return Path(args.data_dir).resolve()
    from huggingface_hub import snapshot_download

    print(f"snapshot_download from {DATASET_REPO} (cached after first run)...", flush=True)
    return Path(
        snapshot_download(
            repo_id=DATASET_REPO,
            repo_type="dataset",
            allow_patterns=DATASET_FILES,
        )
    ).resolve()


def pick_device():
    import torch

    if torch.backends.mps.is_available():
        return "mps"
    return "cuda" if torch.cuda.is_available() else "cpu"


# --------------------------------------------------------------------------
# phase: catalog
# --------------------------------------------------------------------------
def phase_catalog(args, paths):
    """Stream the raw XML tarball; build display titles + embedding texts.

    The tarball is read member-by-member and parsed in memory -- extracting it
    is ~7.7GB on disk. Everything is emitted in shipped product_ids.json order.
    """
    data, old_titles, pids = load_corpus(resolve_data_dir(args))
    if len(pids) != CATALOG_SIZE:
        raise SystemExit(
            f"expected {CATALOG_SIZE:,} shipped SKUs, product_ids.json has {len(pids)}"
        )
    if len(old_titles) != len(pids):
        raise SystemExit("titles.json / product_ids.json length mismatch in the shipped dataset")
    wanted = set(pids)
    if len(wanted) != len(pids):
        raise SystemExit("shipped product_ids.json contains duplicate SKUs")
    print(f"  {len(pids):,} shipped SKUs to look up in the raw XML", flush=True)

    tar_path = Path(args.tarball)
    if not tar_path.exists():
        raise SystemExit(f"raw catalog tarball not found: {tar_path}")

    found = {}
    t0 = time.time()
    n_files = 0
    n_products = 0
    n_skipped_extra = 0
    with tarfile.open(tar_path, "r|gz") as tf:
        for member in tf:
            if "/products/" not in member.name or not member.name.endswith(".xml"):
                continue
            fh = tf.extractfile(member)
            if fh is None:
                continue
            n_files += 1
            for _event, el in ET.iterparse(fh, events=("end",)):
                if el.tag != "product":
                    continue
                n_products += 1
                sku = (el.findtext("sku") or "").strip()
                if sku not in wanted:
                    # The raw archive carries ~276 SKUs the shipped catalog does
                    # not. Adding them would change coverage as well as text,
                    # which would make OLD-vs-NEW two changes instead of one.
                    n_skipped_extra += 1
                    el.clear()
                    continue
                if sku not in found:
                    found[sku] = (
                        html.unescape((el.findtext("name") or "").strip()),
                        html.unescape((el.findtext("manufacturer") or "").strip()),
                        _leaf_category(el),
                        html.unescape((el.findtext("class") or "").strip()),
                    )
                el.clear()
            if n_files % 25 == 0:
                print(
                    f"    [{n_files} files] {n_products:,} products scanned, "
                    f"{len(found):,}/{len(wanted):,} matched, {time.time() - t0:.0f}s",
                    flush=True,
                )
    scan_s = time.time() - t0
    print(
        f"  scan done: {n_files} files, {n_products:,} products, "
        f"{len(found):,} matched, {n_skipped_extra:,} raw-only SKUs skipped, {scan_s:.0f}s",
        flush=True,
    )

    n_fallback = 0
    fallback_examples = []
    new_titles = []
    texts = []
    with open(paths["fields"], "w") as ff:
        for i, sku in enumerate(pids):
            rec = found.get(sku)
            if rec is None:
                # Not in the raw XML (15-16 digit pseudo-SKUs are BestBuy
                # category landing pages, not products). Fall back to the
                # currently-shipped title for BOTH display and embedding, so
                # this row is byte-identical in intent to what ships today.
                n_fallback += 1
                if len(fallback_examples) < 20:
                    fallback_examples.append({"row": i, "sku": sku, "old_title": old_titles[i]})
                name = html.unescape(old_titles[i])
                fields = {"sku": sku, "name": name, "manufacturer": "", "class": ""}
                fields["category_leaf"] = ""
                fields["fallback"] = True
            else:
                name, manufacturer, leaf, cls = rec
                fields = {
                    "sku": sku,
                    "name": name,
                    "manufacturer": manufacturer,
                    "class": cls,
                    "category_leaf": leaf,
                    "fallback": False,
                }
            new_titles.append(name)
            texts.append(enriched_text(fields, fallback_title=html.unescape(old_titles[i])))
            ff.write(json.dumps(fields) + "\n")

    with open(paths["titles"], "w") as f:
        json.dump(new_titles, f)
    # product_ids.json is carried forward byte-for-byte: it is NOT regenerated.
    with open(data / "product_ids.json") as src, open(paths["product_ids"], "w") as dst:
        dst.write(src.read())
    with open(paths["embed_texts"], "w") as f:
        json.dump(texts, f)

    n_entity_fixed = sum(1 for i, t in enumerate(new_titles) if t != old_titles[i])
    meta = {
        "tarball": str(tar_path),
        "shipped_catalog_size": len(pids),
        "n_files_scanned": n_files,
        "n_products_in_raw_xml": n_products,
        "n_raw_only_skus_skipped": n_skipped_extra,
        "n_matched_in_raw_xml": len(found),
        "n_fallback_to_old_title": n_fallback,
        "raw_xml_coverage_pct": 100.0 * len(found) / len(pids),
        "n_titles_changed_by_html_unescape": n_entity_fixed,
        "scan_seconds": scan_s,
        "fallback_examples": fallback_examples,
        "example_embedding_texts": texts[:: max(len(texts) // 5, 1)][:5],
    }
    with open(paths["catalog_meta"], "w") as f:
        json.dump(meta, f, indent=2)
    print(json.dumps({k: v for k, v in meta.items() if k != "fallback_examples"}, indent=2))
    print(f"\nwrote {paths['titles']}, {paths['product_ids']}, {paths['embed_texts']}", flush=True)


# --------------------------------------------------------------------------
# phase: embed
# --------------------------------------------------------------------------
def phase_embed(args, paths):
    from sentence_transformers import SentenceTransformer

    with open(paths["embed_texts"]) as f:
        texts = json.load(f)
    n = len(texts)
    if n != CATALOG_SIZE:
        raise SystemExit(f"embed_texts has {n:,} rows, expected {CATALOG_SIZE:,}")
    device = pick_device()
    print(f"  {n:,} texts, device={device}, batch_size={args.batch_size}", flush=True)

    for key, model_id in (("base", args.base_model), ("bod", args.bod_model)):
        out_path = paths["vecs"][key]
        prog_path = paths["embed_progress"][key]
        done = 0
        if args.resume and out_path.exists() and prog_path.exists():
            with open(prog_path) as f:
                done = int(json.load(f).get("rows_done", 0))
            arr = np.lib.format.open_memmap(out_path, mode="r+")
            if arr.shape != (n, EMBED_DIM) or arr.dtype != np.float16:
                raise SystemExit(f"{out_path} has wrong shape/dtype {arr.shape}/{arr.dtype}")
        else:
            arr = np.lib.format.open_memmap(
                out_path, mode="w+", dtype=np.float16, shape=(n, EMBED_DIM)
            )
        if done >= n:
            print(f"  {key}: already complete ({done:,} rows)", flush=True)
            del arr
            continue
        print(f"  encoding {key} ({model_id}) from row {done:,}...", flush=True)

        m = SentenceTransformer(model_id, device=device)
        t0 = time.time()
        chunk = args.chunk
        while done < n:
            end = min(done + chunk, n)
            vecs = m.encode(
                texts[done:end],
                normalize_embeddings=True,
                batch_size=args.batch_size,
                show_progress_bar=False,
            )
            arr[done:end] = np.asarray(vecs, dtype=np.float32).astype(np.float16)
            arr.flush()
            done = end
            with open(prog_path, "w") as f:
                json.dump({"rows_done": done, "model": model_id}, f)
            el = time.time() - t0
            rate = done / max(el, 1e-9)
            print(
                f"    [{key} {done:,}/{n:,}] {rate:,.0f} items/s "
                f"eta {(n - done) / max(rate, 1e-9) / 60:.1f}m",
                flush=True,
            )
        del m, arr
        print(f"  {key} done in {(time.time() - t0) / 60:.1f}m -> {out_path}", flush=True)


# --------------------------------------------------------------------------
# phase: verify
# --------------------------------------------------------------------------
def phase_verify(args, paths):
    data, old_titles, pids = load_corpus(resolve_data_dir(args))
    with open(paths["product_ids"]) as f:
        new_pids = json.load(f)
    with open(paths["titles"]) as f:
        new_titles = json.load(f)

    checks = {
        "product_ids_identical_and_same_order": new_pids == pids,
        "titles_length_matches": len(new_titles) == len(pids),
    }
    per_model = {}
    for key, fname in (
        ("base", "base_catalog.vecs.fp16.npy"),
        ("bod", "bod_catalog.vecs.fp16.npy"),
    ):
        old = np.load(data / fname, mmap_mode="r")
        new = np.load(paths["vecs"][key], mmap_mode="r")
        idx = np.linspace(0, len(pids) - 1, 5000, dtype=np.int64)
        new_s = np.asarray(new[idx]).astype(np.float32)
        old_s = np.asarray(old[idx]).astype(np.float32)
        cos = float(np.mean(np.sum(new_s * old_s, axis=1)))
        per_model[key] = {
            "old_shape": list(old.shape),
            "new_shape": list(new.shape),
            "old_dtype": str(old.dtype),
            "new_dtype": str(new.dtype),
            "shape_matches": tuple(old.shape) == tuple(new.shape) == (len(pids), EMBED_DIM),
            "dtype_matches": old.dtype == new.dtype == np.float16,
            "new_mean_norm": float(np.mean(np.linalg.norm(new_s, axis=1))),
            "old_mean_norm": float(np.mean(np.linalg.norm(old_s, axis=1))),
            "new_rows_all_finite_sample": bool(np.all(np.isfinite(new_s))),
            "new_zero_rows_in_sample": int(np.sum(np.linalg.norm(new_s, axis=1) < 1e-3)),
            "mean_cosine_old_vs_new_sample": cos,
            "file_bytes": paths["vecs"][key].stat().st_size,
        }
        del old, new

    out = {"checks": checks, "per_model": per_model}
    with open(paths["verify"], "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))
    bad = [k for k, v in checks.items() if not v]
    bad += [
        f"{k}.{c}"
        for k, v in per_model.items()
        for c in ("shape_matches", "dtype_matches")
        if not v[c]
    ]
    if bad:
        raise SystemExit(f"VERIFY FAILED: {bad}")
    print("\nall structural checks passed", flush=True)


# --------------------------------------------------------------------------
# phase: retrieve  (genuine full-corpus top-10, not a closed pool)
# --------------------------------------------------------------------------
def _sample_rows(args, data, pids):
    """Reproduce the exact seed-0 250-query sample every eval this session used."""
    qrels, queries_all = load_split(data, args.queries_file, args.qrels_file)
    pid_set = set(pids)
    eval_qids = sorted(
        qid
        for qid, q in queries_all.items()
        if qid in qrels
        and any(g >= args.min_relevance and p in pid_set for p, g in qrels[qid].items())
    )
    rng = random.Random(args.seed)
    if args.sample and args.sample < len(eval_qids):
        sample_qids = sorted(rng.sample(eval_qids, args.sample))
        how = f"random.Random({args.seed}).sample over sorted eval-eligible qids"
    else:
        sample_qids = eval_qids
        how = "all eval-eligible queries"
    lowered = {queries_all[q].strip().lower() for q in sample_qids}
    manual = [q for q in args.manual_queries if q.strip().lower() not in lowered]
    rows = [{"key": qid, "query": queries_all[qid], "is_manual": False} for qid in sample_qids] + [
        {"key": f"manual:{q}", "query": q, "is_manual": True} for q in manual
    ]
    return rows, qrels, eval_qids, how


def phase_retrieve(args, paths):
    from sentence_transformers import SentenceTransformer

    data, old_titles, pids = load_corpus(resolve_data_dir(args))
    rows, _qrels, eval_qids, how = _sample_rows(args, data, pids)
    queries = [r["query"] for r in rows]
    print(f"  {len(rows)} queries ({len(eval_qids):,} eval-eligible; {how})", flush=True)

    device = pick_device()
    for key, model_id, old_vecs in (
        ("base", args.base_model, "base_catalog.vecs.fp16.npy"),
        ("bod", args.bod_model, "bod_catalog.vecs.fp16.npy"),
    ):
        print(f"  encoding queries with {model_id} on {device}...", flush=True)
        m = SentenceTransformer(model_id, device=device)
        qv = m.encode(
            queries, normalize_embeddings=True, batch_size=64, show_progress_bar=False
        ).astype(np.float32)
        del m
        # The encoder is unchanged between conditions -- only the doc side moves,
        # so the same query vectors are reused for OLD and NEW.
        for cond, vec_path in (("old", data / old_vecs), ("new", paths["vecs"][key])):
            print(f"    full-corpus top-{args.top_k}: {key}/{cond}", flush=True)
            idx, sims = _topk_over_catalog(qv, vec_path, args.top_k)
            for i, r in enumerate(rows):
                r[f"{key}_{cond}"] = [
                    {
                        "rank": rank + 1,
                        "row": int(j),
                        "product_id": pids[int(j)],
                        "title_old": old_titles[int(j)],
                        "sim": float(sims[i, rank]),
                    }
                    for rank, j in enumerate(idx[i])
                    if j >= 0
                ]

    payload = {
        "dataset_repo": DATASET_REPO,
        "data_dir": str(data),
        "new_artifacts_dir": str(paths["artifacts"]),
        "base_model": args.base_model,
        "bod_model": args.bod_model,
        "seed": args.seed,
        "sample_size": sum(1 for r in rows if not r["is_manual"]),
        "n_eval_eligible": len(eval_qids),
        "selection": how,
        "top_k": args.top_k,
        "catalog_size": len(pids),
        "retrieval": "exact cosine over the FULL catalog (no candidate pool)",
        "rows": rows,
    }
    with open(paths["retrieval"], "w") as f:
        json.dump(payload, f)

    pairs = {
        pair_key(r["query"], d["product_id"])
        for r in rows
        for k in MODELS
        for c in CONDITIONS
        for d in r[f"{k}_{c}"]
    }
    slots = sum(len(r[f"{k}_{c}"]) for r in rows for k in MODELS for c in CONDITIONS)
    print(
        f"saved retrieval -> {paths['retrieval']}  "
        f"({slots:,} slots -> {len(pairs):,} unique (query, product) pairs)",
        flush=True,
    )


# --------------------------------------------------------------------------
# phase: estimate / judge
# --------------------------------------------------------------------------
def load_prior_labels():
    """Cached labels from the junk-rate and BM25-gate runs, keyed (query, pid)."""
    out = {}
    for p in PRIOR_JUDGE_CACHES:
        out.update(_load_judged(p))
    return out


def _unique_pairs(payload, max_title_chars):
    """Deduped (query, product) -> prompt, built from the OLD shipped title.

    Using the old title is deliberate: it is the exact string the cached labels
    were produced from, so a label stays a pure function of (query, product)
    and OLD-vs-NEW cannot be confounded by the judge seeing different text.
    """
    out = {}
    for r in payload["rows"]:
        for k in MODELS:
            for c in CONDITIONS:
                for d in r[f"{k}_{c}"]:
                    key = pair_key(r["query"], d["product_id"])
                    if key not in out:
                        out[key] = {
                            "query": r["query"],
                            "product_id": d["product_id"],
                            "title": d["title_old"],
                            "prompt": CATEGORY_PROMPT.format(
                                query=r["query"], title=d["title_old"][:max_title_chars]
                            ),
                        }
    return out


def phase_estimate(args, paths, quiet=False):
    with open(paths["retrieval"]) as f:
        payload = json.load(f)
    pairs = _unique_pairs(payload, args.max_title_chars)
    have = set(load_prior_labels())
    if args.resume:
        have |= set(_load_judged(paths["judge"]))
    todo = [v for k, v in pairs.items() if k not in have]

    envelope = 8  # role/format wrapper the API adds
    tin = sum(len(p["prompt"]) / 4.0 + envelope for p in todo)
    tout = len(todo) * 1  # max_tokens=1
    cost = estimate_cost(args.model, tin, tout)
    breakdown = {
        "model": args.model,
        "n_queries": len(payload["rows"]),
        "n_unique_pairs": len(pairs),
        "n_reused_from_cache": len(pairs) - len(todo),
        "cache_reuse_pct": 100.0 * (len(pairs) - len(todo)) / max(len(pairs), 1),
        "n_new_pairs_to_judge": len(todo),
        "est_tokens_in": int(tin),
        "est_tokens_out": int(tout),
        "est_cost_usd_total": cost,
        "ceiling_usd": COST_CEILING_USD,
    }
    if not quiet:
        print(json.dumps(breakdown, indent=2), flush=True)
        print(
            f"\nPROJECTED COST: ${cost:.4f} ({len(todo):,} new calls) "
            f"vs ceiling ${COST_CEILING_USD:.2f}",
            flush=True,
        )
        if cost > COST_CEILING_USD:
            print("OVER CEILING -- the judge phase will refuse to run.", flush=True)
    return breakdown, todo


async def _run_judge(args, todo, usage, out_f):
    client = make_client()
    sem = asyncio.Semaphore(args.concurrency)

    async def one(p):
        ch = await _chat(client, sem, usage, args.model, p["prompt"], 1, logprobs=True)
        try:
            margin, p_yes = _score_from_logprobs(ch)
        except Exception:  # never let one odd token kill a paid run
            usage.errors += 1
            margin, p_yes = float("nan"), float("nan")
        return p, margin, p_yes

    t0 = time.time()
    done = 0
    for i in range(0, len(todo), 500):
        batch = todo[i : i + 500]
        for p, margin, p_yes in await asyncio.gather(*[one(x) for x in batch]):
            out_f.write(
                json.dumps(
                    {
                        "query": p["query"],
                        "product_id": p["product_id"],
                        "title": p["title"],
                        "margin": None if math.isnan(margin) else margin,
                        "p_yes": None if math.isnan(p_yes) else p_yes,
                    }
                )
                + "\n"
            )
        out_f.flush()
        done += len(batch)
        el = time.time() - t0
        print(
            f"  [judge {done}/{len(todo)}] {done / max(el, 1e-9):.1f} pairs/s "
            f"errors={usage.errors} spent=${estimate_cost(args.model, usage.tin, usage.tout):.4f}",
            flush=True,
        )


def phase_judge(args, paths):
    est, todo = phase_estimate(args, paths, quiet=True)
    c = est["est_cost_usd_total"]
    print(
        f"[cost guard] projected ${c:.4f} for {len(todo):,} new pairs "
        f"({est['cache_reuse_pct']:.1f}% reused) vs ceiling ${COST_CEILING_USD:.2f}",
        flush=True,
    )
    if c > COST_CEILING_USD:
        raise SystemExit(
            f"Refusing to run: projected ${c:.4f} exceeds ceiling ${COST_CEILING_USD:.2f}."
        )
    if not todo:
        print("  fully cached; nothing to do", flush=True)
        return

    usage = Usage()
    t0 = time.time()
    try:
        with open(paths["judge"], "a") as out_f:
            asyncio.run(_run_judge(args, todo, usage, out_f))
    finally:
        cost = estimate_cost(args.model, usage.tin, usage.tout)
        if usage.calls:
            record_spend(
                args.model, usage.tin, usage.tout, cost, "bestbuy full re-index validation"
            )

    cost = estimate_cost(args.model, usage.tin, usage.tout)
    prev = {}
    if Path(paths["judge_meta"]).exists():
        with open(paths["judge_meta"]) as f:
            prev = json.load(f)
    meta = {
        "judge_model": args.model,
        "prompt": CATEGORY_PROMPT,
        "prompt_title_source": "OLD shipped titles.json (matches the cached labels)",
        "n_pairs_judged_this_run": len(todo),
        "n_pairs_reused_from_cache": est["n_reused_from_cache"],
        "tokens_in": usage.tin,
        "tokens_out": usage.tout,
        "api_calls": usage.calls,
        "api_errors": usage.errors,
        "cost_usd": cost,
        "cost_usd_cumulative": round(cost + float(prev.get("cost_usd_cumulative", 0.0)), 6),
        "wall_clock_s": time.time() - t0,
    }
    with open(paths["judge_meta"], "w") as f:
        json.dump(meta, f, indent=2)
    print(
        f"\njudge done in {meta['wall_clock_s'] / 60:.1f}m calls={usage.calls:,} "
        f"errors={usage.errors} cost=${cost:.4f}",
        flush=True,
    )


# --------------------------------------------------------------------------
# phase: eval
# --------------------------------------------------------------------------
def phase_eval(args, paths):
    with open(paths["retrieval"]) as f:
        payload = json.load(f)
    data = Path(payload["data_dir"])
    qrels, _ = load_split(data, args.queries_file, args.qrels_file)
    judged = load_prior_labels()
    n_prior = len(judged)
    judged.update(_load_judged(paths["judge"]))
    print(
        f"  {n_prior:,} cached + {len(judged) - n_prior:,} new = {len(judged):,} labels", flush=True
    )

    with open(paths["catalog_meta"]) as f:
        catalog_meta = json.load(f)
    verify = {}
    if Path(paths["verify"]).exists():
        with open(paths["verify"]) as f:
            verify = json.load(f)

    thr = args.junk_threshold
    holdout = [r for r in payload["rows"] if not r["is_manual"]]
    manual = [r for r in payload["rows"] if r["is_manual"]]
    n_missing = 0

    def score_docs(r, arm):
        nonlocal n_missing
        out = []
        for d in r[arm]:
            j = judged.get(pair_key(r["query"], d["product_id"]))
            p_yes = j["p_yes"] if j and j.get("p_yes") is not None else None
            if p_yes is None:
                n_missing += 1
            out.append({**d, "p_yes": p_yes, "junk": (p_yes is not None and p_yes < thr)})
        return out

    arms = [f"{k}_{c}" for k in MODELS for c in CONDITIONS]
    scored = {arm: [score_docs(r, arm) for r in holdout] for arm in arms}
    if n_missing:
        print(f"  WARNING: {n_missing} (query, doc) slots have no judge score", flush=True)

    def nm(v):
        a = np.asarray(v, dtype=np.float64)
        a = a[~np.isnan(a)]
        return float(a.mean()) if a.size else float("nan")

    per_arm = {}
    raw = {}
    for arm in arms:
        junk, recalls, hits, e1s, ndcgs, pyes = [], [], [], [], [], []
        for r, docs in zip(holdout, scored[arm]):
            top = docs[:K_EVAL]
            ok = [d for d in top if d["p_yes"] is not None]
            junk.append(sum(1 for d in ok if d["junk"]) / len(ok) if ok else float("nan"))
            pyes.append(float(np.mean([d["p_yes"] for d in ok])) if ok else float("nan"))
            m = per_query_metrics(
                [d["product_id"] for d in top],
                qrels[r["key"]],
                k=K_EVAL,
                min_rel=args.min_relevance,
                exact_rel=args.exact_relevance,
            )
            recall, ndcg, e1, _ = m if m else (float("nan"),) * 4
            gold = {p for p, g in qrels[r["key"]].items() if g >= args.min_relevance}
            recalls.append(recall)
            ndcgs.append(ndcg)
            e1s.append(e1)
            hits.append(1.0 if any(d["product_id"] in gold for d in top) else 0.0)
        raw[arm] = {
            "junk": junk,
            "recall": recalls,
            "e1": e1s,
            "hit": hits,
            "ndcg": ndcgs,
        }
        ci = {
            f"{name}_ci95": list(_bootstrap_ci(vals, args.n_boot, args.seed))
            for name, vals in (
                ("junk_rate_at_10", junk),
                ("recall_at_10_fraction_recovered", recalls),
                ("e_at_1", e1s),
            )
        }
        per_arm[arm] = {
            "junk_rate_at_10": nm(junk),
            "mean_p_yes": nm(pyes),
            "recall_at_10_fraction_recovered": nm(recalls),
            "hit_rate_at_10": nm(hits),
            "ndcg_at_10": nm(ndcgs),
            "e_at_1": nm(e1s),
            "queries_with_zero_junk": float(np.mean([j == 0.0 for j in junk])),
            "queries_majority_junk": float(np.mean([j > 0.5 for j in junk])),
            **ci,
        }

    def paired(a, b, field):
        d = [x - y for x, y in zip(raw[a][field], raw[b][field])]
        lo, hi = _bootstrap_ci(d, args.n_boot, args.seed)
        return {"delta": float(np.nanmean(d)), "ci95": [lo, hi]}

    deltas = {}
    for k in MODELS:  # the headline: NEW minus OLD, same model, paired by query
        deltas[f"{k}_new_minus_old"] = {
            "junk_rate_at_10": paired(f"{k}_new", f"{k}_old", "junk"),
            "recall_at_10_fraction_recovered": paired(f"{k}_new", f"{k}_old", "recall"),
            "e_at_1": paired(f"{k}_new", f"{k}_old", "e1"),
            "hit_rate_at_10": paired(f"{k}_new", f"{k}_old", "hit"),
            "ndcg_at_10": paired(f"{k}_new", f"{k}_old", "ndcg"),
        }
    for c in CONDITIONS:  # secondary: BoD vs base within a condition
        deltas[f"bod_minus_base_{c}"] = {
            "junk_rate_at_10": paired(f"bod_{c}", f"base_{c}", "junk"),
            "recall_at_10_fraction_recovered": paired(f"bod_{c}", f"base_{c}", "recall"),
            "e_at_1": paired(f"bod_{c}", f"base_{c}", "e1"),
        }

    # How much of the top-10 actually moved? A near-zero churn would mean the
    # enrichment barely touched retrieval and the metrics are trivially flat.
    churn = {}
    for k in MODELS:
        overlaps, top1 = [], []
        for r in holdout:
            o = {d["product_id"] for d in r[f"{k}_old"][:K_EVAL]}
            nw = {d["product_id"] for d in r[f"{k}_new"][:K_EVAL]}
            overlaps.append(len(o & nw) / max(len(o), 1))
            top1.append(
                1.0
                if r[f"{k}_old"]
                and r[f"{k}_new"]
                and r[f"{k}_old"][0]["product_id"] == r[f"{k}_new"][0]["product_id"]
                else 0.0
            )
        churn[k] = {
            "mean_top10_overlap_old_new": float(np.mean(overlaps)),
            "top1_unchanged_rate": float(np.mean(top1)),
        }

    def example(r, arm, docs=None):
        docs = docs if docs is not None else score_docs(r, arm)
        ok = [d for d in docs if d["p_yes"] is not None]
        return {
            "query": r["query"],
            "junk_rate": sum(1 for d in ok if d["junk"]) / max(len(ok), 1),
            "top_10": [
                {
                    "rank": d["rank"],
                    "title_old": d["title_old"],
                    "p_yes": d["p_yes"],
                    "junk": d["junk"],
                }
                for d in docs[:K_EVAL]
            ],
        }

    # Biggest junk wins from the re-index, per model.
    biggest = {}
    for k in MODELS:
        gains = sorted(
            range(len(holdout)),
            key=lambda i: raw[f"{k}_new"]["junk"][i] - raw[f"{k}_old"]["junk"][i],
        )
        biggest[k] = [
            {
                "query": holdout[i]["query"],
                "junk_old": raw[f"{k}_old"]["junk"][i],
                "junk_new": raw[f"{k}_new"]["junk"][i],
                "old": example(holdout[i], f"{k}_old", scored[f"{k}_old"][i]),
                "new": example(holdout[i], f"{k}_new", scored[f"{k}_new"][i]),
            }
            for i in gains[: args.n_examples]
        ]

    manual_checks = {
        r["query"]: {arm: example(r, arm) for arm in arms}
        for r in manual
        + [
            r
            for r in holdout
            if r["query"].strip().lower() in {q.lower() for q in args.manual_queries}
        ]
    }

    judge_meta = {}
    if Path(paths["judge_meta"]).exists():
        with open(paths["judge_meta"]) as f:
            judge_meta = json.load(f)

    print(f"\n=== FULL-CORPUS retrieval, {len(holdout)} holdout queries, k={K_EVAL} ===")
    print(f"{'arm':<10} {'junk@10':>9} {'R@10':>8} {'E@1':>8} {'hit@10':>8} {'nDCG@10':>9}")
    for arm in arms:
        s = per_arm[arm]
        print(
            f"{arm:<10} {s['junk_rate_at_10']:>9.4f} "
            f"{s['recall_at_10_fraction_recovered']:>8.4f} {s['e_at_1']:>8.4f} "
            f"{s['hit_rate_at_10']:>8.4f} {s['ndcg_at_10']:>9.4f}"
        )
    print("\nΔ NEW - OLD (paired, CI95):")
    for k in MODELS:
        d = deltas[f"{k}_new_minus_old"]
        for metric in ("junk_rate_at_10", "recall_at_10_fraction_recovered", "e_at_1"):
            v = d[metric]
            print(
                f"  {k:<5} {metric:<34} {v['delta']:+.4f} "
                f"[{v['ci95'][0]:+.4f}, {v['ci95'][1]:+.4f}]"
            )
    print(f"\ntop-10 churn: {json.dumps(churn)}")

    out = {
        "experiment": (
            "BestBuy full 1.27M re-index from enriched catalog text: OLD (name-only) vs "
            "NEW (name + manufacturer + categoryPath leaf + class) under genuine "
            "full-corpus top-10 retrieval"
        ),
        "question": (
            "The closed-pool prototype could only reorder an already-retrieved top-10. "
            "Does re-embedding the entire catalog from BestBuy's own taxonomy actually "
            "lower the top-10 category-junk rate in real retrieval, and at what cost to "
            "R@10/E@1?"
        ),
        "config": {
            "dataset_repo": DATASET_REPO,
            "base_model": payload["base_model"],
            "bod_model": payload["bod_model"],
            "catalog_size": payload["catalog_size"],
            "sample_size": len(holdout),
            "seed": payload["seed"],
            "selection": payload["selection"],
            "top_k": payload["top_k"],
            "k_eval": K_EVAL,
            "retrieval": payload["retrieval"],
            "enrichment_format": "enriched_text() from eval_bestbuy_category_enrichment_prototype",
            "display_title_format": "html.unescape(name) only -- no taxonomy jargon",
            "judge_model": args.model,
            "judge_prompt": CATEGORY_PROMPT,
            "judge_title_source": "OLD shipped titles.json for every pair (cache-compatible)",
            "junk_threshold_p_yes": thr,
            "n_boot": args.n_boot,
            "new_artifacts_dir": payload["new_artifacts_dir"],
            "published": False,
        },
        "catalog_build": {k: v for k, v in catalog_meta.items() if k != "fallback_examples"},
        "fallback_examples": catalog_meta.get("fallback_examples", []),
        "artifact_verification": verify,
        "judge_run": judge_meta,
        "summary": per_arm,
        "deltas": deltas,
        "top10_churn": churn,
        "n_unjudged_slots": n_missing,
        "biggest_junk_improvements": biggest,
        "manual_checks": manual_checks,
        "per_query": {
            arm: {
                str(holdout[i]["key"]): {
                    "query": holdout[i]["query"],
                    "junk_rate": raw[arm]["junk"][i],
                    "recall_at_10": raw[arm]["recall"][i],
                    "e_at_1": raw[arm]["e1"][i],
                }
                for i in range(len(holdout))
            }
            for arm in arms
        },
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {out_path}", flush=True)


# --------------------------------------------------------------------------
def main():
    global COST_CEILING_USD
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--phase",
        required=True,
        choices=["catalog", "embed", "verify", "retrieve", "estimate", "judge", "eval"],
    )
    ap.add_argument("--data-dir", default=None, help="local artifact dir; default = HF snapshot")
    ap.add_argument("--tarball", default="/tmp/bestbuy_raw_inspect/product_data.tar.gz")
    ap.add_argument("--queries-file", default="holdout_queries.jsonl")
    ap.add_argument("--qrels-file", default="holdout_qrels.jsonl")
    ap.add_argument("--base-model", default="all-MiniLM-L6-v2")
    ap.add_argument("--bod-model", default="dtunkelang/bag-of-documents-bestbuy-minilm")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--chunk", type=int, default=100_000, help="rows per memmap flush/resume point")
    ap.add_argument("--model", default="gpt-4o-mini", help="OpenAI judge model")
    ap.add_argument("--concurrency", type=int, default=32)
    ap.add_argument("--cost-ceiling", type=float, default=COST_CEILING_USD)
    ap.add_argument("--sample", type=int, default=250)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--top-k", type=int, default=K_EVAL)
    ap.add_argument("--max-title-chars", type=int, default=300)
    ap.add_argument("--min-relevance", type=int, default=1)
    ap.add_argument("--exact-relevance", type=int, default=1)
    ap.add_argument("--junk-threshold", type=float, default=0.5)
    ap.add_argument("--n-examples", type=int, default=8)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--manual-queries", nargs="*", default=list(DEFAULT_MANUAL_QUERIES))
    ap.add_argument("--resume", action="store_true", default=True)
    ap.add_argument("--no-resume", dest="resume", action="store_false")
    ap.add_argument("--work-dir", default="/tmp/bestbuy_reindex_output")
    ap.add_argument("--tag", default="bestbuy")
    ap.add_argument("--out", default="evaluation/results/bestbuy_full_reindex_validation.json")
    args = ap.parse_args()

    COST_CEILING_USD = args.cost_ceiling
    paths = work_paths(args.work_dir, args.tag)
    {
        "catalog": lambda: phase_catalog(args, paths),
        "embed": lambda: phase_embed(args, paths),
        "verify": lambda: phase_verify(args, paths),
        "retrieve": lambda: phase_retrieve(args, paths),
        "estimate": lambda: phase_estimate(args, paths),
        "judge": lambda: phase_judge(args, paths),
        "eval": lambda: phase_eval(args, paths),
    }[args.phase]()


if __name__ == "__main__":
    main()
