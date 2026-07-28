#!/usr/bin/env python3
"""Can BestBuy's own catalog taxonomy fix the top-10 category junk? (prototype)

Context. `eval_bestbuy_llm_judge_junkrate.py` measured that 25.8% (base MiniLM)
/ 14.1% (BoD MiniLM) of the top-10 for 250 sampled BestBuy holdout queries are
the *wrong product TYPE* -- "apple tablet" returns keyboards.
`eval_bestbuy_bm25_junk_gate.py` tried a BM25 leg as the fix and it was a clean
negative (junk got worse, R@10 flat-to-down).

New information: the raw Kaggle catalog XML carries ~90 fields per product, not
just sku+name. `class` (224 distinct), `subclass` (1407), `categoryPath` (a
nested id+name chain) and `manufacturer` are ~100% populated. These are
*ground-truth* labels on the product side -- no classifier, no misclassification
risk. The iPad is `class=TABLET`, the Apple Wireless Keyboard is
`class=INPUT DEVICES`, the Rocketfish iPad keyboard capsule is
`class=TABLET ACCESSORIES`. So the junk in the "apple tablet" top-10 is, in
principle, separable with fields the catalog already has.

Two mechanisms are prototyped here, both WITHOUT retraining and WITHOUT
re-indexing the full 1.27M-product catalog:

    (a) enriched embedding text -- same encoders, richer input:
            "{name} - {manufacturer} - {categoryPath leaf} - {class}"
    (b) explicit category restriction -- an LLM maps the query to 1-3
        compatible `class` values (from the observed vocabulary, not free
        text), and pool candidates in that set are boosted above those that
        are not (stable, order-preserving; a boost, not a hard filter)

*** LIMITATION, STATED UP FRONT ***
This is a CLOSED-POOL RERANKING test. For each query the candidate pool is the
union of the products already in that query's base top-10 and BoD top-10 from
the cached run. Nothing new can be retrieved: R@10 can only stay flat or fall,
and junk-rate can only improve by demoting junk that was already in the top-10
in favour of non-junk that was also already there. A positive result here is a
NECESSARY-not-sufficient condition for a full re-index to help; a negative
result here is close to decisive against it.

Everything reuses the cached judge labels from the two prior runs (same
(query, product) pairs -- same product, same query, only the ORDER changes, so
the category labels are still valid). The only new OpenAI spend is 250 short
query-classification calls (~$0.05).

Phases (cached to --work-dir, resumable):

    --phase catalog   stream the raw product XML tarball, pull name /
                      manufacturer / class / subclass / categoryPath-leaf for
                      just the products in the judged pools (+ all qrels golds)
    --phase embed     re-encode that subset with base + BoD from ENRICHED text;
                      also encode the sampled queries; pull the ORIGINAL
                      catalog vectors for the same products (the baseline)
    --phase estimate  project the query-classification cost -- spends nothing
    --phase classify  gpt-4o-mini: query -> 1-3 compatible `class` values
    --phase eval      4 conditions x junk-rate/R@10/E@1 x {base, BoD}, paired
                      bootstrap CIs, query-classification recall risk, plus
                      three structural diagnostics on the embedding space
                      itself: per-class centroid coherence/separation,
                      nearest-centroid query classification (is a free
                      classifier already available in today's space?), and
                      product-level k-NN category purity -> results JSON

Usage:
    python evaluation/eval_bestbuy_category_enrichment_prototype.py --phase catalog
    python evaluation/eval_bestbuy_category_enrichment_prototype.py --phase embed
    python evaluation/eval_bestbuy_category_enrichment_prototype.py --phase estimate
    python evaluation/eval_bestbuy_category_enrichment_prototype.py --phase classify
    python evaluation/eval_bestbuy_category_enrichment_prototype.py --phase eval
"""

import argparse
import asyncio
import datetime
import html
import json
import math
import os
import re
import sys
import tarfile
import time
import xml.etree.ElementTree as ET
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np  # noqa: E402
from dotenv import load_dotenv  # noqa: E402

from evaluation.eval_bestbuy_llm_judge_rerank import (  # noqa: E402
    load_corpus,
    load_split,
    per_query_metrics,
)

load_dotenv(override=True)

K_EVAL = 10
UNKNOWN_CLASS = "UNKNOWN"

# Reference point: the already-published BM25 result on the same 250 queries
# (full-corpus, BoD only). Carried into the results JSON so the four conditions
# can be read side by side, but NOT recomputed here.
BM25_REFERENCE_RESULTS = "evaluation/results/bestbuy_bm25_junk_gate.json"

PRICES_PER_M_TOKENS = {
    "gpt-4o-mini": {"in": 0.15, "out": 0.60},
    "gpt-4o": {"in": 2.50, "out": 10.00},
    "gpt-4.1-mini": {"in": 0.40, "out": 1.60},
    "gpt-4.1": {"in": 2.00, "out": 8.00},
}
SPEND_LEDGER = Path(__file__).resolve().parent.parent / ".api_spend.jsonl"
COST_CEILING_USD = 1.0

# The query classifier. The candidate vocabulary is injected so the model
# cannot invent category names that do not exist in the catalog.
CLASSIFY_PROMPT = """You are mapping a shopping search query to product \
categories in an electronics retailer's catalog taxonomy.

Search query: {query}

Allowed categories (choose ONLY from this list, copy the strings exactly):
{vocab}

Pick the 1-3 categories that a product would have to be in to satisfy what \
someone typing this query is looking for. If the query names a main product, \
do NOT include the accessory category for it. Order best first.

Reply with only the category strings, one per line, no numbering, no other \
text."""

MANUAL_QUERIES = ("apple tablet", "nokia phone")
SPOTLIGHT_CLASSES = ("TABLET", "INPUT DEVICES", "TABLET ACCESSORIES")


# --------------------------------------------------------------------------
# OpenAI plumbing (same pattern as eval_bestbuy_llm_judge_junkrate.py)
# --------------------------------------------------------------------------
def estimate_cost(model: str, tokens_in: int, tokens_out: int) -> float:
    p = PRICES_PER_M_TOKENS.get(model)
    if not p:
        return 0.0
    return (tokens_in * p["in"] + tokens_out * p["out"]) / 1_000_000.0


def record_spend(model, tokens_in, tokens_out, cost_usd, purpose):
    rec = {
        "ts": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "provider": "openai",
        "model": model,
        "tokens": int(tokens_in + tokens_out),
        "tokens_in": int(tokens_in),
        "tokens_out": int(tokens_out),
        "cost_usd": round(float(cost_usd), 6),
        "purpose": purpose,
    }
    with open(SPEND_LEDGER, "a") as f:
        f.write(json.dumps(rec) + "\n")
    return rec


def make_client():
    from openai import AsyncOpenAI

    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        raise SystemExit("OPENAI_API_KEY not set (expected in .env)")
    return AsyncOpenAI(api_key=key)


class Usage:
    def __init__(self):
        self.tin = 0
        self.tout = 0
        self.calls = 0
        self.errors = 0


async def _chat(client, sem, usage, model, prompt, max_tokens, max_retries=6):
    backoff = 1.0
    async with sem:
        for _ in range(max_retries):
            try:
                resp = await client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=max_tokens,
                    temperature=0.0,
                )
                u = resp.usage
                usage.tin += int(u.prompt_tokens or 0)
                usage.tout += int(u.completion_tokens or 0)
                usage.calls += 1
                return resp.choices[0].message.content or ""
            except Exception:
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2.0, 30.0)
        usage.errors += 1
        return None


# --------------------------------------------------------------------------
# paths / cached inputs
# --------------------------------------------------------------------------
def work_paths(work_dir, tag):
    w = Path(work_dir)
    w.mkdir(parents=True, exist_ok=True)
    return {
        "fields": w / f"catalog_fields_{tag}.json",
        "vecs": w / f"enriched_vecs_{tag}.npz",
        "classify": w / f"query_classes_{tag}.jsonl",
        "classify_meta": w / f"query_classes_meta_{tag}.json",
    }


def load_pool(args):
    """The cached base/BoD top-10 pool from the junk-rate run."""
    with open(Path(args.junk_dir) / f"junk_pool_{args.tag}.json") as f:
        return json.load(f)


def load_judge_labels(args):
    """(query, product_id) -> p_yes, unioned over both cached judge runs."""
    out = {}
    for path in (
        Path(args.junk_dir) / f"junk_judge_{args.tag}.jsonl",
        Path(args.gate_dir) / f"gate_judge_{args.tag}.jsonl",
    ):
        if not path.exists():
            continue
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if r.get("p_yes") is None:
                    continue
                out[f"{r['query']}\t{r['product_id']}"] = float(r["p_yes"])
    return out


def judged_product_ids(args):
    pids = set()
    for path in (
        Path(args.junk_dir) / f"junk_judge_{args.tag}.jsonl",
        Path(args.gate_dir) / f"gate_judge_{args.tag}.jsonl",
    ):
        if not path.exists():
            continue
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    pids.add(json.loads(line)["product_id"])
                except (json.JSONDecodeError, KeyError):
                    continue
    return pids


# --------------------------------------------------------------------------
# phase: catalog  (stream the raw XML tarball, no multi-GB extraction)
# --------------------------------------------------------------------------
FIELDS_WANTED = ("sku", "name", "manufacturer", "class", "subclass", "type", "department")


def _leaf_category(prod_el):
    """Deepest categoryPath entry's name, HTML-unescaped."""
    cp = prod_el.find("categoryPath")
    if cp is None:
        return ""
    names = [c.findtext("name") or "" for c in cp.findall("category")]
    names = [html.unescape(n).strip() for n in names if n and n.strip()]
    return names[-1] if names else ""


def _category_chain(prod_el, max_depth=3):
    cp = prod_el.find("categoryPath")
    if cp is None:
        return ""
    names = [html.unescape(c.findtext("name") or "").strip() for c in cp.findall("category")]
    names = [n for n in names if n and n != "Best Buy"]
    return " > ".join(names[-max_depth:])


def phase_catalog(args, paths):
    """Pull catalog fields for just the products this prototype touches.

    The tarball is streamed member-by-member and parsed in memory. Nothing is
    written to disk except the small JSON result -- extracting the whole thing
    is ~7.7GB and this machine has had a disk incident.
    """
    data, titles, pids = load_corpus(resolve_data_dir(args))
    qrels, _ = load_split(data, args.queries_file, args.qrels_file)
    pool = load_pool(args)

    wanted = judged_product_ids(args)
    n_judged = len(wanted)
    for r in pool["rows"]:
        if not r["is_manual"] and r["key"] in qrels:
            wanted |= set(qrels[r["key"]].keys())
    print(f"  {n_judged:,} judged products + qrels golds -> {len(wanted):,} skus to look up")

    tar_path = Path(args.tarball)
    if not tar_path.exists():
        raise SystemExit(f"raw catalog tarball not found: {tar_path}")

    found = {}
    t0 = time.time()
    n_files = 0
    n_products = 0
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
                if sku in wanted and sku not in found:
                    rec = {}
                    for f in FIELDS_WANTED:
                        rec[f] = html.unescape((el.findtext(f) or "").strip())
                    rec["category_leaf"] = _leaf_category(el)
                    rec["category_chain"] = _category_chain(el)
                    found[sku] = rec
                el.clear()
            if n_files % 25 == 0:
                print(
                    f"    [{n_files}/257 files] {n_products:,} products scanned, "
                    f"{len(found):,}/{len(wanted):,} found, {time.time() - t0:.0f}s",
                    flush=True,
                )
            if len(found) == len(wanted):
                print("    all wanted skus found; stopping the scan early", flush=True)
                break

    # Anything not in the raw XML (the 15-16 digit pseudo-skus are BestBuy
    # category landing pages, not products) falls back to title-only.
    pid_to_row = {p: i for i, p in enumerate(pids)}
    n_fallback = 0
    for sku in wanted:
        if sku in found:
            continue
        n_fallback += 1
        i = pid_to_row.get(sku)
        found[sku] = {
            "sku": sku,
            "name": html.unescape(titles[i]) if i is not None else "",
            "manufacturer": "",
            "class": "",
            "subclass": "",
            "type": "",
            "department": "",
            "category_leaf": "",
            "category_chain": "",
        }

    payload = {
        "tarball": str(tar_path),
        "n_wanted": len(wanted),
        "n_found_in_raw_xml": len(wanted) - n_fallback,
        "n_fallback_title_only": n_fallback,
        "n_products_scanned": n_products,
        "scan_seconds": time.time() - t0,
        "fields": found,
    }
    with open(paths["fields"], "w") as f:
        json.dump(payload, f)
    print(
        f"saved {len(found):,} product field records -> {paths['fields']} "
        f"(raw-XML coverage {100 * (len(wanted) - n_fallback) / max(len(wanted), 1):.1f}%, "
        f"{time.time() - t0:.0f}s)",
        flush=True,
    )


def resolve_data_dir(args):
    if args.data_dir:
        return Path(args.data_dir).resolve()
    from huggingface_hub import snapshot_download

    return Path(
        snapshot_download(
            repo_id="dtunkelang/bag-of-documents-bestbuy",
            repo_type="dataset",
            allow_patterns=[
                "titles.json",
                "product_ids.json",
                "holdout_queries.jsonl",
                "holdout_qrels.jsonl",
                "base_catalog.vecs.fp16.npy",
                "bod_catalog.vecs.fp16.npy",
            ],
        )
    ).resolve()


# --------------------------------------------------------------------------
# enriched text
# --------------------------------------------------------------------------
def enriched_text(rec, fallback_title=""):
    """'{name} - {manufacturer} - {categoryPath leaf} - {class}', empties dropped.

    HTML entities are unescaped everywhere: the shipped titles.json still has
    raw `&#xAE;` / `&#x2122;` in it, which tokenizes into garbage subwords.
    """
    name = rec.get("name") or fallback_title
    parts = [
        html.unescape(name or "").strip(),
        (rec.get("manufacturer") or "").strip(),
        (rec.get("category_leaf") or "").strip(),
        (rec.get("class") or "").strip().title(),
    ]
    seen, keep = set(), []
    for p in parts:
        if p and p.lower() not in seen:
            seen.add(p.lower())
            keep.append(p)
    return " — ".join(keep)


def product_class(rec):
    c = (rec.get("class") or "").strip().upper()
    return c if c else UNKNOWN_CLASS


# --------------------------------------------------------------------------
# phase: embed
# --------------------------------------------------------------------------
def phase_embed(args, paths):
    import torch
    from sentence_transformers import SentenceTransformer

    with open(paths["fields"]) as f:
        fields = json.load(f)["fields"]
    data, titles, pids = load_corpus(resolve_data_dir(args))
    pool = load_pool(args)

    subset = sorted(fields.keys())
    pid_to_row = {p: i for i, p in enumerate(pids)}
    texts = [
        enriched_text(fields[s], titles[pid_to_row[s]] if s in pid_to_row else "") for s in subset
    ]
    queries = [r["query"] for r in pool["rows"]]
    print(f"  {len(subset):,} products, {len(queries)} queries", flush=True)
    print(f"  example enriched text: {texts[len(texts) // 2]!r}", flush=True)

    device = (
        "mps"
        if torch.backends.mps.is_available()
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    out = {
        "subset_pids": np.array(subset, dtype=object),
        "queries": np.array(queries, dtype=object),
    }

    for key, model_id, vecs_file in (
        ("base", args.base_model, args.base_vecs),
        ("bod", args.bod_model, args.bod_vecs),
    ):
        print(f"  encoding with {model_id} on {device}...", flush=True)
        m = SentenceTransformer(model_id, device=device)
        out[f"{key}_prod_enriched"] = m.encode(
            texts, normalize_embeddings=True, batch_size=128, show_progress_bar=False
        ).astype(np.float32)
        out[f"{key}_query"] = m.encode(
            queries, normalize_embeddings=True, batch_size=64, show_progress_bar=False
        ).astype(np.float32)
        del m

        # ORIGINAL (title-only) catalog vectors for the same subset: these are
        # the exact vectors that produced the cached top-10s, so the baseline
        # condition reproduces the cached ranking bit-for-bit.
        cat = np.load(data / vecs_file, mmap_mode="r")
        rows = np.array([pid_to_row[s] for s in subset if s in pid_to_row], dtype=np.int64)
        have = np.array([s in pid_to_row for s in subset], dtype=bool)
        orig = np.zeros((len(subset), cat.shape[1]), dtype=np.float32)
        order = np.argsort(rows)
        orig[np.where(have)[0][order]] = np.asarray(cat[np.sort(rows)]).astype(np.float32)
        out[f"{key}_prod_orig"] = orig
        out[f"{key}_prod_orig_have"] = have
        del cat

    np.savez_compressed(paths["vecs"], **out)
    print(f"saved vectors -> {paths['vecs']}", flush=True)


# --------------------------------------------------------------------------
# phase: estimate / classify
# --------------------------------------------------------------------------
def class_vocabulary(fields, pool_pids=None):
    """Distinct `class` strings observed in the product subset."""
    src = fields if pool_pids is None else {k: v for k, v in fields.items() if k in pool_pids}
    vocab = sorted({product_class(v) for v in src.values()} - {UNKNOWN_CLASS})
    return vocab


def _classify_prompt(query, vocab):
    return CLASSIFY_PROMPT.format(query=query, vocab="\n".join(vocab))


def phase_estimate(args, paths, quiet=False):
    with open(paths["fields"]) as f:
        fields = json.load(f)["fields"]
    pool = load_pool(args)
    vocab = class_vocabulary(fields)
    prompts = [_classify_prompt(r["query"], vocab) for r in pool["rows"]]
    tin = sum(len(p) / 4.0 + 8 for p in prompts)
    tout = len(prompts) * args.max_classify_tokens
    cost = estimate_cost(args.model, tin, tout)
    breakdown = {
        "model": args.model,
        "n_queries": len(prompts),
        "vocab_size": len(vocab),
        "est_tokens_in": int(tin),
        "est_tokens_out": int(tout),
        "est_cost_usd_total": cost,
        "ceiling_usd": COST_CEILING_USD,
    }
    if not quiet:
        print(json.dumps(breakdown, indent=2), flush=True)
        print(f"\nPROJECTED COST: ${cost:.4f} vs ceiling ${COST_CEILING_USD:.2f}", flush=True)
    return breakdown


def _load_classified(path):
    done = {}
    if not Path(path).exists():
        return done
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("classes"):
                done[r["query"]] = r
    return done


def _parse_classes(text, vocab_upper):
    """Map free-text lines back onto the allowed vocabulary (exact, then fuzzy)."""
    if not text:
        return []
    out = []
    for raw in text.splitlines():
        c = re.sub(r"^\s*[-*\d.)\s]+", "", raw).strip().strip("\"'").upper()
        if not c:
            continue
        if c in vocab_upper:
            out.append(c)
            continue
        hits = [v for v in vocab_upper if v == c or v.replace("&", "AND") == c.replace("&", "AND")]
        if hits:
            out.append(hits[0])
    seen, keep = set(), []
    for c in out:
        if c not in seen:
            seen.add(c)
            keep.append(c)
    return keep[:3]


def phase_classify(args, paths):
    with open(paths["fields"]) as f:
        fields = json.load(f)["fields"]
    pool = load_pool(args)
    vocab = class_vocabulary(fields)
    vocab_upper = set(vocab)

    est = phase_estimate(args, paths, quiet=True)
    print(f"[cost guard] projected ${est['est_cost_usd_total']:.4f}", flush=True)
    if est["est_cost_usd_total"] > COST_CEILING_USD:
        raise SystemExit(
            f"Refusing to run: projected ${est['est_cost_usd_total']:.4f} exceeds "
            f"ceiling ${COST_CEILING_USD:.2f}."
        )

    already = _load_classified(paths["classify"]) if args.resume else {}
    todo = [r["query"] for r in pool["rows"] if r["query"] not in already]
    print(f"  {len(already)} cached, {len(todo)} to classify (vocab {len(vocab)})", flush=True)
    if not todo:
        print("  fully cached; nothing to do", flush=True)
        return

    usage = Usage()
    t0 = time.time()

    async def run():
        client = make_client()
        sem = asyncio.Semaphore(args.concurrency)

        async def one(q):
            txt = await _chat(
                client, sem, usage, args.model, _classify_prompt(q, vocab), args.max_classify_tokens
            )
            return q, txt

        with open(paths["classify"], "a") as out_f:
            for i in range(0, len(todo), 100):
                batch = todo[i : i + 100]
                for q, txt in await asyncio.gather(*[one(x) for x in batch]):
                    out_f.write(
                        json.dumps(
                            {"query": q, "raw": txt, "classes": _parse_classes(txt, vocab_upper)}
                        )
                        + "\n"
                    )
                out_f.flush()
                print(f"  [classify {min(i + 100, len(todo))}/{len(todo)}]", flush=True)

    try:
        asyncio.run(run())
    finally:
        c = estimate_cost(args.model, usage.tin, usage.tout)
        if usage.calls:
            record_spend(
                args.model,
                usage.tin,
                usage.tout,
                c,
                "bestbuy category enrichment: query->class classification",
            )

    cost = estimate_cost(args.model, usage.tin, usage.tout)
    meta = {
        "model": args.model,
        "prompt": CLASSIFY_PROMPT,
        "vocab_size": len(vocab),
        "n_queries": len(todo),
        "tokens_in": usage.tin,
        "tokens_out": usage.tout,
        "api_calls": usage.calls,
        "api_errors": usage.errors,
        "cost_usd": cost,
        "wall_clock_s": time.time() - t0,
    }
    with open(paths["classify_meta"], "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\nclassify done in {time.time() - t0:.0f}s cost=${cost:.4f}", flush=True)


# --------------------------------------------------------------------------
# phase: eval
# --------------------------------------------------------------------------
def _bootstrap_ci(values, n_boot=2000, seed=0):
    v = np.asarray([x for x in values if x is not None and not math.isnan(x)], dtype=np.float64)
    if v.size == 0:
        return None, None
    rng = np.random.default_rng(seed)
    means = v[rng.integers(0, v.size, size=(n_boot, v.size))].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def _paired_delta_ci(a, b, n_boot=2000, seed=0):
    """CI on mean(a) - mean(b) resampling QUERIES (paired)."""
    aa = np.asarray(a, dtype=np.float64)
    bb = np.asarray(b, dtype=np.float64)
    ok = ~(np.isnan(aa) | np.isnan(bb))
    aa, bb = aa[ok], bb[ok]
    if aa.size == 0:
        return float("nan"), (None, None)
    d = aa - bb
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, d.size, size=(n_boot, d.size))
    means = d[idx].mean(axis=1)
    return float(d.mean()), (float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)))


def _rank_metrics(ordered_pids, query, qrels_q, labels, thr, min_rel, exact_rel):
    top = ordered_pids[:K_EVAL]
    scored = [labels[f"{query}\t{p}"] for p in top if f"{query}\t{p}" in labels]
    junk = sum(1 for s in scored if s < thr) / len(scored) if scored else float("nan")
    if qrels_q:
        m = per_query_metrics(top, qrels_q, k=K_EVAL, min_rel=min_rel, exact_rel=exact_rel)
        recall, _ndcg, e1, _e3 = m if m else (float("nan"),) * 4
    else:
        recall = e1 = float("nan")
    return junk, recall, e1


def _coherence(vecs, idx_by_group, min_members):
    """Per-group cosine to own centroid vs to the nearest OTHER group centroid."""
    groups = [g for g, ix in idx_by_group.items() if len(ix) >= min_members]
    if len(groups) < 2:
        return []
    cents = np.stack(
        [_l2(np.asarray(vecs[idx_by_group[g]], dtype=np.float32).mean(axis=0)) for g in groups]
    )
    rows = []
    for gi, g in enumerate(groups):
        v = np.asarray(vecs[idx_by_group[g]], dtype=np.float32)
        intra = float((v @ cents[gi]).mean())
        sims = v @ cents.T  # (n_members, n_groups)
        sims[:, gi] = -np.inf
        best = sims.max(axis=1)
        nearest = int(np.bincount(sims.argmax(axis=1), minlength=len(groups)).argmax())
        rows.append(
            {
                "group": g,
                "n_members": int(len(idx_by_group[g])),
                "intra_cos_to_own_centroid": intra,
                "mean_cos_to_nearest_other_centroid": float(best.mean()),
                "modal_nearest_other": groups[nearest],
                "margin": intra - float(best.mean()),
                "red_flag": intra <= float(best.mean()),
            }
        )
    return sorted(rows, key=lambda r: -r["n_members"])


def _l2(v):
    n = float(np.linalg.norm(v))
    return v / n if n > 0 else v


def phase_eval(args, paths):
    with open(paths["fields"]) as f:
        fields = json.load(f)["fields"]
    z = np.load(paths["vecs"], allow_pickle=True)
    subset = list(z["subset_pids"])
    sub_ix = {p: i for i, p in enumerate(subset)}
    pool = load_pool(args)
    labels = load_judge_labels(args)
    classified = _load_classified(paths["classify"])
    data, titles, pids = load_corpus(resolve_data_dir(args))
    qrels, _ = load_split(data, args.queries_file, args.qrels_file)
    thr = args.junk_threshold

    qtexts = list(z["queries"])
    q_ix = {q: i for i, q in enumerate(qtexts)}
    cls_of = {p: product_class(fields[p]) for p in subset}

    conditions = ("baseline", "enriched", "catboost", "enriched_catboost")
    per_query = {
        m: {c: {"junk": [], "recall": [], "e1": []} for c in conditions} for m in ("base", "bod")
    }
    holdout_keys = []
    orderings = {}  # (model, condition, query) -> top-10 pids, for spot checks
    n_no_class_pred = 0

    for r in pool["rows"]:
        query = r["query"]
        qi = q_ix[query]
        qrels_q = qrels.get(r["key"]) if not r["is_manual"] else None
        cand = sorted({d["product_id"] for m in ("base", "bod") for d in r[m][: args.pool_top_k]})
        cand = [p for p in cand if p in sub_ix]
        ci = np.array([sub_ix[p] for p in cand], dtype=np.int64)

        pred = classified.get(query, {}).get("classes") or []
        if not pred:
            n_no_class_pred += 1
        pred_set = set(pred)
        compat = np.array([1.0 if cls_of[p] in pred_set else 0.0 for p in cand])

        for model_key in ("base", "bod"):
            qv = z[f"{model_key}_query"][qi]
            s_orig = z[f"{model_key}_prod_orig"][ci] @ qv
            s_enr = z[f"{model_key}_prod_enriched"][ci] @ qv

            scores = {
                "baseline": s_orig,
                "enriched": s_enr,
                # boost = stable partition on the compatible flag, ties broken
                # by the ORIGINAL dense score (order preserved inside a group)
                "catboost": compat * 10.0 + _rank01(s_orig),
                "enriched_catboost": compat * 10.0 + _rank01(s_enr),
            }
            for cond, sc in scores.items():
                order = np.argsort(-sc, kind="stable")
                ordered = [cand[j] for j in order]
                orderings[(model_key, cond, query)] = ordered[:K_EVAL]
                if r["is_manual"]:
                    continue
                junk, recall, e1 = _rank_metrics(
                    ordered, query, qrels_q, labels, thr, args.min_relevance, args.exact_relevance
                )
                per_query[model_key][cond]["junk"].append(junk)
                per_query[model_key][cond]["recall"].append(recall)
                per_query[model_key][cond]["e1"].append(e1)
        if not r["is_manual"]:
            holdout_keys.append(r["key"])

    n_q = len(holdout_keys)
    print(f"\n  {n_q} holdout queries scored, {n_no_class_pred} with no parsed class prediction")

    # ---- headline table -------------------------------------------------
    summary = {}
    for model_key in ("base", "bod"):
        summary[model_key] = {}
        for cond in conditions:
            d = per_query[model_key][cond]
            row = {}
            for metric, vals in (
                ("junk_rate_at_10", d["junk"]),
                ("recall_at_10", d["recall"]),
                ("e_at_1", d["e1"]),
            ):
                arr = np.asarray(vals, dtype=np.float64)
                lo, hi = _bootstrap_ci(vals, args.n_boot, args.seed)
                row[metric] = float(np.nanmean(arr))
                row[f"{metric}_ci95"] = [lo, hi]
            if cond != "baseline":
                base_d = per_query[model_key]["baseline"]
                for metric, vals, bvals in (
                    ("junk_rate_at_10", d["junk"], base_d["junk"]),
                    ("recall_at_10", d["recall"], base_d["recall"]),
                    ("e_at_1", d["e1"], base_d["e1"]),
                ):
                    dm, (lo, hi) = _paired_delta_ci(vals, bvals, args.n_boot, args.seed)
                    row[f"delta_{metric}_vs_baseline"] = dm
                    row[f"delta_{metric}_ci95"] = [lo, hi]
            summary[model_key][cond] = row

    # ---- step 4: query-classification recall risk -----------------------
    miss_rows = []
    n_miss = 0
    n_total_miss = 0
    n_checkable = 0
    for r in pool["rows"]:
        if r["is_manual"] or r["key"] not in qrels:
            continue
        golds = [p for p, g in qrels[r["key"]].items() if g >= args.min_relevance]
        gold_classes = [cls_of.get(p, UNKNOWN_CLASS) for p in golds if p in cls_of]
        gold_classes = [c for c in gold_classes if c != UNKNOWN_CLASS]
        if not gold_classes:
            continue
        n_checkable += 1
        pred = set(classified.get(r["query"], {}).get("classes") or [])
        missed = sorted({c for c in gold_classes if c not in pred})
        if not (set(gold_classes) & pred):
            n_total_miss += 1
        if missed:
            n_miss += 1
            miss_rows.append(
                {
                    "query": r["query"],
                    "predicted": sorted(pred),
                    "gold_classes": sorted(set(gold_classes)),
                    "gold_classes_missed": missed,
                    "total_miss": not (set(gold_classes) & pred),
                }
            )
    class_risk = {
        "n_queries_with_resolvable_gold_class": n_checkable,
        "n_queries_missing_at_least_one_gold_class": n_miss,
        "miss_rate": n_miss / n_checkable if n_checkable else float("nan"),
        "n_queries_missing_every_gold_class": n_total_miss,
        "total_miss_rate": n_total_miss / n_checkable if n_checkable else float("nan"),
        "n_queries_no_class_prediction": n_no_class_pred,
        "note": (
            "total_miss_rate is the fraction where a HARD category filter would have "
            "removed every gold. miss_rate is the looser 'at least one gold class was "
            "excluded' measure the brief asked for."
        ),
        "examples": miss_rows[: args.n_examples],
    }

    # ---- step 5: named spot checks --------------------------------------
    def snapshot(model_key, cond, query):
        return [
            {
                "rank": i + 1,
                "product_id": p,
                "class": cls_of.get(p, UNKNOWN_CLASS),
                "text": enriched_text(fields.get(p, {}), ""),
                "p_yes": labels.get(f"{query}\t{p}"),
                "junk": (labels.get(f"{query}\t{p}") is not None and labels[f"{query}\t{p}"] < thr),
            }
            for i, p in enumerate(orderings.get((model_key, cond, query), []))
        ]

    spot = {}
    for query in args.manual_queries:
        if not any((m, "baseline", query) in orderings for m in ("base", "bod")):
            continue
        spot[query] = {
            "predicted_classes": classified.get(query, {}).get("classes") or [],
            **{m: {c: snapshot(m, c, query) for c in conditions} for m in ("base", "bod")},
        }

    # ---- coordinator add-on: are the categories coherent in vector space?
    coherence = {}
    by_class, by_leaf = {}, {}
    for i, p in enumerate(subset):
        by_class.setdefault(cls_of[p], []).append(i)
        leaf = (fields[p].get("category_leaf") or "").strip() or UNKNOWN_CLASS
        by_leaf.setdefault(leaf, []).append(i)
    by_class.pop(UNKNOWN_CLASS, None)
    by_leaf.pop(UNKNOWN_CLASS, None)
    for model_key in ("base", "bod"):
        v = z[f"{model_key}_prod_enriched"]
        coherence[model_key] = {
            "by_class": _coherence(
                v, {k: np.array(x) for k, x in by_class.items()}, args.min_class_members
            ),
            "by_category_leaf": _coherence(
                v, {k: np.array(x) for k, x in by_leaf.items()}, args.min_class_members
            ),
        }
        # pairwise centroid cosines for the three classes in the iPad example
        present = [c for c in SPOTLIGHT_CLASSES if len(by_class.get(c, [])) > 0]
        cents = {c: _l2(np.asarray(v[by_class[c]], dtype=np.float32).mean(axis=0)) for c in present}
        coherence[model_key]["spotlight_pairwise_centroid_cos"] = {
            f"{a} vs {b}": float(cents[a] @ cents[b])
            for i, a in enumerate(present)
            for b in present[i + 1 :]
        }
        coherence[model_key]["spotlight_n_members"] = {
            c: len(by_class.get(c, [])) for c in SPOTLIGHT_CLASSES
        }

    # ---- nearest-centroid query classification on TODAY'S embeddings ----
    # Can the *current, unmodified* space classify a query into its category
    # without any retraining, or is it too polluted by the very same
    # wrong-category geometry that puts keyboards next to "apple tablet"?
    # Centroids are built from the ORIGINAL (title-only) product vectors --
    # the space as deployed -- not the enriched ones.
    gold_classes_by_query = {}
    golds_by_query = {}
    for r in pool["rows"]:
        if r["is_manual"] or r["key"] not in qrels:
            continue
        gc = {
            cls_of.get(p, UNKNOWN_CLASS)
            for p, g in qrels[r["key"]].items()
            if g >= args.min_relevance and p in cls_of
        } - {UNKNOWN_CLASS}
        if gc:
            gold_classes_by_query[r["query"]] = gc
            golds_by_query[r["query"]] = [
                p for p, g in qrels[r["key"]].items() if g >= args.min_relevance and p in sub_ix
            ]

    centroid_probe = {}
    name_ix = {c: i for i, c in enumerate(sorted(by_class.keys()))}
    for model_key in ("base", "bod"):
        for space, vkey in (("original", "prod_orig"), ("enriched", "prod_enriched")):
            v = z[f"{model_key}_{vkey}"]
            names = sorted(by_class.keys())
            cents = np.stack(
                [
                    _l2(np.asarray(v[np.array(by_class[c])], dtype=np.float32).mean(axis=0))
                    for c in names
                ]
            )
            top1 = top3 = n = 0
            for query, gc in gold_classes_by_query.items():
                sims = z[f"{model_key}_query"][q_ix[query]] @ cents.T
                order = np.argsort(-sims)[:3]
                pred3 = [names[j] for j in order]
                n += 1
                top1 += 1 if pred3[0] in gc else 0
                top3 += 1 if set(pred3) & gc else 0
            # Leave-golds-out: the gold products for a query also contribute to
            # their own class centroid, which flatters this probe. Recompute
            # each query's centroids with that query's golds removed.
            sums = np.stack(
                [np.asarray(v[np.array(by_class[c])], dtype=np.float32).sum(axis=0) for c in names]
            )
            counts = np.array([len(by_class[c]) for c in names], dtype=np.float64)
            lo1 = lo3 = 0
            for query, gc in gold_classes_by_query.items():
                s, c2 = sums.copy(), counts.copy()
                for p in golds_by_query.get(query, []):
                    cp = cls_of.get(p, UNKNOWN_CLASS)
                    if cp in name_ix:
                        s[name_ix[cp]] -= np.asarray(v[sub_ix[p]], dtype=np.float32)
                        c2[name_ix[cp]] -= 1
                keep = c2 > 0
                cc = s[keep] / c2[keep, None]
                cc = cc / np.maximum(np.linalg.norm(cc, axis=1, keepdims=True), 1e-9)
                kn = [names[j] for j in np.where(keep)[0]]
                sims = z[f"{model_key}_query"][q_ix[query]] @ cc.T
                pred3 = [kn[j] for j in np.argsort(-sims)[:3]]
                lo1 += 1 if pred3[0] in gc else 0
                lo3 += 1 if set(pred3) & gc else 0
            centroid_probe[f"{model_key}_{space}"] = {
                "n_classes": len(names),
                "n_queries": n,
                "top1_accuracy": top1 / n if n else float("nan"),
                "top3_accuracy": top3 / n if n else float("nan"),
                "top1_accuracy_leave_golds_out": lo1 / n if n else float("nan"),
                "top3_accuracy_leave_golds_out": lo3 / n if n else float("nan"),
            }
        # spot-check queries: raw nearest centroids in the ORIGINAL space
        v = z[f"{model_key}_prod_orig"]
        names = sorted(by_class.keys())
        cents = np.stack(
            [
                _l2(np.asarray(v[np.array(by_class[c])], dtype=np.float32).mean(axis=0))
                for c in names
            ]
        )
        for query in args.manual_queries:
            if query not in q_ix:
                continue
            sims = z[f"{model_key}_query"][q_ix[query]] @ cents.T
            order = np.argsort(-sims)[:5]
            centroid_probe.setdefault("spot_checks", {}).setdefault(query, {})[model_key] = [
                {"class": names[j], "cos": float(sims[j])} for j in order
            ]

    # LLM classifier on the same queries, scored the same way, as the ceiling
    llm_top1 = llm_top3 = 0
    for query, gc in gold_classes_by_query.items():
        pred = classified.get(query, {}).get("classes") or []
        llm_top1 += 1 if (pred and pred[0] in gc) else 0
        llm_top3 += 1 if (set(pred) & gc) else 0
    nq = max(len(gold_classes_by_query), 1)
    centroid_probe["llm_classifier_same_scoring"] = {
        "n_queries": len(gold_classes_by_query),
        "top1_accuracy": llm_top1 / nq,
        "top3_accuracy": llm_top3 / nq,
        "note": "gpt-4o-mini with the full 179-class vocabulary in the prompt; "
        "top3 = any of its 1-3 predictions is a gold class",
    }

    # ---- product-level k-NN category purity on TODAY'S space ------------
    # Centroid coherence is an aggregate: a class can look tight on
    # cosine-to-own-centroid while its individual products still sit next to
    # wrong-class neighbours. This is the local version of the same question:
    # of a product's k nearest OTHER products, how many share its `class`?
    # ORIGINAL (title-only) vectors -- the production space, not the enriched one.
    knn_purity = {}
    knn_ks = (5, 10)
    cls_arr = np.array([cls_of[p] for p in subset])
    labeled = np.where(cls_arr != UNKNOWN_CLASS)[0]
    for model_key in ("base", "bod"):
        v = np.asarray(z[f"{model_key}_prod_orig"], dtype=np.float32)[labeled]
        lab = cls_arr[labeled]
        sims = v @ v.T
        np.fill_diagonal(sims, -np.inf)
        nn = np.argsort(-sims, axis=1)[:, : max(knn_ks)]
        same = lab[nn] == lab[:, None]
        entry = {}
        for k in knn_ks:
            entry[f"mean_purity_at_{k}"] = float(same[:, :k].mean())
        per_class = {}
        for c in sorted(set(lab)):
            rows_c = np.where(lab == c)[0]
            if rows_c.size < args.min_class_members:
                continue
            per_class[c] = {
                "n_members": int(rows_c.size),
                **{f"purity_at_{k}": float(same[rows_c, :k].mean()) for k in knn_ks},
            }
        entry["by_class"] = dict(
            sorted(per_class.items(), key=lambda kv: -kv[1]["n_members"])[: args.n_coherence_rows]
        )
        entry["spotlight_by_class"] = {
            c: per_class.get(c) for c in (*SPOTLIGHT_CLASSES, "AT&T HARDWARE", "MOBILE PHONE ACCY")
        }
        # product-level spot checks from the "apple tablet" anecdote
        spot_prods = {}
        for want in ("iPad™ with Wi-Fi - 32GB", "Apple® - Wireless Keyboard", "Apple® - Keyboard"):
            hit = next(
                (p for p in subset if fields[p]["name"].strip().endswith(want.strip())),
                None,
            ) or next((p for p in subset if want in fields[p]["name"]), None)
            if hit is None or hit not in sub_ix:
                continue
            row = int(np.where(labeled == sub_ix[hit])[0][0]) if sub_ix[hit] in labeled else None
            if row is None:
                continue
            spot_prods[fields[hit]["name"]] = {
                "class": cls_of[hit],
                "top5_neighbors": [
                    {
                        "name": fields[subset[labeled[j]]]["name"],
                        "class": cls_arr[labeled[j]],
                        "cos": float(sims[row, j]),
                    }
                    for j in nn[row, :5]
                ],
            }
        entry["product_spot_checks"] = spot_prods
        knn_purity[model_key] = entry
        del sims

    # ---- reference: the already-published BM25 result -------------------
    bm25_ref = {}
    ref_path = Path(__file__).resolve().parent.parent / BM25_REFERENCE_RESULTS
    if ref_path.exists():
        with open(ref_path) as f:
            ref = json.load(f)
        for k in ("dense", "gate_deep", "rrf60"):
            s = ref["summary"].get(k, {})
            bm25_ref[k] = {
                "junk_rate_at_10": s.get("junk_rate_at_10"),
                "recall_at_10": s.get("recall_at_10_fraction_recovered"),
                "e_at_1": s.get("e_at_1"),
            }

    # ---- print ----------------------------------------------------------
    print(f"\n=== closed-pool rerank, {n_q} holdout queries, k={K_EVAL} ===")
    hdr = (
        f"{'model':<5} {'condition':<19} {'junk@10':>9} {'R@10':>8} {'E@1':>8}   deltas vs baseline"
    )
    print(hdr)
    for model_key in ("base", "bod"):
        for cond in conditions:
            s = summary[model_key][cond]
            extra = ""
            if cond != "baseline":
                extra = (
                    f"  Δjunk {s['delta_junk_rate_at_10_vs_baseline']:+.4f} "
                    f"[{s['delta_junk_rate_at_10_ci95'][0]:+.4f},"
                    f"{s['delta_junk_rate_at_10_ci95'][1]:+.4f}]"
                    f"  ΔR {s['delta_recall_at_10_vs_baseline']:+.4f}"
                    f"  ΔE1 {s['delta_e_at_1_vs_baseline']:+.4f}"
                )
            print(
                f"{model_key:<5} {cond:<19} {s['junk_rate_at_10']:>9.4f} "
                f"{s['recall_at_10']:>8.4f} {s['e_at_1']:>8.4f}{extra}"
            )

    print(
        f"\nquery-classification recall risk: miss rate "
        f"{class_risk['miss_rate']:.3f} "
        f"({n_miss}/{n_checkable} queries lose >=1 gold class); "
        f"total-miss rate {class_risk['total_miss_rate']:.3f} "
        f"({n_total_miss}/{n_checkable} lose EVERY gold class)"
    )
    for ex in class_risk["examples"][:8]:
        print(f"    {ex['query']!r} pred={ex['predicted']} gold={ex['gold_classes']}")

    for model_key in ("base", "bod"):
        rows = coherence[model_key]["by_class"][: args.n_coherence_rows]
        print(f"\n--- class coherence ({model_key}, enriched vectors) ---")
        print(f"{'class':<28} {'n':>5} {'intra':>7} {'nearest':>8} {'margin':>7}  nearest-other")
        for c in rows:
            flag = "  <== RED FLAG" if c["red_flag"] else ""
            print(
                f"{c['group'][:28]:<28} {c['n_members']:>5} "
                f"{c['intra_cos_to_own_centroid']:>7.3f} "
                f"{c['mean_cos_to_nearest_other_centroid']:>8.3f} {c['margin']:>7.3f}  "
                f"{c['modal_nearest_other'][:24]}{flag}"
            )
        print(f"  spotlight: {json.dumps(coherence[model_key]['spotlight_pairwise_centroid_cos'])}")

    print("\n--- query -> class, nearest-centroid on TODAY'S space vs the LLM ---")
    for k in sorted(centroid_probe):
        if k in ("spot_checks",):
            continue
        c = centroid_probe[k]
        loo = (
            f"  (leave-golds-out top1 {c['top1_accuracy_leave_golds_out']:.3f} "
            f"top3 {c['top3_accuracy_leave_golds_out']:.3f})"
            if "top1_accuracy_leave_golds_out" in c
            else ""
        )
        print(f"  {k:<27} top1 {c['top1_accuracy']:.3f}  top3 {c['top3_accuracy']:.3f}{loo}")
    for query, blk in centroid_probe.get("spot_checks", {}).items():
        for model_key, rows in blk.items():
            top = ", ".join(f"{r['class']} ({r['cos']:.3f})" for r in rows[:3])
            print(f"  {query!r} [{model_key}] nearest centroids: {top}")

    print("\n--- product-level k-NN category purity (ORIGINAL vectors) ---")
    for model_key in ("base", "bod"):
        e = knn_purity[model_key]
        print(
            f"  {model_key:<5} purity@5 {e['mean_purity_at_5']:.3f}  "
            f"purity@10 {e['mean_purity_at_10']:.3f}"
        )
        for c, s in e["spotlight_by_class"].items():
            if s:
                print(
                    f"      {c:<22} n={s['n_members']:>4}  @5 {s['purity_at_5']:.3f}  "
                    f"@10 {s['purity_at_10']:.3f}"
                )
        for name, s in e["product_spot_checks"].items():
            print(f"      [{model_key}] {name[:60]}  ({s['class']})")
            for nb in s["top5_neighbors"]:
                print(f"          {nb['cos']:.3f} [{nb['class'][:20]:<20}] {nb['name'][:60]}")

    for query, blk in spot.items():
        print(f"\n--- spot check {query!r} (predicted classes: {blk['predicted_classes']}) ---")
        for model_key in ("base", "bod"):
            for cond in conditions:
                print(f"  [{model_key} / {cond}]")
                for d in blk[model_key][cond][:K_EVAL]:
                    f = "JUNK" if d["junk"] else "ok  "
                    print(f"    {d['rank']:>2}. {f} [{d['class'][:22]:<22}] {d['text'][:72]}")

    out = {
        "experiment": (
            "BestBuy category enrichment prototype: do the catalog's own "
            "class/categoryPath fields fix top-10 product-type junk?"
        ),
        "limitation": (
            "CLOSED-POOL RERANK ONLY. Per query the candidate set is the union of that "
            "query's cached base top-10 and BoD top-10 over the full 1.27M catalog. "
            "Nothing new is retrieved, so R@10 can only fall or stay flat and junk-rate "
            "can only improve by reordering within an already-retrieved set. This is a "
            "necessary-not-sufficient screen for a full re-index, NOT a validation of one."
        ),
        "config": {
            "base_model": args.base_model,
            "bod_model": args.bod_model,
            "n_holdout_queries": n_q,
            "pool_top_k": args.pool_top_k,
            "k_eval": K_EVAL,
            "product_subset_size": len(subset),
            "class_vocab_size": len(class_vocabulary(fields)),
            "junk_threshold_p_yes": thr,
            "n_boot": args.n_boot,
            "seed": args.seed,
            "classifier_model": args.model,
            "classifier_prompt": CLASSIFY_PROMPT,
            "enriched_text_format": "{name} — {manufacturer} — {categoryPath leaf} — {Class}, "
            "empty/duplicate parts dropped, html.unescape() applied",
            "judge_labels_reused_from": [args.junk_dir, args.gate_dir],
        },
        "conditions": {
            "baseline": "rank the closed pool by the ORIGINAL title-only catalog vectors "
            "(reproduces the cached top-10 exactly)",
            "enriched": "rank by cosine(original query vector, ENRICHED product vector); "
            "same encoder, richer product text, no retraining",
            "catboost": "original dense score, stable-partitioned so candidates whose `class` "
            "is in the query's LLM-predicted compatible set come first (boost, not filter)",
            "enriched_catboost": "both mechanisms stacked",
            "_bm25_reference": "full-corpus numbers from eval_bestbuy_bm25_junk_gate.py, "
            "NOT closed-pool and NOT directly comparable to the rows above",
        },
        "summary": summary,
        "bm25_reference_full_corpus": bm25_ref,
        "query_classification_risk": class_risk,
        "category_coherence": coherence,
        "nearest_centroid_query_classification": centroid_probe,
        "knn_category_purity": knn_purity,
        "spot_checks": spot,
        "per_query": {
            m: {
                c: {
                    "junk": per_query[m][c]["junk"],
                    "recall": per_query[m][c]["recall"],
                    "e1": per_query[m][c]["e1"],
                }
                for c in conditions
            }
            for m in ("base", "bod")
        },
        "query_keys": holdout_keys,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {out_path}", flush=True)


def _rank01(scores):
    """Map scores to (0,1) by rank so a +10 boost always dominates."""
    order = np.argsort(-np.asarray(scores, dtype=np.float64), kind="stable")
    out = np.empty(len(scores), dtype=np.float64)
    n = max(len(scores), 1)
    for rank, j in enumerate(order):
        out[j] = 1.0 - rank / n
    return out


# --------------------------------------------------------------------------
def main():
    global COST_CEILING_USD
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--phase", required=True, choices=["catalog", "embed", "estimate", "classify", "eval"]
    )
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--queries-file", default="holdout_queries.jsonl")
    ap.add_argument("--qrels-file", default="holdout_qrels.jsonl")
    ap.add_argument("--base-model", default="all-MiniLM-L6-v2")
    ap.add_argument("--bod-model", default="dtunkelang/bag-of-documents-bestbuy-minilm")
    ap.add_argument("--base-vecs", default="base_catalog.vecs.fp16.npy")
    ap.add_argument("--bod-vecs", default="bod_catalog.vecs.fp16.npy")
    ap.add_argument("--tarball", default="/tmp/bestbuy_raw_inspect/product_data.tar.gz")
    ap.add_argument("--junk-dir", default="/tmp/bestbuy_junkrate")
    ap.add_argument("--gate-dir", default="/tmp/bestbuy_bm25_gate")
    ap.add_argument("--model", default="gpt-4o-mini", help="OpenAI query-classifier model")
    ap.add_argument("--max-classify-tokens", type=int, default=32)
    ap.add_argument("--concurrency", type=int, default=32)
    ap.add_argument("--cost-ceiling", type=float, default=COST_CEILING_USD)
    ap.add_argument("--pool-top-k", type=int, default=10, help="depth of each model's cached list")
    ap.add_argument("--junk-threshold", type=float, default=0.5)
    ap.add_argument("--min-relevance", type=int, default=1)
    ap.add_argument("--exact-relevance", type=int, default=1)
    ap.add_argument("--min-class-members", type=int, default=10)
    ap.add_argument("--n-coherence-rows", type=int, default=20)
    ap.add_argument("--n-examples", type=int, default=15)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--manual-queries", nargs="*", default=list(MANUAL_QUERIES))
    ap.add_argument("--resume", action="store_true", default=True)
    ap.add_argument("--no-resume", dest="resume", action="store_false")
    ap.add_argument("--work-dir", default="/tmp/bestbuy_catenrich")
    ap.add_argument("--tag", default="bestbuy")
    ap.add_argument(
        "--out", default="evaluation/results/bestbuy_category_enrichment_prototype.json"
    )
    args = ap.parse_args()

    COST_CEILING_USD = args.cost_ceiling
    paths = work_paths(args.work_dir, args.tag)
    {
        "catalog": lambda: phase_catalog(args, paths),
        "embed": lambda: phase_embed(args, paths),
        "estimate": lambda: phase_estimate(args, paths),
        "classify": lambda: phase_classify(args, paths),
        "eval": lambda: phase_eval(args, paths),
    }[args.phase]()


if __name__ == "__main__":
    main()
