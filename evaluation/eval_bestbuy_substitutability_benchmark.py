#!/usr/bin/env python3
"""Substitutability geometry benchmark for the BestBuy BoD index.

Everything else in this line of work measures QUERY -> PRODUCT relevance
(junk-rate, R@10, E@1, nDCG). This measures the embedding space directly:
for an anchor PRODUCT, do its neighbours by cosine similarity respect a
substitutability hierarchy?

    E  exact / near-duplicate  same core product, trivial variant only
                               (color, storage, bundle, refurb) -- a buyer
                               would treat them as interchangeable
    S  substitute              a different product that serves the SAME
                               buying need as an alternative (other brand or
                               generation of the same product type)
    C  complement              used WITH the anchor, not instead of it
                               (accessory, case, charger, cable, warranty)
    I  irrelevant              different category / use case

A well-formed index puts E closest, then S, then {C, I}. C and I are scored
identically here -- both mean "not a valid substitute" -- so the required
ordering is  E > S > {C, I}.  An accessory is C, never S: that distinction is
the whole point, because the "apple tablet" pathology is exactly an index that
treats an iPad case as if it were an iPad.

Candidate generation deliberately uses THREE sources, only two of which are
the embeddings under test, to avoid the circularity of "an embedding cannot be
shown blind to a substitute it alone was asked to find":

    dense-OLD  top-10 cosine under the pre-re-index BoD vectors
    dense-NEW  top-10 cosine under the re-indexed BoD vectors
    bm25       top-10 over the full catalog, anchor title as pseudo-query

Metric: per-anchor pairwise concordance over the union candidate set --
fraction of (E,S) pairs with sim(E) > sim(S), likewise (E,{C,I}) and
(S,{C,I}). Macro-averaged over the anchors that have >=1 such pair, with a
paired bootstrap CI on the OLD -> NEW delta.

Bias check (methodology mirrors evaluation/eval_esci_llm_judge_lexical_bias.py,
reusing its toks/overlap_metrics): among candidates that are NOT valid
substitutes (C or I) but share heavy lexical surface with the anchor (shared
brand token, high title-token coverage), does cosine similarity rank them high
anyway? That is the "apple tablet" pathology recurring at the product-product
level rather than the query-product level.

Phases (each cached + resumable):
    --phase anchors     stratified anchor sample from the raw catalog `class`
    --phase bm25        full-catalog BM25 index, top-10 per anchor title
    --phase candidates  dense top-10 OLD + NEW, union with BM25, sims for all
    --phase estimate    project OpenAI cost -- spends nothing
    --phase judge       gpt-4o-mini E/S/C/I over every (anchor, candidate)
    --phase eval        concordance + bias check -> results JSON

Usage (the repo .venv lacks torch/bm25s; layer them over the job-search venv):
    uv run --no-project --python /Users/dtunkelang/job-search/.venv/bin/python \\
        --with bm25s --with PyStemmer \\
        python evaluation/eval_bestbuy_substitutability_benchmark.py --phase anchors
    ... --phase bm25 / candidates / estimate / judge / eval
"""

import argparse
import asyncio
import json
import math
import os
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np  # noqa: E402
from dotenv import load_dotenv  # noqa: E402

from evaluation.eval_bestbuy_llm_judge_junkrate import (  # noqa: E402
    Usage,
    _chat,
    estimate_cost,
    make_client,
    record_spend,
)
from evaluation.eval_esci_llm_judge_lexical_bias import (  # noqa: E402
    bootstrap_ci,
    overlap_metrics,
    pearson,
    spearman,
    toks,
)

load_dotenv(override=True)

TOP_K = 10

# Hard stop: refuse to start a paid phase whose projection exceeds this.
COST_CEILING_USD = 3.0

# The NEW (post-re-index) artifacts, written locally by the re-index run.
DEFAULT_NEW_DIR = "/tmp/bestbuy_reindex_output/artifacts"
# The OLD (pre-re-index) vectors: the HF snapshot cached BEFORE the new data
# was pushed. Deliberately a pinned snapshot hash, not the live dataset --
# the live dataset now serves the NEW vectors.
DEFAULT_OLD_DIR = (
    Path.home()
    / ".cache/huggingface/hub/datasets--dtunkelang--bag-of-documents-bestbuy"
    / "snapshots/15ef813587f8958928d77c1e9ff905c9d8165b5c"
)
# Full-catalog sku -> {name, manufacturer, class, category_leaf}, produced by
# the re-index catalog scan of /tmp/bestbuy_raw_inspect/product_data.tar.gz
# (same `phase_catalog` XML parsing as
# evaluation/eval_bestbuy_category_enrichment_prototype.py).
DEFAULT_FIELDS_JSONL = "/tmp/bestbuy_reindex_output/catalog_fields_bestbuy.jsonl"

# Anchor strata. The catalog is 63% COMPACT DISC by count, so a uniform sample
# would be almost all music and would exercise no interesting substitutability
# judgement at all. These are HardGood electronics classes where E/S/C/I is a
# real distinction, plus a deliberately small media tail as a control.
HARDGOOD_CLASSES = (
    "TABLET",
    "PHONES",
    "MOBILE COMPUTING",
    "DESK TOP COMPUTERS",
    "DIGITAL CAMERAS",
    "DIGITAL CAMCORDERS",
    'LARGE FPTV 46"+',
    "MP-3 DEVICES",
    "HEADPHONES-MP3 SPKRS",
    "HARD DRIVES",
    "GPS NAVIGATION",
    "MONITORS",
    "COMPUTER PRINTERS",
    "SPEAKERS",
    "VIDEO GAME HARDWARE",
    "TRAFFIC APPLIANCES",
)
MEDIA_CLASSES = (
    "COMPACT DISC",
    "DVD SOFTWARE",
    "BLU RAY MOVIES",
    "VIDEO GAME SOFTWARE",
)

LABELS = ("E", "S", "C", "I")
LABEL_GAIN = {"E": 3, "S": 2, "C": 1, "I": 0}

JUDGE_PROMPT = """Anchor product: {a_title}{a_cat}
Candidate product: {c_title}{c_cat}

Label the candidate's relationship to the anchor with ONE letter:
E = same core product, differing only in a trivial variant (color, storage \
size, bundle, refurbished) - a buyer would treat them as interchangeable
S = a different product that could reasonably serve the SAME buying need as \
an alternative (different brand or generation of the same product type)
C = used WITH the anchor, not instead of it (accessory, case, charger, cable, \
mount, warranty, add-on)
I = different category or use case; neither a reasonable alternative nor a \
complement

An accessory for the anchor is always C, never S, even if it names the same \
brand. Answer with one letter only."""


# --------------------------------------------------------------------------
# paths / small IO helpers
# --------------------------------------------------------------------------
def work_paths(work_dir, tag):
    w = Path(work_dir)
    w.mkdir(parents=True, exist_ok=True)
    return {
        "dir": w,
        "anchors": w / f"anchors_{tag}.json",
        "bm25": w / f"bm25_top_{tag}.json",
        "cands": w / f"candidates_{tag}.json",
        "judge": w / f"judge_{tag}.jsonl",
        "judge_meta": w / f"judge_meta_{tag}.json",
    }


def _load_json(path, what):
    p = Path(path)
    if not p.exists():
        raise SystemExit(f"missing {what}: {p} -- run the earlier phase first")
    with open(p) as f:
        return json.load(f)


def load_catalog_fields(path):
    """sku -> {name, manufacturer, class, category_leaf} for the whole catalog."""
    p = Path(path)
    if not p.exists():
        raise SystemExit(
            f"catalog fields not found: {p}\n"
            "Expected the re-index catalog scan output. Regenerate with "
            "eval_bestbuy_category_enrichment_prototype.py --phase catalog "
            "over /tmp/bestbuy_raw_inspect/product_data.tar.gz."
        )
    fields = {}
    with open(p) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            fields[r["sku"]] = {
                "name": r.get("name", ""),
                "manufacturer": r.get("manufacturer", ""),
                "class": r.get("class", ""),
                "category_leaf": r.get("category_leaf", ""),
            }
    return fields


def load_index(data_dir, vecs_name="bod_catalog.vecs.fp16.npy"):
    """(product_ids, titles, mmapped fp16 vectors) for one index snapshot."""
    d = Path(data_dir)
    pids = _load_json(d / "product_ids.json", "product_ids.json")
    tpath = d / "titles.json"
    titles = _load_json(tpath, "titles.json") if tpath.exists() else None
    vecs = np.load(d / vecs_name, mmap_mode="r")
    if len(pids) != vecs.shape[0]:
        raise SystemExit(f"{d}: {len(pids)} ids vs {vecs.shape[0]} vectors")
    return pids, titles, vecs


def cat_suffix(rec, use_cat):
    if not use_cat or rec is None:
        return ""
    bits = [b for b in (rec.get("category_leaf", ""), rec.get("class", "")) if b]
    return f" ({' / '.join(bits)})" if bits else ""


# --------------------------------------------------------------------------
# phase: anchors
# --------------------------------------------------------------------------
def phase_anchors(args, paths):
    pids, titles, _ = load_index(args.new_dir)
    print(f"catalog: {len(pids):,} products", flush=True)
    fields = load_catalog_fields(args.fields_jsonl)
    print(f"catalog fields: {len(fields):,} skus", flush=True)

    by_class = defaultdict(list)
    for i, pid in enumerate(pids):
        rec = fields.get(pid)
        if not rec:
            continue
        cls = (rec.get("class") or "").strip().upper()
        if cls:
            by_class[cls].append(i)

    rng = random.Random(args.seed)
    rows, missing = [], []
    plan = [(c, args.per_hardgood) for c in HARDGOOD_CLASSES]
    plan += [(c, args.per_media) for c in MEDIA_CLASSES]

    for cls, n_want in plan:
        pool = by_class.get(cls, [])
        if not pool:
            missing.append(cls)
            continue
        # Skip degenerate titles (warranties, blank names) as anchors: they
        # have no meaningful substitute structure.
        pool = [i for i in pool if len(titles[i]) >= args.min_title_chars]
        take = rng.sample(pool, min(n_want, len(pool)))
        for i in take:
            rec = fields[pids[i]]
            rows.append(
                {
                    "row": int(i),
                    "product_id": pids[i],
                    "title": titles[i],
                    "class": cls,
                    "manufacturer": rec.get("manufacturer", ""),
                    "category_leaf": rec.get("category_leaf", ""),
                    "stratum": "media" if cls in MEDIA_CLASSES else "hardgood",
                }
            )

    rng.shuffle(rows)
    payload = {
        "n_anchors": len(rows),
        "seed": args.seed,
        "per_hardgood": args.per_hardgood,
        "per_media": args.per_media,
        "classes_requested": len(plan),
        "classes_missing": missing,
        "class_counts": dict(Counter(r["class"] for r in rows)),
        "stratum_counts": dict(Counter(r["stratum"] for r in rows)),
        "anchors": rows,
    }
    with open(paths["anchors"], "w") as f:
        json.dump(payload, f, indent=2)
    print(
        f"sampled {len(rows)} anchors across {len(payload['class_counts'])} classes "
        f"({payload['stratum_counts']}) -> {paths['anchors']}",
        flush=True,
    )
    if missing:
        print(f"  WARNING: no products for classes {missing}", flush=True)


# --------------------------------------------------------------------------
# phase: bm25
# --------------------------------------------------------------------------
def build_bm25(titles, k1, b):
    """Full-catalog BM25. Same conventions as eval_bestbuy_bm25_junk_gate.py."""
    import bm25s
    from Stemmer import Stemmer

    stemmer = Stemmer("english")
    t0 = time.time()
    print(f"  tokenizing {len(titles):,} docs (stem=en, stopwords=en)...", flush=True)
    tok = bm25s.tokenize(titles, stopwords="en", stemmer=stemmer, show_progress=False)
    print(f"  indexing BM25 k1={k1} b={b}...", flush=True)
    idx = bm25s.BM25(k1=k1, b=b)
    idx.index(tok, show_progress=False)
    print(f"  built in {time.time() - t0:.0f}s", flush=True)
    return idx, stemmer


def phase_bm25(args, paths):
    anchors = _load_json(paths["anchors"], "anchors")["anchors"]
    pids, titles, _ = load_index(args.new_dir)

    import bm25s

    idx, stemmer = build_bm25(titles, args.k1, args.b)
    queries = [a["title"] for a in anchors]
    qtok = bm25s.tokenize(queries, stopwords="en", stemmer=stemmer, show_progress=False)
    # +1 so we can drop the anchor's own row and still keep top_k.
    docs, scores = idx.retrieve(qtok, k=args.top_k + 1, show_progress=False)

    out = {}
    for qi, a in enumerate(anchors):
        hits = []
        for j in range(docs.shape[1]):
            r = int(docs[qi, j])
            if r == a["row"]:
                continue
            hits.append({"row": r, "product_id": pids[r], "score": float(scores[qi, j])})
            if len(hits) >= args.top_k:
                break
        out[a["product_id"]] = hits

    payload = {
        "k1": args.k1,
        "b": args.b,
        "tokenizer": "bm25s.tokenize(stopwords='en', stemmer=Stemmer('english'))",
        "n_docs": len(titles),
        "top_k": args.top_k,
        "hits": out,
    }
    with open(paths["bm25"], "w") as f:
        json.dump(payload, f)
    print(f"wrote BM25 top-{args.top_k} for {len(out)} anchors -> {paths['bm25']}", flush=True)


# --------------------------------------------------------------------------
# phase: candidates
# --------------------------------------------------------------------------
def dense_topk(vecs, anchor_rows, top_k, chunk=100_000, exclude_self=True):
    """Brute-force cosine top-k for a handful of anchors over the full catalog.

    Vectors are already L2-normalised, so a dot product is the cosine. Streamed
    in row chunks to keep the fp32 upcast off the peak-memory path.
    """
    a = np.asarray(vecs[anchor_rows], dtype=np.float32)
    n = vecs.shape[0]
    best_s = np.full((len(anchor_rows), 0), 0.0, dtype=np.float32)
    best_i = np.zeros((len(anchor_rows), 0), dtype=np.int64)
    t0 = time.time()
    for start in range(0, n, chunk):
        stop = min(start + chunk, n)
        blk = np.asarray(vecs[start:stop], dtype=np.float32)
        sims = a @ blk.T
        kk = min(top_k + 1, sims.shape[1])
        part = np.argpartition(-sims, kk - 1, axis=1)[:, :kk]
        ps = np.take_along_axis(sims, part, axis=1)
        best_s = np.concatenate([best_s, ps], axis=1)
        best_i = np.concatenate([best_i, part + start], axis=1)
        # Re-trim so the running buffer stays small.
        kk2 = min(top_k + 1, best_s.shape[1])
        sel = np.argpartition(-best_s, kk2 - 1, axis=1)[:, :kk2]
        best_s = np.take_along_axis(best_s, sel, axis=1)
        best_i = np.take_along_axis(best_i, sel, axis=1)
        if (start // chunk) % 4 == 0:
            print(f"    {stop:,}/{n:,} rows  {time.time() - t0:.0f}s", flush=True)

    out = []
    for r, arow in enumerate(anchor_rows):
        order = np.argsort(-best_s[r])
        hits = []
        for j in order:
            row = int(best_i[r, j])
            if exclude_self and row == arow:
                continue
            hits.append((row, float(best_s[r, j])))
            if len(hits) >= top_k:
                break
        out.append(hits)
    return out


def sims_for(vecs, anchor_rows, cand_rows_per_anchor):
    """Cosine of each anchor against its own candidate list."""
    out = []
    for arow, crows in zip(anchor_rows, cand_rows_per_anchor):
        if not crows:
            out.append([])
            continue
        av = np.asarray(vecs[arow], dtype=np.float32)
        cv = np.asarray(vecs[crows], dtype=np.float32)
        out.append([float(x) for x in cv @ av])
    return out


def phase_candidates(args, paths):
    anchors = _load_json(paths["anchors"], "anchors")["anchors"]
    bm25 = _load_json(paths["bm25"], "BM25 hits")["hits"]
    pids, titles, new_v = load_index(args.new_dir)
    old_pids, _, old_v = load_index(args.old_dir)
    if old_pids != pids:
        raise SystemExit("OLD and NEW product_ids differ -- row indices are not comparable")
    fields = load_catalog_fields(args.fields_jsonl)

    rows = [a["row"] for a in anchors]
    print(f"dense top-{args.top_k} under NEW ({new_v.shape})...", flush=True)
    new_hits = dense_topk(new_v, rows, args.top_k, chunk=args.chunk)
    print(f"dense top-{args.top_k} under OLD ({old_v.shape})...", flush=True)
    old_hits = dense_topk(old_v, rows, args.top_k, chunk=args.chunk)

    out_rows, src_counter = [], Counter()
    for ai, a in enumerate(anchors):
        srcs = defaultdict(set)
        for r, _ in old_hits[ai]:
            srcs[r].add("dense_old")
        for r, _ in new_hits[ai]:
            srcs[r].add("dense_new")
        for h in bm25.get(a["product_id"], []):
            srcs[int(h["row"])].add("bm25")
        srcs.pop(a["row"], None)
        crows = sorted(srcs)
        for key in ("dense_old", "dense_new", "bm25"):
            src_counter[key] += sum(1 for r in crows if key in srcs[r])
        src_counter["union"] += len(crows)

        s_new = sims_for(new_v, [a["row"]], [crows])[0]
        s_old = sims_for(old_v, [a["row"]], [crows])[0]
        bm_score = {int(h["row"]): float(h["score"]) for h in bm25.get(a["product_id"], [])}
        cands = []
        for j, r in enumerate(crows):
            rec = fields.get(pids[r], {})
            cands.append(
                {
                    "row": r,
                    "product_id": pids[r],
                    "title": titles[r],
                    "class": (rec.get("class") or "").strip().upper(),
                    "manufacturer": rec.get("manufacturer", ""),
                    "category_leaf": rec.get("category_leaf", ""),
                    "sources": sorted(srcs[r]),
                    "sim_old": s_old[j],
                    "sim_new": s_new[j],
                    "bm25_score": bm_score.get(r),
                }
            )
        cands.sort(key=lambda c: -c["sim_new"])
        out_rows.append({**a, "candidates": cands})

    n_pairs = sum(len(r["candidates"]) for r in out_rows)
    payload = {
        "n_anchors": len(out_rows),
        "top_k": args.top_k,
        "n_pairs": n_pairs,
        "mean_candidates_per_anchor": n_pairs / max(len(out_rows), 1),
        "source_counts": dict(src_counter),
        "old_dir": str(args.old_dir),
        "new_dir": str(args.new_dir),
        "rows": out_rows,
    }
    with open(paths["cands"], "w") as f:
        json.dump(payload, f)
    print(
        f"{n_pairs:,} unique (anchor, candidate) pairs over {len(out_rows)} anchors "
        f"(mean {payload['mean_candidates_per_anchor']:.1f}) -> {paths['cands']}",
        flush=True,
    )
    print(f"  source coverage: {dict(src_counter)}", flush=True)


# --------------------------------------------------------------------------
# phase: estimate
# --------------------------------------------------------------------------
def pair_key(anchor_id, cand_id):
    return f"{anchor_id}\t{cand_id}"


def build_prompts(args, cands):
    """[(key, {...prompt...})] for every (anchor, candidate) pair."""
    items = []
    for r in cands["rows"]:
        a_title = r["title"][: args.max_title_chars]
        a_cat = cat_suffix(r, args.with_category)
        for c in r["candidates"]:
            prompt = JUDGE_PROMPT.format(
                a_title=a_title,
                a_cat=a_cat,
                c_title=c["title"][: args.max_title_chars],
                c_cat=cat_suffix(c, args.with_category),
            )
            items.append(
                (
                    pair_key(r["product_id"], c["product_id"]),
                    {
                        "anchor_id": r["product_id"],
                        "candidate_id": c["product_id"],
                        "prompt": prompt,
                    },
                )
            )
    return items


def phase_estimate(args, paths, quiet=False):
    cands = _load_json(paths["cands"], "candidates")
    items = build_prompts(args, cands)
    # ~4 chars/token, the same crude ratio the other judge scripts use.
    tin = sum(len(p["prompt"]) for _, p in items) / 4.0
    tout = len(items) * 1.0
    cost = estimate_cost(args.model, tin, tout)
    breakdown = {
        "model": args.model,
        "n_anchors": cands["n_anchors"],
        "n_pairs": len(items),
        "judge_calls": len(items),
        "est_tokens_in": int(tin),
        "est_tokens_out": int(tout),
        "est_cost_usd_total": cost,
        "ceiling_usd": COST_CEILING_USD,
    }
    if not quiet:
        print(json.dumps(breakdown, indent=2), flush=True)
        print(
            f"\nPROJECTED COST: ${cost:.4f} ({len(items):,} calls) "
            f"vs ceiling ${COST_CEILING_USD:.2f}",
            flush=True,
        )
        if cost > COST_CEILING_USD:
            print("OVER CEILING -- the judge phase will refuse to run.", flush=True)
    return breakdown


def _guard_cost(args, paths):
    est = phase_estimate(args, paths, quiet=True)
    c = est["est_cost_usd_total"]
    print(f"[cost guard] projected ${c:.4f} (ceiling ${COST_CEILING_USD:.2f})", flush=True)
    if c > COST_CEILING_USD:
        raise SystemExit(
            f"Refusing to run: projected ${c:.4f} exceeds ceiling ${COST_CEILING_USD:.2f}. "
            f"Lower --per-hardgood/--top-k or raise --cost-ceiling deliberately."
        )
    return est


# --------------------------------------------------------------------------
# phase: judge
# --------------------------------------------------------------------------
def _label_from_choice(choice):
    """(label, p_label) from the first-token distribution over E/S/C/I."""
    if choice is None:
        return None, float("nan")
    txt = (choice.message.content or "").strip().upper()
    label = next((ch for ch in txt if ch in LABELS), None)
    p = float("nan")
    lp = getattr(choice, "logprobs", None)
    if lp and lp.content:
        top = lp.content[0].top_logprobs
        if top:
            mass = {}
            for t in top:
                tk = (t.token or "").strip().upper()
                if tk and tk[0] in LABELS:
                    mass[tk[0]] = mass.get(tk[0], 0.0) + math.exp(t.logprob)
            if mass:
                if label is None:
                    label = max(mass, key=mass.get)
                tot = sum(mass.values())
                p = mass.get(label, 0.0) / tot if tot > 0 else float("nan")
    return label, p


def _load_judged(path):
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
            except json.JSONDecodeError:  # truncated final line from a kill
                continue
            if r.get("label") in LABELS:
                done[pair_key(r["anchor_id"], r["candidate_id"])] = r
    return done


async def _run_judge(args, todo, usage, out_f):
    client = make_client()
    sem = asyncio.Semaphore(args.concurrency)
    done_n = [0]

    async def one(item):
        key, p = item
        ch = await _chat(client, sem, usage, args.model, p["prompt"], 4, logprobs=True)
        try:
            label, conf = _label_from_choice(ch)
        except Exception:  # never let one odd token kill a paid run
            usage.errors += 1
            label, conf = None, float("nan")
        if label is None:
            usage.errors += 1
            return
        rec = {
            "anchor_id": p["anchor_id"],
            "candidate_id": p["candidate_id"],
            "label": label,
            "p_label": None if math.isnan(conf) else round(conf, 4),
        }
        out_f.write(json.dumps(rec) + "\n")
        done_n[0] += 1
        if done_n[0] % 250 == 0:
            out_f.flush()
            print(f"    {done_n[0]:,}/{len(todo):,} judged", flush=True)

    await asyncio.gather(*(one(it) for it in todo))
    out_f.flush()


def phase_judge(args, paths):
    _guard_cost(args, paths)
    cands = _load_json(paths["cands"], "candidates")
    items = build_prompts(args, cands)
    done = _load_judged(paths["judge"]) if args.resume else {}
    todo = [(k, p) for k, p in items if k not in done]
    print(f"{len(items):,} pairs, {len(done):,} cached, {len(todo):,} to judge", flush=True)
    if not todo:
        print("nothing to do", flush=True)
        return

    usage = Usage()
    t0 = time.time()
    mode = "a" if (args.resume and Path(paths["judge"]).exists()) else "w"
    with open(paths["judge"], mode) as f:
        asyncio.run(_run_judge(args, todo, usage, f))
    cost = estimate_cost(args.model, usage.tin, usage.tout)
    record_spend(args.model, usage.tin, usage.tout, cost, "bestbuy substitutability benchmark")
    meta = {
        "model": args.model,
        "n_pairs_total": len(items),
        "n_judged_now": usage.calls,
        "errors": usage.errors,
        "tokens_in": usage.tin,
        "tokens_out": usage.tout,
        "cost_usd": cost,
        "seconds": time.time() - t0,
    }
    with open(paths["judge_meta"], "w") as f:
        json.dump(meta, f, indent=2)
    print(json.dumps(meta, indent=2), flush=True)


# --------------------------------------------------------------------------
# phase: eval -- concordance
# --------------------------------------------------------------------------
def _concordance(sims, labels, group_a, group_b):
    """(concordant, total) for every (a in group_a, b in group_b) pair.

    A tie in similarity counts 0.5 -- exact-duplicate rows in this catalog do
    produce exact ties, and calling those a win would flatter the index.
    """
    ia = [i for i, lb in enumerate(labels) if lb in group_a]
    ib = [i for i, lb in enumerate(labels) if lb in group_b]
    if not ia or not ib:
        return 0.0, 0
    conc = 0.0
    for i in ia:
        for j in ib:
            if sims[i] > sims[j]:
                conc += 1.0
            elif sims[i] == sims[j]:
                conc += 0.5
    return conc, len(ia) * len(ib)


COMPARISONS = (
    ("E_gt_S", ("E",), ("S",)),
    ("E_gt_CI", ("E",), ("C", "I")),
    ("S_gt_CI", ("S",), ("C", "I")),
)


def per_anchor_concordance(rows, labels_by_pair, sim_key):
    """{comparison: {anchor_id: rate}} plus pooled pair counts."""
    per = {name: {} for name, _, _ in COMPARISONS}
    pooled = {name: [0.0, 0] for name, _, _ in COMPARISONS}
    for r in rows:
        cands = [
            c
            for c in r["candidates"]
            if pair_key(r["product_id"], c["product_id"]) in labels_by_pair
        ]
        if len(cands) < 2:
            continue
        sims = [c[sim_key] for c in cands]
        labels = [labels_by_pair[pair_key(r["product_id"], c["product_id"])] for c in cands]
        for name, ga, gb in COMPARISONS:
            conc, tot = _concordance(sims, labels, ga, gb)
            if tot:
                per[name][r["product_id"]] = conc / tot
                pooled[name][0] += conc
                pooled[name][1] += tot
    return per, pooled


def paired_bootstrap_delta(per_old, per_new, n_boot, seed):
    """CI on macro(NEW) - macro(OLD), resampling anchors (not pairs)."""
    keys = sorted(set(per_old) & set(per_new))
    if len(keys) < 2:
        return None
    o = np.array([per_old[k] for k in keys], dtype=np.float64)
    n = np.array([per_new[k] for k in keys], dtype=np.float64)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(keys), size=(n_boot, len(keys)))
    d = n[idx].mean(axis=1) - o[idx].mean(axis=1)
    return {
        "delta": float(n.mean() - o.mean()),
        "ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
        "p_gt_0": float((d > 0).mean()),
        "n_anchors": len(keys),
    }


# --------------------------------------------------------------------------
# phase: eval -- lexical bias
# --------------------------------------------------------------------------
_BRAND_STOP = {"inc", "co", "corp", "ltd", "llc", "the", "brand", "records", "electronics"}


def brand_tokens(rec):
    m = (rec.get("manufacturer") or "").strip()
    return {t for t in toks(m) if len(t) > 2 and t not in _BRAND_STOP}


def shared_brand(anchor, cand):
    """Does the candidate name/brand carry the anchor's brand token?"""
    ab = brand_tokens(anchor)
    if not ab:
        return False
    return bool(ab & (brand_tokens(cand) | set(toks(cand["title"]))))


def zscore(vals):
    v = np.asarray(vals, dtype=np.float64)
    sd = v.std()
    return (v - v.mean()) / sd if sd > 0 else np.zeros_like(v)


def bias_analysis(rows, labels_by_pair, args):
    """Per-pair lexical-overlap features x label x within-anchor sim rank."""
    recs = []
    for r in rows:
        cands = [
            c
            for c in r["candidates"]
            if pair_key(r["product_id"], c["product_id"]) in labels_by_pair
        ]
        if len(cands) < 2:
            continue
        for sim_key in ("sim_old", "sim_new"):
            z = zscore([c[sim_key] for c in cands])
            order = np.argsort(-np.asarray([c[sim_key] for c in cands]))
            rank = np.empty(len(cands), dtype=np.float64)
            rank[order] = np.arange(len(cands), dtype=np.float64)
            for j, c in enumerate(cands):
                c[f"_z_{sim_key}"] = float(z[j])
                # 1.0 = closest neighbour, 0.0 = furthest in the candidate set
                c[f"_nrank_{sim_key}"] = 1.0 - rank[j] / (len(cands) - 1) if len(cands) > 1 else 1.0
        for c in cands:
            cov, jac = overlap_metrics(r["title"], c["title"])
            lab = labels_by_pair[pair_key(r["product_id"], c["product_id"])]
            recs.append(
                {
                    "anchor_id": r["product_id"],
                    "candidate_id": c["product_id"],
                    "label": lab,
                    "gain": LABEL_GAIN[lab],
                    "valid_sub": lab in ("E", "S"),
                    "coverage": cov,
                    "jaccard": jac,
                    "shared_brand": shared_brand(r, c),
                    "same_class": bool(r["class"] and r["class"] == c["class"]),
                    "sim_old": c["sim_old"],
                    "sim_new": c["sim_new"],
                    "z_old": c["_z_sim_old"],
                    "z_new": c["_z_sim_new"],
                    "nrank_old": c["_nrank_sim_old"],
                    "nrank_new": c["_nrank_sim_new"],
                }
            )

    def grp(pred):
        return [x for x in recs if pred(x)]

    out = {"n_pairs": len(recs), "label_counts": dict(Counter(x["label"] for x in recs))}

    # 1. does raw similarity track the label at all?
    for tag in ("old", "new"):
        out[f"spearman_sim_vs_gain_{tag}"] = spearman(
            [x[f"z_{tag}"] for x in recs], [x["gain"] for x in recs]
        )
        out[f"pearson_coverage_vs_z_{tag}"] = pearson(
            [x["coverage"] for x in recs], [x[f"z_{tag}"] for x in recs]
        )
    out["spearman_coverage_vs_gain"] = spearman(
        [x["coverage"] for x in recs], [x["gain"] for x in recs]
    )

    # 2. THE diagnostic: non-substitutes (C/I) that look lexically like the
    #    anchor. If the index is clean their similarity should not be
    #    inflated; if the "apple tablet" pathology is structural it is.
    hi = lambda x: x["shared_brand"] or x["coverage"] >= args.high_overlap  # noqa: E731
    buckets = {
        "CI_high_overlap": grp(lambda x: not x["valid_sub"] and hi(x)),
        "CI_low_overlap": grp(lambda x: not x["valid_sub"] and not hi(x)),
        "S_high_overlap": grp(lambda x: x["label"] == "S" and hi(x)),
        "S_low_overlap": grp(lambda x: x["label"] == "S" and not hi(x)),
        "E_all": grp(lambda x: x["label"] == "E"),
    }
    out["buckets"] = {}
    for name, g in buckets.items():
        if not g:
            out["buckets"][name] = {"n": 0}
            continue
        out["buckets"][name] = {
            "n": len(g),
            "mean_coverage": float(np.mean([x["coverage"] for x in g])),
            "mean_z_old": float(np.mean([x["z_old"] for x in g])),
            "mean_z_new": float(np.mean([x["z_new"] for x in g])),
            "mean_z_old_ci95": bootstrap_ci([x["z_old"] for x in g], args.n_boot, args.seed),
            "mean_z_new_ci95": bootstrap_ci([x["z_new"] for x in g], args.n_boot, args.seed),
            "mean_nrank_old": float(np.mean([x["nrank_old"] for x in g])),
            "mean_nrank_new": float(np.mean([x["nrank_new"] for x in g])),
        }

    # 3. the pathology as a head-to-head: within an anchor, how often does a
    #    high-overlap non-substitute outrank a genuine substitute?
    by_anchor = defaultdict(list)
    for x in recs:
        by_anchor[x["anchor_id"]].append(x)
    for tag in ("old", "new"):
        for bucket, pred in (
            ("high_overlap", hi),
            ("low_overlap", lambda x: not hi(x)),
        ):
            wins, tot, anchors_seen = 0.0, 0, 0
            for _aid, g in by_anchor.items():
                subs = [x for x in g if x["label"] == "S"]
                junk = [x for x in g if not x["valid_sub"] and pred(x)]
                if not subs or not junk:
                    continue
                anchors_seen += 1
                for s in subs:
                    for j in junk:
                        tot += 1
                        if j[f"sim_{tag}"] > s[f"sim_{tag}"]:
                            wins += 1
                        elif j[f"sim_{tag}"] == s[f"sim_{tag}"]:
                            wins += 0.5
            out[f"nonsub_beats_substitute_{bucket}_{tag}"] = {
                "rate": (wins / tot) if tot else None,
                "n_pairs": tot,
                "n_anchors": anchors_seen,
            }

    # 3b. same head-to-head restricted to anchors that supply BOTH a
    #     high-overlap and a low-overlap non-substitute. The unmatched version
    #     above compares two different anchor sets, so the high-vs-low gap
    #     there is partly a difference between anchors; this one is not.
    for tag in ("old", "new"):
        acc = {"high_overlap": [0.0, 0], "low_overlap": [0.0, 0]}
        anchors_seen = 0
        for _aid, g in by_anchor.items():
            subs = [x for x in g if x["label"] == "S"]
            junk_hi = [x for x in g if not x["valid_sub"] and hi(x)]
            junk_lo = [x for x in g if not x["valid_sub"] and not hi(x)]
            if not subs or not junk_hi or not junk_lo:
                continue
            anchors_seen += 1
            for bucket, junk in (("high_overlap", junk_hi), ("low_overlap", junk_lo)):
                for s in subs:
                    for j in junk:
                        acc[bucket][1] += 1
                        if j[f"sim_{tag}"] > s[f"sim_{tag}"]:
                            acc[bucket][0] += 1.0
                        elif j[f"sim_{tag}"] == s[f"sim_{tag}"]:
                            acc[bucket][0] += 0.5
        out[f"nonsub_beats_substitute_matched_{tag}"] = {
            "n_anchors": anchors_seen,
            **{
                b: {"rate": (v[0] / v[1]) if v[1] else None, "n_pairs": v[1]}
                for b, v in acc.items()
            },
            "gap_high_minus_low": (
                acc["high_overlap"][0] / acc["high_overlap"][1]
                - acc["low_overlap"][0] / acc["low_overlap"][1]
            )
            if acc["high_overlap"][1] and acc["low_overlap"][1]
            else None,
        }

    # 4. partial view: within C/I only, does overlap still predict similarity?
    ci = [x for x in recs if not x["valid_sub"]]
    for tag in ("old", "new"):
        out[f"within_CI_pearson_coverage_vs_z_{tag}"] = pearson(
            [x["coverage"] for x in ci], [x[f"z_{tag}"] for x in ci]
        )
    return out, recs


# --------------------------------------------------------------------------
# phase: eval
# --------------------------------------------------------------------------
def phase_eval(args, paths):
    cands = _load_json(paths["cands"], "candidates")
    judged = _load_judged(paths["judge"])
    if not judged:
        raise SystemExit(f"no judgements in {paths['judge']} -- run --phase judge")
    labels_by_pair = {k: v["label"] for k, v in judged.items()}
    print(f"{len(labels_by_pair):,} judged pairs, {cands['n_anchors']} anchors", flush=True)

    rows = cands["rows"]
    results = {}
    per = {}
    for tag, sim_key in (("old", "sim_old"), ("new", "sim_new")):
        per[tag], pooled = per_anchor_concordance(rows, labels_by_pair, sim_key)
        block = {}
        for name, _, _ in COMPARISONS:
            vals = list(per[tag][name].values())
            block[name] = {
                "n_anchors_qualifying": len(vals),
                "macro_mean": float(np.mean(vals)) if vals else None,
                "macro_ci95": bootstrap_ci(vals, args.n_boot, args.seed),
                "micro_pooled": (pooled[name][0] / pooled[name][1]) if pooled[name][1] else None,
                "n_pairs": pooled[name][1],
            }
        results[tag] = block

    results["delta_new_minus_old"] = {
        name: paired_bootstrap_delta(per["old"][name], per["new"][name], args.n_boot, args.seed)
        for name, _, _ in COMPARISONS
    }

    bias, pair_recs = bias_analysis(rows, labels_by_pair, args)

    # label mix by candidate source -- shows what each generator contributes
    src_labels = defaultdict(Counter)
    for r in rows:
        for c in r["candidates"]:
            lab = labels_by_pair.get(pair_key(r["product_id"], c["product_id"]))
            if not lab:
                continue
            for s in c["sources"]:
                src_labels[s][lab] += 1
            src_labels["|".join(c["sources"])][lab] += 1
    source_label_mix = {k: dict(v) for k, v in sorted(src_labels.items())}

    # raw judged pairs, so a follow-up audit can sample without recomputing
    raw = []
    for r in rows:
        for c in r["candidates"]:
            key = pair_key(r["product_id"], c["product_id"])
            j = judged.get(key)
            if not j:
                continue
            raw.append(
                {
                    "anchor_id": r["product_id"],
                    "anchor_title": r["title"],
                    "anchor_class": r["class"],
                    "anchor_manufacturer": r["manufacturer"],
                    "candidate_id": c["product_id"],
                    "candidate_title": c["title"],
                    "candidate_class": c["class"],
                    "candidate_manufacturer": c["manufacturer"],
                    "label": j["label"],
                    "p_label": j.get("p_label"),
                    "sources": c["sources"],
                    "sim_old": c["sim_old"],
                    "sim_new": c["sim_new"],
                    "bm25_score": c["bm25_score"],
                }
            )
    cov = {p["anchor_id"] + "\t" + p["candidate_id"]: p for p in pair_recs}
    for p in raw:
        f = cov.get(p["anchor_id"] + "\t" + p["candidate_id"])
        if f:
            p["coverage"] = round(f["coverage"], 4)
            p["jaccard"] = round(f["jaccard"], 4)
            p["shared_brand"] = f["shared_brand"]
            p["same_class"] = f["same_class"]

    anchors_meta = _load_json(paths["anchors"], "anchors")
    jmeta = {}
    if Path(paths["judge_meta"]).exists():
        jmeta = _load_json(paths["judge_meta"], "judge meta")

    out = {
        "benchmark": "bestbuy product-product substitutability geometry (E>S>{C,I})",
        "config": {
            "judge_model": args.model,
            "old_dir": cands["old_dir"],
            "new_dir": cands["new_dir"],
            "top_k_per_source": cands["top_k"],
            "candidate_sources": ["dense_old", "dense_new", "bm25"],
            "bm25": _load_json(paths["bm25"], "bm25")["tokenizer"],
            "high_overlap_coverage_threshold": args.high_overlap,
            "n_boot": args.n_boot,
            "seed": args.seed,
            "with_category_in_prompt": args.with_category,
        },
        "anchors": {
            "n": anchors_meta["n_anchors"],
            "class_counts": anchors_meta["class_counts"],
            "stratum_counts": anchors_meta["stratum_counts"],
        },
        "candidates": {
            "n_pairs": cands["n_pairs"],
            "mean_per_anchor": cands["mean_candidates_per_anchor"],
            "source_counts": cands["source_counts"],
            "n_judged": len(judged),
        },
        "judge_cost": jmeta,
        "label_distribution": dict(Counter(v for v in labels_by_pair.values())),
        "source_label_mix": source_label_mix,
        "concordance": results,
        "lexical_bias": bias,
        "raw_pairs": raw,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)

    print("\n=== concordance (macro over qualifying anchors) ===", flush=True)
    for name, _, _ in COMPARISONS:
        o, n = results["old"][name], results["new"][name]
        d = results["delta_new_minus_old"][name]
        print(
            f"  {name:<9} OLD {o['macro_mean']:.4f} {o['macro_ci95']}  "
            f"NEW {n['macro_mean']:.4f} {n['macro_ci95']}  "
            f"delta {d['delta']:+.4f} {d['ci95']}  "
            f"({n['n_anchors_qualifying']} anchors, {n['n_pairs']:,} pairs)",
            flush=True,
        )
    print("\n=== lexical bias ===", flush=True)
    for k in ("CI_high_overlap", "CI_low_overlap", "S_high_overlap", "S_low_overlap", "E_all"):
        b = bias["buckets"][k]
        if not b.get("n"):
            continue
        print(
            f"  {k:<17} n={b['n']:<5} mean z(sim) OLD {b['mean_z_old']:+.3f} "
            f"NEW {b['mean_z_new']:+.3f}  nrank NEW {b['mean_nrank_new']:.3f}",
            flush=True,
        )
    for tag in ("old", "new"):
        for bucket in ("high_overlap", "low_overlap"):
            v = bias[f"nonsub_beats_substitute_{bucket}_{tag}"]
            if v["rate"] is not None:
                print(
                    f"  non-substitute ({bucket}) outranks a true substitute, {tag.upper()}: "
                    f"{v['rate']:.3f}  ({v['n_pairs']:,} pairs, {v['n_anchors']} anchors)",
                    flush=True,
                )
    for tag in ("old", "new"):
        m = bias[f"nonsub_beats_substitute_matched_{tag}"]
        if m["gap_high_minus_low"] is not None:
            print(
                f"  [matched anchors, n={m['n_anchors']}] {tag.upper()} high {m['high_overlap']['rate']:.3f} "
                f"vs low {m['low_overlap']['rate']:.3f}  gap {m['gap_high_minus_low']:+.3f}",
                flush=True,
            )
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
        choices=["anchors", "bm25", "candidates", "estimate", "judge", "eval"],
    )
    ap.add_argument("--new-dir", default=DEFAULT_NEW_DIR, help="re-indexed artifacts")
    ap.add_argument("--old-dir", default=str(DEFAULT_OLD_DIR), help="pre-re-index HF snapshot")
    ap.add_argument("--fields-jsonl", default=DEFAULT_FIELDS_JSONL)
    ap.add_argument("--per-hardgood", type=int, default=8, help="anchors per HardGood class")
    ap.add_argument("--per-media", type=int, default=4, help="anchors per media class")
    ap.add_argument("--min-title-chars", type=int, default=12)
    ap.add_argument("--top-k", type=int, default=TOP_K, help="per candidate source")
    ap.add_argument("--chunk", type=int, default=100_000, help="dense scan block size")
    ap.add_argument("--k1", type=float, default=1.5)
    ap.add_argument("--b", type=float, default=0.75)
    ap.add_argument("--model", default="gpt-4o-mini", help="OpenAI judge model")
    ap.add_argument("--concurrency", type=int, default=32)
    ap.add_argument("--max-title-chars", type=int, default=200)
    ap.add_argument("--with-category", action="store_true", default=True)
    ap.add_argument("--no-category", dest="with_category", action="store_false")
    ap.add_argument(
        "--cost-ceiling",
        type=float,
        default=COST_CEILING_USD,
        help="refuse to start the judge phase if projected above this (USD)",
    )
    ap.add_argument(
        "--high-overlap",
        type=float,
        default=0.5,
        help="anchor-token coverage at/above which a candidate counts as lexically similar",
    )
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", action="store_true", default=True)
    ap.add_argument("--no-resume", dest="resume", action="store_false")
    ap.add_argument("--work-dir", default="/tmp/bestbuy_substitutability")
    ap.add_argument("--tag", default="bestbuy")
    ap.add_argument("--out", default="evaluation/results/bestbuy_substitutability_benchmark.json")
    args = ap.parse_args()

    COST_CEILING_USD = args.cost_ceiling

    paths = work_paths(args.work_dir, args.tag)
    {
        "anchors": lambda: phase_anchors(args, paths),
        "bm25": lambda: phase_bm25(args, paths),
        "candidates": lambda: phase_candidates(args, paths),
        "estimate": lambda: phase_estimate(args, paths),
        "judge": lambda: phase_judge(args, paths),
        "eval": lambda: phase_eval(args, paths),
    }[args.phase]()


if __name__ == "__main__":
    main()
