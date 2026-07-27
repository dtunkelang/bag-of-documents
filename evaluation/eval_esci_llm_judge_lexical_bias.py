#!/usr/bin/env python3
"""Is an LLM relevance judge biased toward literal query-token overlap,
and does paraphrasing the query before judging debias it?

Motivation
----------
arXiv:2501.17969 (Alaofi et al., "LLMs can be Fooled into Labelling a Document
as Relevant", SIGIR-AP '24) reports that LLM relevance judges over-reward
literal query-token overlap between query and document. ESCI is the right
corpus to test that on because its labels are graded and *human*:

    Exact (E)      = 3   the product is what the query asked for
    Substitute (S) = 2   somewhat relevant, but NOT what was asked for
    Complement (C) = 1   does not fulfil the query, but goes with it
    Irrelevant (I) = 0

"Substitute" is exactly the adversarial case the paper describes: an S item is
routinely *lexically* very close to the query while being the wrong answer
("iphone 13 case" -> an iphone 12 case). If the judge is lexically biased, the
S items it wrongly scores like E items should be the high-overlap S items.

Why not BestBuy: its click-derived qrels are binary and average ~2 golds per
query among near-identical SKU colorways, so there is no E-vs-S contrast to
measure. See `evaluation/eval_bestbuy_llm_judge_rerank.py`.

Judge model
-----------
gpt-4o-mini via the OpenAI API. The Alaofi paper only validated human-level
judgment agreement for GPT-4o/GPT-4-class models, so a local 7B-4bit judge is
below the tier the finding was established at (an earlier MLX Qwen2.5-7B
attempt was also ~30s/query). API auth/cost-logging follow
`evaluation/llm_relevance_judge.py`: load_dotenv(override=True), AsyncOpenAI,
roll-up record appended to .api_spend.jsonl.

Design
------
Pool = ESCI's own human-judged candidate set for each query. This deliberately
differs from `eval_bestbuy_llm_judge_rerank.py`, which reranks a BoD-retrieve
top-N pool: here we need *every* candidate to carry a graded human label, and
a retrieval pool would be mostly unjudged. So there is no first-stage retriever
in this experiment; the "baseline order" for the ranking metric is BM25 over
the judged set (a pure-lexical reference point) plus the ESCI file order.

Scoring is *logprob-based*, not text-parsed: one call per (query, doc) with
max_tokens=1, logprobs=True, top_logprobs=20. Two scorers off the same
mechanism:

    graded  (PRIMARY)  expected value of a 0-3 relevance rating under the
                       model's first-token distribution:  sum_d d*p(d)/sum p(d).
                       Continuous, tie-free, and -- unlike yes/no -- not
                       saturated, so within-label variation is measurable.
    yesno   (CHECK)    log p("yes") - log p("no") over case/space token
                       variants, the scorer used by the BestBuy run. Retained
                       as a robustness check because gpt-4o-mini drives it to
                       ~p=1.0 on easy pairs, which compresses the E band and
                       could by itself manufacture a correlation.

Reporting both guards the headline against "your result is an artifact of a
saturated binary scorer". Probability mass for a level absent from the top-20
is floored at exp(min observed top-20 logprob), a valid upper bound.

Two conditions, same pairs, same judge:
    literal      judge sees the original ESCI query string
    paraphrased  judge sees a gpt-4o-mini paraphrase of that query

Both conditions are correlated against the *literal* query's lexical overlap
with the document -- that is the apples-to-apples debiasing test. (The
paraphrase's own overlap is also reported.)

Phases (cached to --work-dir):
    --phase data        build esci_us_data/ test-split files from HF parquets
    --phase sample      choose the query sample + candidate sets
    --phase estimate    print projected API cost; spends nothing
    --phase paraphrase  one paraphrase per sampled query (gpt-4o-mini)
    --phase judge       graded + yesno scores, literal + paraphrased
    --phase eval        correlations, discrimination, nDCG -> results JSON

Usage (this repo's .venv lacks numpy; use uv):
    RUN="uv run --no-project --with numpy --with openai --with python-dotenv python"
    $RUN evaluation/eval_esci_llm_judge_lexical_bias.py --phase sample --sample 250
    $RUN evaluation/eval_esci_llm_judge_lexical_bias.py --phase estimate
    $RUN evaluation/eval_esci_llm_judge_lexical_bias.py --phase paraphrase
    $RUN evaluation/eval_esci_llm_judge_lexical_bias.py --phase judge
    $RUN evaluation/eval_esci_llm_judge_lexical_bias.py --phase eval
"""

import argparse
import asyncio
import datetime
import json
import math
import os
import random
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np  # noqa: E402
from dotenv import load_dotenv  # noqa: E402

load_dotenv(override=True)

K_EVAL = 10

# $ per 1M tokens. Mirrors evaluation/llm_relevance_judge.py.
PRICES_PER_M_TOKENS = {
    "gpt-4o-mini": {"in": 0.15, "out": 0.60},
    "gpt-4o": {"in": 2.50, "out": 10.00},
    "gpt-4.1-mini": {"in": 0.40, "out": 1.60},
    "gpt-4.1": {"in": 2.00, "out": 8.00},
}
SPEND_LEDGER = Path(__file__).resolve().parent.parent / ".api_spend.jsonl"

# Hard stop: refuse to start a paid phase whose projection exceeds this.
COST_CEILING_USD = 5.0

GRADE_LEVELS = (0, 1, 2, 3)

# ESCI grade <-> letter. Matches download/download_esci_us.py.
GRADE_LETTER = {3: "E", 2: "S", 1: "C", 0: "I"}
LABEL_GRADE = {"Exact": 3, "Substitute": 2, "Complement": 1, "Irrelevant": 0}
LETTERS = ("E", "S", "C", "I")

# Verbatim from evaluation/eval_bestbuy_llm_judge_rerank.py so the judge
# mechanics are identical across the two experiments.
POINTWISE_PROMPT = """Search query: {query}
Product: {title}

Is this product a relevant result for that search query? Answer yes or no."""

# Generic graded relevance rubric. Deliberately NOT ESCI's E/S/C/I wording:
# naming "substitute" would hand the judge the answer key for the exact
# contrast under test and understate the bias.
GRADED_PROMPT = """Search query: {query}
Product: {title}

On a scale of 0 to 3, how relevant is this product to the search query?
3 = the product is exactly what the query asks for
2 = the product is highly relevant but not exactly what was asked for
1 = the product is only marginally relevant
0 = the product is not relevant

Answer with a single digit (0, 1, 2, or 3)."""

PARAPHRASE_PROMPT = """Rewrite this online shopping search query so it means \
exactly the same thing but uses different words.

Rules:
- Keep it a short search query, not a sentence.
- Replace words with synonyms wherever a synonym exists.
- Do not add, drop, or loosen any requirement (brand, model, size, colour, \
quantity, negations).
- Output only the rewritten query, nothing else.

Query: {query}

Rewritten query:"""

# Minimal stoplist: the bias under test is about *content* token overlap, and
# leaving function words in mostly adds noise to short product queries.
STOP = {
    "a", "an", "and", "the", "for", "of", "with", "in", "on", "to", "or",
    "by", "at", "from", "is", "are", "be", "my", "your", "it", "that", "this",
}

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def toks(s, drop_stop=True):
    t = _TOKEN_RE.findall((s or "").lower())
    return [w for w in t if not (drop_stop and w in STOP)]


def overlap_metrics(query, title):
    """Lexical overlap between a query and a document title.

    coverage : |q & d| / |q|   fraction of query content tokens literally
                               present in the title. This is the quantity the
                               Alaofi paper's attack manipulates, so it is the
                               primary metric here.
    jaccard  : |q & d| / |q | d|
    """
    q = set(toks(query))
    d = set(toks(title))
    if not q:
        return 0.0, 0.0
    inter = len(q & d)
    return inter / len(q), inter / max(len(q | d), 1)


# --------------------------------------------------------------------------
# stats helpers (no scipy dependency -- the mlx venv is deliberately thin)
# --------------------------------------------------------------------------
def pearson(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.size < 3:
        return float("nan")
    xs, ys = x - x.mean(), y - y.mean()
    den = math.sqrt(float((xs * xs).sum()) * float((ys * ys).sum()))
    return float((xs * ys).sum() / den) if den > 0 else float("nan")


def rankdata(a):
    """Average-tie ranks, mirroring scipy.stats.rankdata(method='average')."""
    a = np.asarray(a, dtype=np.float64)
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(a.size, dtype=np.float64)
    sa = a[order]
    i = 0
    while i < a.size:
        j = i
        while j + 1 < a.size and sa[j + 1] == sa[i]:
            j += 1
        ranks[order[i : j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return ranks


def spearman(x, y):
    if len(x) < 3:
        return float("nan")
    return pearson(rankdata(x), rankdata(y))


def bootstrap_ci(values, n_boot=2000, seed=0, stat=np.mean):
    v = np.asarray([x for x in values if x is not None and np.isfinite(x)], dtype=np.float64)
    if v.size < 2:
        return None
    rng = np.random.default_rng(seed)
    draws = stat(v[rng.integers(0, v.size, size=(n_boot, v.size))], axis=1)
    return [float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))]


def ndcg_at_k(grades_in_rank_order, all_grades, k=K_EVAL):
    """Standard nDCG@k with linear gains (gain = ESCI grade 3/2/1/0)."""
    g = list(grades_in_rank_order)[:k]
    dcg = sum(gr / math.log2(i + 2) for i, gr in enumerate(g))
    ideal = sorted(all_grades, reverse=True)[:k]
    idcg = sum(gr / math.log2(i + 2) for i, gr in enumerate(ideal))
    return dcg / idcg if idcg > 0 else float("nan")


# --------------------------------------------------------------------------
# paths
# --------------------------------------------------------------------------
SCORERS = ("graded", "yesno")
CONDITIONS = ("literal", "paraphrased")


def work_paths(work_dir, tag):
    w = Path(work_dir)
    w.mkdir(parents=True, exist_ok=True)
    p = {
        "sample": w / f"esci_lexbias_sample_{tag}.json",
        "para": w / f"esci_lexbias_paraphrases_{tag}.json",
        "judge_meta": w / f"esci_lexbias_judge_meta_{tag}.json",
    }
    for sc in SCORERS:
        for cond in CONDITIONS:
            p[f"score_{sc}_{cond}"] = w / f"esci_lexbias_{sc}_{cond}_{tag}.npy"
    return p


# --------------------------------------------------------------------------
# OpenAI plumbing (auth + cost pattern from evaluation/llm_relevance_judge.py)
# --------------------------------------------------------------------------
def estimate_cost(model: str, tokens_in: int, tokens_out: int) -> float:
    p = PRICES_PER_M_TOKENS.get(model)
    if not p:
        return 0.0
    return (tokens_in * p["in"] + tokens_out * p["out"]) / 1_000_000.0


def record_spend(model, tokens_in, tokens_out, cost_usd, purpose):
    rec = {
        "ts": datetime.datetime.utcnow().isoformat(timespec="seconds") + "Z",
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
    """Mutable token accumulator shared across coroutines."""

    def __init__(self):
        self.tin = 0
        self.tout = 0
        self.calls = 0
        self.errors = 0


async def _chat(client, sem, usage, model, prompt, max_tokens, logprobs, max_retries=6):
    """One chat completion with exponential backoff. Returns the choice or None."""
    backoff = 1.0
    async with sem:
        for _ in range(max_retries):
            try:
                kw = dict(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=max_tokens,
                    temperature=0.0,
                )
                if logprobs:
                    kw["logprobs"] = True
                    kw["top_logprobs"] = 20
                resp = await client.chat.completions.create(**kw)
                u = resp.usage
                usage.tin += int(u.prompt_tokens or 0)
                usage.tout += int(u.completion_tokens or 0)
                usage.calls += 1
                return resp.choices[0]
            except Exception:  # rate limit / transient API error
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2.0, 30.0)
        usage.errors += 1
        return None


def _score_from_logprobs(choice, scorer):
    """Continuous score from the first-token distribution.

    Mass for a target token missing from the top-20 is floored at
    exp(min observed top-20 logprob) -- a valid upper bound on it.
    """
    if choice is None or not choice.logprobs or not choice.logprobs.content:
        return float("nan")
    top = choice.logprobs.content[0].top_logprobs
    if not top:
        return float("nan")
    floor = math.exp(min(x.logprob for x in top))

    if scorer == "yesno":
        py = sum(math.exp(x.logprob) for x in top if x.token.strip().lower() == "yes")
        pn = sum(math.exp(x.logprob) for x in top if x.token.strip().lower() == "no")
        py = py if py > 0 else floor
        pn = pn if pn > 0 else floor
        return math.log(py) - math.log(pn)

    # graded: expected value of the 0-3 rating.
    # Match on the literal ASCII digit: str.isdigit() is True for exotica like
    # the subscript '₂', which int() then rejects.
    p = {d: 0.0 for d in GRADE_LEVELS}
    _ASCII = {str(d): d for d in GRADE_LEVELS}
    for x in top:
        t = x.token.strip()
        if t in _ASCII:
            p[_ASCII[t]] += math.exp(x.logprob)
    tot = sum(p.values())
    if tot <= 0:
        return float("nan")
    return sum(d * v for d, v in p.items()) / tot


# --------------------------------------------------------------------------
# phase: data -- materialize esci_us_data/ (test split only)
# --------------------------------------------------------------------------
TEST_SHARDS = [
    "data/test-00000-of-00004-d48474212b95f33b.parquet",
    "data/test-00001-of-00004-b7602f1b5c136953.parquet",
    "data/test-00002-of-00004-a81cff173329b486.parquet",
    "data/test-00003-of-00004-22af4ca7fa1313b2.parquet",
]


def phase_data(args):
    """Write titles.json / product_ids.json / test_queries.jsonl / test_qrels.jsonl.

    Same file contract as download/download_esci_us.py, but only the *test*
    split -- this experiment scores ESCI's judged candidate sets directly and
    never retrieves over the full catalog, so the 11-shard train split (~2GB)
    is not needed.
    """
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    out = Path(args.data_dir)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "test_qrels.jsonl").exists() and not args.force:
        print(f"{out}/test_qrels.jsonl exists; skipping (use --force to rebuild)")
        return

    cols = ["query_id", "query", "product_id", "product_title", "product_locale", "esci_label"]
    products, queries, qrels = {}, {}, []
    for shard in TEST_SHARDS:
        p = hf_hub_download("tasksource/esci", shard, repo_type="dataset")
        tbl = pq.read_table(p, columns=cols)
        d = tbl.to_pydict()
        n = 0
        for qid, q, pid, title, loc, lab in zip(
            d["query_id"], d["query"], d["product_id"], d["product_title"],
            d["product_locale"], d["esci_label"],
        ):
            if loc != "us":
                continue
            products.setdefault(pid, title or "")
            queries.setdefault(qid, q)
            qrels.append((qid, pid, LABEL_GRADE[lab]))
            n += 1
        print(f"  {Path(shard).name}: +{n:,} us rows", flush=True)

    pids = list(products)
    with open(out / "titles.json", "w") as f:
        json.dump([products[p] for p in pids], f)
    with open(out / "product_ids.json", "w") as f:
        json.dump(pids, f)
    with open(out / "test_queries.jsonl", "w") as f:
        for qid, qt in queries.items():
            f.write(json.dumps({"query_id": qid, "query": qt}) + "\n")
    with open(out / "test_qrels.jsonl", "w") as f:
        for qid, pid, g in qrels:
            f.write(json.dumps({"query_id": qid, "product_id": pid, "relevance": g}) + "\n")
    print(f"{len(products):,} products / {len(queries):,} queries / {len(qrels):,} qrels -> {out}")


def load_esci(data_dir):
    d = Path(data_dir)
    with open(d / "titles.json") as f:
        titles = json.load(f)
    with open(d / "product_ids.json") as f:
        pids = json.load(f)
    title_of = dict(zip(pids, titles))
    queries = {}
    with open(d / "test_queries.jsonl") as f:
        for line in f:
            r = json.loads(line)
            queries[r["query_id"]] = r["query"]
    qrels = defaultdict(list)  # qid -> [(pid, grade)] in file order
    seen = set()
    with open(d / "test_qrels.jsonl") as f:
        for line in f:
            r = json.loads(line)
            key = (r["query_id"], r["product_id"])
            if key in seen:
                continue
            seen.add(key)
            qrels[r["query_id"]].append((r["product_id"], r["relevance"]))
    return title_of, queries, qrels


# --------------------------------------------------------------------------
# phase: sample
# --------------------------------------------------------------------------
def phase_sample(args, paths):
    title_of, queries, qrels = load_esci(args.data_dir)
    print(f"loaded {len(queries):,} test queries, {len(qrels):,} with qrels", flush=True)

    # Eligibility: the E-vs-S discrimination test is only defined on queries
    # that actually contain both an Exact and a Substitute.
    eligible = []
    for qid, cands in qrels.items():
        if qid not in queries:
            continue
        grades = [g for _p, g in cands if title_of.get(_p)]
        if len(grades) < args.min_cands:
            continue
        if 3 not in grades or 2 not in grades:
            continue
        eligible.append(qid)
    eligible.sort()
    print(
        f"  {len(eligible):,} eligible queries "
        f"(>={args.min_cands} judged candidates, >=1 Exact and >=1 Substitute)",
        flush=True,
    )

    rng = random.Random(args.seed)
    if args.sample and args.sample < len(eligible):
        qids = sorted(rng.sample(eligible, args.sample))
        how = f"random.Random({args.seed}).sample over sorted eligible qids"
    else:
        qids = eligible
        how = "all eligible queries"

    rows = []
    for qid in qids:
        cands = [(p, g) for p, g in qrels[qid] if title_of.get(p)]
        # deterministic cap; keep the label mix by sorting on (grade, pid)
        # then round-robin so a cap never drops a whole grade
        if len(cands) > args.max_cands:
            by_grade = defaultdict(list)
            for p, g in cands:
                by_grade[g].append(p)
            for g in by_grade:
                by_grade[g].sort()
            kept, i = [], 0
            while len(kept) < args.max_cands:
                added = False
                for g in (3, 2, 1, 0):
                    if i < len(by_grade.get(g, [])) and len(kept) < args.max_cands:
                        kept.append((by_grade[g][i], g))
                        added = True
                if not added:
                    break
                i += 1
            keep_set = {p for p, _ in kept}
            cands = [(p, g) for p, g in cands if p in keep_set]  # restore file order
        rows.append(
            {
                "query_id": qid,
                "query": queries[qid],
                "product_ids": [p for p, _ in cands],
                "grades": [g for _, g in cands],
                "titles": [title_of[p] for p, _ in cands],
            }
        )

    n_pairs = sum(len(r["product_ids"]) for r in rows)
    counts = defaultdict(int)
    for r in rows:
        for g in r["grades"]:
            counts[GRADE_LETTER[g]] += 1
    meta = {
        "n_queries": len(rows),
        "n_pairs": n_pairs,
        "n_eligible": len(eligible),
        "seed": args.seed,
        "selection": how,
        "min_cands": args.min_cands,
        "max_cands": args.max_cands,
        "label_counts": dict(counts),
        "data_dir": str(Path(args.data_dir).resolve()),
        "rows": rows,
    }
    with open(paths["sample"], "w") as f:
        json.dump(meta, f)
    print(
        f"sample: {len(rows):,} queries / {n_pairs:,} pairs "
        f"({n_pairs / max(len(rows), 1):.1f} cand/query)  labels={dict(counts)}",
        flush=True,
    )
    print(f"saved -> {paths['sample']}", flush=True)


def _prompt_for(scorer, query, title, max_title_chars):
    tmpl = GRADED_PROMPT if scorer == "graded" else POINTWISE_PROMPT
    return tmpl.format(query=query, title=title[:max_title_chars])


# --------------------------------------------------------------------------
# phase: estimate  (spends nothing)
# --------------------------------------------------------------------------
def phase_estimate(args, paths, quiet=False):
    """Project API cost before any paid phase runs.

    Token counts are measured on the *actual* prompt strings (chars/4 plus a
    fixed chat-envelope allowance), not guessed. The smoke test on this prompt
    family measured 41-47 real prompt tokens, which this reproduces closely.
    """
    with open(paths["sample"]) as f:
        sample = json.load(f)
    rows = sample["rows"]

    CHAT_ENVELOPE_TOKENS = 8  # role/format wrapper the API adds

    def est_tok(s):
        return len(s) / 4.0 + CHAT_ENVELOPE_TOKENS

    # judge: every (query, candidate) pair x len(SCORERS) x len(CONDITIONS)
    judge_in = 0.0
    n_pairs = 0
    for r in rows:
        for t in r["titles"]:
            for sc in SCORERS:
                # paraphrase length is unknown here; the literal query is a
                # good proxy, so charge both conditions at the literal length
                judge_in += len(CONDITIONS) * est_tok(
                    _prompt_for(sc, r["query"], t, args.max_title_chars)
                )
            n_pairs += 1
    judge_calls = n_pairs * len(SCORERS) * len(CONDITIONS)
    judge_out = judge_calls * 1  # max_tokens=1

    # paraphrase: one call per query
    para_in = sum(est_tok(PARAPHRASE_PROMPT.format(query=r["query"])) for r in rows)
    para_out = len(rows) * args.para_max_tokens  # worst case

    tin = judge_in + para_in
    tout = judge_out + para_out
    cost = estimate_cost(args.model, tin, tout)
    breakdown = {
        "model": args.model,
        "n_queries": len(rows),
        "n_pairs": n_pairs,
        "scorers": list(SCORERS),
        "conditions": list(CONDITIONS),
        "judge_calls": judge_calls,
        "paraphrase_calls": len(rows),
        "total_calls": judge_calls + len(rows),
        "est_tokens_in": int(tin),
        "est_tokens_out": int(tout),
        "est_cost_usd_judge": estimate_cost(args.model, judge_in, judge_out),
        "est_cost_usd_paraphrase": estimate_cost(args.model, para_in, para_out),
        "est_cost_usd_total": cost,
        "ceiling_usd": COST_CEILING_USD,
    }
    if not quiet:
        print(json.dumps(breakdown, indent=2), flush=True)
        print(
            f"\nPROJECTED COST: ${cost:.4f} "
            f"({breakdown['total_calls']:,} calls) vs ceiling ${COST_CEILING_USD:.2f}",
            flush=True,
        )
        if cost > COST_CEILING_USD:
            print("OVER CEILING -- paid phases will refuse to run.", flush=True)
    return breakdown


def _guard_cost(args, paths):
    est = phase_estimate(args, paths, quiet=True)
    c = est["est_cost_usd_total"]
    print(f"[cost guard] projected total ${c:.4f} (ceiling ${COST_CEILING_USD:.2f})", flush=True)
    if c > COST_CEILING_USD:
        raise SystemExit(
            f"Refusing to run: projected ${c:.4f} exceeds ceiling ${COST_CEILING_USD:.2f}. "
            f"Lower --sample/--max-cands or raise --cost-ceiling deliberately."
        )
    return est


# --------------------------------------------------------------------------
# phase: paraphrase
# --------------------------------------------------------------------------
def _clean_paraphrase(resp, original):
    """Take the first non-empty line, strip quoting/labels/trailing punctuation."""
    text = (resp or "").strip()
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        line = re.sub(r'^(rewritten query|query|answer)\s*[:\-]\s*', "", line, flags=re.I)
        line = line.strip().strip('"').strip("'").strip()
        line = re.sub(r"[.\s]+$", "", line)
        if line:
            return line
    return original


async def _run_paraphrase(args, rows, usage):
    client = make_client()
    sem = asyncio.Semaphore(args.concurrency)

    async def one(r):
        ch = await _chat(
            client, sem, usage, args.model,
            PARAPHRASE_PROMPT.format(query=r["query"]),
            args.para_max_tokens, logprobs=False,
        )
        return (ch.message.content if ch is not None else "") or ""

    done = 0
    out_raw = [None] * len(rows)
    chunk = 200
    t0 = time.time()
    for i in range(0, len(rows), chunk):
        batch = rows[i : i + chunk]
        res = await asyncio.gather(*[one(r) for r in batch])
        for j, v in enumerate(res):
            out_raw[i + j] = v
        done += len(batch)
        print(
            f"  [paraphrase {done}/{len(rows)}] {done / max(time.time() - t0, 1e-9):.1f}/s "
            f"errors={usage.errors}",
            flush=True,
        )
    return out_raw


def phase_paraphrase(args, paths):
    with open(paths["sample"]) as f:
        sample = json.load(f)
    rows = sample["rows"]
    _guard_cost(args, paths)

    usage = Usage()
    t0 = time.time()
    raws = asyncio.run(_run_paraphrase(args, rows, usage))
    elapsed = time.time() - t0

    out = []
    for r, resp in zip(rows, raws):
        para = _clean_paraphrase(resp, r["query"])
        qt, pt = set(toks(r["query"])), set(toks(para))
        jac = len(qt & pt) / max(len(qt | pt), 1)
        identical = para.strip().lower() == r["query"].strip().lower()
        degenerate = (not pt) or len(para) > 8 * max(len(r["query"]), 12)
        if identical or degenerate:
            para_use = r["query"]
            status = "identical" if identical else "degenerate"
        else:
            para_use = para
            status = "ok"
        out.append(
            {
                "query_id": r["query_id"],
                "query": r["query"],
                "paraphrase_raw": (resp or "").strip()[:400],
                "paraphrase": para_use,
                "status": status,
                "jaccard_with_original": jac,
                "retained_token_frac": (len(qt & pt) / len(qt)) if qt else 1.0,
            }
        )

    cost = estimate_cost(args.model, usage.tin, usage.tout)
    record_spend(args.model, usage.tin, usage.tout, cost, "esci lexical-bias: paraphrase")
    stat = defaultdict(int)
    for o in out:
        stat[o["status"]] += 1
    jacs = [o["jaccard_with_original"] for o in out]
    payload = {
        "model": args.model,
        "decoding": "temperature=0.0",
        "max_tokens": args.para_max_tokens,
        "prompt": PARAPHRASE_PROMPT,
        "wall_clock_s": elapsed,
        "status_counts": dict(stat),
        "api_errors": usage.errors,
        "tokens_in": usage.tin,
        "tokens_out": usage.tout,
        "cost_usd": cost,
        "mean_jaccard_with_original": float(np.mean(jacs)),
        "median_jaccard_with_original": float(np.median(jacs)),
        "frac_zero_overlap": float(np.mean([j == 0 for j in jacs])),
        "paraphrases": out,
    }
    with open(paths["para"], "w") as f:
        json.dump(payload, f, indent=2)
    print(
        f"paraphrase done in {elapsed:.0f}s  status={dict(stat)}  "
        f"mean jaccard(orig,para)={payload['mean_jaccard_with_original']:.3f}  "
        f"cost=${cost:.4f}",
        flush=True,
    )


# --------------------------------------------------------------------------
# phase: judge  (logprob scorers over the OpenAI API)
# --------------------------------------------------------------------------
async def _run_judge_combo(args, rows, qfn, scorer, scores, usage):
    """Fill `scores` (n_q x max_c) for one (scorer, condition) combo."""
    client = make_client()
    sem = asyncio.Semaphore(args.concurrency)

    tasks = []  # (qi, ci, prompt)
    for qi, r in enumerate(rows):
        if args.resume and not np.isnan(scores[qi, : len(r["titles"])]).any():
            continue
        q = qfn(r)
        for ci, t in enumerate(r["titles"]):
            tasks.append((qi, ci, _prompt_for(scorer, q, t, args.max_title_chars)))
    if not tasks:
        print(f"  [{scorer}] fully cached; skipping", flush=True)
        return 0

    async def one(item):
        qi, ci, prompt = item
        ch = await _chat(client, sem, usage, args.model, prompt, 1, logprobs=True)
        try:
            s = _score_from_logprobs(ch, scorer)
        except Exception:  # never let one odd token kill a paid run
            usage.errors += 1
            s = float("nan")
        return qi, ci, s

    t0 = time.time()
    done = 0
    chunk = 500
    for i in range(0, len(tasks), chunk):
        batch = tasks[i : i + chunk]
        for qi, ci, s in await asyncio.gather(*[one(x) for x in batch]):
            scores[qi, ci] = s
        done += len(batch)
        el = time.time() - t0
        print(
            f"  [{scorer} {done}/{len(tasks)}] {done / max(el, 1e-9):.1f} pairs/s "
            f"eta {(len(tasks) - done) / max(done / max(el, 1e-9), 1e-9) / 60:.1f}m "
            f"errors={usage.errors} spent=${estimate_cost(args.model, usage.tin, usage.tout):.4f}",
            flush=True,
        )
    return len(tasks)


def phase_judge(args, paths):
    with open(paths["sample"]) as f:
        sample = json.load(f)
    rows = sample["rows"]
    n_q = len(rows)
    max_c = max(len(r["product_ids"]) for r in rows)
    _guard_cost(args, paths)

    with open(paths["para"]) as f:
        para_payload = json.load(f)
    para_of = {p["query_id"]: p["paraphrase"] for p in para_payload["paraphrases"]}

    qfns = {
        "literal": lambda r: r["query"],
        "paraphrased": lambda r: para_of[r["query_id"]],
    }

    usage = Usage()
    stats = {}
    t_all = time.time()
    # Any crash mid-run must still bank the partial scores AND log the money
    # already spent, otherwise the ledger under-reports the true cost.
    try:
        for scorer in SCORERS:
            for cond in CONDITIONS:
                key = f"score_{scorer}_{cond}"
                path = paths[key]
                scores = np.full((n_q, max_c), np.nan, dtype=np.float32)
                if args.resume and Path(path).exists():
                    prev = np.load(path)
                    if prev.shape == scores.shape:
                        scores = prev
                        print(f"  [{scorer}/{cond}] loaded cache", flush=True)
                t0 = time.time()
                print(f"== judging {scorer} / {cond} ==", flush=True)
                try:
                    n = await_run(
                        _run_judge_combo(args, rows, qfns[cond], scorer, scores, usage)
                    )
                finally:
                    np.save(path, scores)  # bank partial progress for --resume
                fin = scores[np.isfinite(scores)]
                stats[f"{scorer}_{cond}"] = {
                    "wall_clock_s": time.time() - t0,
                    "n_pairs_scored": int(n),
                    "n_nan": int(
                        np.isnan(scores).sum()
                        - (n_q * max_c - sum(len(r["titles"]) for r in rows))
                    ),
                    "score_mean": float(fin.mean()) if fin.size else None,
                    "score_std": float(fin.std()) if fin.size else None,
                    "score_min": float(fin.min()) if fin.size else None,
                    "score_max": float(fin.max()) if fin.size else None,
                }
                print(f"  [{scorer}/{cond}] -> {path}", flush=True)
    finally:
        _c = estimate_cost(args.model, usage.tin, usage.tout)
        if usage.calls:
            record_spend(args.model, usage.tin, usage.tout, _c, "esci lexical-bias: judge")

    cost = estimate_cost(args.model, usage.tin, usage.tout)
    stats["judge_model"] = args.model
    stats["mode"] = (
        "logprob scorers, max_tokens=1, top_logprobs=20; "
        "graded = E[0-3 rating], yesno = log p(yes) - log p(no)"
    )
    stats["graded_prompt"] = GRADED_PROMPT
    stats["yesno_prompt"] = POINTWISE_PROMPT
    stats["tokens_in"] = usage.tin
    stats["tokens_out"] = usage.tout
    stats["api_calls"] = usage.calls
    stats["api_errors"] = usage.errors
    stats["cost_usd"] = cost
    stats["total_wall_clock_s"] = time.time() - t_all
    with open(paths["judge_meta"], "w") as f:
        json.dump(stats, f, indent=2)
    print(
        f"\njudge done in {stats['total_wall_clock_s'] / 60:.1f}m  "
        f"calls={usage.calls:,} errors={usage.errors} cost=${cost:.4f}",
        flush=True,
    )


def await_run(coro):
    """Run a coroutine from sync code (one event loop per combo)."""
    return asyncio.run(coro)

# --------------------------------------------------------------------------
# phase: eval
# --------------------------------------------------------------------------
def bm25_scores(rows, k1=1.2, b=0.75):
    """BM25 over the judged candidate sets, IDF from the sampled title pool."""
    docs = {}
    for r in rows:
        for pid, t in zip(r["product_ids"], r["titles"]):
            docs.setdefault(pid, toks(t))
    N = len(docs)
    df = defaultdict(int)
    for tl in docs.values():
        for w in set(tl):
            df[w] += 1
    avgdl = sum(len(t) for t in docs.values()) / max(N, 1)
    idf = {w: math.log(1 + (N - n + 0.5) / (n + 0.5)) for w, n in df.items()}
    out = []
    for r in rows:
        qt = toks(r["query"])
        row = []
        for pid in r["product_ids"]:
            dl = len(docs[pid])
            tf = defaultdict(int)
            for w in docs[pid]:
                tf[w] += 1
            s = 0.0
            for w in qt:
                if w not in tf:
                    continue
                f = tf[w]
                s += idf.get(w, 0.0) * f * (k1 + 1) / (f + k1 * (1 - b + b * dl / max(avgdl, 1e-9)))
            row.append(s)
        out.append(row)
    return out


def _analyze_scorer(args, rows, n_q, para_of, lit, par):
    """Full bias / discrimination / ranking analysis for one scorer.

    `lit` and `par` are (n_q x max_c) score matrices for the literal and
    paraphrased conditions of the SAME scorer. Returns
    (results, recs, n_nan, corr, disc, fooled).
    """
    # ---- flatten to a per-pair table -------------------------------------
    recs = []
    for qi, r in enumerate(rows):
        pinfo = para_of[r["query_id"]]
        for ci, (pid, g, title) in enumerate(
            zip(r["product_ids"], r["grades"], r["titles"])
        ):
            cov_lit, jac_lit = overlap_metrics(r["query"], title)
            cov_par, jac_par = overlap_metrics(pinfo["paraphrase"], title)
            recs.append(
                {
                    "qi": qi,
                    "qid": r["query_id"],
                    "pid": pid,
                    "grade": g,
                    "label": GRADE_LETTER[g],
                    "cov_lit": cov_lit,
                    "jac_lit": jac_lit,
                    "cov_par": cov_par,
                    "jac_par": jac_par,
                    "s_lit": float(lit[qi, ci]),
                    "s_par": float(par[qi, ci]),
                    "para_ok": pinfo["status"] == "ok",
                }
            )
    n_nan = sum(1 for x in recs if not (np.isfinite(x["s_lit"]) and np.isfinite(x["s_par"])))
    recs = [x for x in recs if np.isfinite(x["s_lit"]) and np.isfinite(x["s_par"])]
    print(f"{len(recs):,} scored pairs ({n_nan} dropped for NaN)", flush=True)

    def sub(pred):
        return [x for x in recs if pred(x)]

    results = {}

    # ---- 1. lexical-overlap vs judge-score correlation, by label ---------
    corr = {}
    for lab in LETTERS:
        g = sub(lambda x, L=lab: x["label"] == L)
        if len(g) < 10:
            corr[lab] = {"n": len(g)}
            continue
        cov = [x["cov_lit"] for x in g]
        entry = {
            "n": len(g),
            "mean_cov_literal": float(np.mean(cov)),
            # both conditions correlated against the LITERAL query's overlap:
            # the apples-to-apples debiasing test
            "spearman_literalcov_vs_score_literal": spearman(cov, [x["s_lit"] for x in g]),
            "spearman_literalcov_vs_score_paraphrased": spearman(cov, [x["s_par"] for x in g]),
            "pearson_literalcov_vs_score_literal": pearson(cov, [x["s_lit"] for x in g]),
            "pearson_literalcov_vs_score_paraphrased": pearson(cov, [x["s_par"] for x in g]),
            # each condition against the overlap the judge actually saw
            "spearman_owncov_vs_score_literal": spearman(cov, [x["s_lit"] for x in g]),
            "spearman_owncov_vs_score_paraphrased": spearman(
                [x["cov_par"] for x in g], [x["s_par"] for x in g]
            ),
            "spearman_jaccard_vs_score_literal": spearman(
                [x["jac_lit"] for x in g], [x["s_lit"] for x in g]
            ),
            "spearman_jaccard_vs_score_paraphrased": spearman(
                [x["jac_lit"] for x in g], [x["s_par"] for x in g]
            ),
        }
        entry["delta_spearman_literalcov"] = (
            entry["spearman_literalcov_vs_score_paraphrased"]
            - entry["spearman_literalcov_vs_score_literal"]
        )
        corr[lab] = entry
    allg = recs
    corr["ALL"] = {
        "n": len(allg),
        "spearman_literalcov_vs_score_literal": spearman(
            [x["cov_lit"] for x in allg], [x["s_lit"] for x in allg]
        ),
        "spearman_literalcov_vs_score_paraphrased": spearman(
            [x["cov_lit"] for x in allg], [x["s_par"] for x in allg]
        ),
        "spearman_grade_vs_score_literal": spearman(
            [x["grade"] for x in allg], [x["s_lit"] for x in allg]
        ),
        "spearman_grade_vs_score_paraphrased": spearman(
            [x["grade"] for x in allg], [x["s_par"] for x in allg]
        ),
        "spearman_grade_vs_literalcov": spearman(
            [x["grade"] for x in allg], [x["cov_lit"] for x in allg]
        ),
    }
    corr["ALL"]["delta_spearman_literalcov"] = (
        corr["ALL"]["spearman_literalcov_vs_score_paraphrased"]
        - corr["ALL"]["spearman_literalcov_vs_score_literal"]
    )
    results["overlap_score_correlation_by_label"] = corr

    # per-query bootstrap CI on the within-S correlation delta
    s_by_q = defaultdict(list)
    for x in recs:
        if x["label"] == "S":
            s_by_q[x["qi"]].append(x)
    rng = np.random.default_rng(args.seed)
    qkeys = sorted(s_by_q)
    deltas = []
    for _ in range(args.n_boot):
        pick = rng.integers(0, len(qkeys), size=len(qkeys))
        pool = [x for i in pick for x in s_by_q[qkeys[i]]]
        if len(pool) < 10:
            continue
        cov = [x["cov_lit"] for x in pool]
        deltas.append(
            spearman(cov, [x["s_par"] for x in pool]) - spearman(cov, [x["s_lit"] for x in pool])
        )
    if deltas:
        results["overlap_score_correlation_by_label"]["S"]["delta_spearman_ci95"] = [
            float(np.percentile(deltas, 2.5)),
            float(np.percentile(deltas, 97.5)),
        ]
        results["overlap_score_correlation_by_label"]["S"]["delta_spearman_p_lt_0"] = float(
            np.mean([d < 0 for d in deltas])
        )

    # ---- 2. E vs S discrimination ---------------------------------------
    def zrow(field):
        """Per-query z-normalized judge score (removes query-level offset)."""
        out = {}
        for qi in range(n_q):
            g = [x for x in recs if x["qi"] == qi]
            v = np.array([x[field] for x in g])
            mu, sd = v.mean(), v.std()
            for x, vv in zip(g, v):
                out[(qi, x["pid"])] = (vv - mu) / sd if sd > 1e-9 else 0.0
        return out

    z_lit, z_par = zrow("s_lit"), zrow("s_par")
    for x in recs:
        x["z_lit"] = z_lit[(x["qi"], x["pid"])]
        x["z_par"] = z_par[(x["qi"], x["pid"])]

    # median split of S items on literal overlap
    s_items = sub(lambda x: x["label"] == "S")
    s_cov_median = float(np.median([x["cov_lit"] for x in s_items])) if s_items else 0.0
    disc = {"s_cov_literal_median": s_cov_median}
    for cond, fld, zfld in (("literal", "s_lit", "z_lit"), ("paraphrased", "s_par", "z_par")):
        e = sub(lambda x: x["label"] == "E")
        s = sub(lambda x: x["label"] == "S")
        s_hi = [x for x in s if x["cov_lit"] > s_cov_median]
        s_lo = [x for x in s if x["cov_lit"] <= s_cov_median]
        d = {
            "mean_E": float(np.mean([x[fld] for x in e])),
            "mean_S": float(np.mean([x[fld] for x in s])),
            "mean_S_high_overlap": float(np.mean([x[fld] for x in s_hi])) if s_hi else None,
            "mean_S_low_overlap": float(np.mean([x[fld] for x in s_lo])) if s_lo else None,
            "mean_C": (
                float(np.mean([x[fld] for x in sub(lambda x: x["label"] == "C")]))
                if sub(lambda x: x["label"] == "C") else None
            ),
            "mean_I": (
                float(np.mean([x[fld] for x in sub(lambda x: x["label"] == "I")]))
                if sub(lambda x: x["label"] == "I") else None
            ),
            "n_E": len(e), "n_S": len(s), "n_S_high": len(s_hi), "n_S_low": len(s_lo),
        }
        d["gap_E_minus_S"] = d["mean_E"] - d["mean_S"]
        if d["mean_S_high_overlap"] is not None:
            d["gap_E_minus_S_high_overlap"] = d["mean_E"] - d["mean_S_high_overlap"]
            d["gap_E_minus_S_low_overlap"] = d["mean_E"] - d["mean_S_low_overlap"]
        # query-normalized versions (the honest ones: query-level score offset
        # otherwise dominates the raw means)
        d["z_mean_E"] = float(np.mean([x[zfld] for x in e]))
        d["z_mean_S"] = float(np.mean([x[zfld] for x in s]))
        d["z_gap_E_minus_S"] = d["z_mean_E"] - d["z_mean_S"]
        if s_hi:
            d["z_mean_S_high_overlap"] = float(np.mean([x[zfld] for x in s_hi]))
            d["z_mean_S_low_overlap"] = float(np.mean([x[zfld] for x in s_lo]))
            d["z_gap_E_minus_S_high_overlap"] = d["z_mean_E"] - d["z_mean_S_high_overlap"]
            d["z_gap_E_minus_S_low_overlap"] = d["z_mean_E"] - d["z_mean_S_low_overlap"]
        # per-query paired gap (mean E - mean S within each query)
        pq = []
        for qi in range(n_q):
            ge = [x[fld] for x in recs if x["qi"] == qi and x["label"] == "E"]
            gs = [x[fld] for x in recs if x["qi"] == qi and x["label"] == "S"]
            if ge and gs:
                pq.append(float(np.mean(ge) - np.mean(gs)))
        d["per_query_gap_E_minus_S_mean"] = float(np.mean(pq)) if pq else None
        d["per_query_gap_E_minus_S_ci95"] = bootstrap_ci(pq, args.n_boot, args.seed)
        d["per_query_gap_frac_positive"] = float(np.mean([x > 0 for x in pq])) if pq else None
        d["_pq"] = pq
        disc[cond] = d

    # paired delta on the per-query E-S gap
    pq_l, pq_p = disc["literal"]["_pq"], disc["paraphrased"]["_pq"]
    if len(pq_l) == len(pq_p) and pq_l:
        dif = [p - l for p, l in zip(pq_p, pq_l)]
        disc["paired_delta_per_query_gap"] = {
            "mean": float(np.mean(dif)),
            "ci95": bootstrap_ci(dif, args.n_boot, args.seed),
            "frac_improved": float(np.mean([d > 0 for d in dif])),
            "n_queries": len(dif),
        }
    disc["literal"].pop("_pq")
    disc["paraphrased"].pop("_pq")
    results["e_vs_s_discrimination"] = disc

    # ---- 2b. "fooled" rate: within-query E/S inversions ------------------
    fooled = {}
    for cond, fld in (("literal", "s_lit"), ("paraphrased", "s_par")):
        tot = inv = tot_hi = inv_hi = 0
        for qi in range(n_q):
            es = [x for x in recs if x["qi"] == qi and x["label"] == "E"]
            ss = [x for x in recs if x["qi"] == qi and x["label"] == "S"]
            for a in es:
                for bnd in ss:
                    tot += 1
                    if bnd[fld] > a[fld]:
                        inv += 1
                    if bnd["cov_lit"] > a["cov_lit"]:  # S is MORE lexical than the E
                        tot_hi += 1
                        if bnd[fld] > a[fld]:
                            inv_hi += 1
        fooled[cond] = {
            "n_ES_pairs": tot,
            "inversion_rate": inv / tot if tot else None,
            "n_ES_pairs_S_more_lexical": tot_hi,
            "inversion_rate_when_S_more_lexical": inv_hi / tot_hi if tot_hi else None,
        }
    if fooled["literal"]["inversion_rate"] is not None:
        fooled["delta_inversion_rate"] = (
            fooled["paraphrased"]["inversion_rate"] - fooled["literal"]["inversion_rate"]
        )
        fooled["delta_inversion_rate_when_S_more_lexical"] = (
            fooled["paraphrased"]["inversion_rate_when_S_more_lexical"]
            - fooled["literal"]["inversion_rate_when_S_more_lexical"]
        )
    results["es_inversion_rates"] = fooled

    # ---- 3. ranking metric ------------------------------------------------
    bm25 = bm25_scores(rows)
    by_q = defaultdict(list)
    for x in recs:
        by_q[x["qi"]].append(x)
    pid_pos = [{p: i for i, p in enumerate(r["product_ids"])} for r in rows]

    rankers = {
        "ESCI file order (baseline)": lambda qi, g: [-pid_pos[qi][x["pid"]] for x in g],
        "BM25 (pure lexical)": lambda qi, g: [bm25[qi][pid_pos[qi][x["pid"]]] for x in g],
        "literal-query token coverage": lambda qi, g: [x["cov_lit"] for x in g],
        "LLM judge, literal query": lambda qi, g: [x["s_lit"] for x in g],
        "LLM judge, paraphrased query": lambda qi, g: [x["s_par"] for x in g],
        "LLM judge, literal+para mean": lambda qi, g: [
            0.5 * (x["z_lit"] + x["z_par"]) for x in g
        ],
    }
    ranking = {}
    per_query_ndcg = {}
    for name, fn in rankers.items():
        vals = []
        for qi in sorted(by_q):
            g = by_q[qi]
            sc = fn(qi, g)
            order = np.argsort(-np.asarray(sc, dtype=np.float64), kind="mergesort")
            grades_ranked = [g[i]["grade"] for i in order]
            vals.append(ndcg_at_k(grades_ranked, [x["grade"] for x in g]))
        vals = [v for v in vals if np.isfinite(v)]
        per_query_ndcg[name] = vals
        ranking[name] = {
            "ndcg10": float(np.mean(vals)),
            "ci95": bootstrap_ci(vals, args.n_boot, args.seed),
            "n": len(vals),
        }
        print(f"  {name:<34s} nDCG@10 {ranking[name]['ndcg10']:.4f}", flush=True)
    a = per_query_ndcg["LLM judge, paraphrased query"]
    b = per_query_ndcg["LLM judge, literal query"]
    dif = [x - y for x, y in zip(a, b)]
    ranking["_paired_delta_para_minus_literal"] = {
        "mean": float(np.mean(dif)),
        "ci95": bootstrap_ci(dif, args.n_boot, args.seed),
        "wins": int(sum(1 for d in dif if d > 0)),
        "losses": int(sum(1 for d in dif if d < 0)),
        "ties": int(sum(1 for d in dif if d == 0)),
    }
    results["ranking_ndcg10"] = ranking

    # ---- 4. overlap-stratified judge means -------------------------------
    strat = {}
    edges = [0.0, 0.34, 0.67, 1.01]
    for lab in LETTERS:
        g = sub(lambda x, L=lab: x["label"] == L)
        buckets = []
        for lo, hi in zip(edges[:-1], edges[1:]):
            bkt = [x for x in g if lo <= x["cov_lit"] < hi]
            buckets.append(
                {
                    "cov_range": [lo, min(hi, 1.0)],
                    "n": len(bkt),
                    "z_mean_literal": float(np.mean([x["z_lit"] for x in bkt])) if bkt else None,
                    "z_mean_paraphrased": (
                        float(np.mean([x["z_par"] for x in bkt])) if bkt else None
                    ),
                }
            )
        strat[lab] = buckets
    results["judge_score_by_overlap_bucket"] = strat
    return results, recs, n_nan, corr, disc, fooled


def _print_scorer_summary(scorer, corr, disc, fooled, ranking):
    print(f"\n################ SCORER: {scorer} ################", flush=True)
    print("\n== overlap(literal query) vs judge score, Spearman ==", flush=True)
    print(f"  {'label':<6s} {'n':>6s} {'literal':>9s} {'para':>9s} {'delta':>9s}", flush=True)
    for lab in list(LETTERS) + ["ALL"]:
        e = corr[lab]
        if "spearman_literalcov_vs_score_literal" not in e:
            continue
        print(
            f"  {lab:<6s} {e['n']:>6d} "
            f"{e['spearman_literalcov_vs_score_literal']:>9.4f} "
            f"{e['spearman_literalcov_vs_score_paraphrased']:>9.4f} "
            f"{e['delta_spearman_literalcov']:>+9.4f}",
            flush=True,
        )
    s_ci = corr["S"].get("delta_spearman_ci95")
    if s_ci:
        print(
            f"  S delta 95% CI [{s_ci[0]:+.4f}, {s_ci[1]:+.4f}]  "
            f"P(delta<0)={corr['S'].get('delta_spearman_p_lt_0'):.3f}",
            flush=True,
        )
    print("\n== E vs S discrimination (per-query z-normalized judge score) ==", flush=True)
    for cond in CONDITIONS:
        d = disc[cond]
        print(
            f"  {cond:<12s} E {d['z_mean_E']:+.4f}  S {d['z_mean_S']:+.4f}  "
            f"gap {d['z_gap_E_minus_S']:+.4f} | S-high-ovl {d.get('z_mean_S_high_overlap', 0):+.4f} "
            f"(gap {d.get('z_gap_E_minus_S_high_overlap', 0):+.4f})  "
            f"S-low-ovl {d.get('z_mean_S_low_overlap', 0):+.4f} "
            f"(gap {d.get('z_gap_E_minus_S_low_overlap', 0):+.4f})",
            flush=True,
        )
    print("  raw (un-normalized) means:", flush=True)
    for cond in CONDITIONS:
        d = disc[cond]
        print(
            f"    {cond:<12s} E {d['mean_E']:+.4f}  S {d['mean_S']:+.4f}  "
            f"S-hi {d['mean_S_high_overlap']:+.4f}  S-lo {d['mean_S_low_overlap']:+.4f}  "
            f"C {d['mean_C']:+.4f}  I {d['mean_I']:+.4f}",
            flush=True,
        )
    print("\n== E/S inversion rate (judge scores an S above an E) ==", flush=True)
    for cond in CONDITIONS:
        f_ = fooled[cond]
        print(
            f"  {cond:<12s} all pairs {f_['inversion_rate']:.4f} "
            f"(n={f_['n_ES_pairs']:,})   when S is more lexical than E "
            f"{f_['inversion_rate_when_S_more_lexical']:.4f} "
            f"(n={f_['n_ES_pairs_S_more_lexical']:,})",
            flush=True,
        )
    print("\n== nDCG@10 ==", flush=True)
    for name, v in ranking.items():
        if name.startswith("_"):
            continue
        ci = v.get("ci95")
        ci_s = f" [{ci[0]:.4f}, {ci[1]:.4f}]" if ci else ""
        print(f"  {name:<34s} {v['ndcg10']:.4f}{ci_s}", flush=True)
    pd_ = ranking.get("_paired_delta_para_minus_literal")
    if pd_:
        print(
            f"  paired delta (para - literal): {pd_['mean']:+.4f} "
            f"CI [{pd_['ci95'][0]:+.4f}, {pd_['ci95'][1]:+.4f}] "
            f"W/L/T {pd_['wins']}/{pd_['losses']}/{pd_['ties']}",
            flush=True,
        )


def _actual_spend(purpose_prefix="esci lexical-bias"):
    """Sum cost_usd from .api_spend.jsonl for this experiment's records."""
    if not SPEND_LEDGER.exists():
        return {"total_usd": 0.0, "records": []}
    recs = []
    for line in open(SPEND_LEDGER):
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except Exception:
            continue
        if str(r.get("purpose", "")).startswith(purpose_prefix):
            recs.append(r)
    return {
        "total_usd": round(sum(float(r.get("cost_usd", 0.0)) for r in recs), 6),
        "tokens_in": sum(int(r.get("tokens_in", 0)) for r in recs),
        "tokens_out": sum(int(r.get("tokens_out", 0)) for r in recs),
        "records": recs,
    }


def phase_eval(args, paths):
    with open(paths["sample"]) as f:
        sample = json.load(f)
    rows = sample["rows"]
    n_q = len(rows)
    with open(paths["para"]) as f:
        para_payload = json.load(f)
    para_of = {p["query_id"]: p for p in para_payload["paraphrases"]}
    judge_meta = {}
    if Path(paths["judge_meta"]).exists():
        with open(paths["judge_meta"]) as f:
            judge_meta = json.load(f)

    by_scorer = {}
    for scorer in SCORERS:
        lp, pp = paths[f"score_{scorer}_literal"], paths[f"score_{scorer}_paraphrased"]
        if not (Path(lp).exists() and Path(pp).exists()):
            print(f"  [{scorer}] score files missing; skipping", flush=True)
            continue
        print(f"\n--- analyzing scorer: {scorer} ---", flush=True)
        res, recs, n_nan, corr, disc, fooled = _analyze_scorer(
            args, rows, n_q, para_of, np.load(lp), np.load(pp)
        )
        res["n_pairs_scored"] = len(recs)
        res["n_pairs_dropped_nan"] = n_nan
        by_scorer[scorer] = res
        _print_scorer_summary(scorer, corr, disc, fooled, res["ranking_ndcg10"])

    spend = _actual_spend()
    payload = {
        "experiment": "esci_llm_judge_lexical_bias",
        "question": (
            "Is a pointwise LLM relevance judge biased toward literal query-token "
            "overlap, and does paraphrasing the query before judging reduce it?"
        ),
        "reference": "arXiv:2501.17969 (Alaofi et al., SIGIR-AP '24)",
        "corpus": "ESCI-US (tasksource/esci, test split), human graded E/S/C/I",
        "data_dir": sample["data_dir"],
        "pool": "ESCI's own human-judged candidate set per query (no first-stage retriever)",
        "judge_model": args.model,
        "judge_mode": (
            "logprob scorers over the OpenAI API, max_tokens=1, top_logprobs=20. "
            "graded (PRIMARY) = expected value of a 0-3 rating; "
            "yesno (robustness check) = log p(yes) - log p(no)."
        ),
        "primary_scorer": "graded",
        "graded_prompt": GRADED_PROMPT,
        "yesno_prompt": POINTWISE_PROMPT,
        "n_queries": sample["n_queries"],
        "n_pairs": sample["n_pairs"],
        "n_eligible_queries": sample["n_eligible"],
        "sample_seed": sample["seed"],
        "sample_selection": sample["selection"],
        "eligibility": (
            f">={sample['min_cands']} judged candidates with a title, "
            f">=1 Exact and >=1 Substitute; candidates capped at {sample['max_cands']}"
        ),
        "label_counts": sample["label_counts"],
        "lexical_overlap_metric": (
            "coverage = |content tokens of query present in title| / |content tokens of query|; "
            "lowercased [a-z0-9]+ tokens, minimal stoplist"
        ),
        "ndcg": "standard nDCG@10, linear gains = ESCI grade (E=3,S=2,C=1,I=0)",
        "paraphrase": {k: v for k, v in para_payload.items() if k != "paraphrases"},
        "judge_runtime": judge_meta,
        "api_spend": spend,
        "n_boot": args.n_boot,
        "results_by_scorer": by_scorer,
        "results": by_scorer.get("graded"),  # primary, for backward-compatible reads
    }
    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    with open(outp, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nsaved -> {outp}", flush=True)
    print(
        f"actual API spend for this experiment: ${spend['total_usd']:.4f} "
        f"(in={spend['tokens_in']:,} out={spend['tokens_out']:,} tokens)",
        flush=True,
    )

# --------------------------------------------------------------------------
def main():
    global COST_CEILING_USD
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--phase", required=True,
                    choices=["data", "sample", "estimate", "paraphrase", "judge", "eval"])
    ap.add_argument("--data-dir", default="esci_us_data")
    ap.add_argument("--work-dir", default="/tmp/esci_lexbias")
    ap.add_argument("--tag", default="esci_us")
    ap.add_argument("--out", default="evaluation/results/esci_llm_judge_lexical_bias.json")
    ap.add_argument("--model", default="gpt-4o-mini", help="OpenAI judge/paraphrase model")
    ap.add_argument("--concurrency", type=int, default=32)
    ap.add_argument("--cost-ceiling", type=float, default=COST_CEILING_USD,
                    help="refuse to start a paid phase projected above this (USD)")
    ap.add_argument("--sample", type=int, default=250)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min-cands", type=int, default=8)
    ap.add_argument("--max-cands", type=int, default=24)
    ap.add_argument("--max-title-chars", type=int, default=300)
    ap.add_argument("--para-max-tokens", type=int, default=48)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--resume", action="store_true", default=True)
    ap.add_argument("--no-resume", dest="resume", action="store_false")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    COST_CEILING_USD = args.cost_ceiling

    paths = work_paths(args.work_dir, args.tag)
    {
        "data": lambda: phase_data(args),
        "sample": lambda: phase_sample(args, paths),
        "estimate": lambda: phase_estimate(args, paths),
        "paraphrase": lambda: phase_paraphrase(args, paths),
        "judge": lambda: phase_judge(args, paths),
        "eval": lambda: phase_eval(args, paths),
    }[args.phase]()


if __name__ == "__main__":
    main()
