#!/usr/bin/env python3
"""LLM-as-relevance-judge for arbitrary (query, doc) pairs.

Generalizes evaluation/llm_judge_qrels.py (pair-judge, MLX) to single-doc
judgments via the OpenAI API. Powers the LLM-relevance-judge track:
- Branch A: re-judge existing qrels for label noise (precision)
- Branch B: judge top-K retrieved candidates to find missed positives (recall)
- Branch C: generate qrels from scratch on unlabeled query slices (de novo)
- Branch D: filter candidates inside the bag-construction pipeline

Verdicts (per-(query, doc)):
  EXACT       - the document fully satisfies the query (the user's true target)
  RELATED     - the document is on-topic but not what the user is looking for
  IRRELEVANT  - the document does not address the query
  UNCERTAIN   - cannot decide from the available text

Input JSONL row: {"qid": ..., "query": ..., "did": ..., "doc_text": ...}
  Optional fields {"prior_label": ...} are passed through to output unchanged.

Output JSONL row: {"qid": ..., "did": ..., "verdict": ..., "reason": ...,
                   "raw": ..., "tokens_in": ..., "tokens_out": ..., "elapsed_s": ...}

Resumable: skips (qid, did) pairs already present in --output.
Cost-logged: appends a single roll-up record to .api_spend.jsonl on exit.

Usage:
  .venv/bin/python evaluation/llm_relevance_judge.py \\
      --input /tmp/nfcorpus_topK_pairs.jsonl \\
      --output evaluation/llm_judge_outputs/nfcorpus_topk.jsonl \\
      --model gpt-4o-mini --concurrency 24 --purpose "nfcorpus dense qrels pilot B"
"""

import argparse
import asyncio
import datetime
import json
import os
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(override=True)

from openai import AsyncOpenAI  # noqa: E402

# gpt-4o-mini pricing (2026-05, $ per 1M tokens)
PRICES_PER_M_TOKENS = {
    "gpt-4o-mini": {"in": 0.15, "out": 0.60},
    "gpt-4o": {"in": 2.50, "out": 10.00},
    "gpt-4.1-mini": {"in": 0.40, "out": 1.60},
    "gpt-4.1": {"in": 2.00, "out": 8.00},
}

SPEND_LEDGER = Path(__file__).resolve().parent.parent / ".api_spend.jsonl"

PROMPT_TEMPLATE = """You are evaluating whether a single document satisfies a search query.

QUERY: {query}

DOCUMENT:
{doc_text}

Judge whether the document is an EXACT match, RELATED, or IRRELEVANT to the query.

Definitions:
- EXACT: a user searching with this query would consider this document to be exactly what they wanted. The core intent of the query is fully satisfied by the document.
- RELATED: the document is on-topic and shares concepts with the query, but it is not what the user is actually looking for (wrong specific subtype, missing a required constraint, adjacent product/topic).
- IRRELEVANT: the document does not address the query.

Important:
- Judge the document against the query, not against any assumed label.
- If the query has a constraint like "without X", the document must NOT have X to be EXACT.
- If you cannot decide from the available text, respond UNCERTAIN.

Format your response EXACTLY as:
VERDICT: EXACT / RELATED / IRRELEVANT / UNCERTAIN
REASON: <one short sentence>
"""


def parse_response(text: str):
    verdict = None
    reason = ""
    for line in text.strip().splitlines():
        s = line.strip()
        if s.upper().startswith("VERDICT:"):
            v = s.split(":", 1)[1].strip().upper()
            for tok in ("EXACT", "RELATED", "IRRELEVANT", "UNCERTAIN"):
                if tok in v:
                    verdict = tok
                    break
        elif s.upper().startswith("REASON:"):
            reason = s.split(":", 1)[1].strip()
    if verdict is None:
        # last-chance heuristic on raw text
        upper = text.upper()
        for tok in ("EXACT", "RELATED", "IRRELEVANT", "UNCERTAIN"):
            if re.search(rf"\b{tok}\b", upper):
                verdict = tok
                break
    return verdict or "UNCERTAIN", reason


def record_spend(model: str, tokens_in: int, tokens_out: int, cost_usd: float, purpose: str):
    rec = {
        "ts": datetime.datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "provider": "openai",
        "model": model,
        "tokens": int(tokens_in + tokens_out),
        "tokens_in": int(tokens_in),
        "tokens_out": int(tokens_out),
        "cost_usd": round(float(cost_usd), 4),
        "purpose": purpose,
    }
    with open(SPEND_LEDGER, "a") as f:
        f.write(json.dumps(rec) + "\n")


def estimate_cost(model: str, tokens_in: int, tokens_out: int) -> float:
    if model not in PRICES_PER_M_TOKENS:
        return 0.0
    p = PRICES_PER_M_TOKENS[model]
    return (tokens_in * p["in"] + tokens_out * p["out"]) / 1_000_000.0


async def judge_one(
    client: AsyncOpenAI,
    sem: asyncio.Semaphore,
    rec: dict,
    model: str,
    max_tokens: int,
    max_retries: int = 5,
) -> dict:
    prompt = PROMPT_TEMPLATE.format(query=rec["query"], doc_text=rec["doc_text"])
    backoff = 1.0
    last_err = None
    async with sem:
        t0 = time.time()
        for _ in range(max_retries):
            try:
                resp = await client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=max_tokens,
                    temperature=0.0,
                )
                break
            except Exception as e:  # rate limit, transient API error
                last_err = e
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2.0, 30.0)
        else:
            return {
                "qid": rec["qid"],
                "did": rec["did"],
                "verdict": "ERROR",
                "reason": str(last_err)[:200],
                "raw": "",
                "tokens_in": 0,
                "tokens_out": 0,
                "elapsed_s": time.time() - t0,
            }
        dt = time.time() - t0
        raw = resp.choices[0].message.content or ""
        verdict, reason = parse_response(raw)
        usage = resp.usage
        out = {
            "qid": rec["qid"],
            "did": rec["did"],
            "verdict": verdict,
            "reason": reason,
            "raw": raw,
            "tokens_in": int(usage.prompt_tokens or 0),
            "tokens_out": int(usage.completion_tokens or 0),
            "elapsed_s": round(dt, 2),
        }
        if "prior_label" in rec:
            out["prior_label"] = rec["prior_label"]
        return out


async def run(args):
    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"loading input from {input_path}...", flush=True)
    items = []
    with open(input_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    print(f"  {len(items)} (query, doc) pairs in input", flush=True)

    if args.limit > 0:
        items = items[: args.limit]
        print(f"  limited to first {len(items)}", flush=True)

    done = set()
    if output_path.exists() and args.resume:
        with open(output_path) as f:
            for line in f:
                try:
                    r = json.loads(line)
                    done.add((r["qid"], r["did"]))
                except Exception:
                    pass
        print(f"  resuming: {len(done)} already judged", flush=True)

    todo = [r for r in items if (r["qid"], r["did"]) not in done]
    print(f"  {len(todo)} judgments to run", flush=True)
    if not todo:
        print("  nothing to do; exiting", flush=True)
        return

    client = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    sem = asyncio.Semaphore(args.concurrency)

    fout = open(output_path, "a")  # noqa: SIM115 (long-lived JSONL writer)
    n_done = 0
    n_err = 0
    tokens_in_total = 0
    tokens_out_total = 0
    t_start = time.time()

    # process in chunks so we flush + report progress regularly
    chunk = 100
    for i in range(0, len(todo), chunk):
        batch = todo[i : i + chunk]
        tasks = [judge_one(client, sem, r, args.model, args.max_tokens) for r in batch]
        results = await asyncio.gather(*tasks)
        for res in results:
            fout.write(json.dumps(res) + "\n")
            tokens_in_total += res["tokens_in"]
            tokens_out_total += res["tokens_out"]
            if res["verdict"] == "ERROR":
                n_err += 1
            n_done += 1
        fout.flush()

        elapsed = time.time() - t_start
        rate = n_done / elapsed if elapsed > 0 else 0
        remaining = len(todo) - n_done
        eta_s = remaining / rate if rate > 0 else 0
        cost_so_far = estimate_cost(args.model, tokens_in_total, tokens_out_total)
        cost_eta = (cost_so_far / n_done) * len(todo) if n_done > 0 else 0
        print(
            f"  [{n_done}/{len(todo)}] err={n_err} rate={rate * 60:.0f}/min "
            f"ETA={eta_s / 60:.0f}min spent=${cost_so_far:.3f} proj=${cost_eta:.3f}",
            flush=True,
        )

    fout.close()
    final_cost = estimate_cost(args.model, tokens_in_total, tokens_out_total)
    record_spend(args.model, tokens_in_total, tokens_out_total, final_cost, args.purpose)
    print(
        f"\ndone. {n_done} judged ({n_err} errors). "
        f"tokens in={tokens_in_total} out={tokens_out_total} cost=${final_cost:.4f} "
        f"wall={time.time() - t_start:.0f}s",
        flush=True,
    )
    print(f"output: {output_path}", flush=True)
    print(f"spend logged to {SPEND_LEDGER}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="JSONL with {qid, query, did, doc_text}")
    ap.add_argument("--output", required=True, help="JSONL output (one judgment per line)")
    ap.add_argument("--model", default="gpt-4o-mini")
    ap.add_argument("--max-tokens", type=int, default=80)
    ap.add_argument("--concurrency", type=int, default=24)
    ap.add_argument("--limit", type=int, default=0, help="0 = all; otherwise process first N")
    ap.add_argument(
        "--resume",
        action="store_true",
        default=True,
        help="Skip (qid, did) pairs already present in --output (default: on)",
    )
    ap.add_argument(
        "--no-resume",
        dest="resume",
        action="store_false",
        help="Re-judge from scratch, overwriting existing output",
    )
    ap.add_argument("--purpose", default="llm-relevance-judge run", help="ledger purpose label")
    args = ap.parse_args()

    if not args.resume:
        out_path = Path(args.output)
        if out_path.exists():
            print(f"--no-resume: truncating existing {out_path}", flush=True)
            out_path.unlink()

    asyncio.run(run(args))


if __name__ == "__main__":
    main()
