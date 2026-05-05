#!/usr/bin/env python3
"""LLM-judge for ESCI qrels labeling noise.

For each (query, pos product, neg product) triple from the inversion-pairs
data, ask a local LLM whether the labels look correct. Produces verdicts:
  KEEP       - both labels look correct
  FLIP_POS   - the labeled positive does not match the query
  FLIP_NEG   - the labeled negative actually matches the query
  BOTH_BAD   - neither labeled item matches the query
  UNCERTAIN  - cannot tell from titles alone

Output: /tmp/llm_judge_results.jsonl with one row per pair.

Usage:
  python evaluation/llm_judge_qrels.py \
      --pairs /tmp/inversion_pairs.json \
      --model mlx-community/Qwen2.5-32B-Instruct-4bit \
      --max-tokens 200 \
      --limit 0   # 0 means all pairs
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

PROMPT_TEMPLATE = """You are evaluating whether two products are relevant to a search query.

QUERY: {query}

PRODUCT A:
{pos_title}

PRODUCT B:
{neg_title}

For each product independently, decide whether it is an EXACT match for the query.
"Exact match" means a customer searching with this query would consider the product to be exactly what they wanted (correct product type, correct specifications, correct constraints like negation).

Important:
- Judge each product on its own merit. Do not assume the labels are correct.
- If the query has a constraint like "without X", the product must NOT have X.
- If the products are essentially identical to each other, they should get the same judgment.

Format your response EXACTLY as:
A_MATCHES: YES / NO / UNCLEAR
B_MATCHES: YES / NO / UNCLEAR
REASON: <one short sentence per product>
"""


def parse_response(text):
    """Parse A_MATCHES / B_MATCHES decisions, then derive verdict.

    Verdict mapping:
      A=YES, B=NO  -> KEEP        (existing labels are correct)
      A=NO,  B=YES -> SWAP        (positive and negative labels are swapped)
      A=NO,  B=NO  -> BOTH_BAD    (labeled positive does not match)
      A=YES, B=YES -> FLIP_NEG    (labeled negative actually matches)
      A=NO,  B=UNCLEAR -> FLIP_POS (positive does not match; neg uncertain)
      A=UNCLEAR, B=YES -> FLIP_NEG (neg matches; pos uncertain)
      else         -> UNCERTAIN
    """
    a = b = None
    reason = ""
    for line in text.strip().splitlines():
        s = line.strip().upper()
        if s.startswith("A_MATCHES:"):
            v = s.split(":", 1)[1].strip()
            for tok in ("YES", "NO", "UNCLEAR"):
                if tok in v:
                    a = tok
                    break
        elif s.startswith("B_MATCHES:"):
            v = s.split(":", 1)[1].strip()
            for tok in ("YES", "NO", "UNCLEAR"):
                if tok in v:
                    b = tok
                    break
        elif s.startswith("REASON:"):
            reason = line.split(":", 1)[1].strip()

    if a == "YES" and b == "NO":
        verdict = "KEEP"
    elif a == "NO" and b == "YES":
        verdict = "SWAP"
    elif a == "NO" and b == "NO":
        verdict = "BOTH_BAD"
    elif a == "YES" and b == "YES":
        verdict = "FLIP_NEG"
    elif a == "NO" and b == "UNCLEAR":
        verdict = "FLIP_POS"
    elif a == "UNCLEAR" and b == "YES":
        verdict = "FLIP_NEG"
    else:
        verdict = "UNCERTAIN"
    return verdict, reason, a, b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default="/tmp/inversion_pairs.json")
    ap.add_argument(
        "--model",
        default="mlx-community/Qwen2.5-32B-Instruct-4bit",
        help="MLX model id; supports any mlx-community/* 4-bit Qwen",
    )
    ap.add_argument("--max-tokens", type=int, default=200)
    ap.add_argument("--limit", type=int, default=0, help="0 = all pairs; otherwise process first N")
    ap.add_argument(
        "--output",
        default="/tmp/llm_judge_results.jsonl",
        help="Output JSONL file (one record per pair)",
    )
    ap.add_argument(
        "--resume",
        action="store_true",
        help="If set, skip pairs whose qid is already in the output file",
    )
    args = ap.parse_args()

    print(f"loading pairs from {args.pairs}...", flush=True)
    with open(args.pairs) as f:
        pairs = json.load(f)
    print(f"  {len(pairs)} pairs", flush=True)

    pairs.sort(key=lambda r: -r["pn_max"])  # process most-suspicious first

    if args.limit > 0:
        pairs = pairs[: args.limit]
        print(f"  limited to first {len(pairs)} pairs", flush=True)

    done_qids = set()
    if args.resume and os.path.exists(args.output):
        with open(args.output) as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    done_qids.add(rec["qid"])
                except Exception:
                    pass
        print(f"  resuming: {len(done_qids)} already judged", flush=True)

    print(f"\nloading model {args.model}...", flush=True)
    from mlx_lm import generate, load

    t0 = time.time()
    model, tokenizer = load(args.model)
    print(f"  load took {time.time() - t0:.0f}s", flush=True)

    n_done = 0
    n_skipped = 0
    t_start = time.time()
    fout = open(args.output, "a" if args.resume else "w")  # noqa: SIM115 (long-lived JSONL writer)
    for rec in pairs:
        if rec["qid"] in done_qids:
            n_skipped += 1
            continue

        prompt = PROMPT_TEMPLATE.format(
            query=rec["query"], pos_title=rec["pos_for_pn_title"], neg_title=rec["neg_for_pn_title"]
        )
        messages = [{"role": "user", "content": prompt}]
        text_in = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )

        t0 = time.time()
        response = generate(
            model,
            tokenizer,
            prompt=text_in,
            max_tokens=args.max_tokens,
            verbose=False,
        )
        dt = time.time() - t0

        verdict, reason, a_match, b_match = parse_response(response)
        out = {
            "qid": rec["qid"],
            "query": rec["query"],
            "pos_title": rec["pos_for_pn_title"],
            "neg_title": rec["neg_for_pn_title"],
            "pn_max": rec["pn_max"],
            "pp_max": rec["pp_max"],
            "a_matches": a_match,
            "b_matches": b_match,
            "verdict": verdict,
            "reason": reason,
            "raw_response": response,
            "elapsed_s": dt,
        }
        fout.write(json.dumps(out) + "\n")
        fout.flush()
        n_done += 1

        if n_done % 25 == 0 or n_done <= 5:
            elapsed_total = time.time() - t_start
            rate = n_done / elapsed_total if elapsed_total > 0 else 0
            remaining = len(pairs) - n_skipped - n_done
            eta_s = remaining / rate if rate > 0 else 0
            print(
                f"  [{n_done}/{len(pairs) - n_skipped}] verdict={verdict} "
                f"pn_max={rec['pn_max']:.3f} dt={dt:.1f}s "
                f"rate={rate * 60:.1f}/min ETA={eta_s / 60:.0f}min",
                flush=True,
            )

    fout.close()
    print(f"\ndone. {n_done} judged, {n_skipped} skipped. results in {args.output}", flush=True)


if __name__ == "__main__":
    main()
