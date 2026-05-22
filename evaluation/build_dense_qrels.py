#!/usr/bin/env python3
"""Convert llm_relevance_judge.py output into qrels JSONL files.

Reads a JSONL of LLM verdicts (one per (qid, did) pair) and writes 1-3 qrels
files matching the bagofdocs schema:
    {"query_id": ..., "product_id": ..., "relevance": <int>}

Two output flavors:
  --strict   :  relevance=2 for EXACT only (sparse but high precision)
  --liberal  :  relevance=2 for EXACT, relevance=1 for RELATED (denser)
  --graded   :  same as --liberal but additionally writes a graded file
                where IRRELEVANT/UNCERTAIN do not appear (i.e., omitted)

When multiple original qrels are also passed via --merge-original-qrels, the
LLM-derived qrels are UNIONED with whatever the original file marked at
>= --keep-original-min-relevance (default: 2). The original-derived rows
get LLM-IRRELEVANT removed if the LLM ALSO judged that pair.

Usage:
  .venv/bin/python evaluation/build_dense_qrels.py \\
      --input evaluation/llm_judge_outputs/nfcorpus_pilot_b.jsonl \\
      --strict-output nfcorpus_data/test_qrels_llm_strict.jsonl \\
      --liberal-output nfcorpus_data/test_qrels_llm_liberal.jsonl
"""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="LLM judge output JSONL")
    ap.add_argument("--strict-output", default=None, help="path for strict qrels (EXACT->rel=2)")
    ap.add_argument(
        "--liberal-output",
        default=None,
        help="path for liberal qrels (EXACT->rel=2, RELATED->rel=1)",
    )
    ap.add_argument(
        "--merge-original-qrels",
        default=None,
        help="optional original qrels.jsonl to union with (keeps rows >= --keep-original-min-relevance "
        "unless the LLM judged them IRRELEVANT)",
    )
    ap.add_argument("--keep-original-min-relevance", type=int, default=2)
    args = ap.parse_args()

    if not args.strict_output and not args.liberal_output:
        raise SystemExit("must specify at least one of --strict-output / --liberal-output")

    verdicts = {}  # (qid, did) -> verdict
    with open(args.input) as f:
        for line in f:
            r = json.loads(line)
            verdicts[(r["qid"], r["did"])] = r["verdict"]
    print(f"loaded {len(verdicts):,} judgments from {args.input}", flush=True)
    print(f"verdict distribution: {Counter(verdicts.values())}", flush=True)

    # Build strict / liberal qrels from LLM
    strict = defaultdict(dict)
    liberal = defaultdict(dict)
    for (qid, did), v in verdicts.items():
        if v == "EXACT":
            strict[qid][did] = 2
            liberal[qid][did] = 2
        elif v == "RELATED":
            liberal[qid][did] = 1

    # Optionally merge original qrels
    if args.merge_original_qrels:
        n_added = 0
        n_dropped = 0
        with open(args.merge_original_qrels) as f:
            for line in f:
                r = json.loads(line)
                qid = r["query_id"]
                did = r["product_id"]
                rel = r["relevance"]
                if rel < args.keep_original_min_relevance:
                    continue
                v = verdicts.get((qid, did))
                if v == "IRRELEVANT":
                    n_dropped += 1
                    continue
                # add to strict only if not already there
                if did not in strict[qid]:
                    strict[qid][did] = rel
                    n_added += 1
                if did not in liberal[qid]:
                    liberal[qid][did] = rel
        print(
            f"merged original qrels: added {n_added:,} (not in LLM pool), "
            f"dropped {n_dropped:,} LLM-IRRELEVANT",
            flush=True,
        )

    def _write(out_path: str, q: dict):
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        n = 0
        with open(out_path, "w") as f:
            for qid in sorted(q):
                for did in sorted(q[qid]):
                    f.write(
                        json.dumps(
                            {
                                "query_id": qid,
                                "product_id": did,
                                "relevance": q[qid][did],
                            }
                        )
                        + "\n"
                    )
                    n += 1
        return n

    if args.strict_output:
        n = _write(args.strict_output, strict)
        avg = n / max(len(strict), 1)
        print(
            f"wrote {n:,} strict qrels across {len(strict):,} queries "
            f"(avg {avg:.1f}/q) to {args.strict_output}",
            flush=True,
        )
    if args.liberal_output:
        n = _write(args.liberal_output, liberal)
        avg = n / max(len(liberal), 1)
        print(
            f"wrote {n:,} liberal qrels across {len(liberal):,} queries "
            f"(avg {avg:.1f}/q) to {args.liberal_output}",
            flush=True,
        )


if __name__ == "__main__":
    main()
