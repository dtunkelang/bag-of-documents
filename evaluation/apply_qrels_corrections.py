#!/usr/bin/env python3
"""Apply LLM-judge verdicts to test_qrels.jsonl, producing test_qrels_cleaned.jsonl.

Corrections applied (per LLM verdict on the (pos_for_pn, neg_for_pn) pair only;
other qrels rows for the query are left alone):

  KEEP      : no change
  FLIP_NEG  : neg relevance 0 -> 3 (the labeled negative actually matches)
  BOTH_BAD  : pos relevance 3 -> 0 (the labeled positive does not match)
  SWAP      : pos 3 -> 0 AND neg 0 -> 3
  UNCERTAIN : no change

Outputs:
  esci_us_data/test_qrels_cleaned.jsonl
  /tmp/qrels_corrections_log.jsonl  (record of every change)
"""

import json
import os
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

SCRIPT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    qrels_in = os.path.join(SCRIPT_DIR, "esci_us_data/test_qrels.jsonl")
    qrels_out = os.path.join(SCRIPT_DIR, "esci_us_data/test_qrels_cleaned.jsonl")
    judge_path = "/tmp/llm_judge_results.jsonl"
    pairs_path = "/tmp/inversion_pairs.json"
    log_path = "/tmp/qrels_corrections_log.jsonl"

    print(f"loading judge results from {judge_path}...", flush=True)
    judge_by_qid = {}
    with open(judge_path) as f:
        for line in f:
            r = json.loads(line)
            judge_by_qid[r["qid"]] = r
    print(f"  {len(judge_by_qid)} verdicts", flush=True)

    print(f"loading inversion pairs from {pairs_path}...", flush=True)
    with open(pairs_path) as f:
        pairs = json.load(f)
    pair_by_qid = {p["qid"]: p for p in pairs}
    print(f"  {len(pair_by_qid)} pairs", flush=True)

    # Map (qid, pid) -> change_to_relevance
    changes = {}
    log = []
    counts = defaultdict(int)
    for qid, judge in judge_by_qid.items():
        verdict = judge["verdict"]
        counts["verdict_" + verdict] += 1
        pair = pair_by_qid.get(qid)
        if pair is None:
            counts["missing_pair"] += 1
            continue
        pos_pid, neg_pid = pair["pn_pair"]
        if verdict == "KEEP":
            continue
        elif verdict == "FLIP_NEG":
            changes[(qid, neg_pid)] = 3
            log.append(
                {
                    "qid": qid,
                    "pid": neg_pid,
                    "from": 0,
                    "to": 3,
                    "verdict": verdict,
                    "query": judge["query"],
                }
            )
            counts["flips_neg_to_pos"] += 1
        elif verdict == "BOTH_BAD":
            changes[(qid, pos_pid)] = 0
            log.append(
                {
                    "qid": qid,
                    "pid": pos_pid,
                    "from": 3,
                    "to": 0,
                    "verdict": verdict,
                    "query": judge["query"],
                }
            )
            counts["flips_pos_to_neg"] += 1
        elif verdict == "SWAP":
            changes[(qid, pos_pid)] = 0
            changes[(qid, neg_pid)] = 3
            log.append(
                {
                    "qid": qid,
                    "pid": pos_pid,
                    "from": 3,
                    "to": 0,
                    "verdict": verdict,
                    "query": judge["query"],
                }
            )
            log.append(
                {
                    "qid": qid,
                    "pid": neg_pid,
                    "from": 0,
                    "to": 3,
                    "verdict": verdict,
                    "query": judge["query"],
                }
            )
            counts["swap_pairs"] += 1
        elif verdict == "UNCERTAIN":
            counts["uncertain_skipped"] += 1

    print("\nplanned corrections:", flush=True)
    for k, v in sorted(counts.items()):
        print(f"  {k}: {v}", flush=True)
    print(f"  total qrels rows changed: {len(changes)}", flush=True)

    # Read original qrels, apply changes, write cleaned
    print(f"\nrewriting {qrels_in} -> {qrels_out}...", flush=True)
    n_total = 0
    n_changed = 0
    with open(qrels_in) as fin, open(qrels_out, "w") as fout:
        for line in fin:
            r = json.loads(line)
            n_total += 1
            key = (r["query_id"], r["product_id"])
            if key in changes:
                new_rel = changes[key]
                if new_rel != r["relevance"]:
                    r["relevance"] = new_rel
                    n_changed += 1
            fout.write(json.dumps(r) + "\n")
    print(
        f"  {n_total:,} qrels rows scanned, {n_changed:,} changed ({n_changed / n_total:.2%})",
        flush=True,
    )

    with open(log_path, "w") as f:
        for r in log:
            f.write(json.dumps(r) + "\n")
    print(f"  log written to {log_path} ({len(log)} entries)", flush=True)


if __name__ == "__main__":
    main()
