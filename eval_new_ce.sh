#!/usr/bin/env bash
# Post-training validation pipeline for a new CE.
#
# 1. Measure E-vs-S gap on judged pairs
# 2. Re-bag the regime eval set with the new CE
# 3. Compare per-regime entity-purity before/after
#
# Usage: ./eval_new_ce.sh <ce_model_path> [<label_for_logs>]

set -euo pipefail

CE_MODEL="${1:?usage: eval_new_ce.sh <ce_model_path> [label]}"
LABEL="${2:-$(basename "$CE_MODEL")}"

OUT_DIR="/tmp/eval_${LABEL}"
mkdir -p "$OUT_DIR"

echo "==> 1. CE E-vs-S gap on 500 judged pairs"
.venv/bin/python eval_ce_es_gap.py --ce-model "$CE_MODEL" 2>&1 | tee "$OUT_DIR/es_gap.log"

echo
echo "==> 2. Re-bagging 45 regime eval queries with new CE"
.venv/bin/python compute_bags.py eval/regime_queries.jsonl "$OUT_DIR/bags.jsonl" \
    --ce-rerank "$CE_MODEL" --index-dir combined_index_amazon 2>&1 | tail -20

echo
echo "==> 3. Per-regime entity-purity diff (vs current esci-us-ce baseline at /tmp/bags_eval_default.jsonl)"
.venv/bin/python - <<EOF
import json, statistics
from collections import defaultdict

def load_bags(path):
    out = {}
    with open(path) as f:
        for line in f:
            b = json.loads(line)
            out[b["query"]] = b
    return out

def load_eval():
    out = {}
    with open("eval/regime_queries.jsonl") as f:
        for line in f:
            d = json.loads(line)
            out[d["query"]] = d
    return out

eval_q = load_eval()
old_bags = load_bags("/tmp/bags_eval_default.jsonl")
new_bags = load_bags("$OUT_DIR/bags.jsonl")

def entity_pct(bag, e):
    n = len(bag["results"])
    if n == 0:
        return 0
    return sum(1 for m in bag["results"] if e in m["title"].lower()) / n * 100

stats = defaultdict(lambda: {"old": [], "new": []})
for q, info in eval_q.items():
    if q not in old_bags or q not in new_bags:
        continue
    e = info["entity_token"]
    stats[info["regime"]]["old"].append(entity_pct(old_bags[q], e))
    stats[info["regime"]]["new"].append(entity_pct(new_bags[q], e))

print(f"{'regime':<6} {'old %ent':>10} {'new %ent':>10} {'delta':>8} {'wins':>5} {'losses':>7} {'tied':>5}")
print("-" * 60)
for r in ("easy", "mid", "hard"):
    old_v = stats[r]["old"]
    new_v = stats[r]["new"]
    if not old_v:
        continue
    om = statistics.mean(old_v)
    nm = statistics.mean(new_v)
    n = len(old_v)
    wins = sum(1 for o, w in zip(old_v, new_v) if w > o + 1)
    losses = sum(1 for o, w in zip(old_v, new_v) if o > w + 1)
    tied = n - wins - losses
    print(f"{r:<6} {om:>10.1f} {nm:>10.1f} {nm - om:>+8.1f} {wins:>5} {losses:>7} {tied:>5}")
EOF

echo
echo "Done. Artifacts in $OUT_DIR/"
