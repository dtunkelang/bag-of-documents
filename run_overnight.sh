#!/bin/bash
# Overnight pipeline: wait for bag computation, then sanity check + eval + fine-tune
set -e
DIR="$(cd "$(dirname "$0")" && pwd)"
VENV="$DIR/.venv/bin/python3"
BAGS="$DIR/bags.jsonl"
LOG="$DIR/overnight.log"

exec > >(tee -a "$LOG") 2>&1
echo "=== Overnight pipeline started: $(date) ==="

# Prevent sleep for the duration of this script
command -v caffeinate > /dev/null && caffeinate -i -w $$ &

# --- Step 0: Wait for bag computation to finish ---
echo "Waiting for compute_bags.py to finish..."
while pgrep -f compute_bags > /dev/null 2>&1; do
    sleep 60
done
echo "Bag computation finished: $(date)"

# --- Step 1: Sanity check bags ---
echo ""
echo "=== SANITY CHECK: $(date) ==="
$VENV -c "
import json, numpy as np, random
from collections import Counter

bags_path = '$BAGS'
bags = []
with open(bags_path) as f:
    for line in f:
        bags.append(json.loads(line))
print(f'Total bags: {len(bags)}')

sizes = [b['num_results'] for b in bags]
specs = [b['specificity'] for b in bags if b['specificity'] > 0]
empties = sum(1 for s in sizes if s == 0)

print(f'Empty bags: {empties} ({empties/len(bags)*100:.1f}%)')
print(f'Bag size: mean={np.mean(sizes):.1f}, median={np.median(sizes):.0f}, min={min(sizes)}, max={max(sizes)}')
print(f'Specificity: mean={np.mean(specs):.3f}, p5={np.percentile(specs,5):.3f}, p25={np.percentile(specs,25):.3f}, min={min(specs):.3f}')
print(f'Bags <5 members: {sum(1 for s in sizes if 0 < s < 5)}')
print(f'Bags <10 members: {sum(1 for s in sizes if 0 < s < 10)}')
print(f'Bags with 50 members (full): {sum(1 for s in sizes if s == 50)}')

# Category diversity check — sample queries to see breadth
print()
print('=== QUERY DIVERSITY CHECK ===')
non_empty = [b for b in bags if b['num_results'] > 0]
random.seed(42)
sample = random.sample(non_empty, min(15, len(non_empty)))
for b in sample:
    top = b['results'][0]['title'][:65] if b['results'] else '(empty)'
    print(f'  {b[\"num_results\"]:2d} results  spec={b[\"specificity\"]:.3f}  \"{b[\"query\"]}\"')
    print(f'      -> {top}')

# Check for queries that might be outside our product categories
print()
print('=== EMPTY BAG ANALYSIS ===')
empty_bags = [b for b in bags if b['num_results'] == 0]
if empty_bags:
    random.seed(99)
    empty_sample = random.sample(empty_bags, min(10, len(empty_bags)))
    for b in empty_sample:
        print(f'  Empty: \"{b[\"query\"]}\"')
else:
    print('  No empty bags!')
"

# --- Step 2: Remove empty bags ---
echo ""
echo "=== REMOVING EMPTY BAGS: $(date) ==="
$VENV -c "
import json
bags_path = '$BAGS'
kept = 0
with open(bags_path) as f:
    lines = f.readlines()
with open(bags_path, 'w') as f:
    for line in lines:
        bag = json.loads(line)
        if bag['num_results'] > 0:
            f.write(line)
            kept += 1
print(f'Kept {kept} non-empty bags (removed {len(lines) - kept} empty)')
"

# --- Step 3: Fine-tune model ---
echo ""
echo "=== FINE-TUNING: $(date) ==="
rm -rf "$DIR/query_model_tmp"
$VENV "$DIR/finetune_query_model.py" "$BAGS" "$DIR/query_model_tmp"

# --- Step 4: Eval new model ---
echo ""
echo "=== EVAL NEW MODEL: $(date) ==="
$VENV "$DIR/eval_finetuned.py" "$DIR/query_model_tmp/" --base

# --- Step 5: Swap model ---
echo ""
echo "=== INSTALLING NEW MODEL ==="
rm -rf "$DIR/query_model"
mv "$DIR/query_model_tmp" "$DIR/query_model"
rm -f "$DIR/retrieval_model"
ln -s query_model "$DIR/retrieval_model"
echo "  query_model updated, retrieval_model symlink refreshed"

echo ""
echo "=== Overnight pipeline complete: $(date) ==="
