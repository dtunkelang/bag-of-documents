"""Ensemble rerank: combine top-10 lists from two rerankers via RRF.

Inputs (from prior run):
  - /tmp/rerank_eval/queries.json          (list of query strings)
  - /tmp/rerank_eval/candidates.json       (list of top-100 titles per query)
  - /tmp/rerank_eval/rerank_*.npy          ((22458, 10) position arrays)

For each query:
  - Each candidate position (0-99) gets a rank in each model (1-10 if present, else inf).
  - RRF score = 1/(rh + 60) + 1/(r6 + 60).
  - Top-10 positions by RRF score.

Compare to base R@10 / nDCG@10 and the individual rerankers.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
import math
import os
import statistics
from collections import defaultdict

os.environ["TOKENIZERS_PARALLELISM"] = "false"
# Run all relative paths from the repo root regardless of where the script
# is invoked. (Previously hardcoded to a developer's home directory.)
from pathlib import Path as _Path  # noqa: E402

os.chdir(_Path(__file__).resolve().parent.parent)

import numpy as np  # noqa: E402  (placed after os.environ to control thread count)

WORKDIR = "/tmp/rerank_eval"
SCRIPT_DIR = str(_Path(__file__).resolve().parent.parent)

# Load eval data
print("loading qrels + queries + product map...", flush=True)
with open(os.path.join(SCRIPT_DIR, "esci_us_data/test_queries.jsonl")) as f:
    queries_id = {}
    for line in f:
        d = json.loads(line)
        queries_id[d["query_id"]] = d["query"]

with open(os.path.join(SCRIPT_DIR, "esci_us_data/test_qrels.jsonl")) as f:
    qrels = defaultdict(dict)
    for line in f:
        r = json.loads(line)
        qrels[r["query_id"]][r["product_id"]] = r["relevance"]
qrels = dict(qrels)

with open(os.path.join(SCRIPT_DIR, "esci_us_data/product_ids.json")) as f:
    esci_pids = json.load(f)
with open(os.path.join(SCRIPT_DIR, "esci_us_data/titles.json")) as f:
    esci_titles_arr = json.load(f)
title_to_pid = {t: p for p, t in zip(esci_pids, esci_titles_arr)}

# Reload eval queries (same order as the prior run)
with open(os.path.join(WORKDIR, "queries.json")) as f:
    queries = json.load(f)
print(f"  {len(queries):,} eval queries", flush=True)

# Map eval-query strings back to qids (matching the prior run's filter logic)
# We need the QID order — the eval queries.json was built from filtered qids in qids order.
# Reproduce that filter:
qids = []
for qid in queries_id:
    if qid not in qrels:
        continue
    if any(g >= 2 for g in qrels[qid].values()):
        qids.append(qid)
# Full run had no --limit, so no shuffle
assert [queries_id[qid] for qid in qids] == queries, "query order mismatch with prior run"

# Load candidate titles per query
print("loading candidates.json...", flush=True)
with open(os.path.join(WORKDIR, "candidates.json")) as f:
    candidates_titles = json.load(f)
print(f"  {len(candidates_titles):,} candidate lists", flush=True)

# Map candidates to pids
candidates_pids = []
for ts in candidates_titles:
    candidates_pids.append([title_to_pid.get(t) for t in ts])

# Load rerank top-10 positions
hn_pos = np.load(os.path.join(WORKDIR, "rerank_query_model_us_qrels_mnrl_hardneg.npy"))
m6_pos = np.load(os.path.join(WORKDIR, "rerank_query_model_us_full_6m_mnrl.npy"))
print(f"  shapes: hn={hn_pos.shape}, 6m={m6_pos.shape}", flush=True)

GAIN = {3: 1.0, 2: 0.1, 1: 0.01, 0: 0.0}


def metrics_for(pids, qrels_q, k=10):
    relevant_e_s = {p for p, g in qrels_q.items() if g >= 2}
    if not relevant_e_s:
        return None
    top_k = pids[:k]
    recall = sum(1 for p in top_k if p in relevant_e_s) / len(relevant_e_s)
    gains = [GAIN.get(qrels_q.get(p, 0), 0.0) for p in top_k]
    dcg = sum(g / math.log2(i + 2) for i, g in enumerate(gains))
    ideal = sorted(qrels_q.values(), reverse=True)[:k]
    idcg = sum(GAIN.get(g, 0) / math.log2(i + 2) for i, g in enumerate(ideal))
    return {"recall": recall, "ndcg": dcg / idcg if idcg > 0 else 0.0}


# Build per-query orderings:
#   - base: positions 0..9 of candidates_pids[qi]
#   - hardneg: positions hn_pos[qi]
#   - 6m: positions m6_pos[qi]
#   - ensemble (RRF): positions ranked by 1/(rh+60) + 1/(r6+60)

K_RRF = 60  # standard RRF constant


def ensemble_top10(qi):
    rh = {int(pos): rank + 1 for rank, pos in enumerate(hn_pos[qi]) if pos >= 0}
    r6 = {int(pos): rank + 1 for rank, pos in enumerate(m6_pos[qi]) if pos >= 0}
    pool = set(rh) | set(r6)
    scores = []
    for pos in pool:
        s = 0.0
        if pos in rh:
            s += 1.0 / (rh[pos] + K_RRF)
        if pos in r6:
            s += 1.0 / (r6[pos] + K_RRF)
        scores.append((s, pos))
    scores.sort(reverse=True)
    return [pos for _, pos in scores[:10]]


def measure_all(orderings_fn, name):
    recalls = []
    ndcgs = []
    for qi, qid in enumerate(qids):
        positions = orderings_fn(qi)
        pids = [candidates_pids[qi][int(p)] for p in positions if p >= 0]
        m = metrics_for(pids, qrels[qid], k=10)
        if m:
            recalls.append(m["recall"])
            ndcgs.append(m["ndcg"])
    return {
        "name": name,
        "n": len(recalls),
        "R@10": statistics.mean(recalls),
        "nDCG@10": statistics.mean(ndcgs),
    }


# Compute
res_base = measure_all(lambda qi: list(range(10)), "base (MiniLM)")
res_hn = measure_all(lambda qi: hn_pos[qi].tolist(), "base+hardneg")
res_6m = measure_all(lambda qi: m6_pos[qi].tolist(), "base+6M MNRL")
res_ens = measure_all(ensemble_top10, "base+ensemble (RRF k=60)")


# Try also a sum-rank fusion variant (rank_h + rank_6m, lower is better)
def sumrank_top10(qi):
    rh = {int(pos): rank + 1 for rank, pos in enumerate(hn_pos[qi]) if pos >= 0}
    r6 = {int(pos): rank + 1 for rank, pos in enumerate(m6_pos[qi]) if pos >= 0}
    pool = set(rh) | set(r6)
    scores = []
    for pos in pool:
        s = rh.get(pos, 100) + r6.get(pos, 100)  # absent = sentinel rank 100
        scores.append((s, pos))
    scores.sort()  # lower sum-rank is better
    return [pos for _, pos in scores[:10]]


res_sr = measure_all(sumrank_top10, "base+sumrank")

print(f"\n=== Ensemble eval ({len(qids):,} queries) ===")
print(f"{'Model':<32} {'R@10':>8} {'nDCG@10':>9}  ΔR / ΔnDCG vs base")
print("-" * 75)
for r in (res_base, res_hn, res_6m, res_ens, res_sr):
    if r is res_base:
        delta = ""
    else:
        delta = f"  +{(r['R@10'] - res_base['R@10']) * 100:+.2f}pp / +{r['nDCG@10'] - res_base['nDCG@10']:+.4f}"
    print(f"{r['name']:<32} {r['R@10']:>8.2%} {r['nDCG@10']:>9.4f}{delta}")
