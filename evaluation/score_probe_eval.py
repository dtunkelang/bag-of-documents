#!/usr/bin/env python3
"""Compute Hit@K for each retriever on the probe set, overall and per-archetype.

Strict: label >= 2 (relevant). Lenient: label >= 1 (relevant or marginal).
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path


def hit_at_k(retriever_doc_ids, labels_for_query, k, threshold):
    for doc_id in retriever_doc_ids[:k]:
        lab = labels_for_query.get(doc_id, 0)
        if lab >= threshold:
            return 1
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--topk", required=True, help="probe_topk_v2.jsonl")
    ap.add_argument("--labels", required=True, help="probe_labels.jsonl")
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--out-md", required=True)
    args = ap.parse_args()

    # labels[query_id][doc_id] = label
    labels = defaultdict(dict)
    with open(args.labels) as f:
        for line in f:
            r = json.loads(line)
            labels[r["query_id"]][r["doc_id"]] = r["label"]

    with open(args.topk) as f:
        rows = [json.loads(l) for l in f]
    retrievers = sorted({rn for r in rows for rn in r["retrievers"]})
    print(f"retrievers: {retrievers}")
    print(f"queries: {len(rows)}")

    # Hit@K storage
    K_vals = [1, 5, 10]
    overall = {r: {(k, t): [] for k in K_vals for t in ("strict", "lenient")} for r in retrievers}
    per_arch = defaultdict(
        lambda: {r: {(k, t): [] for k in K_vals for t in ("strict", "lenient")} for r in retrievers}
    )

    for row in rows:
        qid = row["query_id"]
        arch = row.get("archetype", "?")
        q_labels = labels.get(qid, {})
        for rn, retr in row["retrievers"].items():
            doc_ids = retr["doc_ids"]
            for k in K_vals:
                for t, thresh in (("strict", 2), ("lenient", 1)):
                    h = hit_at_k(doc_ids, q_labels, k, thresh)
                    overall[rn][(k, t)].append(h)
                    per_arch[arch][rn][(k, t)].append(h)

    def avg(xs):
        return 100.0 * sum(xs) / len(xs) if xs else 0.0

    # build output
    out = {
        "n_queries": len(rows),
        "retrievers": retrievers,
        "overall": {
            rn: {
                f"H@{k}_{t}": round(avg(overall[rn][(k, t)]), 2)
                for k in K_vals
                for t in ("strict", "lenient")
            }
            for rn in retrievers
        },
        "per_archetype": {
            arch: {
                rn: {
                    f"H@{k}_{t}": round(avg(per_arch[arch][rn][(k, t)]), 2)
                    for k in K_vals
                    for t in ("strict", "lenient")
                }
                for rn in retrievers
            }
            for arch in sorted(per_arch.keys())
        },
        "archetype_counts": {
            arch: len(per_arch[arch][retrievers[0]][(1, "strict")])
            for arch in sorted(per_arch.keys())
        },
    }

    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(out, f, indent=2)
    print(f"wrote {args.out_json}")

    # label distribution from labels file
    label_counts = defaultdict(int)
    with open(args.labels) as f:
        for line in f:
            r = json.loads(line)
            label_counts[r["label"]] += 1
    total = sum(label_counts.values())

    # markdown output
    md = []
    md.append(f"# Probe Set Hit@K Evaluation ({len(rows)} queries, {total:,} judged candidates)\n")
    md.append(
        "**Built 2026-05-27.** Hand-curated 102-query probe set across 14 archetypes, judged in-session "
        "by Claude Opus 4.7 (label scheme: 0=not relevant, 1=related/marginal, 2=relevant).\n"
    )
    md.append("## Label distribution\n")
    md.append("| Label | Count | % |")
    md.append("|---|---:|---:|")
    for lab in (0, 1, 2):
        c = label_counts[lab]
        pct = 100.0 * c / total
        names = {0: "not relevant", 1: "related/marginal", 2: "relevant"}
        md.append(f"| {lab} ({names[lab]}) | {c:,} | {pct:.1f}% |")
    md.append(f"| **Total** | **{total:,}** | 100% |\n")

    md.append("## Overall Hit@K\n")
    hdr_cells = ["Retriever"] + [f"H@{k} {t}" for k in K_vals for t in ("strict", "lenient")]
    md.append("| " + " | ".join(hdr_cells) + " |")
    md.append("|" + "|".join(["---"] + ["---:"] * (len(hdr_cells) - 1)) + "|")
    for rn in retrievers:
        cells = [rn]
        for k in K_vals:
            for t in ("strict", "lenient"):
                cells.append(f"{avg(overall[rn][(k, t)]):.1f}")
        md.append("| " + " | ".join(cells) + " |")
    md.append("")

    md.append("## Per-archetype Hit@10 strict\n")
    md.append("| Archetype | N | " + " | ".join(retrievers) + " |")
    md.append("|---|---:|" + "|".join(["---:"] * len(retrievers)) + "|")
    for arch in sorted(per_arch.keys()):
        n = len(per_arch[arch][retrievers[0]][(10, "strict")])
        cells = [arch, str(n)]
        for rn in retrievers:
            cells.append(f"{avg(per_arch[arch][rn][(10, 'strict')]):.1f}")
        md.append("| " + " | ".join(cells) + " |")

    Path(args.out_md).write_text("\n".join(md) + "\n")
    print(f"wrote {args.out_md}")


if __name__ == "__main__":
    main()
