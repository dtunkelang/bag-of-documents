#!/usr/bin/env python3
"""Gradio demo: base MiniLM vs BoD-trained retrieval on the BestBuy ACM catalog.

Side-by-side top-10. Holdout queries (with ground-truth click labels) get
their hits highlighted so the reader can see the lift directly.

Usage:
    python demo_bestbuy.py                # http://localhost:7860
    python demo_bestbuy.py --port 8080
    python demo_bestbuy.py --share        # public Gradio link
"""

import argparse
import json
import os
from collections import defaultdict

import gradio as gr
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "bestbuy_acm_data")


def load_resources(base_name, bod_path):
    print("loading data...", flush=True)
    with open(os.path.join(DATA_DIR, "product_ids.json")) as f:
        pids = json.load(f)
    with open(os.path.join(DATA_DIR, "titles.json")) as f:
        titles = json.load(f)
    qrels = defaultdict(set)
    queries_by_qid = {}
    with open(os.path.join(DATA_DIR, "holdout_queries.jsonl")) as f:
        for line in f:
            d = json.loads(line)
            queries_by_qid[d["query"]] = d["query_id"]
    with open(os.path.join(DATA_DIR, "holdout_qrels.jsonl")) as f:
        for line in f:
            r = json.loads(line)
            qrels[r["query_id"]].add(r["product_id"])
    print(f"  catalog={len(pids):,}  holdout={len(queries_by_qid):,}", flush=True)

    device = (
        "mps"
        if torch.backends.mps.is_available()
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    base_cache = os.path.join(DATA_DIR, "base_catalog.vecs.fp16.npy")
    print(f"loading base catalog from {base_cache}...", flush=True)
    base_pv = np.load(base_cache).astype(np.float32)
    base_model = SentenceTransformer(base_name, device=device)

    print(f"loading BoD model from {bod_path} and encoding catalog...", flush=True)
    bod_model = SentenceTransformer(bod_path, device=device)
    bod_pv = bod_model.encode(
        titles, normalize_embeddings=True, batch_size=128, show_progress_bar=True
    ).astype(np.float32)

    return {
        "pids": pids,
        "titles": titles,
        "qrels": qrels,
        "queries_by_qid": queries_by_qid,
        "base_model": base_model,
        "bod_model": bod_model,
        "base_pv": base_pv,
        "bod_pv": bod_pv,
    }


def topk(model, pv, query, k=10):
    qv = model.encode([query], normalize_embeddings=True, show_progress_bar=False).astype(
        np.float32
    )[0]
    sims = pv @ qv
    idx = np.argpartition(-sims, k)[:k]
    idx = idx[np.argsort(-sims[idx])]
    return [(int(i), float(sims[i])) for i in idx]


def format_results(hits, titles, pids, gold_pids):
    rows = []
    for rank, (i, s) in enumerate(hits, 1):
        title = titles[i]
        pid = pids[i]
        is_gold = pid in gold_pids
        marker = "✅" if is_gold else "&nbsp;&nbsp;"
        if is_gold:
            row = f"| {rank} | **{marker}** | **{title}** | `{s:.3f}` |"
        else:
            row = f"| {rank} | {marker} | {title} | `{s:.3f}` |"
        rows.append(row)
    header = "| # | hit | title | sim |\n|---:|:---:|:---|---:|"
    return header + "\n" + "\n".join(rows)


def build_app(R):
    def run(query):
        query = (query or "").strip().lower()
        if not query:
            return "_(enter a query)_", "_(enter a query)_", ""
        qid = R["queries_by_qid"].get(query)
        gold_set = set()
        gold_idxs = set()
        if qid is not None:
            gold_set = R["qrels"].get(qid, set())
            pid_to_idx = {p: i for i, p in enumerate(R["pids"])}
            gold_idxs = {pid_to_idx[p] for p in gold_set if p in pid_to_idx}

        base_hits = topk(R["base_model"], R["base_pv"], query, k=10)
        bod_hits = topk(R["bod_model"], R["bod_pv"], query, k=10)
        base_md = format_results(base_hits, R["titles"], R["pids"], gold_set)
        bod_md = format_results(bod_hits, R["titles"], R["pids"], gold_set)

        # Lift summary line.
        if gold_idxs:
            base_hit_ranks = [r for r, (i, _) in enumerate(base_hits, 1) if i in gold_idxs]
            bod_hit_ranks = [r for r, (i, _) in enumerate(bod_hits, 1) if i in gold_idxs]
            n_pos = len(gold_idxs)
            base_n = len(base_hit_ranks)
            bod_n = len(bod_hit_ranks)
            note = (
                f"**Holdout query** — {n_pos} clicked product(s) in catalog.  "
                f"Base hits in top-10: **{base_n}/{n_pos}** "
                f"(ranks {base_hit_ranks or '—'}).  "
                f"BoD hits in top-10: **{bod_n}/{n_pos}** "
                f"(ranks {bod_hit_ranks or '—'})."
            )
        else:
            note = "_Free-form query — no ground-truth labels to highlight._"
        return base_md, bod_md, note

    # Holdout queries where BoD finds 8-10/10 gold and base finds 0-1.
    # Each illustrates a different failure mode of off-the-shelf semantic search.
    examples = [
        "ati",  # brand abbreviation -> Radeon graphics
        "dvd storage",  # abstract intent -> cases/wallets, not DVDs
        "turtlebeach",  # joined brand name -> headsets
        "stereo system",  # product class -> bundles, not components
        "i pad 2",  # spaced tokenization -> iPad 2
        "definitive technology",  # generic-sounding brand -> speakers
        "iphone 4 incase",  # InCase brand vs preposition
        "ear phones",  # spelling variant -> earphones
    ]

    with gr.Blocks(title="BoD demo: BestBuy click data") as demo:
        gr.Markdown(
            "# Bag-of-Documents on BestBuy click data\n"
            "Same catalog (53,048 products), same query encoder architecture "
            "(`all-MiniLM-L6-v2`). The only difference: the right-hand model was "
            "fine-tuned on 48,516 query→clicked-SKU bags from BestBuy's 2012 "
            "ACM hackathon clickthrough log via MNRL.\n"
            "\nHoldout R@10: base **0.5559** → BoD **0.7308** (+17.49pp).  "
            "E@1: base **0.2538** → BoD **0.3718** (+11.80pp)."
        )
        with gr.Row():
            q = gr.Textbox(
                label="Query",
                placeholder="try one of the examples or type your own",
                lines=1,
            )
        gr.Examples(examples=examples, inputs=q)
        note = gr.Markdown()
        with gr.Row():
            with gr.Column():
                gr.Markdown("### Base MiniLM (off-the-shelf)")
                base_out = gr.Markdown()
            with gr.Column():
                gr.Markdown("### BoD-trained (this work)")
                bod_out = gr.Markdown()
        q.submit(run, inputs=q, outputs=[base_out, bod_out, note])

    return demo


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-model", default="all-MiniLM-L6-v2")
    ap.add_argument("--bod-model", default="query_model_bestbuy_bod")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--share", action="store_true")
    args = ap.parse_args()

    R = load_resources(args.base_model, args.bod_model)
    app = build_app(R)
    app.launch(server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
