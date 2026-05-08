#!/usr/bin/env python3
"""HF Space: BestBuy ACM bag-of-documents retrieval demo.

Side-by-side comparison of off-the-shelf MiniLM vs a BoD-fine-tuned MiniLM
on the BestBuy 2012 ACM Hackathon clickthrough dataset (53,048 products,
12,128 holdout queries). The BoD model was fine-tuned via MNRL on 48,516
click-derived bags. Holdout R@10 (binary hit-rate): base 0.556 → BoD
0.731 (+17.49pp). E@1: 0.254 → 0.372 (+11.80pp).

When you type a holdout query, products that were actually clicked for
that query are highlighted with ✅. The summary line above the columns
shows how many clicked products each model surfaces in its top-10.

Repo: https://github.com/dtunkelang/bag-of-documents
"""

import json
import os
from collections import defaultdict

import gradio as gr
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

# Embedded artifacts in this Space repo.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "data")
MODEL_DIR = os.path.join(SCRIPT_DIR, "query_model_bestbuy_bod")
BASE_MODEL = "all-MiniLM-L6-v2"


def load_resources():
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
    print(f"  catalog={len(pids):,}  holdout queries={len(queries_by_qid):,}", flush=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    base_pv = np.load(os.path.join(DATA_DIR, "base_catalog.vecs.fp16.npy")).astype(np.float32)
    bod_pv = np.load(os.path.join(DATA_DIR, "bod_catalog.vecs.fp16.npy")).astype(np.float32)
    print("  catalog vectors loaded (base + BoD)", flush=True)

    base_model = SentenceTransformer(BASE_MODEL, device=device)
    bod_model = SentenceTransformer(MODEL_DIR, device=device)
    print("  models loaded", flush=True)

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
        if is_gold:
            rows.append(f"| {rank} | **✅** | **{title}** | `{s:.3f}` |")
        else:
            rows.append(f"| {rank} | &nbsp;&nbsp; | {title} | `{s:.3f}` |")
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

    # Example queries selected for clear, demo-friendly contrasts.
    # Each illustrates a distinct failure mode of off-the-shelf semantic search:
    examples = [
        "ati",  # brand abbreviation -> Radeon graphics cards
        "dvd storage",  # abstract intent -> cases/wallets, not DVDs themselves
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
            "ACM hackathon clickthrough log via MultipleNegativesRankingLoss "
            "([code](https://github.com/dtunkelang/bag-of-documents)).\n"
            "\n"
            "Holdout R@10 (binary hit-rate, n=12,128): "
            "base **0.5559** → BoD **0.7308** (+17.49pp).  "
            "E@1: **0.2538** → **0.3718** (+11.80pp). "
            "The largest single-corpus BoD lift in the project.\n"
            "\n"
            "Holdout queries are highlighted with ✅ on rows where the product "
            "was actually clicked for that query in the original 2012 log.\n"
        )
        with gr.Row():
            q = gr.Textbox(
                label="Query",
                placeholder="try one of the examples or type your own",
                lines=1,
                scale=4,
            )
            search_btn = gr.Button("Search", variant="primary", scale=1)
        note = gr.Markdown()
        with gr.Row():
            with gr.Column():
                gr.Markdown("### Base MiniLM (off-the-shelf)")
                base_out = gr.Markdown()
            with gr.Column():
                gr.Markdown("### BoD-trained (this work)")
                bod_out = gr.Markdown()
        # Examples populate the textbox AND trigger the search on click.
        gr.Examples(
            examples=examples,
            inputs=q,
            outputs=[base_out, bod_out, note],
            fn=run,
            run_on_click=True,
            cache_examples=False,
        )
        search_btn.click(run, inputs=q, outputs=[base_out, bod_out, note])
        q.submit(run, inputs=q, outputs=[base_out, bod_out, note])

    return demo


def main():
    R = load_resources()
    app = build_app(R)
    app.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))


if __name__ == "__main__":
    main()
