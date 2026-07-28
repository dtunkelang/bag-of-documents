#!/usr/bin/env python3
"""Local Gradio demo: BestBuy full 1.27M catalog, OLD vs NEW (re-indexed) vectors.

Side-by-side dense retrieval over the *real* full 1,274,801-product BestBuy
catalog, comparing two encodings of the same catalog with the same unmodified
embedding model:

  OLD ("Currently Live")  -- the catalog vectors that were shipped in the
      dtunkelang/bag-of-documents-bestbuy HF dataset. Indexing text was the
      raw product name only, with HTML entities left unescaped.
  NEW ("New Re-Index")    -- Pattern 30 re-index. Indexing text is
      name + manufacturer + categoryPath + class, HTML-unescaped. No model
      retraining: identical encoder weights, richer input text.

Both encodings are in the same product_ids.json row order, so row i refers to
the same SKU in both matrices; no SKU re-alignment is needed.

IMPORTANT CAVEAT: the aggregate validation (Pattern 30, Part F) found the
junk-rate to be *flat* overall at full catalog scale. This demo exists to
inspect individual queries, not as evidence of a general improvement.

Run:
    /Users/dtunkelang/job-search/.venv/bin/python \
        evaluation/local_demo_bestbuy_full_reindex.py --port 7862
"""

import argparse
import gc
import json
import os
import time

import gradio as gr
import numpy as np
from sentence_transformers import SentenceTransformer

# --------------------------------------------------------------------------
# paths / constants
# --------------------------------------------------------------------------
# The OLD vectors are read from the huggingface_hub cache snapshot that was
# pulled *before* the re-indexed vectors were published to the dataset repo.
# We deliberately do NOT snapshot_download here: the remote now serves the NEW
# vectors, so re-fetching would silently turn this into a NEW-vs-NEW demo.
DEFAULT_OLD_DIR = os.path.expanduser(
    "~/.cache/huggingface/hub/datasets--dtunkelang--bag-of-documents-bestbuy"
    "/snapshots/15ef813587f8958928d77c1e9ff905c9d8165b5c"
)
DEFAULT_NEW_DIR = "/tmp/bestbuy_reindex_output/artifacts"

BASE_MODEL = "all-MiniLM-L6-v2"
BOD_MODEL = "dtunkelang/bag-of-documents-bestbuy-minilm"
MODEL_CHOICES = {
    "base MiniLM (all-MiniLM-L6-v2)": ("base", BASE_MODEL),
    "BoD MiniLM (fine-tuned)": ("bod", BOD_MODEL),
}
DEFAULT_MODEL_LABEL = "base MiniLM (all-MiniLM-L6-v2)"

CATALOG_SIZE = 1_274_801
TOP_K = 10
# Rows per chunk when upcasting fp16 -> fp32 for the dot product. 200k x 384
# fp32 is ~307MB of scratch, which keeps peak RSS well under control on a
# 16GB machine while still letting BLAS do the matmul at full speed.
CHUNK_ROWS = 200_000

EXAMPLE_QUERIES = ["apple tablet", "nokia phone", "roomba"]


# --------------------------------------------------------------------------
# data loading
# --------------------------------------------------------------------------
def _vec_path(directory, key):
    return os.path.join(directory, f"{key}_catalog.vecs.fp16.npy")


def load_titles(old_dir, new_dir):
    """Load both title lists once; they are model-independent."""
    print("loading titles...", flush=True)
    with open(os.path.join(old_dir, "titles.json")) as f:
        old_titles = json.load(f)
    with open(os.path.join(new_dir, "titles.json")) as f:
        new_titles = json.load(f)
    if len(old_titles) != len(new_titles):
        raise SystemExit(f"title count mismatch: old={len(old_titles)} new={len(new_titles)}")
    if len(old_titles) != CATALOG_SIZE:
        raise SystemExit(f"expected {CATALOG_SIZE:,} titles, got {len(old_titles):,}")
    print(f"  titles loaded: {len(old_titles):,} rows (old + new)", flush=True)
    return old_titles, new_titles


class VectorCache:
    """Holds exactly one model's OLD+NEW catalog matrices (~1.96GB as fp16).

    Switching models drops the previous pair before loading the next one, so
    at most two 979MB matrices are resident at any time.
    """

    def __init__(self, old_dir, new_dir):
        self.old_dir = old_dir
        self.new_dir = new_dir
        self.key = None
        self.old = None
        self.new = None
        self.model = None

    def ensure(self, key, model_name):
        if self.key == key:
            return
        # Drop the outgoing model's arrays *before* allocating the new ones.
        self.old = None
        self.new = None
        self.model = None
        self.key = None
        gc.collect()

        t0 = time.time()
        print(f"loading vectors for model '{key}'...", flush=True)
        old = np.load(_vec_path(self.old_dir, key))
        new = np.load(_vec_path(self.new_dir, key))
        for name, arr in (("old", old), ("new", new)):
            if arr.dtype != np.float16:
                raise SystemExit(f"{name} {key} vecs: expected float16, got {arr.dtype}")
            if arr.shape[0] != CATALOG_SIZE:
                raise SystemExit(
                    f"{name} {key} vecs: expected {CATALOG_SIZE:,} rows, got {arr.shape}"
                )
        print(f"  old={old.shape} new={new.shape} ({time.time() - t0:.1f}s)", flush=True)

        print(f"loading encoder {model_name}...", flush=True)
        model = SentenceTransformer(model_name, device="cpu")

        self.old, self.new, self.model, self.key = old, new, model, key
        print(f"  ready ({time.time() - t0:.1f}s total)", flush=True)


# --------------------------------------------------------------------------
# retrieval
# --------------------------------------------------------------------------
def chunked_topk(vecs_fp16, qvec_fp32, k=TOP_K):
    """Top-k cosine sim against an fp16 matrix without upcasting the whole thing.

    numpy has no BLAS path for fp16 matmul, so we upcast CHUNK_ROWS at a time
    and let BLAS handle each slice in fp32. Peak scratch is one chunk.
    """
    n = vecs_fp16.shape[0]
    sims = np.empty(n, dtype=np.float32)
    for start in range(0, n, CHUNK_ROWS):
        stop = min(start + CHUNK_ROWS, n)
        sims[start:stop] = vecs_fp16[start:stop].astype(np.float32) @ qvec_fp32
    idx = np.argpartition(-sims, k)[:k]
    idx = idx[np.argsort(-sims[idx])]
    return [(int(i), float(sims[i])) for i in idx]


def encode_query(model, query):
    return model.encode([query], normalize_embeddings=True, show_progress_bar=False).astype(
        np.float32
    )[0]


# --------------------------------------------------------------------------
# formatting
# --------------------------------------------------------------------------
def _html_escape(s):
    return (
        s.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace("\n", " ")
        .replace("\r", " ")
        .strip()
    )


def format_results(hits, titles):
    """HTML table (rank / title / sim) -- same style as space_demo_bestbuy/app.py.

    HTML sidesteps Markdown's column-count fragility on titles containing '|'
    or newlines, and keeps short fields on one line so '10' doesn't wrap.
    """
    nw = "white-space:nowrap;padding:2px 6px"
    rows = []
    for rank, (i, s) in enumerate(hits, 1):
        rows.append(
            f"<tr><td style='{nw};text-align:right'>{rank}</td>"
            f"<td style='padding:2px 6px'>{_html_escape(titles[i])}</td>"
            f"<td style='{nw};text-align:right;font-family:monospace'>{s:.3f}</td></tr>"
        )
    return (
        "<table style='width:100%;font-size:14px;border-collapse:collapse'>"
        f"<tr style='border-bottom:2px solid #ccc'>"
        f"<th style='{nw};text-align:right'>#</th>"
        "<th style='padding:2px 6px;text-align:left'>title</th>"
        f"<th style='{nw};text-align:right'>sim</th></tr>" + "".join(rows) + "</table>"
    )


# --------------------------------------------------------------------------
# app
# --------------------------------------------------------------------------
def build_search_fn(cache, old_titles, new_titles):
    def search(query, model_label):
        query = (query or "").strip()
        if not query:
            empty = "<i>enter a query</i>"
            return empty, empty, ""
        key, model_name = MODEL_CHOICES[model_label]
        cache.ensure(key, model_name)

        t0 = time.time()
        qvec = encode_query(cache.model, query)
        old_hits = chunked_topk(cache.old, qvec)
        new_hits = chunked_topk(cache.new, qvec)
        elapsed = time.time() - t0

        status = (
            f"<b>{_html_escape(query)}</b> &nbsp;|&nbsp; model: <code>{key}</code> "
            f"&nbsp;|&nbsp; {CATALOG_SIZE:,} products scanned per side "
            f"&nbsp;|&nbsp; {elapsed:.2f}s"
        )
        return (
            format_results(old_hits, old_titles),
            format_results(new_hits, new_titles),
            status,
        )

    return search


HEADER = f"""
# BestBuy full-catalog re-index: Currently Live vs. New Re-Index

Dense retrieval over the **real, full {CATALOG_SIZE:,}-product BestBuy catalog** — not a
judged closed pool. Any query is valid here; that is the point of this demo versus the
closed-pool one.

Both columns use the **same, unmodified embedding model** (selectable below). The only
difference is the text each product was indexed with:

| | indexing text |
|---|---|
| **Currently Live** | product `name` only, HTML entities left unescaped (`&amp;amp;`, `&amp;quot;` …) |
| **New Re-Index** | `name` + `manufacturer` + `categoryPath` + `class`, HTML-unescaped |

No retraining, no new model — richer indexing text and cleaner display titles only.

> **Read this before drawing conclusions.** Aggregate validation (Pattern 30, Part F) found
> the LLM-judged **junk-rate is flat overall at full catalog scale**. The re-index fixes some
> queries and regresses others; it is not a measured general improvement. This demo is a
> tool for **inspecting individual queries**, not evidence of aggregate lift.

*Note on "Currently Live": the re-indexed vectors were being uploaded to the
`dtunkelang/bag-of-documents-bestbuy` dataset while this demo was built, and that upload has
since completed. The left column therefore shows the encoding that was live up until
publication, read from a local pre-upload cache snapshot — it is the "before" side, not a
live re-fetch.*
"""


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--old-dir", default=DEFAULT_OLD_DIR, help="pre-upload HF cache snapshot")
    ap.add_argument("--new-dir", default=DEFAULT_NEW_DIR, help="re-indexed artifacts dir")
    ap.add_argument("--port", type=int, default=7862)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--warm", default=None, help="run this query at startup, print, and exit")
    args = ap.parse_args()

    for d in (args.old_dir, args.new_dir):
        if not os.path.isdir(d):
            raise SystemExit(f"missing directory: {d}")
    for key in ("base", "bod"):
        for d in (args.old_dir, args.new_dir):
            if not os.path.exists(_vec_path(d, key)):
                raise SystemExit(f"missing {_vec_path(d, key)}")

    old_titles, new_titles = load_titles(args.old_dir, args.new_dir)
    cache = VectorCache(args.old_dir, args.new_dir)
    search = build_search_fn(cache, old_titles, new_titles)

    if args.warm:
        for q in args.warm.split("|"):
            out_old, out_new, status = search(q, DEFAULT_MODEL_LABEL)
            print(f"\n===== {q} =====\n{status}")
            print("--- OLD ---\n", out_old)
            print("--- NEW ---\n", out_new)
        return

    with gr.Blocks(title="BestBuy re-index: live vs new") as demo:
        gr.Markdown(HEADER)
        with gr.Row():
            model_sel = gr.Dropdown(
                choices=list(MODEL_CHOICES),
                value=DEFAULT_MODEL_LABEL,
                label="embedding model (both columns use the same one)",
            )
        with gr.Row():
            qbox = gr.Textbox(
                label="query", placeholder="e.g. apple tablet", scale=4, submit_btn=False
            )
            btn = gr.Button("Search", variant="primary", scale=1)
        status = gr.HTML()
        with gr.Row():
            with gr.Column():
                gr.Markdown("### Currently Live\n*name-only indexing, HTML entities unescaped*")
                out_old = gr.HTML()
            with gr.Column():
                gr.Markdown("### New Re-Index\n*name + manufacturer + categoryPath + class*")
                out_new = gr.HTML()
        gr.Examples(examples=[[q] for q in EXAMPLE_QUERIES], inputs=[qbox])

        for trigger in (btn.click, qbox.submit):
            trigger(search, inputs=[qbox, model_sel], outputs=[out_old, out_new, status])

    demo.launch(server_name=args.host, server_port=args.port, share=False)


if __name__ == "__main__":
    main()
