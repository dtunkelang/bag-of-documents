#!/usr/bin/env python3
"""LOCAL-ONLY Gradio prototype: query->category nearest-centroid boosting.

This is the interactive companion to Pattern 30, Part E in
`evaluation/CHS_RESULTS.md` -- the *category-classification* mechanism, not
the enriched-embedding-text mechanism (Part D-a).

The idea under test: the deployed embedding space already knows what category
a query belongs to, without any retraining and without an LLM call. Take the
per-`class` centroid of the ORIGINAL (title-only) product vectors, cosine the
query vector against those 179 centroids, keep the top-3, and then stably
reorder the dense ranking so candidates whose `class` is in that top-3 sort
ahead of the ones that aren't. Ties inside each group keep their original
dense order, and non-matching candidates are demoted rather than removed --
a soft boost, because top-3 category recall misses ~25-37% of the time and a
hard filter would turn those misses into empty result sets.

Everything here runs against the CLOSED POOL of 6,890 products cached by
`eval_bestbuy_category_enrichment_prototype.py` (phases catalog/embed). That
pool is the union of the base-MiniLM and BoD-MiniLM top-10s over a 250-query
eval sample. It is NOT the 1.27M-product BestBuy catalog. Queries outside
that sample's neighbourhood will return whatever happens to be nearest in a
6,890-product grab bag, which is usually nonsense.

Not deployed anywhere. Does not touch `space_demo_bestbuy/`.

Run:
    python evaluation/local_demo_bestbuy_enrichment.py [--port 7861]
"""

import argparse
import html as html_mod
import json
import os
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

WORK_DIR = Path(os.environ.get("CATENRICH_DIR", "/tmp/bestbuy_catenrich"))
TAG = "bestbuy"
FIELDS_PATH = WORK_DIR / f"catalog_fields_{TAG}.json"
VECS_PATH = WORK_DIR / f"enriched_vecs_{TAG}.npz"

# Per-(query, product) LLM junk labels, unioned over the two cached judge runs
# that produced evaluation/results/bestbuy_llm_judge_junkrate.json and
# evaluation/results/bestbuy_bm25_junk_gate.json. The summary JSONs in
# evaluation/results/ don't carry product_ids, so the per-pair JSONLs those
# runs wrote are the usable source.
JUDGE_JSONLS = (
    Path(os.environ.get("JUNK_DIR", "/tmp/bestbuy_junkrate")) / f"junk_judge_{TAG}.jsonl",
    Path(os.environ.get("GATE_DIR", "/tmp/bestbuy_bm25_gate")) / f"gate_judge_{TAG}.jsonl",
)
JUNK_THRESHOLD = 0.5  # p_yes < 0.5 == judged wrong-category junk

BASE_MODEL = "all-MiniLM-L6-v2"
BOD_MODEL = "dtunkelang/bag-of-documents-bestbuy-minilm"
MODEL_CHOICES = ("base MiniLM (off-the-shelf)", "BoD MiniLM (fine-tuned)")
MODEL_KEYS = {MODEL_CHOICES[0]: "base", MODEL_CHOICES[1]: "bod"}

UNKNOWN_CLASS = "UNKNOWN"
TOP_K = 10
TOP_CLASSES = 3

# Queries verified to be in the cached eval sample AND to change the top-10
# once the boost is applied. Chosen by sweeping all 252 cached queries and
# keeping the ones with the largest drop in judged-junk count between the two
# columns. "ford focus" is the interesting one: the predicted classes look
# nonsensical (MOBILE ACCESSORIES / LENSES / GADGETS) yet the boost still
# clears out a top-10 of "Focus" CDs and cassettes in favour of Ford car
# install kits -- the centroid landed in roughly the right neighbourhood of
# the space even though the class *name* reads wrong.
EXAMPLES = [
    "apple tablet",
    "nokia phone",
    "sam sung galaxy tab",
    "motorola zoom tablet",
    "canon 50d",
    "joy sticks",
    "touch screen car audio",
    "ford focus",
]


# --------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------
def product_class(rec):
    c = (rec.get("class") or "").strip().upper()
    return c if c else UNKNOWN_CLASS


def _l2(v):
    n = float(np.linalg.norm(v))
    return v / n if n > 0 else v


def load_judge_labels():
    """(query, product_id) -> p_yes, unioned over both cached judge runs."""
    out = {}
    for path in JUDGE_JSONLS:
        if not path.exists():
            print(f"  WARNING: judge labels missing at {path}", flush=True)
            continue
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if r.get("p_yes") is None:
                    continue
                out[(r["query"], r["product_id"])] = float(r["p_yes"])
    return out


def load_resources():
    for p in (FIELDS_PATH, VECS_PATH):
        if not p.exists():
            raise SystemExit(
                f"missing cached artifact {p}\n"
                "Run: python evaluation/eval_bestbuy_category_enrichment_prototype.py "
                "--phase catalog then --phase embed"
            )
    print("loading cached closed-pool artifacts...", flush=True)
    with open(FIELDS_PATH) as f:
        fields = json.load(f)["fields"]
    z = np.load(VECS_PATH, allow_pickle=True)
    pids = [str(p) for p in z["subset_pids"]]
    sample_queries = sorted(str(q) for q in z["queries"])

    titles = [html_mod.unescape(fields[p].get("name") or p).strip() for p in pids]
    classes = [product_class(fields[p]) for p in pids]

    # Per-class centroids over the ORIGINAL (title-only) vectors -- the space
    # as deployed. Part E measured the centroid probe on exactly these, so the
    # demo stays consistent with the reported numbers.
    by_class = {}
    for i, c in enumerate(classes):
        if c != UNKNOWN_CLASS:
            by_class.setdefault(c, []).append(i)
    class_names = sorted(by_class)

    device = (
        "mps"
        if torch.backends.mps.is_available()
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"  {len(pids):,} products, {len(class_names)} classes, device={device}", flush=True)

    models, prod_vecs, centroids = {}, {}, {}
    for key, model_id in (("base", BASE_MODEL), ("bod", BOD_MODEL)):
        v = np.ascontiguousarray(z[f"{key}_prod_orig"], dtype=np.float32)
        prod_vecs[key] = v
        centroids[key] = np.stack([_l2(v[np.array(by_class[c])].mean(axis=0)) for c in class_names])
        models[key] = SentenceTransformer(model_id, device=device)
        print(f"  loaded {model_id}", flush=True)

    labels = load_judge_labels()
    print(f"  {len(labels):,} cached (query, product) judge labels", flush=True)

    return {
        "pids": pids,
        "titles": titles,
        "classes": classes,
        "class_names": class_names,
        "prod_vecs": prod_vecs,
        "centroids": centroids,
        "models": models,
        "labels": labels,
        "sample_queries": set(sample_queries),
        "n_sample_queries": len(sample_queries),
    }


# --------------------------------------------------------------------------
# retrieval
# --------------------------------------------------------------------------
def _rank01(x):
    """Dense score -> [0,1] by rank, so the class flag dominates cleanly."""
    order = np.argsort(-x, kind="stable")
    out = np.empty(len(x), dtype=np.float64)
    out[order] = np.linspace(1.0, 0.0, num=len(x), endpoint=True)
    return out


def classify_query(R, model_key, qv, top_n=TOP_CLASSES):
    """Nearest-centroid query -> class. No LLM call: this is the point."""
    sims = R["centroids"][model_key] @ qv
    order = np.argsort(-sims)[:top_n]
    return [(R["class_names"][j], float(sims[j])) for j in order]


def rank(R, model_key, query, k=TOP_K, top_n=TOP_CLASSES):
    """(predicted classes, baseline top-k, category-boosted top-k).

    `pred` always carries TOP_CLASSES entries so the UI can show the full
    shortlist, but only the first `top_n` of them are treated as compatible.
    """
    model = R["models"][model_key]
    qv = model.encode([query], normalize_embeddings=True, show_progress_bar=False).astype(
        np.float32
    )[0]
    pred = classify_query(R, model_key, qv, top_n=max(TOP_CLASSES, top_n))
    pred_set = {c for c, _ in pred[:top_n]}

    sims = R["prod_vecs"][model_key] @ qv
    compat = np.array([1.0 if c in pred_set else 0.0 for c in R["classes"]])
    # Stable partition on the compatibility flag, ties broken by the ORIGINAL
    # dense score. Non-matching candidates are demoted, never dropped.
    boosted_score = compat * 10.0 + _rank01(sims)

    def topk(score):
        idx = np.argsort(-score, kind="stable")[:k]
        return [(int(i), float(sims[i]), bool(compat[i])) for i in idx]

    return pred, topk(sims), topk(boosted_score)


# --------------------------------------------------------------------------
# rendering
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


def format_results(hits, R, query):
    """HTML table -- avoids Markdown's column-count fragility on titles with
    '|' or newlines, and keeps short fields on one line so '10' doesn't wrap.

    The badge column is the cached LLM junk label for this exact
    (query, product) pair, where one exists. Unlike the deployed Space there
    are no clicked-product labels to highlight here.
    """
    nw = "white-space:nowrap;padding:2px 6px"
    rows = []
    for rank_i, (i, s, is_compat) in enumerate(hits, 1):
        p_yes = R["labels"].get((query, R["pids"][i]))
        if p_yes is None:
            badge = "<span style='color:#bbb'>-</span>"
        elif p_yes < JUNK_THRESHOLD:
            badge = "<span title='LLM judge: wrong category'>&#9888;&#65039; junk</span>"
        else:
            badge = "<span style='color:#2a2' title='LLM judge: right category'>ok</span>"
        cls = _html_escape(R["classes"][i])
        cls_cell = (
            f"<b style='color:#176'>{cls}</b>"
            if is_compat
            else f"<span style='color:#999'>{cls}</span>"
        )
        rows.append(
            f"<tr><td style='{nw};text-align:right'>{rank_i}</td>"
            f"<td style='{nw};text-align:center;font-size:12px'>{badge}</td>"
            f"<td style='padding:2px 6px'>{_html_escape(R['titles'][i])}</td>"
            f"<td style='{nw};font-size:11px'>{cls_cell}</td>"
            f"<td style='{nw};text-align:right;font-family:monospace'>{s:.3f}</td></tr>"
        )
    return (
        "<table style='width:100%;font-size:14px;border-collapse:collapse'>"
        "<tr style='border-bottom:2px solid #ccc'>"
        f"<th style='{nw};text-align:right'>#</th>"
        f"<th style='{nw};text-align:center'>judge</th>"
        "<th style='padding:2px 6px;text-align:left'>title</th>"
        f"<th style='{nw};text-align:left'>class</th>"
        f"<th style='{nw};text-align:right'>sim</th></tr>" + "".join(rows) + "</table>"
    )


def format_classes(pred, top_n):
    """Predicted classes as chips. The first `top_n` are the ones actually
    boosted; the rest are shown greyed so the shortlist stays inspectable."""
    chips = []
    for rank_i, (c, s) in enumerate(pred, 1):
        on = rank_i <= top_n
        style = (
            ("background:#0b4f4a;color:#fff" if on else "background:#eee;color:#888")
            + ";display:inline-block;margin:0 6px 0 0;padding:4px 12px;border-radius:12px;font-size:14px"
        )
        weight = "700" if on else "400"
        chips.append(
            f"<span style='{style};font-weight:{weight}'>{rank_i}. {_html_escape(c)} "
            f"<span style='font-weight:400;opacity:.75;font-family:monospace'>{s:.3f}</span></span>"
        )
    return (
        "<div style='padding:10px 0'>"
        "<span style='font-size:13px;color:#666'>query classified by nearest class "
        "centroid (no LLM, no retraining) &rarr;&nbsp;</span><br>"
        "<div style='padding-top:6px'>" + "".join(chips) + "</div></div>"
    )


# --------------------------------------------------------------------------
# app
# --------------------------------------------------------------------------
NOTICE = """<div style="border:2px solid #c0392b;background:#fdf0ee;border-radius:8px;
padding:12px 16px;margin-bottom:8px">
<b style="color:#c0392b">LOCAL PROTOTYPE &mdash; CLOSED POOL OF {n_prod:,} PRODUCTS, NOT THE
1.27M-PRODUCT CATALOG.</b><br>
Everything below ranks within the union of the base-MiniLM and BoD-MiniLM top-10s
over a {n_q}-query eval sample. Nothing new is retrieved. Queries unrelated to that
sample will return poor or meaningless results &mdash; this is a mechanism inspector,
not a search engine. Full writeup: <code>evaluation/CHS_RESULTS.md</code>, Pattern 30
(Part E, query&rarr;category nearest-centroid classification).
</div>"""

HEADER = """# Category-boosted ranking (Pattern 30, Part E)

Same query, same model, same product vectors on both sides. The only difference
is what happens *after* the dense scoring:

* **Baseline** &mdash; plain cosine ranking over the original (title-only) product
  embeddings. This is the space as deployed.
* **Category-boosted** &mdash; the query is classified into a product `class` by
  cosine against the 179 per-class centroids of those same embeddings, then the
  ranking is stably reordered so products in the query's top-N predicted classes
  sort ahead of products that aren't. Ties keep their dense order; non-matching
  products are demoted, **not** filtered out (top-3 class recall misses 25-37% of
  the time, so a hard filter would turn those misses into empty result pages).

No retraining, no re-indexing, no LLM call at query time &mdash; the category
signal is read straight out of the embedding space that is already deployed.

**The N slider is the interesting knob.** N=3 is the safe setting Part E measured,
but it lets in near-miss classes: for `apple tablet` the base model's #1 class is
`TABLET` and its #2 is `TABLET ACCESSORIES`, so at N=3 the iPad keyboards survive
the boost. Drop to N=1 and they don't. That is the whole precision/recall tradeoff
in one control.

The **judge** column is the cached `gpt-4o-mini` wrong-category label for that
exact (query, product) pair, and only exists for pairs that appeared in the
original sample's top-10s. `-` means "never judged", not "fine". Note that this
judge is itself unreliable on the flagship anecdote &mdash; it waved through the
`apple tablet` keyboards and flagged actual Acer tablets as junk.
"""


def build_app(R):
    # Imported lazily: gradio is a local-demo-only dependency, not in
    # requirements.txt, and tests/test_imports.py imports every evaluation/
    # script in clean CI.
    import gradio as gr

    def run(query, model_choice, top_n):
        query = (query or "").strip().lower()
        top_n = int(top_n)
        if not query:
            empty = "_(enter a query)_"
            return "", empty, empty, ""
        model_key = MODEL_KEYS.get(model_choice, "base")
        pred, base_hits, boost_hits = rank(R, model_key, query, top_n=top_n)

        moved = [
            r for r, ((i, _, _), (j, _, _)) in enumerate(zip(base_hits, boost_hits), 1) if i != j
        ]
        n_compat = sum(1 for _, _, c in base_hits if c)

        def n_junk(hits):
            return sum(
                1
                for i, _, _ in hits
                if (R["labels"].get((query, R["pids"][i])) or 1.0) < JUNK_THRESHOLD
            )

        in_sample = query in R["sample_queries"]
        if in_sample:
            head = (
                f"**In the cached eval sample** &mdash; judged junk in top-10: "
                f"baseline **{n_junk(base_hits)}**, boosted **{n_junk(boost_hits)}**."
            )
        else:
            head = (
                "**Free-form query** &mdash; outside the cached sample, so there are no "
                "judge labels (the judge column will be all `-`) and the closed pool "
                "probably doesn't contain what you want."
            )
        note = (
            f"{head}  Baseline top-10 already in a predicted class: **{n_compat}/10**. "
            f"Ranks changed by the boost: **{len(moved)}/10**."
        )
        return (
            format_classes(pred, top_n),
            format_results(base_hits, R, query),
            format_results(boost_hits, R, query),
            note,
        )

    with gr.Blocks(title="Category-boosted BestBuy retrieval (local prototype)") as demo:
        gr.HTML(NOTICE.format(n_prod=len(R["pids"]), n_q=R["n_sample_queries"]))
        gr.Markdown(HEADER)
        with gr.Row():
            model_sel = gr.Radio(
                choices=list(MODEL_CHOICES),
                value=MODEL_CHOICES[0],
                label="Model (the category effect is strongest on the off-the-shelf base)",
                scale=2,
            )
            top_n_sel = gr.Slider(
                minimum=1,
                maximum=TOP_CLASSES,
                value=TOP_CLASSES,
                step=1,
                label="N: how many predicted classes count as compatible",
                scale=1,
            )
        with gr.Row():
            q = gr.Textbox(
                label="Query",
                placeholder="try one of the examples or type your own",
                lines=1,
                scale=4,
            )
            search_btn = gr.Button("Search", variant="primary", scale=1)
        classes_out = gr.HTML()
        note = gr.Markdown()
        with gr.Row():
            with gr.Column():
                gr.Markdown("### Baseline (dense only)")
                base_out = gr.HTML()
            with gr.Column():
                gr.Markdown("### Category-boosted (predicted classes first)")
                boost_out = gr.HTML()

        inputs = [q, model_sel, top_n_sel]
        outputs = [classes_out, base_out, boost_out, note]
        gr.Examples(
            examples=[[e, MODEL_CHOICES[0], TOP_CLASSES] for e in EXAMPLES],
            inputs=inputs,
            outputs=outputs,
            fn=run,
            run_on_click=True,
            cache_examples=False,
        )
        search_btn.click(run, inputs=inputs, outputs=outputs)
        q.submit(run, inputs=inputs, outputs=outputs)
        model_sel.change(run, inputs=inputs, outputs=outputs)
        top_n_sel.release(run, inputs=inputs, outputs=outputs)

    return demo, run


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--port", type=int, default=int(os.environ.get("PORT", 7861)))
    ap.add_argument("--host", default="127.0.0.1")
    args = ap.parse_args()

    R = load_resources()
    demo, _ = build_app(R)
    demo.launch(server_name=args.host, server_port=args.port, share=False)


if __name__ == "__main__":
    main()
