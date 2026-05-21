#!/usr/bin/env python3
"""Lightweight ColBERTv2 max-sim scorer for (query, doc) pair reranking.

Implements ColBERT-style late interaction without the colbert-ai package
(which has a transformers 5.x incompatibility and a voyager/fast-plaid
Python 3.14 wheel gap). Skips the PLAID index entirely — designed to
*rerank* an existing BM25 top-K candidate pool with multi-vector max-sim
scores. This is the cleanest way to test whether multi-vector adds
orthogonal signal to the CC5 stack on ESCI-US.

Architecture loaded directly from `colbert-ir/colbertv2.0`:
- BertModel (bert.* keys) — 768-dim hidden
- Linear projection (linear.weight, shape (128, 768)) — 768 → 128
- Per-token outputs L2-normalized

Score(q, d) = sum_i max_j (Q_i · D_j) over valid (non-[PAD], non-[MASK]) D tokens.

Usage:
    .venv/bin/python evaluation/colbert_maxsim.py \\
        --data-dir esci_us_data \\
        --queries-file test_queries.jsonl \\
        --qrels-file test_qrels.jsonl \\
        --candidate-pool combined_index_us_minilm/ce_top100_candidates.npy \\
        --max-queries 100
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import os  # noqa: E402
import time  # noqa: E402
from collections import defaultdict  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
from huggingface_hub import hf_hub_download  # noqa: E402
from safetensors.torch import load_file  # noqa: E402
from transformers import AutoTokenizer, BertModel  # noqa: E402

COLBERT_MODEL = "colbert-ir/colbertv2.0"
K_EVAL = 10


class ColBERTScorer:
    def __init__(self, device=None, doc_maxlen=180, query_maxlen=32):
        self.device = device or (
            "mps"
            if torch.backends.mps.is_available()
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        print(f"loading ColBERTv2 from {COLBERT_MODEL} on {self.device}...", flush=True)
        t0 = time.time()
        self.tokenizer = AutoTokenizer.from_pretrained(COLBERT_MODEL)
        self.bert = BertModel.from_pretrained(COLBERT_MODEL).to(self.device).eval()
        # Load the 768->128 projection manually
        weights_path = hf_hub_download(COLBERT_MODEL, "model.safetensors")
        sd = load_file(weights_path)
        self.linear = torch.nn.Linear(768, 128, bias=False).to(self.device)
        with torch.no_grad():
            self.linear.weight.copy_(sd["linear.weight"])
        self.linear.eval()
        self.doc_maxlen = doc_maxlen
        self.query_maxlen = query_maxlen
        # ColBERT uses [unused0] / [unused1] as [Q] / [D] markers.
        self.q_marker = self.tokenizer.convert_tokens_to_ids("[unused0]")
        self.d_marker = self.tokenizer.convert_tokens_to_ids("[unused1]")
        # MASK token is the padding character used for query expansion.
        self.mask_id = self.tokenizer.mask_token_id
        self.pad_id = self.tokenizer.pad_token_id
        print(f"  loaded in {time.time() - t0:.1f}s", flush=True)

    @torch.inference_mode()
    def encode_queries(self, queries: list[str], batch_size: int = 32):
        """Returns (N, query_maxlen, 128) tensor of L2-normalized query token vecs.

        ColBERT query expansion: queries are padded with [MASK] to query_maxlen
        before encoding, so attention sees the masks as "ghost" tokens that
        get free per-position predictions. This is part of the trained behavior.
        """
        all_qv = []
        for i in range(0, len(queries), batch_size):
            batch = queries[i : i + batch_size]
            # Tokenize with truncation but no padding (we'll pad manually with MASK)
            enc = self.tokenizer(
                batch,
                truncation=True,
                max_length=self.query_maxlen - 2,  # reserve [CLS], [Q]
                return_attention_mask=False,
                padding=False,
                return_tensors=None,
            )
            ids_list = []
            for ids in enc["input_ids"]:
                # Strip the auto-added [CLS]/[SEP] then build: [CLS] [Q] tokens...
                if ids[0] == self.tokenizer.cls_token_id:
                    ids = ids[1:]
                if ids and ids[-1] == self.tokenizer.sep_token_id:
                    ids = ids[:-1]
                full = [self.tokenizer.cls_token_id, self.q_marker, *ids]
                # Pad to query_maxlen with [MASK]
                while len(full) < self.query_maxlen:
                    full.append(self.mask_id)
                full = full[: self.query_maxlen]
                ids_list.append(full)
            input_ids = torch.tensor(ids_list, device=self.device)
            attn = torch.ones_like(input_ids)
            h = self.bert(input_ids, attention_mask=attn).last_hidden_state
            v = self.linear(h)
            v = torch.nn.functional.normalize(v, dim=-1)
            all_qv.append(v)
        return torch.cat(all_qv, dim=0)  # (N, query_maxlen, 128)

    @torch.inference_mode()
    def encode_docs(self, docs: list[str], batch_size: int = 32):
        """Returns (N, max_actual_len, 128) padded fp32 tensor + (N, max_actual_len) bool mask.

        For docs we DON'T pad-with-MASK (only [CLS] [D] tokens... [SEP] then PAD).
        The returned mask excludes [PAD] tokens from max-sim aggregation.
        """
        all_dv = []
        all_mask = []
        max_len = 0
        for i in range(0, len(docs), batch_size):
            batch = docs[i : i + batch_size]
            enc = self.tokenizer(
                batch,
                truncation=True,
                max_length=self.doc_maxlen - 2,
                padding=False,
                return_attention_mask=False,
                return_tensors=None,
            )
            ids_list = []
            for ids in enc["input_ids"]:
                if ids[0] == self.tokenizer.cls_token_id:
                    ids = ids[1:]
                if ids and ids[-1] == self.tokenizer.sep_token_id:
                    ids = ids[:-1]
                full = [
                    self.tokenizer.cls_token_id,
                    self.d_marker,
                    *ids,
                    self.tokenizer.sep_token_id,
                ]
                ids_list.append(full)
            # Pad batch to its own max length
            batch_max = max(len(x) for x in ids_list)
            input_ids = torch.full(
                (len(ids_list), batch_max), self.pad_id, dtype=torch.long, device=self.device
            )
            attn = torch.zeros((len(ids_list), batch_max), dtype=torch.long, device=self.device)
            for j, ids in enumerate(ids_list):
                input_ids[j, : len(ids)] = torch.tensor(ids, device=self.device)
                attn[j, : len(ids)] = 1
            h = self.bert(input_ids, attention_mask=attn).last_hidden_state
            v = self.linear(h)
            v = torch.nn.functional.normalize(v, dim=-1)
            # Mask: True where attn=1 AND token != [CLS] AND != [D] marker
            # (ColBERT typically includes [CLS] and [D] in scoring; keep all attn=1.)
            mask = attn.bool()
            all_dv.append(v.cpu())
            all_mask.append(mask.cpu())
            max_len = max(max_len, batch_max)
        # Pad everything to global max_len
        N = sum(v.shape[0] for v in all_dv)
        D = torch.zeros(N, max_len, 128, dtype=all_dv[0].dtype)
        M = torch.zeros(N, max_len, dtype=torch.bool)
        off = 0
        for v, m in zip(all_dv, all_mask):
            n, ln, _ = v.shape
            D[off : off + n, :ln] = v
            M[off : off + n, :ln] = m
            off += n
        return D, M

    @staticmethod
    def max_sim(Q: torch.Tensor, D: torch.Tensor, D_mask: torch.Tensor) -> torch.Tensor:
        """Q: (Nq, Lq, dim), D: (Nd, Ld, dim), D_mask: (Nd, Ld) bool.

        Returns (Nq, Nd) max-sim score matrix.
        """
        # (Nq, Lq, Nd, Ld)
        sim = torch.einsum("qld,nkd->qlnk", Q, D)
        # Mask invalid doc tokens to -inf so max ignores them
        sim = sim.masked_fill(~D_mask[None, None, :, :], -1e9)
        # max over doc tokens, sum over query tokens
        per_q_per_d = sim.max(dim=-1).values  # (Nq, Lq, Nd)
        return per_q_per_d.sum(dim=1)  # (Nq, Nd)


def per_query_metrics(retrieved_pids, qrels_q, k=K_EVAL, min_rel=2, exact_rel=3):
    pos_e = {pid for pid, g in qrels_q.items() if g >= exact_rel}
    pos_es = {pid for pid, g in qrels_q.items() if g >= min_rel}
    if not pos_es:
        return None
    top_k = retrieved_pids[:k]
    recall = sum(1 for p in top_k if p in pos_es) / len(pos_es)
    gains = [1.0 if p in pos_e else (0.1 if p in pos_es else 0.0) for p in top_k]
    dcg = sum(g / math.log2(i + 2) for i, g in enumerate(gains))
    ideal = sorted((1.0 if p in pos_e else 0.1 for p in pos_es), reverse=True)[:k]
    idcg = sum(g / math.log2(i + 2) for i, g in enumerate(ideal))
    ndcg = dcg / idcg if idcg > 0 else 0.0
    e1 = 1.0 if pos_e and top_k and top_k[0] in pos_e else 0.0 if pos_e else float("nan")
    e3 = sum(1 for p in top_k[:3] if p in pos_e) / min(3, len(pos_e)) if pos_e else float("nan")
    return recall, ndcg, e1, e3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--queries-file", default="test_queries.jsonl")
    ap.add_argument("--qrels-file", default="test_qrels.jsonl")
    ap.add_argument(
        "--candidate-pool",
        required=True,
        help="path to (N_q, K) int array of catalog positions (e.g. ce_top100_candidates.npy)",
    )
    ap.add_argument(
        "--index-titles",
        default=None,
        help="path to the titles.json corresponding to the candidate-pool positions "
        "(default: combined_index_us_minilm/titles.json for ESCI-US)",
    )
    ap.add_argument(
        "--qids-source",
        default="combined_index_us_minilm/bm25s_qids.json",
        help="json file with the qid order matching candidate-pool rows",
    )
    ap.add_argument("--top-k", type=int, default=100)
    ap.add_argument("--max-queries", type=int, default=0, help="0 = all")
    ap.add_argument("--batch-size-docs", type=int, default=32)
    ap.add_argument("--min-relevance", type=int, default=2)
    ap.add_argument("--exact-relevance", type=int, default=3)
    ap.add_argument("--out-path", default=None)
    args = ap.parse_args()

    data = Path(args.data_dir).resolve()
    print(f"corpus: {data.name}  top_k: {args.top_k}", flush=True)

    qrels = defaultdict(dict)
    with open(data / args.qrels_file) as f:
        for line in f:
            r = json.loads(line)
            qrels[r["query_id"]][r["product_id"]] = r["relevance"]
    queries_all = {}
    with open(data / args.queries_file) as f:
        for line in f:
            d = json.loads(line)
            queries_all[d["query_id"]] = d["query"]

    cand = np.load(args.candidate_pool)
    with open(args.qids_source) as f:
        qids = json.load(f)
    if cand.shape[0] != len(qids):
        raise SystemExit(f"candidate-pool rows ({cand.shape[0]}) != qids ({len(qids)})")
    K = min(args.top_k, cand.shape[1])

    # Load index titles (the doc text indexed by the candidate positions)
    index_titles_path = args.index_titles or os.path.join(
        os.path.dirname(args.candidate_pool), "titles.json"
    )
    with open(index_titles_path) as f:
        index_titles = json.load(f)

    # Load corpus pid->title map for evaluation
    with open(data / "product_ids.json") as f:
        esci_pids = json.load(f)
    with open(data / "titles.json") as f:
        esci_titles = json.load(f)
    title_to_pid = {t: p for p, t in zip(esci_pids, esci_titles)}
    pos_to_pid = [title_to_pid.get(t) for t in index_titles]

    # Subsample queries if requested
    if args.max_queries and args.max_queries < len(qids):
        rng = np.random.default_rng(42)
        sample_idx = sorted(rng.choice(len(qids), size=args.max_queries, replace=False).tolist())
    else:
        sample_idx = list(range(len(qids)))

    scorer = ColBERTScorer()

    # Encode all queries up front (cheap)
    queries = [queries_all[qids[i]] for i in sample_idx]
    print(f"\nencoding {len(queries):,} queries...", flush=True)
    t0 = time.time()
    Q = scorer.encode_queries(queries, batch_size=32)
    print(f"  Q shape: {tuple(Q.shape)}  time: {time.time() - t0:.1f}s", flush=True)

    # For each query, encode its K candidate docs and score
    n_pairs_total = len(sample_idx) * K
    out_scores = np.full((len(sample_idx), K), np.nan, dtype=np.float32)

    print(
        f"\nscoring {len(sample_idx):,} queries × {K} candidates = {n_pairs_total:,} pairs...",
        flush=True,
    )
    t0 = time.time()
    n_done = 0
    for qi_local, qi_global in enumerate(sample_idx):
        positions = cand[qi_global, :K]
        valid_positions = positions[positions >= 0]
        if len(valid_positions) == 0:
            continue
        docs = [index_titles[int(p)] for p in valid_positions]
        D, D_mask = scorer.encode_docs(docs, batch_size=args.batch_size_docs)
        D = D.to(scorer.device)
        D_mask = D_mask.to(scorer.device)
        # Score against this single query
        scores = ColBERTScorer.max_sim(Q[qi_local : qi_local + 1], D, D_mask)
        scores = scores.cpu().numpy().squeeze()
        # Place into valid positions
        j = 0
        for k_idx in range(K):
            if positions[k_idx] >= 0:
                out_scores[qi_local, k_idx] = scores[j]
                j += 1
        n_done += len(valid_positions)
        if (qi_local + 1) % 50 == 0 or qi_local == len(sample_idx) - 1:
            elapsed = time.time() - t0
            done_q = qi_local + 1
            rate_q = done_q / max(elapsed, 1e-3)
            eta_min = (len(sample_idx) - done_q) / max(rate_q, 1e-3) / 60
            print(
                f"  {done_q:,}/{len(sample_idx):,} queries  ({rate_q:.2f} q/s)  eta {eta_min:.1f}m",
                flush=True,
            )

    # Save scores
    if args.out_path:
        os.makedirs(os.path.dirname(args.out_path) or ".", exist_ok=True)
        np.save(args.out_path, out_scores)
        print(f"\nsaved ColBERT scores -> {args.out_path}", flush=True)

    # Eval: rerank candidate pool by ColBERT score, compute R@10/E@1/nDCG
    rs, ns, e1s, e3s = [], [], [], []
    for qi_local, qi_global in enumerate(sample_idx):
        qid = qids[qi_global]
        s = out_scores[qi_local].copy()
        s[np.isnan(s)] = -np.inf
        order = np.argsort(-s)[:K_EVAL]
        ordering = [pos_to_pid[int(cand[qi_global, j])] for j in order if cand[qi_global, j] >= 0]
        ordering = [p for p in ordering if p is not None]
        m = per_query_metrics(
            ordering, qrels[qid], min_rel=args.min_relevance, exact_rel=args.exact_relevance
        )
        if m is None:
            continue
        r, nd, e1, e3 = m
        rs.append(r)
        ns.append(nd)
        if not math.isnan(e1):
            e1s.append(e1)
            e3s.append(e3)

    print(
        f"\nColBERTv2 max-sim over BM25 top-{K} candidates  "
        f"R@10 {np.mean(rs):.4f}  nDCG@10 {np.mean(ns):.4f}  "
        f"E@1 {np.mean(e1s):.4f}  E@3 {np.mean(e3s):.4f}  (n={len(rs):,})",
        flush=True,
    )


if __name__ == "__main__":
    main()
