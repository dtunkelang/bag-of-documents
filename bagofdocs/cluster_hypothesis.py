"""Cluster Hypothesis Score (CHS) — a-priori metric for BoD-readiness of a corpus.

Operationalizes the van Rijsbergen cluster hypothesis: "closely related
documents tend to be relevant to the same requests." For BoD specifically:
documents that share a query (i.e., would land in the same bag) should be
close to each other in embedding space, and closer to each other than to
documents that don't share the query.

For a (corpus, encoder) pair, this module computes:

    SCHS (Simple Cluster Hypothesis Score):
        How much closer in-bag pairs are to each other than random pairs.
            SCHS = (mean_intra - mean_random) / (1 - mean_random)
        Range [0, 1]. Computable on any corpus with positive labels.

    HCHS (Hard Cluster Hypothesis Score):
        How much closer in-bag pairs are to each other than to within-query
        labeled negatives. Stronger test — measures separability against
        confusable negatives.
            HCHS = (mean_intra - mean_inter_neg) / (mean_intra - mean_random)
        Range [0, 1]. Requires explicit hard negatives in qrels.

    strong_inv_rate:
        Fraction of queries where some pos-neg cosine exceeds the best
        pos-pos cosine. The cluster hypothesis fails for that query.
        Computable when explicit negatives are available.

Calibration (under all-MiniLM-L6-v2 unless noted):

    Empirically BoD-positive corpora tend to have SCHS >= 0.5; empirically
    BoD-negative corpora tend to have SCHS < 0.4. The metric also tracks
    BoD lift magnitude — ESCI-Spanish, with a smaller empirical BoD lift
    than ESCI-US, sits between the two clusters at SCHS ~0.42-0.45.

    SEE evaluation/CHS_RESULTS.md FOR THE FULL CALIBRATION TABLE.

How to apply to a NEW corpus:

    1. Get qrels (query -> {doc_id: relevance_grade}), product/doc IDs in
       a list, and parallel titles/text in another list.
    2. Pick an encoder appropriate for the corpus's language(s).
    3. Call:
            from bagofdocs.cluster_hypothesis import compute_chs
            result = compute_chs(qrels, pids, titles, "all-MiniLM-L6-v2")
       or use the CLI tool: evaluation/cluster_hypothesis_score.py
    4. Compare result["schs"] to the calibration table:
         >= 0.50  GREEN   — BoD likely to generalize
         0.40-0.50 YELLOW  — BoD may give a smaller lift; pilot first
         < 0.40   RED     — BoD unlikely to lift over base
       Joint factor to consider: result["n_pos_bearing"]. With <500 multi-
       positive queries, training scale will limit BoD even if SCHS is high.

Caveats:

    - SCHS is a (corpus, encoder) pair property. A different encoder will
      produce different numbers; the cross-corpus comparison is meaningful
      only with a fixed encoder.
    - HCHS depends on whether the qrels include genuine hard negatives.
      For BEIR-style positives-only corpora, HCHS is undefined.
    - The thresholds above are anchored to ESCI-US; adding more validated
      positive/negative corpora should refine them.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np


@dataclass
class ChsResult:
    """Structured CHS output for a (corpus, encoder) pair.

    SCHS / HCHS / strong_inv_rate are NaN when the corpus does not support
    the corresponding metric (e.g., HCHS requires >= 50 queries with
    explicit negatives).
    """

    n_pos_bearing: int
    n_explicit_neg: int
    n_products_touched: int
    mean_intra: float
    mean_inter_neg: float
    mean_random: float
    schs: float
    hchs: float
    strong_inv_rate: float
    has_explicit_negs: bool
    pp_means: list  # per-query intra-bag mean cosines (over all pos-bearing)
    pp_mins: list  # per-query intra-bag min cosines (over all pos-bearing)
    pn_means: list  # per-query inter-neg mean cosines (over explicit-neg subset)

    def to_dict(self) -> dict:
        d = {k: v for k, v in self.__dict__.items()}
        # Keep distributions out of summary serialization by default.
        for k in ("pp_means", "pp_mins", "pn_means"):
            d.pop(k, None)
        return d


def _resolve_partition(partition: str, pos_grade: int, neg_grade: int):
    """Return (is_pos, is_neg, label) for the chosen partition mode."""
    if partition == "strict":
        is_pos = lambda g: g == pos_grade  # noqa: E731
        is_neg = lambda g: g == neg_grade  # noqa: E731
        label = f"strict (grade={pos_grade} vs {neg_grade})"
    elif partition == "relaxed":
        # Relaxed: include the next grade above neg as positive too, and the
        # next grade below pos as negative. For ESCI 0/1/2/3 this is E+S vs I+C.
        is_pos = lambda g: g >= max(1, pos_grade - 1)  # noqa: E731
        is_neg = lambda g: g <= min(neg_grade + 1, pos_grade - 2)  # noqa: E731
        label = f"relaxed (>= {max(1, pos_grade - 1)} vs <= {min(neg_grade + 1, pos_grade - 2)})"
    else:
        raise ValueError(f"unknown partition: {partition!r}; expected 'strict' or 'relaxed'")
    return is_pos, is_neg, label


def _encode(texts, encoder_name, batch_size=128, show_progress=False):
    """Encode texts with a sentence-transformers model. Returns L2-normalized fp32."""
    import torch
    from sentence_transformers import SentenceTransformer

    device = (
        "mps"
        if torch.backends.mps.is_available()
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    model = SentenceTransformer(encoder_name, device=device)
    vecs = model.encode(
        texts,
        normalize_embeddings=True,
        batch_size=batch_size,
        show_progress_bar=show_progress,
    )
    return np.asarray(vecs, dtype=np.float32)


def compute_chs(
    qrels: dict,
    pids: list,
    titles: list,
    encoder_name: str,
    partition: str = "strict",
    pos_grade: int | None = None,
    neg_grade: int = 0,
    seed: int = 42,
    n_random_pairs: int = 200_000,
    cache_vecs: np.ndarray | None = None,
    min_pos_bearing: int = 50,
    min_explicit_neg: int = 50,
    verbose: bool = True,
) -> ChsResult:
    """Compute Cluster Hypothesis Score for (corpus, encoder).

    Args:
        qrels: dict mapping query_id -> {item_id: relevance_grade}.
        pids: list of all item ids in the catalog.
        titles: parallel list of titles/text for each item in `pids`.
        encoder_name: sentence-transformers model id (e.g. "all-MiniLM-L6-v2").
        partition: "strict" (only top-grade vs bottom-grade) or "relaxed"
            (include the adjacent grades).
        pos_grade: relevance grade considered positive. None = max grade in qrels.
        neg_grade: relevance grade considered negative.
        seed: PRNG seed for the random-pair baseline.
        n_random_pairs: number of random pairs to sample for mean_random.
        cache_vecs: optional precomputed product embeddings aligned with `pids`.
            If provided, skips the encoding step.
        min_pos_bearing: minimum number of queries with >=2 positives required
            to compute SCHS. Below this returns a ChsResult with NaN scores.
        min_explicit_neg: minimum number of queries with >=1 explicit negative
            required to compute HCHS. Below this leaves HCHS as NaN.
        verbose: print progress.

    Returns:
        ChsResult.
    """
    pid_to_pos = {p: i for i, p in enumerate(pids)}

    if pos_grade is None:
        pos_grade = max((max(qr.values()) for qr in qrels.values() if qr), default=1)

    is_pos, is_neg, _label = _resolve_partition(partition, pos_grade, neg_grade)

    pos_bearing = []  # list of (qid, [pos_pids])
    explicit_neg = []  # list of (qid, [pos_pids], [neg_pids])
    for qid, qr in qrels.items():
        pos = [p for p, g in qr.items() if is_pos(g) and p in pid_to_pos]
        if len(pos) < 2:
            continue
        pos_bearing.append((qid, pos))
        neg = [p for p, g in qr.items() if is_neg(g) and p in pid_to_pos]
        if neg:
            explicit_neg.append((qid, pos, neg))

    # Early exit: too few pos-bearing queries to make SCHS meaningful.
    if len(pos_bearing) < min_pos_bearing:
        if verbose:
            print(
                f"  only {len(pos_bearing)} queries with >=2 positives "
                f"(need >={min_pos_bearing}); returning NaN scores",
                flush=True,
            )
        return ChsResult(
            n_pos_bearing=len(pos_bearing),
            n_explicit_neg=len(explicit_neg),
            n_products_touched=0,
            mean_intra=float("nan"),
            mean_inter_neg=float("nan"),
            mean_random=float("nan"),
            schs=float("nan"),
            hchs=float("nan"),
            strong_inv_rate=float("nan"),
            has_explicit_negs=False,
            pp_means=[],
            pp_mins=[],
            pn_means=[],
        )

    has_explicit_negs = len(explicit_neg) >= min_explicit_neg

    # Encode (or load cached) the union of products touched by all eligible queries.
    touched = set()
    for _qid, pos in pos_bearing:
        for p in pos:
            touched.add(pid_to_pos[p])
    for _qid, _pos, neg in explicit_neg:
        for p in neg:
            touched.add(pid_to_pos[p])
    touched = sorted(touched)

    if cache_vecs is not None:
        if cache_vecs.shape[0] != len(pids):
            raise ValueError(
                f"cache_vecs has {cache_vecs.shape[0]} rows, expected {len(pids)} "
                f"(must align with pids order)"
            )
        pv = cache_vecs[touched].astype(np.float32, copy=False)
    else:
        if verbose:
            print(
                f"  encoding {len(touched):,} products with {encoder_name}...",
                flush=True,
            )
        t0 = time.time()
        pv = _encode([titles[i] for i in touched], encoder_name)
        if verbose:
            print(f"  encoded in {time.time() - t0:.0f}s", flush=True)

    pos_remap = {orig: subset for subset, orig in enumerate(touched)}

    # Per-query intra-bag metrics over ALL pos-bearing queries (input to SCHS).
    pp_means_all, pp_mins_all = [], []
    for _qid, pos in pos_bearing:
        idx = [pos_remap[pid_to_pos[p]] for p in pos]
        pp_sims = pv[idx] @ pv[idx].T
        iu = np.triu_indices(len(pp_sims), k=1)
        pairs = pp_sims[iu]
        pp_means_all.append(float(pairs.mean()))
        pp_mins_all.append(float(pairs.min()))

    # Per-query metrics over the explicit-neg subset (input to HCHS / strong_inv).
    pp_means_exp, pn_means_exp = [], []
    n_strong_inv = 0
    for _qid, pos, neg in explicit_neg:
        pi = [pos_remap[pid_to_pos[p]] for p in pos]
        ni = [pos_remap[pid_to_pos[p]] for p in neg]
        pp = pv[pi]
        nn = pv[ni]
        pp_sims = pp @ pp.T
        iu = np.triu_indices(len(pp_sims), k=1)
        pairs = pp_sims[iu]
        pp_means_exp.append(float(pairs.mean()))
        pp_max = float(pairs.max())
        pn_sims = pp @ nn.T
        pn_means_exp.append(float(pn_sims.mean()))
        if float(pn_sims.max()) > pp_max:
            n_strong_inv += 1

    # Random-pair baseline cosine over the touched subset.
    rng = np.random.default_rng(seed)
    n_subset = pv.shape[0]
    a = rng.integers(0, n_subset, n_random_pairs)
    b = rng.integers(0, n_subset, n_random_pairs)
    keep = a != b
    mean_random = float((pv[a[keep]] * pv[b[keep]]).sum(axis=1).mean())

    mean_intra = float(np.mean(pp_means_all))
    schs = (mean_intra - mean_random) / max(1 - mean_random, 1e-6)

    if has_explicit_negs:
        # HCHS uses mean_intra computed over the SAME explicit-neg subset
        # (apples-to-apples with mean_inter_neg).
        mean_intra_exp = float(np.mean(pp_means_exp))
        mean_inter_neg = float(np.mean(pn_means_exp))
        hchs = (mean_intra_exp - mean_inter_neg) / max(mean_intra_exp - mean_random, 1e-6)
        strong_inv = n_strong_inv / len(explicit_neg)
    else:
        mean_inter_neg = float("nan")
        hchs = float("nan")
        strong_inv = float("nan")

    return ChsResult(
        n_pos_bearing=len(pos_bearing),
        n_explicit_neg=len(explicit_neg),
        n_products_touched=len(touched),
        mean_intra=mean_intra,
        mean_inter_neg=mean_inter_neg,
        mean_random=mean_random,
        schs=schs,
        hchs=hchs,
        strong_inv_rate=strong_inv,
        has_explicit_negs=has_explicit_negs,
        pp_means=pp_means_all,
        pp_mins=pp_mins_all,
        pn_means=pn_means_exp if has_explicit_negs else [],
    )


def schs_verdict(schs: float, n_pos_bearing: int) -> str:
    """One-line GREEN/YELLOW/RED verdict from SCHS, accounting for training scale.

    Anchored to the empirical calibration in evaluation/CHS_RESULTS.md.
    """
    if not np.isfinite(schs):
        return "UNDETERMINED — too few multi-positive queries to score"
    if n_pos_bearing < 500:
        scale = " (low n_pos_bearing — training scale may limit BoD even at high SCHS)"
    else:
        scale = ""
    if schs >= 0.50:
        return f"GREEN — BoD likely to generalize{scale}"
    if schs >= 0.40:
        return f"YELLOW — BoD may give a smaller lift; pilot first{scale}"
    return f"RED — BoD unlikely to lift over base{scale}"
