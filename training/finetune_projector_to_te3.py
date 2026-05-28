#!/usr/bin/env python3
"""Pure-projection distillation: train a Linear(d_base -> 1024) projector that
maps frozen base-encoder query vectors toward te3-large query vectors.

Unlike full distillation (which fine-tunes the encoder and collapsed in prior
runs: see project_te3_distillation_collapse), this script keeps the base encoder
frozen and learns only the linear projector. Lower-variance objective.

Training data: union of train_queries.jsonl across the 4 jobs sub-corpora,
intersected with cached train_queries_te3_1024.{ids.json,vecs.fp16.npy}.

Evaluation: eval_queries_unified.jsonl (3,311 queries; held out from train),
scored against the unified te3 catalog (347,900 x 1024 fp16).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402


def load_jsonl_queries(path: Path) -> list[str]:
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            out.append(row["query"])
    return out


def load_te3_query_cache(dirs: list[Path], split: str) -> dict[str, np.ndarray]:
    """Returns {query_text: te3_vector_1024} from the cached
    {split}_queries_te3_1024.{ids.json,vecs.fp16.npy} files."""
    qmap: dict[str, np.ndarray] = {}
    for d in dirs:
        ids_p = d / f"{split}_queries_te3_1024.ids.json"
        vec_p = d / f"{split}_queries_te3_1024.vecs.fp16.npy"
        if not (ids_p.exists() and vec_p.exists()):
            print(f"  {d.name}/{split}: missing cache files, skipping")
            continue
        with open(ids_p) as f:
            ids = json.load(f)
        vecs = np.load(vec_p).astype(np.float32)
        for q, v in zip(ids, vecs):
            qmap[q] = v
    return qmap


def l2norm_np(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return x / n


def encode_st_queries(
    model_id: str,
    queries: list[str],
    device: str,
    query_prefix: str,
    batch_size: int,
) -> np.ndarray:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_id, device=device)
    inputs = [query_prefix + q for q in queries] if query_prefix else queries
    vecs = model.encode(
        inputs,
        normalize_embeddings=True,
        batch_size=batch_size,
        show_progress_bar=True,
    ).astype(np.float32)
    return vecs


def train_projector(
    src: torch.Tensor,
    tgt: torch.Tensor,
    d_out: int,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    device: str,
    seed: int,
) -> torch.nn.Module:
    """Train a Linear(d_in -> d_out) projector with cosine loss."""
    torch.manual_seed(seed)
    n, d_in = src.shape
    proj = torch.nn.Linear(d_in, d_out, bias=True).to(device)
    src = src.to(device)
    tgt = tgt.to(device)
    opt = torch.optim.AdamW(proj.parameters(), lr=lr, weight_decay=weight_decay)

    for ep in range(1, epochs + 1):
        proj.train()
        perm = torch.randperm(n, device=device)
        ep_loss = 0.0
        ep_cos = 0.0
        n_seen = 0
        for s in range(0, n, batch_size):
            idx = perm[s : s + batch_size]
            x = src[idx]
            y = tgt[idx]
            opt.zero_grad()
            y_hat = F.normalize(proj(x), dim=-1)
            loss = (1.0 - (y_hat * y).sum(dim=-1)).mean()
            loss.backward()
            opt.step()
            bs = x.size(0)
            ep_loss += loss.item() * bs
            ep_cos += (y_hat * y).sum(dim=-1).mean().item() * bs
            n_seen += bs
        print(
            f"  ep {ep:>3}/{epochs}: train_loss={ep_loss / n_seen:.4f} "
            f"train_cos={ep_cos / n_seen:.4f}",
            flush=True,
        )
    return proj


def eval_against_catalog(
    proj_eval_vecs: np.ndarray,
    cat_norm: np.ndarray,
    golds: list[int],
    k: int,
) -> dict:
    """Compute R@1/5/10 for projected eval vectors against te3 catalog."""
    n_q = proj_eval_vecs.shape[0]
    QB = 256
    h1 = h5 = h10 = n = 0
    for s in range(0, n_q, QB):
        e = min(s + QB, n_q)
        sc = proj_eval_vecs[s:e] @ cat_norm.T
        idx = np.argpartition(-sc, kth=k, axis=1)[:, :k]
        for r in range(idx.shape[0]):
            gi = golds[s + r]
            if gi < 0:
                continue
            ordered = idx[r][np.argsort(-sc[r, idx[r]])]
            n += 1
            if ordered[0] == gi:
                h1 += 1
            if gi in ordered[:5]:
                h5 += 1
            if gi in ordered[:10]:
                h10 += 1
    return {
        "n": n,
        "r_at_1": round(h1 / max(n, 1), 4),
        "r_at_5": round(h5 / max(n, 1), 4),
        "r_at_10": round(h10 / max(n, 1), 4),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-model", default="intfloat/e5-base-v2", help="frozen base encoder")
    ap.add_argument("--query-prefix", default="query: ")
    ap.add_argument(
        "--source-dirs",
        default="jobs_data,jobs_data_linkedin,jobs_data_usajobs,jobs_data_jobstreet",
        help="comma-separated directories with train/eval jsonl + te3 caches",
    )
    ap.add_argument("--unified-data-dir", default="unified_jobs")
    ap.add_argument(
        "--unified-eval-file",
        default="eval_queries_unified.jsonl",
        help="held-out queries file under --unified-data-dir",
    )
    ap.add_argument(
        "--catalog-vecs",
        default="te3_catalog.vecs.fp16.npy",
        help="te3 catalog filename under --unified-data-dir",
    )
    ap.add_argument("--device", default="mps")
    ap.add_argument("--encode-batch-size", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--train-batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--output-dir",
        default="training_artifacts/projector_e5base_to_te3",
        help="where to write projector weights + eval JSON",
    )
    ap.add_argument(
        "--cache-encoded",
        action="store_true",
        help="cache e5-encoded train/eval vectors under output-dir to skip re-encoding on rerun",
    )
    args = ap.parse_args()

    src_dirs = [Path(d) for d in args.source_dirs.split(",")]
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    unified_dir = Path(args.unified_data_dir)

    # ---- Step 1: assemble training pairs ----
    print(f"[1/4] assembling training pairs from {len(src_dirs)} dirs", flush=True)
    train_queries = []
    seen = set()
    for d in src_dirs:
        p = d / "train_queries.jsonl"
        if not p.exists():
            print(f"  {d.name}: no train_queries.jsonl, skipping")
            continue
        n_before = len(seen)
        for q in load_jsonl_queries(p):
            if q not in seen:
                seen.add(q)
                train_queries.append(q)
        print(f"  {d.name}: +{len(seen) - n_before} new queries (total {len(seen)})")

    print("  loading cached te3 train vectors", flush=True)
    te3_train_map = load_te3_query_cache(src_dirs, "train")
    print(f"  te3 train cache: {len(te3_train_map):,} entries")

    # Intersect: queries present in both jsonl and te3 cache
    train_pairs = [(q, te3_train_map[q]) for q in train_queries if q in te3_train_map]
    print(f"  intersected: {len(train_pairs):,} (query, te3_vec) pairs")
    if len(train_pairs) < 1000:
        raise SystemExit("too few training pairs; check inputs")

    train_query_texts = [q for q, _ in train_pairs]
    te3_train = np.stack([v for _, v in train_pairs], axis=0).astype(np.float32)
    te3_train = l2norm_np(te3_train)
    print(f"  te3_train shape={te3_train.shape}")

    # ---- Step 2: encode train + eval queries with frozen base encoder ----
    eval_p = unified_dir / args.unified_eval_file
    eval_queries = load_jsonl_queries(eval_p)
    print(f"[2/4] {len(eval_queries):,} eval queries from {eval_p}", flush=True)

    train_enc_path = out_dir / "train_queries_base.vecs.fp16.npy"
    eval_enc_path = out_dir / "eval_queries_base.vecs.fp16.npy"

    if args.cache_encoded and train_enc_path.exists() and eval_enc_path.exists():
        print(f"  using cached base encodings from {out_dir}")
        base_train = np.load(train_enc_path).astype(np.float32)
        base_eval = np.load(eval_enc_path).astype(np.float32)
        if base_train.shape[0] != len(train_query_texts):
            raise SystemExit(
                f"cached train encoding row count {base_train.shape[0]} != "
                f"current pair count {len(train_query_texts)}; rerun without --cache-encoded"
            )
    else:
        print(f"  encoding {len(train_query_texts):,} train queries with {args.base_model}")
        t0 = time.time()
        base_train = encode_st_queries(
            args.base_model,
            train_query_texts,
            args.device,
            args.query_prefix,
            args.encode_batch_size,
        )
        print(f"    train encoded in {time.time() - t0:.1f}s; shape={base_train.shape}")

        print(f"  encoding {len(eval_queries):,} eval queries")
        t0 = time.time()
        base_eval = encode_st_queries(
            args.base_model,
            eval_queries,
            args.device,
            args.query_prefix,
            args.encode_batch_size,
        )
        print(f"    eval encoded in {time.time() - t0:.1f}s; shape={base_eval.shape}")

        if args.cache_encoded:
            np.save(train_enc_path, base_train.astype(np.float16))
            np.save(eval_enc_path, base_eval.astype(np.float16))
            print(f"  cached encodings to {out_dir}")

    base_train = l2norm_np(base_train)
    base_eval = l2norm_np(base_eval)

    # ---- Step 3: train Linear(d_in -> 1024) projector ----
    print(f"[3/4] training projector ({base_train.shape[1]} -> {te3_train.shape[1]})", flush=True)
    src_t = torch.from_numpy(base_train)
    tgt_t = torch.from_numpy(te3_train)
    proj = train_projector(
        src_t,
        tgt_t,
        d_out=te3_train.shape[1],
        epochs=args.epochs,
        batch_size=args.train_batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        device=args.device,
        seed=args.seed,
    )

    proj_path = out_dir / "projector.pt"
    torch.save(
        {
            "state_dict": proj.state_dict(),
            "d_in": int(base_train.shape[1]),
            "d_out": int(te3_train.shape[1]),
            "base_model": args.base_model,
            "query_prefix": args.query_prefix,
        },
        proj_path,
    )
    print(f"  wrote projector -> {proj_path}")

    # ---- Step 4: eval against te3 catalog ----
    print(f"[4/4] loading te3 catalog {unified_dir / args.catalog_vecs}", flush=True)
    cat = np.load(unified_dir / args.catalog_vecs, mmap_mode="r").astype(np.float32)
    cat_norm = l2norm_np(cat)
    print(f"  catalog shape={cat.shape}")

    with open(unified_dir / "doc_ids.json") as f:
        doc_ids = json.load(f)
    pid2idx = {p: i for i, p in enumerate(doc_ids)}
    eval_rows = []
    with open(eval_p) as f:
        for line in f:
            line = line.strip()
            if line:
                eval_rows.append(json.loads(line))
    golds = [pid2idx.get(r["source_doc_id"], -1) for r in eval_rows]

    proj.eval()
    with torch.no_grad():
        be = torch.from_numpy(base_eval).to(args.device)
        proj_eval = F.normalize(proj(be), dim=-1).cpu().numpy().astype(np.float32)

    t0 = time.time()
    m = eval_against_catalog(proj_eval, cat_norm, golds, k=10)
    print(f"  eval done in {time.time() - t0:.1f}s")
    print(f"  R@1={m['r_at_1']:.4f}  R@5={m['r_at_5']:.4f}  R@10={m['r_at_10']:.4f}  (n={m['n']})")

    out_json = out_dir / "eval_metrics.json"
    summary = {
        "base_model": args.base_model,
        "query_prefix": args.query_prefix,
        "n_train_pairs": int(te3_train.shape[0]),
        "n_eval": int(proj_eval.shape[0]),
        "epochs": args.epochs,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "metrics": m,
        "te3_baseline_r_at_10": 0.7049,
    }
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  wrote {out_json}")


if __name__ == "__main__":
    main()
