#!/usr/bin/env python3
"""Stream download + CLIP-encode ESCI-US image-bearing ASINs.

Reads image_head_check_full.jsonl to identify which ASINs have real
images. For each, downloads /P/{ASIN}.01.LZZZZZZZ.jpg, preprocesses,
and batches through CLIP image encoder on MPS.

Output:
    esci_us_data/clip_image_vecs.fp16.npy  — (N, 512) fp16
    esci_us_data/clip_image_idx.npy        — (N,) int32, catalog indices

The JPEG is discarded immediately after preprocessing — disk footprint
stays minimal. Resumes from any partial output on restart.
"""

import io
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import open_clip
import requests
import torch
from PIL import Image
from requests.adapters import HTTPAdapter

URL_TEMPLATE = "https://images-na.ssl-images-amazon.com/images/P/{asin}.01.LZZZZZZZ.jpg"
MODEL_NAME = "ViT-B-32"
PRETRAINED = "laion2b_s34b_b79k"
BATCH_SIZE = 128
NUM_DOWNLOAD_WORKERS = 24
OUT_VECS = Path("esci_us_data/clip_image_vecs.fp16.npy")
OUT_IDX = Path("esci_us_data/clip_image_idx.npy")
SAVE_EVERY = BATCH_SIZE * 20  # checkpoint every 2560 vecs


def load_targets():
    with open("esci_us_data/product_ids.json") as f:
        pids = json.load(f)
    pid_to_idx = {p: i for i, p in enumerate(pids)}
    targets = []
    with open("esci_us_data/image_head_check_full.jsonl") as f:
        for line in f:
            r = json.loads(line)
            if r.get("has_image"):
                targets.append((r["asin"], pid_to_idx[r["asin"]]))
    return targets, len(pids)


def main():
    targets, catalog_size = load_targets()
    print(f"target ASINs: {len(targets):,}  (catalog size {catalog_size:,})", flush=True)

    # Resume
    all_vecs: list[np.ndarray] = []
    all_idx: list[int] = []
    if OUT_VECS.exists() and OUT_IDX.exists():
        existing_vecs = np.load(OUT_VECS)
        existing_idx = np.load(OUT_IDX)
        already_done = {int(i) for i in existing_idx}
        targets = [(a, i) for a, i in targets if i not in already_done]
        all_vecs = [existing_vecs]
        all_idx = list(existing_idx)
        print(f"resume: {len(already_done):,} cached, {len(targets):,} remaining", flush=True)
    if not targets:
        print("nothing to do")
        return

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"loading CLIP {MODEL_NAME} on {device}...", flush=True)
    model, _, preprocess = open_clip.create_model_and_transforms(MODEL_NAME, pretrained=PRETRAINED)
    model = model.to(device).eval()

    session = requests.Session()
    adapter = HTTPAdapter(
        pool_connections=NUM_DOWNLOAD_WORKERS, pool_maxsize=NUM_DOWNLOAD_WORKERS * 2
    )
    session.mount("https://", adapter)
    session.mount("http://", adapter)

    def download_and_preprocess(asin: str, idx: int):
        try:
            r = session.get(URL_TEMPLATE.format(asin=asin), timeout=15)
            if r.status_code != 200 or len(r.content) < 1000:
                return None
            img = Image.open(io.BytesIO(r.content)).convert("RGB")
            return idx, preprocess(img)
        except Exception:
            return None

    t0 = time.time()
    n_start = len(all_idx)
    n_failed = 0
    batch_tensors: list = []
    batch_idx: list = []

    def flush_batch():
        nonlocal batch_tensors, batch_idx
        if not batch_tensors:
            return
        stack = torch.stack(batch_tensors).to(device)
        with torch.no_grad():
            v = model.encode_image(stack)
            v = v / v.norm(dim=-1, keepdim=True)
            v = v.cpu().float().numpy().astype(np.float16)
        all_vecs.append(v)
        all_idx.extend(batch_idx)
        batch_tensors = []
        batch_idx = []

    def checkpoint():
        if not all_idx:
            return
        cur_vecs = (
            np.concatenate(all_vecs, axis=0)
            if isinstance(all_vecs[0], np.ndarray)
            else np.stack(all_vecs)
        )
        np.save(OUT_VECS, cur_vecs)
        np.save(OUT_IDX, np.array(all_idx, dtype=np.int32))

    print(f"streaming {len(targets):,} downloads + CLIP encode (chunked)...", flush=True)
    CHUNK = 5000  # submit work this many at a time to keep Future-list memory bounded
    last_ckpt = 0
    with ThreadPoolExecutor(max_workers=NUM_DOWNLOAD_WORKERS) as ex:
        for chunk_start in range(0, len(targets), CHUNK):
            chunk = targets[chunk_start : chunk_start + CHUNK]
            futs = [ex.submit(download_and_preprocess, a, i) for a, i in chunk]
            for fut in as_completed(futs):
                res = fut.result()
                if res is None:
                    n_failed += 1
                    continue
                idx, tensor = res
                batch_tensors.append(tensor)
                batch_idx.append(idx)
                if len(batch_tensors) >= BATCH_SIZE:
                    flush_batch()
                    n_done_this_run = len(all_idx) - n_start
                    if n_done_this_run - last_ckpt >= SAVE_EVERY:
                        checkpoint()
                        last_ckpt = n_done_this_run
                        elapsed = time.time() - t0
                        rate = n_done_this_run / max(elapsed, 1)
                        eta_min = (len(targets) - n_done_this_run) / max(rate, 1) / 60
                        print(
                            f"  encoded={n_done_this_run:,}/{len(targets):,}  "
                            f"failed={n_failed:,}  rate={rate:.0f}/s  ETA {eta_min:.0f}min",
                            flush=True,
                        )
            futs.clear()  # release Future references for this chunk before submitting next
        # Final flush after all chunks
        flush_batch()
    checkpoint()

    elapsed = time.time() - t0
    print(
        f"\ndone in {elapsed / 60:.0f}min — encoded={len(all_idx):,}  failed={n_failed:,}",
        flush=True,
    )
    print(f"vecs: {OUT_VECS}  idx: {OUT_IDX}")


if __name__ == "__main__":
    main()
