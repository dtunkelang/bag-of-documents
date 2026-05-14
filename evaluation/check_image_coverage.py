#!/usr/bin/env python3
"""Full ESCI-US catalog HEAD check for /P/{ASIN}.01.LZZZZZZZ.jpg coverage."""

import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

with open("esci_us_data/product_ids.json") as f:
    pids = json.load(f)
print(f"loaded {len(pids):,} ASINs", flush=True)

OUT_PATH = Path("esci_us_data/image_head_check_full.jsonl")
# Resume: skip ASINs already checked
done = set()
if OUT_PATH.exists():
    with open(OUT_PATH) as f:
        for line in f:
            try:
                done.add(json.loads(line)["asin"])
            except Exception:
                pass
print(f"resume: {len(done):,} ASINs already checked", flush=True)

todo = [p for p in pids if p not in done]
print(f"to check: {len(todo):,}", flush=True)
if not todo:
    sys.exit(0)

session = requests.Session()


def check(asin):
    url = f"https://images-na.ssl-images-amazon.com/images/P/{asin}.01.LZZZZZZZ.jpg"
    try:
        r = session.head(url, timeout=10, allow_redirects=True)
        cl = int(r.headers.get("Content-Length", 0))
        return {
            "asin": asin,
            "status": r.status_code,
            "content_length": cl,
            "has_image": (r.status_code == 200 and cl > 1000),
        }
    except Exception:
        return {"asin": asin, "status": 0, "content_length": 0, "has_image": False}


t0 = time.time()
n_done = 0
n_has_image = 0
with open(OUT_PATH, "a") as out, ThreadPoolExecutor(max_workers=50) as ex:
    futs = [ex.submit(check, a) for a in todo]
    for fut in as_completed(futs):
        res = fut.result()
        out.write(json.dumps(res) + "\n")
        n_done += 1
        if res["has_image"]:
            n_has_image += 1
        if n_done % 5000 == 0:
            elapsed = time.time() - t0
            rate = n_done / max(elapsed, 1)
            eta_min = (len(todo) - n_done) / max(rate, 1) / 60
            print(
                f"  {n_done:,}/{len(todo):,}  has_image={n_has_image:,} ({100 * n_has_image / n_done:.0f}%)  rate={rate:.0f}/s  ETA {eta_min:.0f}min",
                flush=True,
            )
            out.flush()

elapsed = time.time() - t0
print(
    f"\ndone in {elapsed / 60:.0f}min — {n_has_image:,}/{n_done:,} = {100 * n_has_image / n_done:.0f}% real-image",
    flush=True,
)
