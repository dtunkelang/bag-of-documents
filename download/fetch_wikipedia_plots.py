#!/usr/bin/env python3
"""Fetch English Wikipedia lead + plot section for each bridged movie title.

Wikipedia's prop=extracts API caps exlimit=1 for whole-article requests, so we
fetch one title at a time using a thread pool (network-bound, so threads are OK).

Plot heading variants recognized: Plot, Plot summary, Synopsis, Story, Storyline.

Resumable: a sidecar .processed file lists tconsts already fetched; on restart,
those are skipped.

Output:
  <out-path>: gzipped JSONL with {tconst, enwiki_title, lead, plot}

Usage:
  .venv/bin/python download/fetch_wikipedia_plots.py \\
      --bridge movies_data/wikidata_bridge.jsonl \\
      --out-path movies_data/wikipedia_plots.jsonl.gz \\
      --workers 4 --rate-per-sec 8.0
"""

import argparse
import gzip
import json
import re
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

API = "https://en.wikipedia.org/w/api.php"
USER_AGENT = "bagofdocs-movies-demo/1.0 (research; contact: daniel.tunkelang@algolia.com)"

PLOT_HEADINGS = ("plot", "plot summary", "synopsis", "story", "storyline")
SECTION_RE = re.compile(r"(?m)^(={2,6})\s*(.+?)\s*\1\s*$")


def load_targets(bridge_path: Path) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    with open(bridge_path) as f:
        for line in f:
            d = json.loads(line)
            t = d.get("enwiki_title")
            if t:
                out.append((d["tconst"], t))
    return out


def load_processed(path: Path) -> set[str]:
    if not path.exists():
        return set()
    with open(path) as f:
        return {line.strip() for line in f if line.strip()}


def split_lead_plot(extract: str) -> tuple[str, str]:
    if not extract:
        return "", ""
    matches = list(SECTION_RE.finditer(extract))
    if not matches:
        return extract.strip(), ""
    lead = extract[: matches[0].start()].strip()
    plot = ""
    for i, m in enumerate(matches):
        depth = len(m.group(1))
        title = m.group(2).strip().lower()
        if title in PLOT_HEADINGS:
            start = m.end()
            end = len(extract)
            for nxt in matches[i + 1 :]:
                if len(nxt.group(1)) <= depth:
                    end = nxt.start()
                    break
            plot = extract[start:end].strip()
            break
    return lead, plot


class RateLimiter:
    """Simple token-bucket-ish global rate limiter shared across threads."""

    def __init__(self, rate_per_sec: float):
        self.min_interval = 1.0 / max(rate_per_sec, 0.01)
        self.lock = threading.Lock()
        self.next_t = 0.0

    def acquire(self) -> None:
        with self.lock:
            now = time.time()
            wait = self.next_t - now
            if wait > 0:
                time.sleep(wait)
                now = time.time()
            self.next_t = now + self.min_interval


def fetch_one(
    title: str,
    limiter: RateLimiter,
    max_retries: int = 5,
    timeout: float = 60.0,
) -> tuple[str, str]:
    """Return (resolved_title, extract_text). Empty extract on missing/error."""
    params = {
        "action": "query",
        "format": "json",
        "prop": "extracts",
        "explaintext": "1",
        "exsectionformat": "wiki",
        "redirects": "1",
        "titles": title,
    }
    url = API + "?" + urllib.parse.urlencode(params)
    backoff = 2.0
    for _ in range(max_retries):
        limiter.acquire()
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                data = json.loads(r.read())
            for page in data.get("query", {}).get("pages", {}).values():
                if "missing" in page:
                    return page.get("title", title), ""
                return page.get("title", title), page.get("extract", "")
            return title, ""
        except urllib.error.HTTPError as e:
            if e.code == 429:
                time.sleep(backoff)
                backoff = min(backoff * 2, 60.0)
                continue
            return title, ""
        except (urllib.error.URLError, TimeoutError):
            time.sleep(backoff)
            backoff = min(backoff * 2, 60.0)
    return title, ""


def worker(
    tconst: str,
    title: str,
    limiter: RateLimiter,
    max_retries: int,
) -> dict | None:
    resolved, extract = fetch_one(title, limiter, max_retries=max_retries)
    lead, plot = split_lead_plot(extract)
    if not lead and not plot:
        return {"tconst": tconst, "enwiki_title": title, "lead": "", "plot": ""}
    return {
        "tconst": tconst,
        "enwiki_title": resolved or title,
        "lead": lead,
        "plot": plot,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bridge", default="movies_data/wikidata_bridge.jsonl")
    ap.add_argument("--out-path", default="movies_data/wikipedia_plots.jsonl.gz")
    ap.add_argument(
        "--processed-path",
        default="movies_data/wikipedia_plots.processed",
    )
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument(
        "--rate-per-sec",
        type=float,
        default=8.0,
        help="Global request rate cap across all workers.",
    )
    ap.add_argument("--max-retries", type=int, default=5)
    ap.add_argument("--limit", type=int, default=0, help="Stop after N new fetches (0=unlimited).")
    args = ap.parse_args()

    out_path = Path(args.out_path)
    proc_path = Path(args.processed_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    targets = load_targets(Path(args.bridge))
    targets.sort(key=lambda x: x[0])
    processed = load_processed(proc_path)
    pending = [(tc, t) for tc, t in targets if tc not in processed]
    n_total = len(targets)
    n_pending = len(pending)
    print(
        f"[plots] {n_total:,} bridged titles | {len(processed):,} already done | "
        f"{n_pending:,} to fetch",
        flush=True,
    )

    if args.limit > 0:
        pending = pending[: args.limit]
        print(f"[plots] limit={args.limit} → fetching {len(pending):,} this run", flush=True)

    limiter = RateLimiter(args.rate_per_sec)
    out_mode = "ab" if out_path.exists() else "wb"

    write_lock = threading.Lock()
    n_done = 0
    n_with_plot = 0
    n_with_lead = 0
    t_start = time.time()

    def submit_and_drain(out_f, proc_f):
        nonlocal n_done, n_with_plot, n_with_lead
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futures = {ex.submit(worker, tc, t, limiter, args.max_retries): tc for tc, t in pending}
            for fut in as_completed(futures):
                tc = futures[fut]
                try:
                    row = fut.result()
                except Exception as e:
                    print(f"  {tc}: worker exception: {e}", flush=True)
                    row = None
                if row is None:
                    continue
                with write_lock:
                    out_f.write((json.dumps(row, ensure_ascii=False) + "\n").encode("utf-8"))
                    proc_f.write(tc + "\n")
                    n_done += 1
                    if row.get("lead"):
                        n_with_lead += 1
                    if row.get("plot"):
                        n_with_plot += 1
                    if n_done % 200 == 0:
                        out_f.flush()
                        elapsed = time.time() - t_start
                        rate = n_done / max(elapsed, 1e-6)
                        eta_min = (len(pending) - n_done) / max(rate, 1e-6) / 60
                        print(
                            f"  {n_done:,}/{len(pending):,} | lead {n_with_lead:,} "
                            f"plot {n_with_plot:,} | {rate:.1f} req/s | ETA {eta_min:.0f} min",
                            flush=True,
                        )

    with gzip.open(out_path, out_mode) as out_f, open(proc_path, "a", buffering=1) as proc_f:
        submit_and_drain(out_f, proc_f)

    elapsed = (time.time() - t_start) / 60
    print(
        f"[plots] done: {n_done:,} fetched ({n_with_lead:,} w/ lead, {n_with_plot:,} w/ plot) "
        f"in {elapsed:.1f} min",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
