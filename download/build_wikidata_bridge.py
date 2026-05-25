#!/usr/bin/env python3
"""Build a Wikidata bridge mapping IMDb tconst → (qid, enwiki_title).

For each tconst in the input IMDb titles JSONL, queries the Wikidata Query
Service in batches via VALUES clauses, looking up entities with property
P345 (IMDb ID) and their English Wikipedia sitelink (optional).

Checkpointed: on interrupt, resume from the last completed chunk.

Output:
  <out-path>: JSONL with {tconst, qid, enwiki_title (may be null)}

Usage:
  .venv/bin/python download/build_wikidata_bridge.py \\
      --imdb-titles movies_data/titles_imdb.jsonl \\
      --out-path movies_data/wikidata_bridge.jsonl \\
      --chunk-size 200 --rate-per-sec 1.0
"""

import argparse
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

WDQS = "https://query.wikidata.org/sparql"
USER_AGENT = "bagofdocs-movies-demo/1.0 (research; contact: daniel.tunkelang@algolia.com)"


def load_tconsts(path: Path) -> list[str]:
    out = []
    with open(path) as f:
        for line in f:
            doc = json.loads(line)
            out.append(doc["tconst"])
    return out


def build_query(tconsts: list[str]) -> str:
    values = " ".join(f'"{tc}"' for tc in tconsts)
    return f"""SELECT ?entity ?imdb ?wp WHERE {{
  VALUES ?imdb {{ {values} }}
  ?entity wdt:P345 ?imdb .
  OPTIONAL {{ ?wp schema:about ?entity ; schema:isPartOf <https://en.wikipedia.org/> }}
}}"""


def run_sparql(query: str, timeout: float = 60.0) -> dict:
    data = urllib.parse.urlencode({"query": query, "format": "json"}).encode()
    req = urllib.request.Request(
        WDQS,
        data=data,
        headers={
            "User-Agent": USER_AGENT,
            "Accept": "application/sparql-results+json",
            "Content-Type": "application/x-www-form-urlencoded",
        },
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def parse_results(res: dict) -> list[dict]:
    rows = []
    for b in res.get("results", {}).get("bindings", []):
        entity_uri = b.get("entity", {}).get("value", "")
        qid = entity_uri.rsplit("/", 1)[-1] if entity_uri else None
        imdb = b.get("imdb", {}).get("value")
        wp_uri = b.get("wp", {}).get("value", "")
        enwiki_title = None
        if wp_uri.startswith("https://en.wikipedia.org/wiki/"):
            enwiki_title = urllib.parse.unquote(wp_uri[len("https://en.wikipedia.org/wiki/") :])
        rows.append({"tconst": imdb, "qid": qid, "enwiki_title": enwiki_title})
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--imdb-titles", default="movies_data/titles_imdb.jsonl")
    ap.add_argument("--out-path", default="movies_data/wikidata_bridge.jsonl")
    ap.add_argument(
        "--checkpoint-path",
        default="movies_data/wikidata_bridge.checkpoint.json",
    )
    ap.add_argument("--chunk-size", type=int, default=200)
    ap.add_argument("--rate-per-sec", type=float, default=1.0)
    ap.add_argument("--max-retries", type=int, default=5)
    args = ap.parse_args()

    out_path = Path(args.out_path)
    ckpt_path = Path(args.checkpoint_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    tconsts = load_tconsts(Path(args.imdb_titles))
    tconsts.sort()
    n = len(tconsts)
    n_chunks = (n + args.chunk_size - 1) // args.chunk_size
    print(f"[bridge] {n:,} tconsts → {n_chunks:,} chunks of {args.chunk_size}", flush=True)

    # Resume
    start = 0
    if ckpt_path.exists():
        with open(ckpt_path) as f:
            ckpt = json.load(f)
        start = int(ckpt.get("next_chunk", 0))
        print(f"[bridge] resuming from chunk {start}", flush=True)

    out_mode = "a" if start > 0 else "w"
    min_interval = 1.0 / args.rate_per_sec
    last_t = 0.0
    n_rows = 0
    t_start = time.time()

    with open(out_path, out_mode) as out:
        for idx in range(start, n_chunks):
            chunk = tconsts[idx * args.chunk_size : (idx + 1) * args.chunk_size]
            q = build_query(chunk)
            # rate limit
            dt = time.time() - last_t
            if dt < min_interval:
                time.sleep(min_interval - dt)
            # retry loop
            backoff = 2.0
            for attempt in range(args.max_retries):
                try:
                    res = run_sparql(q)
                    break
                except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as e:
                    print(
                        f"  chunk {idx} attempt {attempt + 1}/{args.max_retries} failed: {e}",
                        flush=True,
                    )
                    time.sleep(backoff)
                    backoff = min(backoff * 2, 60.0)
            else:
                print(
                    f"  chunk {idx} GIVEUP after {args.max_retries} retries; skipping", flush=True
                )
                last_t = time.time()
                continue

            rows = parse_results(res)
            for row in rows:
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
                n_rows += 1
            out.flush()
            last_t = time.time()

            # checkpoint
            with open(ckpt_path, "w") as ck:
                json.dump({"next_chunk": idx + 1, "n_rows": n_rows}, ck)

            if (idx + 1) % 50 == 0:
                elapsed = time.time() - t_start
                rate = (idx + 1 - start) / max(elapsed, 1e-6)
                eta_min = (n_chunks - idx - 1) / max(rate, 1e-6) / 60
                print(
                    f"  chunk {idx + 1:,}/{n_chunks:,} | rows {n_rows:,} | "
                    f"{rate:.2f} chunks/s | ETA {eta_min:.0f} min",
                    flush=True,
                )

    print(f"[bridge] done: {n_rows:,} rows in {(time.time() - t_start) / 60:.1f} min", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
