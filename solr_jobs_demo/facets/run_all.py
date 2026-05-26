#!/usr/bin/env python3
"""Run heuristic facet classification over the full 347.9k corpus.

Writes one line per doc to facets.jsonl: {"idx": <int>, ...8 facet fields}.
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from heuristics import classify_record

META = Path("/Users/dtunkelang/bagofdocs/unified_jobs/metadata.jsonl")
OUT = Path("/Users/dtunkelang/bagofdocs/solr_jobs_demo/facets/facets.jsonl")


def main() -> int:
    t0 = time.time()
    n = 0
    with open(META) as f, open(OUT, "w") as out:
        for i, line in enumerate(f):
            rec = json.loads(line)
            facets = classify_record(rec)
            out.write(json.dumps({"idx": i, **facets}) + "\n")
            n += 1
            if n % 50000 == 0:
                rate = n / (time.time() - t0)
                print(f"  {n:,} ({rate:.0f}/s)", flush=True)
    print(f"done: {n:,} docs in {time.time() - t0:.1f}s -> {OUT}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
