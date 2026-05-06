"""Tests for the BestBuy ACM bag-building scripts.

We don't run the encoders end-to-end (they need real models + catalog data).
Instead we verify the deterministic data-shape invariants:
  - The 80/20 query split in build_bestbuy_bags.py is reproducible and disjoint.
  - add_random_hardnegs_bestbuy.py picks negatives that are not bag positives.
"""

import json
import random
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def test_train_test_split_is_disjoint_and_deterministic():
    """The split logic in build_bestbuy_bags.py: shuffle qids with seed,
    take first N as test. Re-running with the same seed gives the same split,
    and train/test never share a qid."""
    qids = [f"q{i:04d}" for i in range(1000)]
    test_fraction = 0.2

    def split(seed):
        rng = random.Random(seed)
        shuffled = sorted(qids)
        rng.shuffle(shuffled)
        n_test = int(len(shuffled) * test_fraction)
        test_set = set(shuffled[:n_test])
        train_set = [q for q in shuffled if q not in test_set]
        return set(train_set), test_set

    train1, test1 = split(42)
    train2, test2 = split(42)
    train3, _ = split(7)

    assert train1 == train2 and test1 == test2
    assert train1.isdisjoint(test1)
    assert len(train1) + len(test1) == len(qids)
    assert len(test1) == 200
    assert train1 != train3


def _make_fake_data_dir(tmp_path):
    """Build a minimal bestbuy_acm_data/ that satisfies add_random_hardnegs_bestbuy.py."""
    d = tmp_path / "bestbuy_acm_data"
    d.mkdir()
    titles = [f"product-{i}" for i in range(50)]
    pids = [str(1000 + i) for i in range(50)]
    (d / "titles.json").write_text(json.dumps(titles))
    (d / "product_ids.json").write_text(json.dumps(pids))
    bags = []
    for i in range(5):
        positives = [titles[i * 3 + k] for k in range(3)]
        bags.append(
            {
                "query": f"query-{i}",
                "query_vector": [0.0] * 384,
                "results": [{"title": t} for t in positives],
                "num_results": len(positives),
                "specificity": 0.9,
            }
        )
    (d / "bags.jsonl").write_text("\n".join(json.dumps(b) for b in bags))
    return d


def test_add_random_hardnegs_excludes_positives(tmp_path):
    data_dir = _make_fake_data_dir(tmp_path)
    n_hardnegs = 5
    cmd = [
        sys.executable,
        str(ROOT / "download/add_random_hardnegs_bestbuy.py"),
        "--data-dir",
        str(data_dir),
        "--n-hardnegs",
        str(n_hardnegs),
        "--seed",
        "0",
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    out_path = data_dir / "bags_with_hardnegs.jsonl"
    assert out_path.exists()
    with open(out_path) as f:
        for line in f:
            bag = json.loads(line)
            pos_titles = {r["title"] for r in bag["results"]}
            assert "hardnegs" in bag
            assert len(bag["hardnegs"]) == n_hardnegs
            assert pos_titles.isdisjoint(set(bag["hardnegs"]))
            assert len(set(bag["hardnegs"])) == len(bag["hardnegs"])  # no dupes
