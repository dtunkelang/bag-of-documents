"""Smoke test: every tracked .py script in download/, indexing/, training/,
evaluation/, scripts/ should be importable.

This catches:
  - Missing or misconfigured sys.path shim after a script moves
  - Broken `from utils import X` and other top-level imports
  - Syntax errors

Each script is imported in a *subprocess* to isolate side effects (some
scripts may load big artifacts on first import).
"""

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SUBDIRS = ["download", "indexing", "training", "evaluation", "scripts"]

# Scripts that do real work at import time (network IO, etc.) and don't
# benefit from this smoke test. Adding them would just make CI slow / flaky.
# Best practice would be to wrap their top-level code in `def main()`, but
# they are one-off data-acquisition utilities and not load-bearing.
IMPORT_SKIP = {
    # Do real work (HTTP downloads) at top level — would slow CI and fail
    # without network. Refactor to def main() if revisiting.
    "download/download_esci_es.py",
    "download/download_esci_us.py",
    "download/download_nfcorpus.py",
    "download/download_fiqa.py",
    "download/download_scifact.py",
    "download/download_beir.py",
    "download/download_catalog.py",
    # Loads eval data files at module top level — fails in clean CI without
    # esci_us_data/. Refactor to def main() if revisiting.
    "evaluation/eval_ensemble.py",
    # Opens esci_us_data/product_ids.json at module top level (one-off
    # data-acquisition utility, same pattern as download/*).
    "evaluation/check_image_coverage.py",
}


def collect_scripts():
    out = []
    for sub in SUBDIRS:
        for p in sorted((ROOT / sub).glob("*.py")):
            if p.name.startswith("__"):
                continue
            rel = str(p.relative_to(ROOT))
            if rel in IMPORT_SKIP:
                continue
            out.append(p)
    return out


@pytest.mark.parametrize("script", collect_scripts(), ids=lambda p: f"{p.parent.name}/{p.name}")
def test_script_imports(script):
    """Importing the script as a module must not raise."""
    rel = script.relative_to(ROOT)
    # Convert path → module dotted name (e.g., evaluation/eval_oracle_routing.py
    # → evaluation.eval_oracle_routing).
    module = str(rel.with_suffix("")).replace("/", ".")
    cmd = [sys.executable, "-c", f"import {module}"]
    result = subprocess.run(
        cmd,
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        timeout=15,
    )
    if result.returncode != 0:
        pytest.fail(
            f"Import of {rel} failed (exit {result.returncode}):\n"
            f"--- stderr ---\n{result.stderr}\n"
            f"--- stdout ---\n{result.stdout}"
        )
