"""Verify shell scripts reference existing Python files.

After the directory reorg, both scripts/run_pipeline.sh and evaluation/eval_new_ce.sh kept
references to scripts at their old top-level paths and would have failed
to run. These tests catch that class of regression: any
`python ... some_script.py` invocation in a .sh file is checked against
the filesystem.
"""

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent

# Match a python invocation followed by a .py file path. Permissive enough to
# catch common forms: `python foo.py`, `.venv/bin/python -u foo.py`,
# `$VENV foo.py`, `.venv/bin/python evaluation/eval_x.py`.
PYTHON_INVOCATION_RE = re.compile(
    r"(?:^|[\s|])(?:[\w./$\-]*python[0-9.]*|\$VENV)\s+(?:-[a-zA-Z]+\s+)*([\w./\-]+\.py)\b"
)


def collect_shell_scripts():
    """Project-owned shell scripts only — exclude .venv and other vendored trees."""
    excluded_parts = {".venv", "node_modules", ".git"}
    return sorted(p for p in ROOT.rglob("*.sh") if not (set(p.parts) & excluded_parts))


@pytest.mark.parametrize(
    "shell_script", collect_shell_scripts(), ids=lambda p: str(p.relative_to(ROOT))
)
def test_shell_script_paths_exist(shell_script):
    """Every `python <path>.py` reference in a .sh file must point at a real file."""
    text = shell_script.read_text()
    matches = PYTHON_INVOCATION_RE.findall(text)
    missing = []
    for path_str in matches:
        # Skip absolute paths (system tools, /tmp/...)
        if path_str.startswith("/"):
            continue
        # Skip stdin/heredoc placeholders
        if path_str in ("-",):
            continue
        candidate = ROOT / path_str
        if not candidate.exists():
            missing.append(path_str)
    if missing:
        pytest.fail(f"{shell_script.relative_to(ROOT)} references missing scripts: {missing}")
