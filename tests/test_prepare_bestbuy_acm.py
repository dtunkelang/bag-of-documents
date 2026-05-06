"""Tests for download/prepare_bestbuy_acm.py.

The script doesn't actually download from Kaggle — it processes already-
downloaded data. Verify that:
  - Importing the module is safe (no work at import time).
  - check_prerequisites raises a clear, actionable error when the Kaggle
    data is missing.
"""

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


def _load_module():
    """Load the script as a module so we can call check_prerequisites directly."""
    spec = importlib.util.spec_from_file_location(
        "prepare_bestbuy_acm",
        ROOT / "download" / "prepare_bestbuy_acm.py",
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["prepare_bestbuy_acm"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_module_imports_without_data(tmp_path, monkeypatch):
    """Importing the module must not require the Kaggle data to exist."""
    monkeypatch.chdir(tmp_path)
    mod = _load_module()
    # Module-level constants should be set; no work was done.
    assert "kaggle.com" in mod.KAGGLE_URL
    assert mod.SOURCE_DIR == "acm-sf-chapter-hackathon-big"


def test_prerequisite_check_exits_with_clear_message(tmp_path, monkeypatch, capsys):
    """When the Kaggle data is missing, check_prerequisites should print an
    actionable message and exit with non-zero status."""
    monkeypatch.chdir(tmp_path)
    mod = _load_module()
    with pytest.raises(SystemExit) as exc:
        mod.check_prerequisites()
    assert exc.value.code != 0
    err = capsys.readouterr().err
    # Error should reference the Kaggle URL and the expected directory.
    assert "kaggle.com" in err
    assert "acm-sf-chapter-hackathon-big" in err
    assert "train.csv" in err


def test_prerequisite_check_passes_when_files_present(tmp_path, monkeypatch):
    """check_prerequisites should not exit when train.csv + an XML file exist."""
    monkeypatch.chdir(tmp_path)
    base = tmp_path / "acm-sf-chapter-hackathon-big"
    (base / "product_data" / "products").mkdir(parents=True)
    (base / "train.csv").write_text("user,sku,query\n,,\n")
    (base / "product_data" / "products" / "products_dummy.xml").write_text("<products/>")
    mod = _load_module()
    # Should NOT raise SystemExit.
    mod.check_prerequisites()
