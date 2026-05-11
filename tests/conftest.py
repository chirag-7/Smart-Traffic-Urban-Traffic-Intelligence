"""Shared pytest fixtures for smart-traffic tests."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent

# Make repo packages importable without installing.
for sub in ("ml_predictor_worker", "ml", "spark_jobs", "producers"):
    sys.path.insert(0, str(REPO_ROOT / sub))


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return REPO_ROOT


@pytest.fixture(scope="session")
def schemas_dir(repo_root) -> Path:
    return repo_root / "schemas"


@pytest.fixture(scope="session")
def load_schema(schemas_dir):
    """Return a function (name) -> parsed JSON Schema dict."""
    def _load(name: str) -> dict:
        with (schemas_dir / name).open("r", encoding="utf-8") as fh:
            return json.load(fh)
    return _load
