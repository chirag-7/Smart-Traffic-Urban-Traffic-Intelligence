"""Tests for ml_predictor_worker per-sensor sliding window + feature computation."""

from __future__ import annotations

import importlib

import numpy as np
import pytest


@pytest.fixture()
def worker(monkeypatch):
    """Import ml_predictor_worker.main with a clean window state."""
    # Patch out the schema-load side effect by setting SCHEMA_DIR to schemas/ on disk.
    import os
    REPO_ROOT_ENV = str(__import__("pathlib").Path(__file__).resolve().parent.parent)
    monkeypatch.setenv("SCHEMA_DIR", os.path.join(REPO_ROOT_ENV, "schemas"))
    # Import fresh each time so global _windows is reset.
    mod = importlib.import_module("main")
    mod = importlib.reload(mod)
    # Wipe per-sensor state for test isolation.
    mod._windows.clear()
    mod._windows.default_factory  # touch defaultdict to ensure import worked
    return mod


def _push(worker, sensor_id, ts_seq, speed_seq):
    """Append a sequence of (timestamp, speed) pairs to a sensor's window."""
    for ts, sp in zip(ts_seq, speed_seq):
        worker._windows[sensor_id].append((float(ts), float(sp)))


def test_warming_up_returns_none(worker):
    # Only 5 entries — not enough for lag_1h (needs 12).
    base = 1715000000.0
    _push(worker, "S1", [base + 300 * i for i in range(5)], [60.0] * 5)
    feats = worker.compute_features("S1", base + 300 * 5)
    assert feats is None


def test_features_with_full_window(worker):
    # 12 entries with linearly increasing speeds gives deterministic lag values.
    base = 1715000000.0
    timestamps = [base + 300 * i for i in range(12)]
    speeds = [50.0 + i for i in range(12)]  # 50, 51, ..., 61
    _push(worker, "S1", timestamps, speeds)

    feats = worker.compute_features("S1", base + 300 * 12)
    assert feats is not None
    assert feats.shape == (7,)
    # Indices match worker.FEATURE_NAMES order.
    hour, dow, weekend, lag_5, lag_15, lag_1h, rolling = feats.tolist()
    assert lag_5 == pytest.approx(61.0)         # most recent
    assert lag_15 == pytest.approx(59.0)        # 3 steps back
    assert lag_1h == pytest.approx(50.0)        # 12 steps back (oldest)
    assert rolling == pytest.approx(np.mean([56, 57, 58, 59, 60, 61]))
    assert 0 <= hour < 24
    assert 1 <= dow <= 7
    assert weekend in (0, 1)


def test_parse_timestamp_handles_iso_and_epoch(worker):
    # ISO with seconds
    ts = worker._parse_timestamp("2022-01-01 00:05:00")
    assert ts is not None and ts > 0
    # Nanosecond integer (METR-LA axis1 format)
    ts2 = worker._parse_timestamp("1640995500000000000")
    assert ts2 is not None and ts2 > 0
    # Unparseable
    assert worker._parse_timestamp("not-a-time") is None


def test_state_isolated_per_sensor(worker):
    base = 1715000000.0
    _push(worker, "S_A", [base + 300 * i for i in range(12)], [40.0 + i for i in range(12)])
    _push(worker, "S_B", [base + 300 * i for i in range(12)], [80.0 + i for i in range(12)])
    fa = worker.compute_features("S_A", base + 300 * 12)
    fb = worker.compute_features("S_B", base + 300 * 12)
    assert fa is not None and fb is not None
    # lag_1h indices show different absolute values per sensor.
    assert fa[5] == pytest.approx(40.0)
    assert fb[5] == pytest.approx(80.0)
