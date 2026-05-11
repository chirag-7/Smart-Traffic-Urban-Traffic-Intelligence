"""Tests for the offline trainer's feature engineering.

The same logic runs inside ml_predictor_worker — we test the trainer side
because it's pure pandas and easy to import.
"""

from __future__ import annotations

import importlib

import numpy as np
import pandas as pd
import pytest


@pytest.fixture(scope="module")
def trainer_module():
    """Import ml/train_speed_model.py without running main()."""
    return importlib.import_module("train_speed_model")  # added to sys.path in conftest


def _make_synthetic_series(n_per_sensor=20, n_sensors=2, start="2022-01-01") -> pd.DataFrame:
    """Build a deterministic METR-LA-like long-format dataframe.

    Speeds = 50 + sensor_idx * 5 + row_idx so we can predict every lag exactly.
    """
    timestamps = pd.date_range(start=start, periods=n_per_sensor, freq="5min")
    rows = []
    for s in range(n_sensors):
        for i, ts in enumerate(timestamps):
            rows.append({
                "timestamp": ts,
                "sensor_id": f"sensor_{s}",
                "speed": 50.0 + s * 5.0 + i,
            })
    return pd.DataFrame(rows)


def test_add_features_drops_warm_up_rows(trainer_module):
    """First 12 rows per sensor should be dropped (lag_1h = 12 steps back)."""
    df = _make_synthetic_series(n_per_sensor=20, n_sensors=2)
    out = trainer_module.add_features(df)
    # 8 rows per sensor survive (20 - 12 warm-up).
    counts = out.groupby("sensor_id").size().to_dict()
    assert counts == {"sensor_0": 8, "sensor_1": 8}


def test_add_features_correct_lag_values(trainer_module):
    """Verify lag_5min, lag_15min, lag_1h, rolling_mean_30min on the simple series."""
    df = _make_synthetic_series(n_per_sensor=20, n_sensors=1)
    out = trainer_module.add_features(df)
    out = out.sort_values("timestamp").reset_index(drop=True)

    # Row 0 of out corresponds to source-row 12 (speed=50+12=62)
    row0 = out.iloc[0]
    assert row0["speed"] == pytest.approx(62.0)
    # lag_5min = 1 step back = row 11 = 61
    assert row0["lag_5min"] == pytest.approx(61.0)
    # lag_15min = 3 steps back = row 9 = 59
    assert row0["lag_15min"] == pytest.approx(59.0)
    # lag_1h = 12 steps back = row 0 = 50
    assert row0["lag_1h"] == pytest.approx(50.0)
    # rolling_mean_30min = mean of speeds at rows 6..11 = mean(56..61) = 58.5
    assert row0["rolling_mean_30min"] == pytest.approx(np.mean([56, 57, 58, 59, 60, 61]))


def test_feature_names_contract(trainer_module):
    """The exported FEATURE_NAMES must match the columns we put into the model."""
    df = _make_synthetic_series(n_per_sensor=20, n_sensors=1)
    out = trainer_module.add_features(df)
    for f in trainer_module.FEATURE_NAMES:
        assert f in out.columns, f"missing feature column: {f}"


def test_hour_dow_is_weekend(trainer_module):
    df = _make_synthetic_series(n_per_sensor=20, n_sensors=1, start="2022-01-08")  # Saturday
    out = trainer_module.add_features(df)
    # Day-of-week should be 6 (Saturday) for all surviving rows in this window.
    assert (out["dow"] >= 6).all()
    assert (out["is_weekend"] == 1).all()
