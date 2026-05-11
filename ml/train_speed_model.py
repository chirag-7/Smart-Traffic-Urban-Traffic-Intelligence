"""
Offline speed-prediction trainer.

Reads the full METR-LA dataset, engineers temporal and lag features per sensor,
trains a LightGBM regressor, exports the model to ONNX, and logs the run to
MLflow.

The exact same feature engineering is implemented inside the online
``ml_predictor_worker`` so that training and serving stay aligned.

Outputs
-------
- ``models/speed_predictor/speed_predictor.onnx``       — ONNX model used by the worker
- ``models/speed_predictor/speed_predictor_meta.json``  — feature list + version metadata
- MLflow run with params, metrics, importances, and artifacts

Run from repo root:

    python ml/train_speed_model.py

Or with custom hyperparameters and a tracking URI:

    python ml/train_speed_model.py --num-leaves 127 \
        --mlflow-uri http://localhost:5000
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from pathlib import Path

import h5py
import lightgbm as lgb
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from onnxconverter_common.data_types import FloatTensorType
from onnxmltools import convert_lightgbm
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

import mlflow

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("train_speed_model")

REPO_ROOT = Path(__file__).resolve().parent.parent
H5_PATH = REPO_ROOT / "data" / "metr-la" / "METR-LA.h5"
MODEL_DIR = REPO_ROOT / "models" / "speed_predictor"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# Feature contract — must match ml_predictor_worker/main.py exactly.
FEATURE_NAMES = [
    "hour",
    "dow",
    "is_weekend",
    "lag_5min",
    "lag_15min",
    "lag_1h",
    "rolling_mean_30min",
]
SAMPLE_INTERVAL_MIN = 5            # METR-LA samples every 5 minutes
LAG_INDICES = {                    # in deque-of-most-recent units
    "lag_5min":  1,                # 1 step back  =  5 min
    "lag_15min": 3,                # 3 steps back = 15 min
    "lag_1h":   12,                # 12 steps back = 60 min
}
ROLLING_WINDOW_STEPS = 6           # 6 steps × 5 min = 30 min


# ---------------------------------------------------------------------------
# Load + reshape METR-LA
# ---------------------------------------------------------------------------

def load_metr_la(h5_path: Path) -> pd.DataFrame:
    if not h5_path.exists():
        raise FileNotFoundError(f"METR-LA not found at {h5_path}")
    with h5py.File(h5_path, "r") as f:
        values = f["df/block0_values"][:]            # (T, N)
        timestamps = f["df/axis1"][:]
        sensors = f["df/block0_items"][:]
    n_t, n_s = values.shape
    logger.info("METR-LA shape: %s timestamps × %s sensors", n_t, n_s)

    ts_series = pd.Series(timestamps)
    if pd.api.types.is_numeric_dtype(ts_series):
        ts_series = pd.to_datetime(ts_series, unit="ns")
    else:
        ts_series = pd.to_datetime(
            ts_series.str.decode("utf-8") if hasattr(ts_series.str, "decode") else ts_series
        )

    sensor_str = np.array(
        [s.decode("utf-8") if isinstance(s, (bytes, bytearray)) else str(s) for s in sensors]
    )

    df = pd.DataFrame(
        {
            "timestamp": np.repeat(ts_series.values, n_s),
            "sensor_id": np.tile(sensor_str, n_t),
            "speed": values.reshape(-1),
        }
    )
    df = df.dropna(subset=["speed"])
    return df.sort_values(["sensor_id", "timestamp"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Feature engineering — replicated 1:1 in the online worker
# ---------------------------------------------------------------------------

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["hour"] = df["timestamp"].dt.hour.astype("int16")
    df["dow"] = (df["timestamp"].dt.dayofweek + 1).astype("int16")
    df["is_weekend"] = (df["dow"] >= 6).astype("int16")

    grouped = df.groupby("sensor_id", sort=False)["speed"]
    df["lag_5min"] = grouped.shift(LAG_INDICES["lag_5min"])
    df["lag_15min"] = grouped.shift(LAG_INDICES["lag_15min"])
    df["lag_1h"] = grouped.shift(LAG_INDICES["lag_1h"])
    df["rolling_mean_30min"] = (
        grouped.shift(1).rolling(window=ROLLING_WINDOW_STEPS, min_periods=2).mean()
    )

    df = df.dropna(subset=FEATURE_NAMES)
    return df


# ---------------------------------------------------------------------------
# ONNX export
# ---------------------------------------------------------------------------

def to_onnx(model: lgb.LGBMRegressor, n_features: int) -> bytes:
    initial_types = [("input", FloatTensorType([None, n_features]))]
    onnx_model = convert_lightgbm(model, initial_types=initial_types, target_opset=14)
    return onnx_model.SerializeToString()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train LightGBM speed predictor and export to ONNX.")
    p.add_argument("--limit-rows", type=int, default=None,
                   help="Optional cap on rows for quick iteration. Defaults to full dataset.")
    p.add_argument("--mlflow-uri", type=str,
                   default=os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"),
                   help="MLflow tracking URI (default: env MLFLOW_TRACKING_URI or localhost:5000).")
    p.add_argument("--mlflow-experiment", type=str,
                   default=os.getenv("MLFLOW_EXPERIMENT_NAME", "smart-traffic-speed"))
    p.add_argument("--num-leaves", type=int, default=63)
    p.add_argument("--learning-rate", type=float, default=0.05)
    p.add_argument("--n-estimators", type=int, default=400)
    p.add_argument("--model-version", type=str, default="lgbm_speed_v1")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    logger.info("Loading METR-LA…")
    t0 = time.time()
    df = load_metr_la(H5_PATH)
    logger.info("Loaded %s rows in %.1fs", f"{len(df):,}", time.time() - t0)
    if args.limit_rows:
        df = df.head(args.limit_rows)
        logger.info("Truncated to %s rows for quick iteration.", f"{len(df):,}")

    logger.info("Engineering features…")
    t0 = time.time()
    df = add_features(df)
    logger.info("Feature dataframe: %s rows × %s features in %.1fs",
                f"{len(df):,}", len(FEATURE_NAMES), time.time() - t0)

    X = df[FEATURE_NAMES].astype("float32").values
    y = df["speed"].astype("float32").values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=True
    )
    logger.info("Train / test split: %s / %s", len(X_train), len(X_test))

    try:
        mlflow.set_tracking_uri(args.mlflow_uri)
        mlflow.set_experiment(args.mlflow_experiment)
        run_ctx = mlflow.start_run(run_name=args.model_version)
        mlflow_enabled = True
        logger.info("MLflow tracking enabled at %s", args.mlflow_uri)
    except Exception as exc:  # noqa: BLE001
        logger.warning("MLflow unavailable (%s) — proceeding without tracking.", exc)
        run_ctx = None
        mlflow_enabled = False

    params = {
        "objective": "regression",
        "metric": "rmse",
        "num_leaves": args.num_leaves,
        "learning_rate": args.learning_rate,
        "n_estimators": args.n_estimators,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.9,
        "bagging_freq": 5,
        "min_child_samples": 50,
        "verbose": -1,
        "random_state": 42,
    }
    logger.info("Training LightGBM with params: %s", params)
    t0 = time.time()
    model = lgb.LGBMRegressor(**params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        eval_metric="rmse",
        callbacks=[lgb.early_stopping(stopping_rounds=20, verbose=False)],
    )
    train_seconds = time.time() - t0
    logger.info("Trained in %.1fs (best iter=%s)", train_seconds, model.best_iteration_)

    y_pred = model.predict(X_test)
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    mae = float(mean_absolute_error(y_test, y_pred))
    r2 = float(r2_score(y_test, y_pred))
    logger.info("Holdout — RMSE %.4f  MAE %.4f  R² %.4f", rmse, mae, r2)

    # --- ONNX export ---
    onnx_bytes = to_onnx(model, n_features=len(FEATURE_NAMES))
    onnx_path = MODEL_DIR / "speed_predictor.onnx"
    onnx_path.write_bytes(onnx_bytes)
    logger.info("Wrote %s (%s KiB)", onnx_path, len(onnx_bytes) // 1024)

    meta = {
        "model_version": args.model_version,
        "feature_names": FEATURE_NAMES,
        "sample_interval_min": SAMPLE_INTERVAL_MIN,
        "lag_indices": LAG_INDICES,
        "rolling_window_steps": ROLLING_WINDOW_STEPS,
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "trained_at": time.time(),
        "n_train": len(X_train),
        "n_test": len(X_test),
        "best_iteration": int(model.best_iteration_) if model.best_iteration_ else None,
    }
    meta_path = MODEL_DIR / "speed_predictor_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2))
    logger.info("Wrote %s", meta_path)

    if mlflow_enabled:
        with run_ctx:
            mlflow.log_params(params)
            mlflow.log_metric("rmse", rmse)
            mlflow.log_metric("mae", mae)
            mlflow.log_metric("r2", r2)
            mlflow.log_metric("n_train", len(X_train))
            mlflow.log_metric("n_test", len(X_test))
            mlflow.log_metric("train_seconds", train_seconds)
            mlflow.log_dict(meta, "speed_predictor_meta.json")
            mlflow.log_artifact(str(onnx_path), artifact_path="onnx")
            importances = dict(zip(FEATURE_NAMES, [float(v) for v in model.feature_importances_]))
            mlflow.log_dict(importances, "feature_importances.json")
            mlflow.set_tag("model_version", args.model_version)
            mlflow.set_tag("framework", "lightgbm+onnx")

    logger.info("Done.")


if __name__ == "__main__":
    main()
