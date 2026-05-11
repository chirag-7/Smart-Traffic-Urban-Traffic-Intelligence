"""
Online speed-prediction Kafka worker.

Flow per message:
  1. Consume reading from topic_sensors (sticky-partitioned by sensor_id).
  2. Validate against schemas/sensors.json.
  3. Maintain a per-sensor sliding window (collections.deque) of recent
     (timestamp, speed) pairs. The window is large enough to cover the
     longest lag feature (lag_1h = 12 steps of 5 minutes).
  4. Compute features identical to the offline trainer:
       hour, dow, is_weekend, lag_5min, lag_15min, lag_1h, rolling_mean_30min
  5. Run onnxruntime InferenceSession.run() to predict speed.
  6. Validate output against schemas/speed_predictions.json.
  7. Publish to topic_speed_predictions (sticky by sensor_id).

Warm start
----------
On boot the worker reads the last ~30 minutes of sensor_speeds from
``delta_tables/sensor_speeds`` via the ``deltalake`` (Rust) package and
primes every sensor's deque. This eliminates the cold-start period during
which lag-1h and rolling features would be NaN. Production-grade state
fault tolerance would require a stateful processor (Faust / Kafka
Streams); for this prototype the warm-start trick is sufficient.

Metrics exposed on /metrics port ML_WORKER_METRICS_PORT (default 9101):
  ml_predictions_total{sensor_id}
  ml_predictions_skipped_total{reason}
  ml_inference_microseconds{sensor_id}
  ml_active_sensors
  ml_warm_start_rows
"""

from __future__ import annotations

import json
import logging
import os
import signal
import socket
import sys
import time
from collections import defaultdict, deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Deque, Dict, Optional, Tuple

import numpy as np
from confluent_kafka import Consumer, KafkaError, KafkaException, Producer
from jsonschema import Draft202012Validator
from prometheus_client import Counter, Gauge, Histogram, start_http_server

# NOTE: onnxruntime is intentionally NOT imported at module level. It's a
# C++ extension that requires platform-specific runtime libraries (e.g.
# vcruntime140_1.dll on Windows) and importing it eagerly would prevent
# the test suite from exercising the pure-Python window/feature logic on
# hosts that don't have the VC++ Redistributable installed. We import it
# lazily inside `main()` instead — production containers always have it.

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s ml-predictor-worker %(message)s",
)
logger = logging.getLogger("ml-predictor-worker")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

WORKER_ID = os.getenv("HOSTNAME", socket.gethostname())

KAFKA_BROKERS = os.getenv("KAFKA_BROKERS", "kafka:29092")
INPUT_TOPIC = os.getenv("INPUT_TOPIC", "topic_sensors")
OUTPUT_TOPIC = os.getenv("OUTPUT_TOPIC", "topic_speed_predictions")
CONSUMER_GROUP = os.getenv("CONSUMER_GROUP", "ml-predictor-workers")

MODEL_PATH = os.getenv("ML_MODEL_PATH", "/app/models/speed_predictor/speed_predictor.onnx")
META_PATH = os.getenv("ML_MODEL_META_PATH", "/app/models/speed_predictor/speed_predictor_meta.json")
SCHEMA_DIR = Path(os.getenv("SCHEMA_DIR", "/app/schemas"))

METRICS_PORT = int(os.getenv("ML_WORKER_METRICS_PORT", "9101"))
MODEL_VERSION = os.getenv("ML_MODEL_VERSION", "lgbm_speed_v1")

DELTA_SENSOR_PATH = os.getenv("DELTA_SENSOR_PATH", "/opt/delta_tables/sensor_speeds")
WARM_START_MINUTES = float(os.getenv("ML_WARM_START_MINUTES", "30"))

# Feature contract — must match ml/train_speed_model.py exactly.
FEATURE_NAMES = [
    "hour",
    "dow",
    "is_weekend",
    "lag_5min",
    "lag_15min",
    "lag_1h",
    "rolling_mean_30min",
]
SAMPLE_INTERVAL_MIN = 5
LAG_INDICES = {"lag_5min": 1, "lag_15min": 3, "lag_1h": 12}
ROLLING_WINDOW_STEPS = 6
MAX_WINDOW = 12  # enough for lag_1h plus rolling_mean_30min context

# ---------------------------------------------------------------------------
# Prometheus metrics
# ---------------------------------------------------------------------------

PRED_TOTAL = Counter(
    "ml_predictions_total",
    "Successful predictions published to topic_speed_predictions.",
    labelnames=("sensor_id",),
)
PRED_SKIPPED = Counter(
    "ml_predictions_skipped_total",
    "Skipped messages, by reason.",
    labelnames=("reason",),
)
INFERENCE_US = Histogram(
    "ml_inference_microseconds",
    "onnxruntime inference time per row (microseconds).",
    labelnames=("sensor_id",),
    buckets=(50, 100, 200, 500, 1000, 2000, 5000, 10000),
)
ACTIVE_SENSORS = Gauge(
    "ml_active_sensors",
    "Number of sensor_ids with a populated sliding window.",
)
WARM_START_ROWS = Gauge(
    "ml_warm_start_rows",
    "Rows read from Delta during warm start.",
)

# ---------------------------------------------------------------------------
# Schema validators
# ---------------------------------------------------------------------------

def _load_validator(name: str) -> Draft202012Validator:
    with (SCHEMA_DIR / name).open("r", encoding="utf-8") as fh:
        return Draft202012Validator(json.load(fh))


try:
    INPUT_VALIDATOR = _load_validator("sensors.json")
    OUTPUT_VALIDATOR = _load_validator("speed_predictions.json")
except FileNotFoundError as exc:
    logger.error("Schema file missing at %s: %s", SCHEMA_DIR, exc)
    sys.exit(2)

# ---------------------------------------------------------------------------
# Per-sensor sliding window state
# ---------------------------------------------------------------------------

WindowEntry = Tuple[float, float]  # (epoch_seconds, speed_mph)
_windows: Dict[str, Deque[WindowEntry]] = defaultdict(lambda: deque(maxlen=MAX_WINDOW))


_DATETIME_FORMATS = (
    "%Y-%m-%dT%H:%M:%S",
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%dT%H:%M:%S.%f",
)


def _parse_timestamp(value: str) -> Optional[float]:
    """Parse the producer's timestamp into epoch seconds. Returns None on failure.

    Handles three common METR-LA / producer formats:
      - Nanoseconds since epoch as a digit string (h5py axis1 default)
      - ISO-8601 (``2022-01-01T00:05:00``)
      - Space-separated ``YYYY-MM-DD HH:MM:SS`` (with or without fractional seconds)
    """
    if value is None:
        return None
    s = str(value)
    if s.isdigit():
        try:
            return float(s) / 1e9
        except (ValueError, TypeError):
            return None
    try:
        return datetime.fromisoformat(s).timestamp()
    except ValueError:
        pass
    for fmt in _DATETIME_FORMATS:
        try:
            return datetime.strptime(s, fmt).timestamp()
        except ValueError:
            continue
    return None


def compute_features(sensor_id: str, ts_epoch: float) -> Optional[np.ndarray]:
    """Compute the 7-feature vector from the window before the current sample.

    Returns None when the window is too short to satisfy lag_1h or the rolling
    mean (i.e. the cold-start period for this sensor).
    """
    win = _windows[sensor_id]
    if len(win) < LAG_INDICES["lag_1h"]:
        return None

    speeds = [v for _, v in win]
    lag_5 = speeds[-LAG_INDICES["lag_5min"]]
    lag_15 = speeds[-LAG_INDICES["lag_15min"]]
    lag_1h = speeds[-LAG_INDICES["lag_1h"]]
    rolling = float(np.mean(speeds[-ROLLING_WINDOW_STEPS:]))

    dt = datetime.fromtimestamp(ts_epoch, tz=timezone.utc)
    dow = dt.isoweekday()  # 1=Mon … 7=Sun (matches pandas dayofweek+1)
    is_weekend = 1 if dow >= 6 else 0

    return np.array(
        [
            dt.hour,
            dow,
            is_weekend,
            lag_5,
            lag_15,
            lag_1h,
            rolling,
        ],
        dtype=np.float32,
    )


# ---------------------------------------------------------------------------
# Warm start from Delta
# ---------------------------------------------------------------------------

def warm_start(path: str, minutes: float) -> int:
    """Prime per-sensor windows from the last ``minutes`` of sensor_speeds.

    No-op when Delta path is missing (first run, before Spark has written
    anything). Returns the number of rows ingested.
    """
    if not os.path.exists(path):
        logger.info("Warm-start skipped — %s not present yet.", path)
        return 0

    try:
        from deltalake import DeltaTable  # imported lazily to avoid hard dep on boot
        import pandas as pd
    except Exception as exc:  # noqa: BLE001
        logger.warning("Warm-start unavailable (%s) — windows will fill organically.", exc)
        return 0

    try:
        dt_table = DeltaTable(path)
        df = dt_table.to_pandas()
    except Exception as exc:  # noqa: BLE001
        logger.warning("Warm-start failed to read Delta (%s).", exc)
        return 0

    if df.empty:
        logger.info("Warm-start: Delta table is empty.")
        return 0

    df = df.dropna(subset=["sensor_id", "speed", "timestamp"]).copy()
    df["ts_epoch"] = df["timestamp"].apply(_parse_timestamp)
    df = df.dropna(subset=["ts_epoch"])
    if df.empty:
        return 0

    cutoff = df["ts_epoch"].max() - minutes * 60.0
    df = df[df["ts_epoch"] >= cutoff]
    df = df.sort_values("ts_epoch")

    rows = 0
    for sensor_id, group in df.groupby("sensor_id"):
        win = _windows[sensor_id]
        for entry in group[["ts_epoch", "speed"]].itertuples(index=False):
            win.append((float(entry.ts_epoch), float(entry.speed)))
            rows += 1

    ACTIVE_SENSORS.set(len(_windows))
    WARM_START_ROWS.set(rows)
    logger.info("Warm-start primed %s sensors with %s rows from %s.",
                len(_windows), rows, path)
    return rows


# ---------------------------------------------------------------------------
# Kafka factories
# ---------------------------------------------------------------------------

def make_consumer() -> Consumer:
    return Consumer({
        "bootstrap.servers": KAFKA_BROKERS,
        "group.id": CONSUMER_GROUP,
        "auto.offset.reset": "latest",
        "enable.auto.commit": False,
        "session.timeout.ms": 30_000,
        "max.poll.interval.ms": 600_000,
        "client.id": f"ml-worker-{WORKER_ID}",
    })


def make_producer() -> Producer:
    return Producer({
        "bootstrap.servers": KAFKA_BROKERS,
        "linger.ms": 50,
        "batch.size": 32 * 1024,
        "compression.type": "lz4",
        "acks": "all",
        "enable.idempotence": True,
        "client.id": f"ml-worker-{WORKER_ID}",
    })


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def process_message(raw: bytes, session, input_name: str, producer: Producer) -> None:
    """Run inference for one Kafka message.

    ``session`` is an ``onnxruntime.InferenceSession`` (typed as Any here so
    this module can be imported without onnxruntime — see top-of-file note).
    """
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        PRED_SKIPPED.labels(reason="json_decode_error").inc()
        return

    errors = list(INPUT_VALIDATOR.iter_errors(data))
    if errors:
        PRED_SKIPPED.labels(reason="schema_invalid").inc()
        return

    sensor_id = data["sensor_id"]
    ts_epoch = _parse_timestamp(data.get("timestamp"))
    if ts_epoch is None:
        PRED_SKIPPED.labels(reason="bad_timestamp").inc()
        return

    speed = float(data["speed"])

    # Compute features *before* appending — features must reference past samples only.
    features = compute_features(sensor_id, ts_epoch)
    _windows[sensor_id].append((ts_epoch, speed))
    ACTIVE_SENSORS.set(len(_windows))

    if features is None:
        PRED_SKIPPED.labels(reason="warming_up").inc()
        return

    inputs = features.reshape(1, -1)
    t0 = time.perf_counter()
    try:
        outputs = session.run(None, {input_name: inputs})
    except Exception as exc:  # noqa: BLE001
        PRED_SKIPPED.labels(reason="inference_error").inc()
        logger.warning("Inference error sensor=%s: %s", sensor_id, exc)
        return
    inf_us = (time.perf_counter() - t0) * 1e6
    INFERENCE_US.labels(sensor_id=sensor_id).observe(inf_us)

    predicted = float(np.asarray(outputs[0]).reshape(-1)[0])

    msg = {
        "sensor_id": sensor_id,
        "timestamp": str(data["timestamp"]),
        "actual_speed": speed,
        "predicted_speed": predicted,
        "horizon_minutes": 0,
        "model_version": MODEL_VERSION,
        "inference_us": inf_us,
        "processed_at": time.time(),
    }
    out_errors = list(OUTPUT_VALIDATOR.iter_errors(msg))
    if out_errors:
        PRED_SKIPPED.labels(reason="output_invalid").inc()
        logger.warning("Output schema invalid for sensor=%s: %s", sensor_id, out_errors[0].message)
        return

    producer.produce(
        OUTPUT_TOPIC,
        key=sensor_id.encode("utf-8"),
        value=json.dumps(msg).encode("utf-8"),
    )
    PRED_TOTAL.labels(sensor_id=sensor_id).inc()


_running = True


def _handle_signal(signum, _frame):  # noqa: ANN001
    global _running
    logger.info("Received signal %s — shutting down gracefully.", signum)
    _running = False


def main() -> None:
    if not os.path.exists(MODEL_PATH):
        logger.error(
            "ONNX model not found at %s. Run `python ml/train_speed_model.py` first "
            "and ensure ./models is mounted into the worker.",
            MODEL_PATH,
        )
        sys.exit(1)

    logger.info(
        "ml-predictor-worker starting | bootstrap=%s in=%s out=%s group=%s "
        "model=%s metrics_port=%s",
        KAFKA_BROKERS, INPUT_TOPIC, OUTPUT_TOPIC, CONSUMER_GROUP, MODEL_PATH, METRICS_PORT,
    )

    start_http_server(METRICS_PORT)
    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    # Lazy-import so the module is import-clean on hosts without the C++
    # runtime that onnxruntime needs (see the top-of-file note).
    import onnxruntime as ort

    sess = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    logger.info("Loaded ONNX model (input=%s, version=%s)", input_name, MODEL_VERSION)

    if os.path.exists(META_PATH):
        try:
            with open(META_PATH, "r", encoding="utf-8") as fh:
                meta = json.load(fh)
            logger.info(
                "Model metadata: trained_at=%s rmse=%s mae=%s r2=%s",
                meta.get("trained_at"), meta.get("rmse"),
                meta.get("mae"), meta.get("r2"),
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not parse %s: %s", META_PATH, exc)

    warm_start(DELTA_SENSOR_PATH, WARM_START_MINUTES)

    consumer = make_consumer()
    producer = make_producer()
    consumer.subscribe([INPUT_TOPIC])

    msg_count = 0
    last_commit = time.time()
    try:
        while _running:
            msg = consumer.poll(timeout=1.0)
            if msg is None:
                producer.poll(0)
                if time.time() - last_commit > 5.0:
                    consumer.commit(asynchronous=True)
                    last_commit = time.time()
                continue
            if msg.error():
                if msg.error().code() == KafkaError._PARTITION_EOF:
                    continue
                raise KafkaException(msg.error())

            process_message(msg.value(), sess, input_name, producer)
            msg_count += 1
            if msg_count % 500 == 0:
                producer.poll(0)
                consumer.commit(asynchronous=True)
                last_commit = time.time()
                logger.info("Processed %s messages so far.", msg_count)
    finally:
        logger.info("Flushing producer and closing consumer…")
        producer.flush(10.0)
        try:
            consumer.commit(asynchronous=False)
        except KafkaException:
            pass
        consumer.close()
        logger.info("ml-predictor-worker stopped. Processed %s messages.", msg_count)


if __name__ == "__main__":
    main()
