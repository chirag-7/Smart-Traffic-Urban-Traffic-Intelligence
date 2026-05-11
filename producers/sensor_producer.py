"""
Publish METR-LA freeway speeds to Kafka ``topic_sensors``.

Reads ``data/metr-la/METR-LA.h5`` (h5py layout: block0_values, axis1, block0_items).
Each message includes ``sensor_index`` (column order = adjacency matrix index) so the
streaming job can join congestion events to ``graph_shortest_paths``.

Phase 3 hardening:
  - Sticky partitioning: ``key=sensor_id`` so the ml_predictor_worker sees a
    coherent per-sensor stream on a single partition (required for the
    sliding window).
  - jsonschema validation against schemas/sensors.json before publishing.
  - SIGINT/SIGTERM handler that flushes in-flight messages on Ctrl+C.
  - --rate flag to control replay throughput.

CLI:
    python producers/sensor_producer.py --rate 200 --limit 100000
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import signal
import sys
import time
from pathlib import Path

import h5py
import numpy as np
from dotenv import load_dotenv
from jsonschema import Draft202012Validator
from kafka import KafkaProducer

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent
SCHEMA_PATH = REPO_ROOT / "schemas" / "sensors.json"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="METR-LA sensor producer.")
    p.add_argument("--rate", type=float, default=2000.0,
                   help="Target send rate in messages per second (default 2000).")
    p.add_argument("--limit", type=int, default=None,
                   help="Optional cap on total messages sent (useful for quick demos).")
    p.add_argument("--source", type=str,
                   default=str(REPO_ROOT / "data" / "metr-la" / "METR-LA.h5"),
                   help="Path to METR-LA.h5.")
    return p.parse_args()


_running = True


def _handle_signal(signum, _frame):  # noqa: ANN001
    global _running
    logger.info("Received signal %s — flushing producer…", signum)
    _running = False


def main() -> None:
    args = parse_args()
    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    src = Path(args.source)
    if not src.exists():
        logger.error("Source not found: %s", src)
        sys.exit(1)

    with SCHEMA_PATH.open("r", encoding="utf-8") as fh:
        validator = Draft202012Validator(json.load(fh))

    bootstrap = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
    topic = os.getenv("TOPIC_SENSORS", "topic_sensors")

    producer = KafkaProducer(
        bootstrap_servers=bootstrap,
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
        key_serializer=lambda k: k.encode("utf-8") if isinstance(k, str) else k,
        linger_ms=20,
        batch_size=64 * 1024,
        acks="all",
        compression_type="gzip",
    )
    logger.info("Kafka %s | topic=%s | rate=%s msg/s", bootstrap, topic, args.rate)
    logger.info("Loading METR-LA from %s…", src)

    sleep_per_msg = 1.0 / max(args.rate, 1.0)
    sent = 0
    invalid = 0

    try:
        with h5py.File(src, "r") as f:
            values = f["df/block0_values"][:]
            timestamps = f["df/axis1"][:]
            sensors = f["df/block0_items"][:]
            logger.info("Shape %s × %s", values.shape[0], values.shape[1])

            for t_idx, timestamp in enumerate(timestamps):
                if not _running:
                    break
                for s_idx, sensor_id in enumerate(sensors):
                    if not _running:
                        break
                    speed_value = values[t_idx, s_idx]
                    if np.isnan(speed_value):
                        continue
                    sensor_id_str = (
                        sensor_id.decode("utf-8")
                        if isinstance(sensor_id, bytes)
                        else str(sensor_id)
                    )
                    msg = {
                        "timestamp": str(timestamp),
                        "sensor_id": sensor_id_str,
                        "sensor_index": int(s_idx),
                        "speed": float(speed_value),
                        "ingested_at": time.time(),
                    }
                    errors = list(validator.iter_errors(msg))
                    if errors:
                        invalid += 1
                        if invalid <= 5:
                            logger.warning("Skip schema-invalid sensor row: %s", errors[0].message)
                        continue

                    producer.send(topic, key=sensor_id_str, value=msg)
                    sent += 1
                    if sent % 5000 == 0:
                        logger.info("Sent %s messages…", f"{sent:,}")
                    if args.limit and sent >= args.limit:
                        break

                    time.sleep(sleep_per_msg)
                if args.limit and sent >= args.limit:
                    break

    finally:
        logger.info("Flushing producer (timeout 10s)…")
        producer.flush(timeout=10)
        producer.close(timeout=5)
        logger.info("Sensor producer finished — sent=%s invalid=%s", f"{sent:,}", invalid)


if __name__ == "__main__":
    main()
