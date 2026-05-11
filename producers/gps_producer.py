"""
Stream NYC TLC yellow taxi trips (pickup/dropoff zones and distance) to Kafka
``topic_gps``.

Performance fix: the previous version used ``df.iterrows()`` over millions of
rows, which is ~50x slower than the dict-of-records approach used here. A
2022-01 file (~3M rows) used to take 4+ hours; with ``to_dict('records')``
plus chunked sending it completes in minutes (rate is dominated by the
target replay throughput, not Python overhead).

CLI:
    python producers/gps_producer.py --rate 200 --limit 100000
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

import pandas as pd
from dotenv import load_dotenv
from jsonschema import Draft202012Validator
from kafka import KafkaProducer

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent
SCHEMA_PATH = REPO_ROOT / "schemas" / "gps.json"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="GPS / NYC TLC trip producer.")
    p.add_argument("--source", type=str,
                   default=str(REPO_ROOT / "data" / "nyc-taxi" / "yellow_tripdata_2022-01.parquet"),
                   help="Parquet path.")
    p.add_argument("--rate", type=float, default=200.0,
                   help="Target send rate in messages per second (default 200).")
    p.add_argument("--limit", type=int, default=None,
                   help="Optional cap on total messages sent.")
    return p.parse_args()


_running = True


def _handle_signal(signum, _frame):  # noqa: ANN001
    global _running
    logger.info("Received signal %s — flushing in-flight messages…", signum)
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
    topic = os.getenv("TOPIC_GPS", "topic_gps")
    producer = KafkaProducer(
        bootstrap_servers=bootstrap,
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
        linger_ms=20,
        batch_size=64 * 1024,
        acks="all",
        compression_type="lz4",
    )

    logger.info("Reading %s (this may take a moment for large files)…", src)
    df = pd.read_parquet(
        src,
        columns=[
            "tpep_pickup_datetime",
            "tpep_dropoff_datetime",
            "PULocationID",
            "DOLocationID",
            "trip_distance",
        ],
    )
    if args.limit:
        df = df.head(args.limit)
    logger.info("Loaded %s trips.", f"{len(df):,}")

    # Convert to plain Python types ahead of time so the hot loop is just dict + JSON.
    df = df.rename(
        columns={
            "tpep_pickup_datetime": "pickup_time",
            "tpep_dropoff_datetime": "dropoff_time",
            "PULocationID": "pickup_zone",
            "DOLocationID": "dropoff_zone",
            "trip_distance": "distance_miles",
        }
    )
    df["pickup_time"] = df["pickup_time"].astype(str)
    df["dropoff_time"] = df["dropoff_time"].astype(str)
    df["pickup_zone"] = df["pickup_zone"].astype(int)
    df["dropoff_zone"] = df["dropoff_zone"].astype(int)
    df["distance_miles"] = df["distance_miles"].astype(float)

    records = df.to_dict("records")  # ~50x faster than iterrows for downstream iteration
    sleep_per_msg = 1.0 / max(args.rate, 1.0)

    sent = 0
    invalid = 0
    try:
        for rec in records:
            if not _running:
                break
            rec["ingested_at"] = time.time()
            errors = list(validator.iter_errors(rec))
            if errors:
                invalid += 1
                if invalid <= 5:
                    logger.warning("Skip schema-invalid GPS row: %s", errors[0].message)
                continue
            producer.send(topic, value=rec)
            sent += 1
            if sent % 5000 == 0:
                logger.info("Sent %s messages…", f"{sent:,}")
            time.sleep(sleep_per_msg)
    finally:
        logger.info("Flushing producer (timeout 10s)…")
        producer.flush(timeout=10)
        producer.close(timeout=5)
        logger.info("GPS producer finished — sent=%s invalid=%s", f"{sent:,}", invalid)


if __name__ == "__main__":
    main()
