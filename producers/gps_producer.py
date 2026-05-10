"""
Stream NYC TLC yellow taxi trips (pickup/dropoff zones and distance) to Kafka ``topic_gps``.

Expects ``data/nyc-taxi/yellow_tripdata_2022-01.parquet`` (or equivalent path in your setup).
"""

from __future__ import annotations

import json
import logging
import os
import time

import pandas as pd
from dotenv import load_dotenv
from kafka import KafkaProducer

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

KAFKA_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")

producer = KafkaProducer(
    bootstrap_servers=KAFKA_SERVERS,
    value_serializer=lambda v: json.dumps(v).encode("utf-8"),
    linger_ms=10,
    batch_size=16384,
)

df = pd.read_parquet(
    "data/nyc-taxi/yellow_tripdata_2022-01.parquet",
    columns=[
        "tpep_pickup_datetime",
        "tpep_dropoff_datetime",
        "PULocationID",
        "DOLocationID",
        "trip_distance",
    ],
)
logger.info("Kafka %s — loaded %s trips", KAFKA_SERVERS, f"{len(df):,}")

row_count = 0
for _, row in df.iterrows():
    producer.send(
        "topic_gps",
        value={
            "pickup_time": str(row["tpep_pickup_datetime"]),
            "dropoff_time": str(row["tpep_dropoff_datetime"]),
            "pickup_zone": int(row["PULocationID"]),
            "dropoff_zone": int(row["DOLocationID"]),
            "distance_miles": float(row["trip_distance"]),
        },
    )
    row_count += 1
    if row_count % 5000 == 0:
        logger.info("Sent %s messages…", f"{row_count:,}")
    time.sleep(0.005)

producer.flush()
logger.info("Finished — %s messages.", f"{row_count:,}")
