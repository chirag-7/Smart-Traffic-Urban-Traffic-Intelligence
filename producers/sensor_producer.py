"""
Publish METR-LA freeway speeds to Kafka ``topic_sensors``.

Reads ``data/metr-la/METR-LA.h5`` (h5py layout: block0_values, axis1, block0_items).
Each message includes ``sensor_index`` (column order = adjacency matrix index) so the
streaming job can join congestion events to ``graph_shortest_paths``.
"""

from __future__ import annotations

import json
import logging
import os
import time

import h5py
import numpy as np
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

logger.info("Kafka %s — loading METR-LA…", KAFKA_SERVERS)

with h5py.File("data/metr-la/METR-LA.h5", "r") as f:
    values = f["df/block0_values"][:]
    timestamps = f["df/axis1"][:]
    sensors = f["df/block0_items"][:]

    logger.info("Shape %s × %s", values.shape[0], values.shape[1])

    row_count = 0
    for t_idx, timestamp in enumerate(timestamps):
        for s_idx, sensor_id in enumerate(sensors):
            speed_value = values[t_idx, s_idx]
            if np.isnan(speed_value):
                continue
            msg = {
                "timestamp": str(timestamp),
                "sensor_id": sensor_id.decode("utf-8")
                if isinstance(sensor_id, bytes)
                else str(sensor_id),
                "sensor_index": int(s_idx),
                "speed": float(speed_value),
            }
            producer.send("topic_sensors", value=msg)
            row_count += 1

        if row_count and row_count % 5000 == 0:
            logger.info("Sent %s messages…", f"{row_count:,}")

        time.sleep(0.01)

producer.flush()
logger.info("Finished — %s messages.", f"{row_count:,}")
