"""
One-shot Kafka topic initialiser.

Runs as a short-lived Docker container at compose-up time. Creates every topic
the smart-traffic pipeline uses with explicit partition count and retention.
Idempotent: topics that already exist are skipped (no error).

Environment variables (all optional, sensible defaults supplied):
  KAFKA_BOOTSTRAP_SERVERS   default kafka:29092 (Docker internal listener)
  KAFKA_NUM_PARTITIONS      default 3
  KAFKA_REPLICATION_FACTOR  default 1 (single-broker local dev)
  KAFKA_RETENTION_MS        default 86400000 (24 h)
  TOPIC_SENSORS / TOPIC_GPS / TOPIC_CCTV / TOPIC_CCTV_INFERRED /
  TOPIC_WEATHER / TOPIC_SPEED_PREDICTIONS   defaults match .env.example

Exit code 0 on success, non-zero on broker unreachable after retries.
"""

from __future__ import annotations

import logging
import os
import sys
import time

from confluent_kafka.admin import AdminClient, NewTopic
from confluent_kafka.cimpl import KafkaError, KafkaException

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("init_topics")

BOOTSTRAP = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "kafka:29092")
PARTITIONS = int(os.getenv("KAFKA_NUM_PARTITIONS", "3"))
REPLICATION = int(os.getenv("KAFKA_REPLICATION_FACTOR", "1"))
RETENTION_MS = os.getenv("KAFKA_RETENTION_MS", "86400000")

TOPICS = [
    os.getenv("TOPIC_SENSORS", "topic_sensors"),
    os.getenv("TOPIC_GPS", "topic_gps"),
    os.getenv("TOPIC_CCTV", "topic_cctv"),
    os.getenv("TOPIC_CCTV_INFERRED", "topic_cctv_inferred"),
    os.getenv("TOPIC_WEATHER", "topic_weather"),
    os.getenv("TOPIC_SPEED_PREDICTIONS", "topic_speed_predictions"),
]


def wait_for_broker(admin: AdminClient, attempts: int = 30, sleep_s: float = 2.0) -> None:
    """Poll broker metadata until reachable; raise if all attempts fail."""
    for i in range(1, attempts + 1):
        try:
            md = admin.list_topics(timeout=5)
            if md.brokers:
                logger.info("Broker reachable on attempt %s (%s brokers).", i, len(md.brokers))
                return
        except KafkaException as exc:
            logger.info("Attempt %s/%s — broker not ready (%s)", i, attempts, exc)
        time.sleep(sleep_s)
    raise SystemExit(f"Kafka broker {BOOTSTRAP} unreachable after {attempts} attempts.")


def main() -> None:
    logger.info(
        "init_topics bootstrap=%s partitions=%s replication=%s retention_ms=%s",
        BOOTSTRAP, PARTITIONS, REPLICATION, RETENTION_MS,
    )

    admin = AdminClient({"bootstrap.servers": BOOTSTRAP})
    wait_for_broker(admin)

    existing = set(admin.list_topics(timeout=10).topics.keys())
    to_create = [t for t in TOPICS if t not in existing]
    already = [t for t in TOPICS if t in existing]

    for t in already:
        logger.info("Skip — exists: %s", t)

    if not to_create:
        logger.info("All %s topics already present. Nothing to do.", len(TOPICS))
        return

    new_topics = [
        NewTopic(
            topic=t,
            num_partitions=PARTITIONS,
            replication_factor=REPLICATION,
            config={
                "retention.ms": RETENTION_MS,
                "cleanup.policy": "delete",
                "compression.type": "producer",
            },
        )
        for t in to_create
    ]

    futures = admin.create_topics(new_topics, request_timeout=30)
    failed = []
    for topic, fut in futures.items():
        try:
            fut.result()
            logger.info("Created topic: %s (partitions=%s)", topic, PARTITIONS)
        except KafkaException as exc:
            err = exc.args[0] if exc.args else None
            if err is not None and err.code() == KafkaError.TOPIC_ALREADY_EXISTS:
                logger.info("Race — topic appeared: %s", topic)
            else:
                logger.error("Failed to create %s: %s", topic, exc)
                failed.append(topic)

    if failed:
        sys.exit(f"Topic creation failed for: {failed}")

    logger.info("init_topics complete.")


if __name__ == "__main__":
    main()
