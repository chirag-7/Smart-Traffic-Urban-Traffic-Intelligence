"""
Kafka consumer-group lag monitor (Phase 4).

Polls the broker every ``POLL_SECONDS`` for every (group, topic, partition)
triple in ``GROUPS`` and ``TOPICS`` and exposes:

    kafka_consumer_lag{group, topic, partition}  Gauge
    kafka_log_end_offset{topic, partition}       Gauge
    kafka_committed_offset{group, topic, partition}  Gauge
    kafka_lag_monitor_poll_seconds               Histogram

So Grafana can plot "messages waiting to be processed per consumer group".
"""

from __future__ import annotations

import logging
import os
import signal
import socket
import time
from typing import Iterable

from confluent_kafka import Consumer, KafkaException, TopicPartition
from confluent_kafka.admin import AdminClient
from prometheus_client import Gauge, Histogram, start_http_server

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s lag-monitor %(message)s",
)
logger = logging.getLogger("lag-monitor")

BOOTSTRAP = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "kafka:29092")
GROUPS = [g.strip() for g in os.getenv(
    "LAG_MONITOR_GROUPS", "yolo-workers,ml-predictor-workers"
).split(",") if g.strip()]
TOPICS = [t.strip() for t in os.getenv(
    "LAG_MONITOR_TOPICS",
    "topic_cctv,topic_sensors,topic_cctv_inferred,topic_speed_predictions,topic_gps,topic_weather",
).split(",") if t.strip()]
POLL_SECONDS = float(os.getenv("LAG_MONITOR_POLL_SECONDS", "15"))
METRICS_PORT = int(os.getenv("LAG_MONITOR_METRICS_PORT", "9102"))

LAG = Gauge(
    "kafka_consumer_lag",
    "Messages between the consumer-group committed offset and the log-end offset.",
    labelnames=("group", "topic", "partition"),
)
LOG_END = Gauge(
    "kafka_log_end_offset",
    "Latest offset (high watermark) per partition.",
    labelnames=("topic", "partition"),
)
COMMITTED = Gauge(
    "kafka_committed_offset",
    "Last committed offset for each (group, topic, partition).",
    labelnames=("group", "topic", "partition"),
)
POLL_LATENCY = Histogram(
    "kafka_lag_monitor_poll_seconds",
    "Time spent on a single full poll cycle.",
    buckets=(0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10, 30),
)


def topic_partitions(admin: AdminClient, topics: Iterable[str]) -> dict[str, list[int]]:
    md = admin.list_topics(timeout=10)
    result = {}
    for t in topics:
        if t in md.topics and md.topics[t].error is None:
            result[t] = list(md.topics[t].partitions.keys())
        else:
            result[t] = []
    return result


def poll_once(admin: AdminClient, topic_parts: dict[str, list[int]]) -> None:
    for topic, parts in topic_parts.items():
        if not parts:
            continue
        tps = [TopicPartition(topic, p) for p in parts]
        # log-end offsets (high watermarks)
        ends: dict[int, int] = {}
        # Use a transient consumer to query offsets.
        consumer = Consumer({
            "bootstrap.servers": BOOTSTRAP,
            "group.id": f"lag-monitor-probe-{socket.gethostname()}",
            "enable.auto.commit": False,
        })
        try:
            for tp in tps:
                _, high = consumer.get_watermark_offsets(tp, timeout=5, cached=False)
                ends[tp.partition] = high
                LOG_END.labels(topic=topic, partition=str(tp.partition)).set(high)
        finally:
            consumer.close()

        for group in GROUPS:
            group_consumer = Consumer({
                "bootstrap.servers": BOOTSTRAP,
                "group.id": group,
                "enable.auto.commit": False,
            })
            try:
                committed = group_consumer.committed(tps, timeout=5)
            except KafkaException as exc:
                logger.warning("committed() failed for group=%s topic=%s: %s", group, topic, exc)
                group_consumer.close()
                continue
            for tp in committed:
                p = tp.partition
                end = ends.get(p, 0)
                # offset == -1001 means "no committed offset for this partition" yet.
                cur = tp.offset if tp.offset is not None and tp.offset >= 0 else 0
                COMMITTED.labels(group=group, topic=topic, partition=str(p)).set(cur)
                LAG.labels(group=group, topic=topic, partition=str(p)).set(max(0, end - cur))
            group_consumer.close()


_running = True


def _handle_signal(signum, _frame):  # noqa: ANN001
    global _running
    logger.info("Received signal %s — exiting.", signum)
    _running = False


def main() -> None:
    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    logger.info(
        "lag-monitor starting | bootstrap=%s groups=%s topics=%s poll=%ss metrics_port=%s",
        BOOTSTRAP, GROUPS, TOPICS, POLL_SECONDS, METRICS_PORT,
    )
    start_http_server(METRICS_PORT)

    admin = AdminClient({"bootstrap.servers": BOOTSTRAP})
    while _running:
        t0 = time.time()
        try:
            tp_map = topic_partitions(admin, TOPICS)
            poll_once(admin, tp_map)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Poll error: %s", exc)
        elapsed = time.time() - t0
        POLL_LATENCY.observe(elapsed)
        for _ in range(int(POLL_SECONDS)):
            if not _running:
                break
            time.sleep(1)


if __name__ == "__main__":
    main()
