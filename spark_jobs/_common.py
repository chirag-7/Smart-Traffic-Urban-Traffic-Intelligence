"""
Shared helpers for the Phase 3 split Spark streaming scripts.

Each stream_*.py builds its own SparkSession so it can be restarted
independently. They all need the same package coordinates, Delta extensions,
local-mode parallelism, and Kafka-bootstrap autodetection — centralised here.
"""

from __future__ import annotations

import logging
import os
import socket
import sys

from pyspark.sql import SparkSession

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)


def detect_kafka_bootstrap() -> str:
    """Use kafka:29092 if we resolve the Docker network DNS, else localhost:9092."""
    try:
        socket.getaddrinfo("kafka", 29092, socket.AF_UNSPEC, socket.SOCK_STREAM)
        return "kafka:29092"
    except (socket.gaierror, socket.timeout):
        return "localhost:9092"


def build_spark(app_name: str, *, cores: int = 2, driver_memory: str | None = None) -> SparkSession:
    """Build a Spark session preconfigured for Kafka + Delta on a single host.

    Default cores=2 (was 4 in the monolith) because we expect up to 5 of these
    drivers to run side-by-side. Override per-script if needed.
    """
    mem = driver_memory or os.getenv("SPARK_DRIVER_MEMORY", "2g")
    return (
        SparkSession.builder.appName(app_name)
        .master(f"local[{cores}]")
        .config(
            "spark.jars.packages",
            "org.apache.spark:spark-sql-kafka-0-10_2.12:3.4.0,io.delta:delta-core_2.12:2.4.0",
        )
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
        .config(
            "spark.sql.catalog.spark_catalog",
            "org.apache.spark.sql.delta.catalog.DeltaCatalog",
        )
        .config("spark.sql.shuffle.partitions", "4")
        .config("spark.driver.memory", mem)
        .getOrCreate()
    )


def silence_chatty_loggers(spark: SparkSession) -> None:
    """Quiet the well-known noisy loggers without hiding actual errors."""
    spark.sparkContext.setLogLevel("WARN")
    log4j = spark.sparkContext._jvm.org.apache.log4j  # type: ignore[attr-defined]
    log4j.LogManager.getLogger("org.apache.spark.sql.kafka010.KafkaDataConsumer").setLevel(
        log4j.Level.ERROR
    )
