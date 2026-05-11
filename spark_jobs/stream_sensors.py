"""
Phase 3: dedicated streaming script for the sensors branch.

Responsibilities:
  - Consume topic_sensors.
  - Validate each row (non-null required fields, speed in [0, 150]).
  - Split valid / invalid; route invalid rows to delta_tables/dlq_sensors.
  - Persist valid rows to delta_tables/sensor_speeds (partitioned by event_date).
  - On congestion (speed < threshold), join broadcast graph_shortest_paths and
    append to delta_tables/rerouting_alerts.
  - Stamp every row with `ingested_at` / `processed_at` and write per-batch
    summaries to delta_tables/pipeline_latency.

Idempotent: `txnAppId` is set on every Delta sink so a Spark restart after a
checkpoint gap does not duplicate rows.
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path

from dotenv import load_dotenv
from pyspark.sql.functions import (
    col, current_timestamp, from_json, lit, to_date,
)
from pyspark.sql.types import (
    DoubleType, FloatType, IntegerType, MapType,
    StringType, StructField, StructType,
)

from _common import build_spark, detect_kafka_bootstrap, silence_chatty_loggers

load_dotenv()
logger = logging.getLogger("stream_sensors")

CONGESTION_THRESHOLD_MPH = float(os.getenv("CONGESTION_THRESHOLD_MPH", "20.0"))
TOPIC_SENSORS = os.getenv("TOPIC_SENSORS", "topic_sensors")
GRAPH_SP_PATH = "delta_tables/graph_shortest_paths"

REPO_ROOT = Path(__file__).resolve().parent.parent
DLQ_PATH = "delta_tables/dlq_sensors"
SPEEDS_PATH = "delta_tables/sensor_speeds"
ALERTS_PATH = "delta_tables/rerouting_alerts"
LATENCY_PATH = "delta_tables/pipeline_latency"


def main() -> None:
    kafka_bootstrap = detect_kafka_bootstrap()
    logger.info("Kafka bootstrap: %s", kafka_bootstrap)
    spark = build_spark("StreamSensors", cores=2)
    silence_chatty_loggers(spark)

    # ---- Broadcast graph_shortest_paths once at startup ----
    sp_bc = None
    try:
        sp_rows = spark.read.format("delta").load(GRAPH_SP_PATH).collect()
        sp_dict = {r["id"]: dict(r["distances"] or {}) for r in sp_rows}
        sp_bc = spark.sparkContext.broadcast(sp_dict)
        logger.info("Broadcasted graph_shortest_paths: %s vertices.", len(sp_dict))
    except Exception as exc:  # noqa: BLE001
        logger.warning("graph_shortest_paths missing (%s) — rerouting disabled.", exc)

    sensor_schema = StructType([
        StructField("timestamp", StringType()),
        StructField("sensor_id", StringType()),
        StructField("sensor_index", IntegerType(), True),
        StructField("speed", FloatType()),
        StructField("ingested_at", DoubleType(), True),
    ])

    def handle_batch(batch_df, batch_id):
        if batch_df.rdd.isEmpty():
            return

        batch_t0 = time.time()
        batch_df = batch_df.cache()

        # ---- Split valid / invalid ----
        valid = batch_df.filter(
            col("sensor_id").isNotNull()
            & col("speed").isNotNull()
            & col("speed").between(0.0, 150.0)
        )
        invalid = batch_df.subtract(valid)

        invalid_count = invalid.count()
        if invalid_count > 0:
            (
                invalid.withColumn("dlq_reason", lit("failed_validation"))
                       .withColumn("dlq_at", current_timestamp())
                       .withColumn("batch_id", lit(int(batch_id)))
                       .write.format("delta").mode("append")
                       .option("mergeSchema", "true")
                       .option("txnAppId", "stream-sensors-dlq")
                       .option("txnVersion", str(batch_id))
                       .save(DLQ_PATH)
            )
            logger.warning("Batch %s: %s invalid sensor row(s) → DLQ.", batch_id, invalid_count)

        # ---- Persist valid readings ----
        valid_out = valid.withColumn("event_date", to_date(col("timestamp").cast("timestamp")))
        valid_count = valid_out.count()
        (
            valid_out.write.format("delta").mode("append")
                .option("mergeSchema", "true")
                .option("txnAppId", "stream-sensors-speeds")
                .option("txnVersion", str(batch_id))
                .partitionBy("event_date")
                .save(SPEEDS_PATH)
        )

        # ---- Congestion rerouting (broadcast lookup, no per-batch Delta read) ----
        alerts = []
        if sp_bc is not None:
            congested = valid_out.filter(col("speed") < lit(CONGESTION_THRESHOLD_MPH))
            for r in congested.collect():
                if r["sensor_index"] is None:
                    continue
                vid = str(int(r["sensor_index"]))
                distances = sp_bc.value.get(vid)
                if not distances:
                    continue
                alerts.append({
                    "id": vid,
                    "sensor_id": r["sensor_id"],
                    "speed": float(r["speed"]),
                    "distances": distances,
                    "batch_id": int(batch_id),
                })

        if alerts:
            alerts_schema = StructType([
                StructField("id", StringType()),
                StructField("sensor_id", StringType()),
                StructField("speed", FloatType()),
                StructField("distances", MapType(StringType(), IntegerType())),
                StructField("batch_id", IntegerType()),
            ])
            (
                spark.createDataFrame(alerts, alerts_schema)
                     .write.format("delta").mode("append")
                     .option("txnAppId", "stream-sensors-alerts")
                     .option("txnVersion", str(batch_id))
                     .save(ALERTS_PATH)
            )
            logger.info("Batch %s: wrote %s rerouting alert(s).", batch_id, len(alerts))

        # ---- Per-batch latency summary ----
        batch_seconds = time.time() - batch_t0
        latency_row = [{
            "topic": TOPIC_SENSORS,
            "batch_id": int(batch_id),
            "valid_count": int(valid_count),
            "invalid_count": int(invalid_count),
            "alerts_count": int(len(alerts)),
            "batch_seconds": float(batch_seconds),
            "batch_finished_at": float(time.time()),
        }]
        latency_schema = StructType([
            StructField("topic", StringType()),
            StructField("batch_id", IntegerType()),
            StructField("valid_count", IntegerType()),
            StructField("invalid_count", IntegerType()),
            StructField("alerts_count", IntegerType()),
            StructField("batch_seconds", FloatType()),
            StructField("batch_finished_at", DoubleType()),
        ])
        (
            spark.createDataFrame(latency_row, latency_schema)
                 .write.format("delta").mode("append")
                 .option("mergeSchema", "true")
                 .option("txnAppId", "stream-sensors-latency")
                 .option("txnVersion", str(batch_id))
                 .save(LATENCY_PATH)
        )

        batch_df.unpersist()

    stream = (
        spark.readStream.format("kafka")
        .option("kafka.bootstrap.servers", kafka_bootstrap)
        .option("subscribe", TOPIC_SENSORS)
        .option("startingOffsets", "latest")
        .option("maxOffsetsPerTrigger", 5000)
        .load()
        .select(from_json(col("value").cast("string"), sensor_schema).alias("d"))
        .select("d.*")
    )

    query = (
        stream.writeStream.foreachBatch(handle_batch)
        .option("checkpointLocation", "delta_tables/checkpoints/sensors_v2")
        .trigger(processingTime="5 seconds")
        .start()
    )

    logger.info("stream_sensors active — Ctrl+C to stop.")
    query.awaitTermination()


if __name__ == "__main__":
    main()
