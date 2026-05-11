"""
Phase 3: dedicated streaming script for the CCTV-inferred branch.

Reads topic_cctv_inferred (output of yolo_worker), validates required fields,
persists to delta_tables/cv_vehicle_counts partitioned by event_date.
Idempotent via Spark Structured Streaming checkpoint.
"""

from __future__ import annotations

import logging
import os

from dotenv import load_dotenv
from pyspark.sql.functions import col, from_json, to_date
from pyspark.sql.types import (
    ArrayType, DoubleType, FloatType, IntegerType,
    StringType, StructField, StructType,
)

from _common import build_spark, detect_kafka_bootstrap, silence_chatty_loggers

load_dotenv()
logger = logging.getLogger("stream_cctv")

TOPIC = os.getenv("TOPIC_CCTV_INFERRED", "topic_cctv_inferred")


def main() -> None:
    kafka_bootstrap = detect_kafka_bootstrap()
    logger.info("Kafka bootstrap: %s", kafka_bootstrap)
    spark = build_spark("StreamCCTV", cores=2)
    silence_chatty_loggers(spark)

    schema = StructType([
        StructField("camera_id", StringType()),
        StructField("frame_id", StringType()),
        StructField("timestamp", DoubleType()),
        StructField("unique_count_5min", IntegerType()),
        StructField("unique_count_session", IntegerType()),
        StructField("track_ids", ArrayType(IntegerType())),
        StructField("frame_url", StringType()),
        StructField("inference_ms", FloatType()),
        StructField("model_version", StringType()),
        StructField("worker_id", StringType()),
        StructField("processed_at", DoubleType()),
    ])

    stream = (
        spark.readStream.format("kafka")
        .option("kafka.bootstrap.servers", kafka_bootstrap)
        .option("subscribe", TOPIC)
        .option("startingOffsets", "latest")
        .option("maxOffsetsPerTrigger", 500)
        .load()
        .select(from_json(col("value").cast("string"), schema).alias("d"))
        .select("d.*")
        .filter(col("camera_id").isNotNull() & col("frame_id").isNotNull())
        .withColumn("event_date", to_date(col("timestamp").cast("timestamp")))
    )

    query = (
        stream.writeStream.format("delta")
        .option("checkpointLocation", "delta_tables/checkpoints/cctv_inferred_v2")
        .partitionBy("event_date")
        .outputMode("append")
        .trigger(processingTime="5 seconds")
        .start("delta_tables/cv_vehicle_counts")
    )

    logger.info("stream_cctv active — Ctrl+C to stop.")
    query.awaitTermination()


if __name__ == "__main__":
    main()
