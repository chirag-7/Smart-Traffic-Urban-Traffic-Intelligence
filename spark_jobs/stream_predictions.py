"""
Streaming ingestion for the speed-predictions branch.

Reads ``topic_speed_predictions`` (output of the ML predictor Kafka worker),
persists to ``delta_tables/speed_predictions`` partitioned by ``event_date``.
"""

from __future__ import annotations

import logging
import os

from dotenv import load_dotenv
from pyspark.sql.functions import col, from_json, to_date
from pyspark.sql.types import (
    DoubleType, FloatType, IntegerType, StringType, StructField, StructType,
)

from _common import build_spark, detect_kafka_bootstrap, silence_chatty_loggers

load_dotenv()
logger = logging.getLogger("stream_predictions")

TOPIC = os.getenv("TOPIC_SPEED_PREDICTIONS", "topic_speed_predictions")


def main() -> None:
    kafka_bootstrap = detect_kafka_bootstrap()
    logger.info("Kafka bootstrap: %s", kafka_bootstrap)
    spark = build_spark("StreamPredictions", cores=2)
    silence_chatty_loggers(spark)

    schema = StructType([
        StructField("sensor_id", StringType()),
        StructField("timestamp", StringType()),
        StructField("actual_speed", FloatType(), True),
        StructField("predicted_speed", FloatType()),
        StructField("horizon_minutes", IntegerType(), True),
        StructField("model_version", StringType()),
        StructField("inference_us", FloatType(), True),
        StructField("processed_at", DoubleType(), True),
    ])

    stream = (
        spark.readStream.format("kafka")
        .option("kafka.bootstrap.servers", kafka_bootstrap)
        .option("subscribe", TOPIC)
        .option("startingOffsets", "latest")
        .option("maxOffsetsPerTrigger", 5000)
        .load()
        .select(from_json(col("value").cast("string"), schema).alias("d"))
        .select("d.*")
        .filter(col("sensor_id").isNotNull() & col("predicted_speed").isNotNull())
        .withColumn("event_date", to_date(col("timestamp").cast("timestamp")))
    )

    query = (
        stream.writeStream.format("delta")
        .option("checkpointLocation", "delta_tables/checkpoints/speed_predictions_v2")
        .partitionBy("event_date")
        .outputMode("append")
        .trigger(processingTime="5 seconds")
        .start("delta_tables/speed_predictions")
    )

    logger.info("stream_predictions active — Ctrl+C to stop.")
    query.awaitTermination()


if __name__ == "__main__":
    main()
