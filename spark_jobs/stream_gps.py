"""
Streaming ingestion for the GPS branch.

Reads ``topic_gps``, persists to ``delta_tables/gps_trips`` partitioned by
``event_date``.
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
logger = logging.getLogger("stream_gps")

TOPIC = os.getenv("TOPIC_GPS", "topic_gps")


def main() -> None:
    kafka_bootstrap = detect_kafka_bootstrap()
    logger.info("Kafka bootstrap: %s", kafka_bootstrap)
    spark = build_spark("StreamGPS", cores=2)
    silence_chatty_loggers(spark)

    schema = StructType([
        StructField("pickup_time", StringType()),
        StructField("dropoff_time", StringType()),
        StructField("pickup_zone", IntegerType()),
        StructField("dropoff_zone", IntegerType()),
        StructField("distance_miles", FloatType()),
        StructField("ingested_at", DoubleType(), True),
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
        .filter(col("pickup_time").isNotNull() & col("dropoff_time").isNotNull())
        .withColumn("event_date", to_date(col("pickup_time").cast("timestamp")))
    )

    query = (
        stream.writeStream.format("delta")
        .option("checkpointLocation", "delta_tables/checkpoints/gps_v2")
        .partitionBy("event_date")
        .outputMode("append")
        .trigger(processingTime="5 seconds")
        .start("delta_tables/gps_trips")
    )

    logger.info("stream_gps active — Ctrl+C to stop.")
    query.awaitTermination()


if __name__ == "__main__":
    main()
