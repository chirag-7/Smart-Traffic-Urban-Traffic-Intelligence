"""
Phase 3: dedicated streaming script for the weather branch.

Reads topic_weather, persists to delta_tables/weather. Small table, no
partitioning needed.
"""

from __future__ import annotations

import logging
import os

from dotenv import load_dotenv
from pyspark.sql.functions import col, from_json
from pyspark.sql.types import (
    DoubleType, FloatType, IntegerType, StringType, StructField, StructType,
)

from _common import build_spark, detect_kafka_bootstrap, silence_chatty_loggers

load_dotenv()
logger = logging.getLogger("stream_weather")

TOPIC = os.getenv("TOPIC_WEATHER", "topic_weather")


def main() -> None:
    kafka_bootstrap = detect_kafka_bootstrap()
    logger.info("Kafka bootstrap: %s", kafka_bootstrap)
    spark = build_spark("StreamWeather", cores=1, driver_memory="1g")
    silence_chatty_loggers(spark)

    schema = StructType([
        StructField("city", StringType()),
        StructField("country", StringType()),
        StructField("observed_at", StringType()),
        StructField("temp_c", FloatType()),
        StructField("feels_like_c", FloatType()),
        StructField("temp_min_c", FloatType()),
        StructField("temp_max_c", FloatType()),
        StructField("humidity", IntegerType()),
        StructField("pressure_hpa", IntegerType()),
        StructField("wind_speed_mps", FloatType()),
        StructField("wind_deg", IntegerType()),
        StructField("clouds_pct", IntegerType()),
        StructField("weather_main", StringType()),
        StructField("weather_description", StringType()),
        StructField("weather_id", IntegerType()),
        StructField("timestamp", DoubleType()),
        StructField("ingested_at", DoubleType(), True),
    ])

    stream = (
        spark.readStream.format("kafka")
        .option("kafka.bootstrap.servers", kafka_bootstrap)
        .option("subscribe", TOPIC)
        .option("startingOffsets", "latest")
        .load()
        .select(from_json(col("value").cast("string"), schema).alias("d"))
        .select("d.*")
    )

    query = (
        stream.writeStream.format("delta")
        .option("checkpointLocation", "delta_tables/checkpoints/weather_v2")
        .outputMode("append")
        .trigger(processingTime="30 seconds")
        .start("delta_tables/weather")
    )

    logger.info("stream_weather active — Ctrl+C to stop.")
    query.awaitTermination()


if __name__ == "__main__":
    main()
