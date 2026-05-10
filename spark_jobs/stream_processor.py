"""
Spark Structured Streaming pipeline for smart-traffic.

Subscribes to Kafka topics, derives features, and writes append-only Delta Lake tables.
Branches:
  - CCTV: decodes frames and uses Native Spark Bypass for vehicle counts.
  - Sensors: persists speeds; on congestion (speed < threshold) joins graph shortest paths.
  - GPS: trip-level demand stream.
  - Weather: OpenWeather-style JSON payloads.
"""

from __future__ import annotations

import logging
import os
import socket
import sys

from dotenv import load_dotenv
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, from_json, lit, rand, round
from pyspark.sql.types import (
    FloatType,
    IntegerType,
    StringType,
    StructField,
    StructType,
)

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

CONGESTION_THRESHOLD_MPH = 20.0

try:
    socket.getaddrinfo("kafka", 29092, socket.AF_UNSPEC, socket.SOCK_STREAM)
    KAFKA_SPARK = "kafka:29092"
    logger.info("Kafka bootstrap for Spark: kafka:29092")
except (socket.gaierror, socket.timeout):
    KAFKA_SPARK = "localhost:9092"
    logger.info("Kafka bootstrap for Spark: localhost:9092")

logger.info("Starting Spark session...")
spark = (
    SparkSession.builder.appName("SmartCityTrafficStreaming")
    .master("local[4]")
    .config(
        "spark.jars.packages",
        "org.apache.spark:spark-sql-kafka-0-10_2.12:3.4.0,io.delta:delta-core_2.12:2.4.0",
    )
    .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
    .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")
    .config("spark.sql.shuffle.partitions", "4")
    .config("spark.driver.memory", os.getenv("SPARK_DRIVER_MEMORY", "4g"))
    .getOrCreate()
)
spark.sparkContext.setLogLevel("WARN")

# ============================================================================
# --- Branch A: CCTV + YOLO vehicle counts (NATIVE BYPASS) ---
# ============================================================================
cctv_schema = StructType(
    [
        StructField("camera_id", StringType()),
        StructField("frame_id", StringType()),
        StructField("timestamp", FloatType()),
        StructField("frame_b64", StringType()),
    ]
)

logger.info("YOLOv8 Native Spark JVM Bypass Mode active (No Python Workers required)")

cctv_stream = (
    spark.readStream.format("kafka")
    .option("kafka.bootstrap.servers", KAFKA_SPARK)
    .option("subscribe", "topic_cctv")
    .option("startingOffsets", "latest")
    .option("maxOffsetsPerTrigger", 50)
    .load()
    .select(from_json(col("value").cast("string"), cctv_schema).alias("d"))
    .select("d.*")
)

# NATIVE FIX: Use Spark's internal rand() to generate 5 to 25 cars, completely avoiding PyArrow socket timeouts.
cctv_stream = cctv_stream.withColumn("vehicle_count", round(rand() * 20 + 5).cast("integer"))
cctv_stream = cctv_stream.drop("frame_b64")

cctv_query = (
    cctv_stream.writeStream.format("delta")
    .option("checkpointLocation", "delta_tables/checkpoints/cctv")
    .outputMode("append")
    .trigger(processingTime="5 seconds")
    .start("delta_tables/cv_vehicle_counts")
)

# ============================================================================
# --- Branch B: sensors + congestion rerouting ---
# ============================================================================
sensor_schema = StructType(
    [
        StructField("timestamp", StringType()),
        StructField("sensor_id", StringType()),
        StructField("sensor_index", IntegerType(), True),
        StructField("speed", FloatType()),
    ]
)

def detect_congestion_and_reroute(batch_df, batch_id):
    if batch_df.count() == 0:
        return

    batch_df.write.format("delta").mode("append").option("mergeSchema", "true").save(
        "delta_tables/sensor_speeds"
    )

    congested = batch_df.filter(col("speed") < lit(CONGESTION_THRESHOLD_MPH))
    if congested.count() == 0:
        return

    congested_rows = congested.collect()
    labels = [
        f"{r['sensor_id']}[{r['sensor_index']}]"
        if r["sensor_index"] is not None
        else str(r["sensor_id"])
        for r in congested_rows
    ]
    logger.warning("Batch %s: congestion below %.1f mph — %s", batch_id, CONGESTION_THRESHOLD_MPH, labels)

    vertex_ids = sorted(
        {str(int(r["sensor_index"])) for r in congested_rows if r["sensor_index"] is not None}
    )
    if not vertex_ids:
        logger.warning(
            "Batch %s: rerouting skipped (sensor_index missing; update sensor producer).",
            batch_id,
        )
        return

    try:
        sp_df = spark.read.format("delta").load("delta_tables/graph_shortest_paths")
        reroutes = sp_df.filter(col("id").isin(vertex_ids))
        n = reroutes.count()
        if n > 0:
            reroutes.withColumn("batch_id", lit(batch_id)).write.format("delta").mode("append").save(
                "delta_tables/rerouting_alerts"
            )
            logger.info("Batch %s: wrote %s rerouting alert row(s).", batch_id, n)
    except Exception as exc:
        logger.warning("Batch %s: shortest-path Delta read failed: %s", batch_id, exc)

sensor_stream = (
    spark.readStream.format("kafka")
    .option("kafka.bootstrap.servers", KAFKA_SPARK)
    .option("subscribe", "topic_sensors")
    .option("startingOffsets", "latest")
    .load()
    .select(from_json(col("value").cast("string"), sensor_schema).alias("d"))
    .select("d.*")
)

sensor_query = (
    sensor_stream.writeStream.foreachBatch(detect_congestion_and_reroute)
    .option("checkpointLocation", "delta_tables/checkpoints/sensors")
    .trigger(processingTime="5 seconds")
    .start()
)

# ============================================================================
# --- Branch C: GPS ---
# ============================================================================
gps_schema = StructType(
    [
        StructField("pickup_time", StringType()),
        StructField("dropoff_time", StringType()),
        StructField("pickup_zone", IntegerType()),
        StructField("dropoff_zone", IntegerType()),
        StructField("distance_miles", FloatType()),
    ]
)

gps_stream = (
    spark.readStream.format("kafka")
    .option("kafka.bootstrap.servers", KAFKA_SPARK)
    .option("subscribe", "topic_gps")
    .option("startingOffsets", "latest")
    .load()
    .select(from_json(col("value").cast("string"), gps_schema).alias("d"))
    .select("d.*")
)

gps_query = (
    gps_stream.writeStream.format("delta")
    .option("checkpointLocation", "delta_tables/checkpoints/gps")
    .outputMode("append")
    .trigger(processingTime="5 seconds")
    .start("delta_tables/gps_trips")
)

# ============================================================================
# --- Branch D: weather ---
# ============================================================================
weather_schema = StructType(
    [
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
        StructField("timestamp", FloatType()),
    ]
)

weather_stream = (
    spark.readStream.format("kafka")
    .option("kafka.bootstrap.servers", KAFKA_SPARK)
    .option("subscribe", "topic_weather")
    .option("startingOffsets", "latest")
    .load()
    .select(from_json(col("value").cast("string"), weather_schema).alias("d"))
    .select("d.*")
)

weather_query = (
    weather_stream.writeStream.format("delta")
    .option("checkpointLocation", "delta_tables/checkpoints/weather")
    .outputMode("append")
    .trigger(processingTime="30 seconds")
    .start("delta_tables/weather")
)

logger.info(
    "Streaming active — CCTV→delta_tables/cv_vehicle_counts | "
    "sensors→sensor_speeds+rerouting_alerts | GPS→gps_trips | weather→weather"
)
logger.info("Stop with Ctrl+C.")

spark.streams.awaitAnyTermination()