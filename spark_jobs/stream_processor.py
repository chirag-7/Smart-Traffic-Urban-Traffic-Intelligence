"""
Spark Structured Streaming pipeline for smart-traffic (Phase 1).

Subscribes to Kafka topics, validates schemas, writes append-only Delta tables.

Branches:
  - CCTV (NEW)  : reads ``topic_cctv_inferred`` (already-inferenced JSON from
                  the yolo-worker). No HTTP calls, no UDFs, no rand(). Pure
                  Kafka-to-Delta sink, partitioned by event_date.
  - Sensors     : persists speeds; on congestion (speed < threshold) joins
                  precomputed ``graph_shortest_paths`` (broadcast once at
                  startup — not re-read per batch).
  - GPS         : trip-level demand stream, partitioned by event_date.
  - Weather     : OpenWeather-style JSON payloads.

Phase 3 will split this into four independent scripts and add DLQ + idempotent
writes. Phase 1 keeps the broadcast-shortest-paths fix and the partitioning
fix in place, plus removes the `rand()` placeholder entirely.
"""

from __future__ import annotations

import logging
import os
import socket
import sys

from dotenv import load_dotenv
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, from_json, lit, to_date
from pyspark.sql.types import (
    ArrayType,
    DoubleType,
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

CONGESTION_THRESHOLD_MPH = float(os.getenv("CONGESTION_THRESHOLD_MPH", "20.0"))
TOPIC_SENSORS = os.getenv("TOPIC_SENSORS", "topic_sensors")
TOPIC_GPS = os.getenv("TOPIC_GPS", "topic_gps")
TOPIC_CCTV_INFERRED = os.getenv("TOPIC_CCTV_INFERRED", "topic_cctv_inferred")
TOPIC_WEATHER = os.getenv("TOPIC_WEATHER", "topic_weather")
TOPIC_SPEED_PREDICTIONS = os.getenv("TOPIC_SPEED_PREDICTIONS", "topic_speed_predictions")

# Autodetect Kafka bootstrap depending on whether we resolve the internal
# Docker DNS name (when running on host, we won't — fall back to localhost).
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
# Broadcast graph_shortest_paths ONCE at startup.
#
# Previously this Delta table was read from disk inside foreachBatch on every
# micro-batch (5s) — a major performance bug. We now materialise a Python
# dict[str, dict[str, int]] once and broadcast it to executors.
# ============================================================================
GRAPH_SP_PATH = "delta_tables/graph_shortest_paths"
SHORTEST_PATHS_BC = None
try:
    sp_rows = spark.read.format("delta").load(GRAPH_SP_PATH).collect()
    sp_dict = {r["id"]: dict(r["distances"] or {}) for r in sp_rows}
    SHORTEST_PATHS_BC = spark.sparkContext.broadcast(sp_dict)
    logger.info("Broadcasted graph_shortest_paths: %s vertices.", len(sp_dict))
except Exception as exc:  # noqa: BLE001
    logger.warning(
        "graph_shortest_paths not available (%s). Run "
        "`python spark_jobs/graph_analytics.py` first. Rerouting will be skipped.",
        exc,
    )

# ============================================================================
# --- Branch A: CCTV inferred (NEW — Kafka-to-Delta sink, no HTTP, no UDF) ---
# ============================================================================
cctv_inferred_schema = StructType(
    [
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
    ]
)

cctv_stream = (
    spark.readStream.format("kafka")
    .option("kafka.bootstrap.servers", KAFKA_SPARK)
    .option("subscribe", TOPIC_CCTV_INFERRED)
    .option("startingOffsets", "latest")
    .option("maxOffsetsPerTrigger", 500)
    .load()
    .select(from_json(col("value").cast("string"), cctv_inferred_schema).alias("d"))
    .select("d.*")
    .withColumn("event_date", to_date(col("timestamp").cast("timestamp")))
)

cctv_query = (
    cctv_stream.writeStream.format("delta")
    .option("checkpointLocation", "delta_tables/checkpoints/cctv_inferred")
    .partitionBy("event_date")
    .outputMode("append")
    .trigger(processingTime="5 seconds")
    .start("delta_tables/cv_vehicle_counts")
)

# ============================================================================
# --- Branch B: sensors + congestion rerouting (broadcast lookup) ---
# ============================================================================
sensor_schema = StructType(
    [
        StructField("timestamp", StringType()),
        StructField("sensor_id", StringType()),
        StructField("sensor_index", IntegerType(), True),
        StructField("speed", FloatType()),
        StructField("ingested_at", DoubleType(), True),
    ]
)


def detect_congestion_and_reroute(batch_df, batch_id):
    if batch_df.rdd.isEmpty():
        return

    # Annotate event_date and persist all sensor readings.
    sensor_out = batch_df.withColumn(
        "event_date", to_date(col("timestamp").cast("timestamp"))
    )
    (
        sensor_out.write.format("delta")
        .mode("append")
        .option("mergeSchema", "true")
        .option("txnAppId", "stream-processor-sensors")
        .option("txnVersion", str(batch_id))
        .partitionBy("event_date")
        .save("delta_tables/sensor_speeds")
    )

    if SHORTEST_PATHS_BC is None:
        return

    congested = batch_df.filter(col("speed") < lit(CONGESTION_THRESHOLD_MPH))
    if congested.rdd.isEmpty():
        return

    rows = congested.collect()
    sp_dict = SHORTEST_PATHS_BC.value
    alerts = []
    for r in rows:
        if r["sensor_index"] is None:
            continue
        vid = str(int(r["sensor_index"]))
        distances = sp_dict.get(vid)
        if not distances:
            continue
        alerts.append(
            {
                "id": vid,
                "sensor_id": r["sensor_id"],
                "speed": float(r["speed"]),
                "distances": distances,
                "batch_id": int(batch_id),
            }
        )

    if not alerts:
        logger.warning(
            "Batch %s: congestion detected but no broadcast matches "
            "(sensor_index missing or vertex not in graph).",
            batch_id,
        )
        return

    from pyspark.sql.types import MapType

    alerts_schema = StructType(
        [
            StructField("id", StringType()),
            StructField("sensor_id", StringType()),
            StructField("speed", FloatType()),
            StructField("distances", MapType(StringType(), IntegerType())),
            StructField("batch_id", IntegerType()),
        ]
    )
    alerts_df = spark.createDataFrame(alerts, alerts_schema)
    (
        alerts_df.write.format("delta")
        .mode("append")
        .option("txnAppId", "stream-processor-alerts")
        .option("txnVersion", str(batch_id))
        .save("delta_tables/rerouting_alerts")
    )
    logger.info("Batch %s: wrote %s rerouting alert(s).", batch_id, len(alerts))


sensor_stream = (
    spark.readStream.format("kafka")
    .option("kafka.bootstrap.servers", KAFKA_SPARK)
    .option("subscribe", TOPIC_SENSORS)
    .option("startingOffsets", "latest")
    .option("maxOffsetsPerTrigger", 5000)
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
        StructField("ingested_at", DoubleType(), True),
    ]
)

gps_stream = (
    spark.readStream.format("kafka")
    .option("kafka.bootstrap.servers", KAFKA_SPARK)
    .option("subscribe", TOPIC_GPS)
    .option("startingOffsets", "latest")
    .option("maxOffsetsPerTrigger", 5000)
    .load()
    .select(from_json(col("value").cast("string"), gps_schema).alias("d"))
    .select("d.*")
    .withColumn("event_date", to_date(col("pickup_time").cast("timestamp")))
)

gps_query = (
    gps_stream.writeStream.format("delta")
    .option("checkpointLocation", "delta_tables/checkpoints/gps")
    .partitionBy("event_date")
    .outputMode("append")
    .trigger(processingTime="5 seconds")
    .start("delta_tables/gps_trips")
)

# ============================================================================
# --- Branch D: weather (unchanged from the original; small table, no partitioning needed) ---
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
        StructField("timestamp", DoubleType()),
        StructField("ingested_at", DoubleType(), True),
    ]
)

weather_stream = (
    spark.readStream.format("kafka")
    .option("kafka.bootstrap.servers", KAFKA_SPARK)
    .option("subscribe", TOPIC_WEATHER)
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

# ============================================================================
# --- Branch E: speed predictions (NEW — Phase 2 ml_predictor_worker output) ---
# ============================================================================
speed_pred_schema = StructType(
    [
        StructField("sensor_id", StringType()),
        StructField("timestamp", StringType()),
        StructField("actual_speed", FloatType(), True),
        StructField("predicted_speed", FloatType()),
        StructField("horizon_minutes", IntegerType(), True),
        StructField("model_version", StringType()),
        StructField("inference_us", FloatType(), True),
        StructField("processed_at", DoubleType(), True),
    ]
)

speed_pred_stream = (
    spark.readStream.format("kafka")
    .option("kafka.bootstrap.servers", KAFKA_SPARK)
    .option("subscribe", TOPIC_SPEED_PREDICTIONS)
    .option("startingOffsets", "latest")
    .option("maxOffsetsPerTrigger", 5000)
    .load()
    .select(from_json(col("value").cast("string"), speed_pred_schema).alias("d"))
    .select("d.*")
    .withColumn("event_date", to_date(col("timestamp").cast("timestamp")))
)

speed_pred_query = (
    speed_pred_stream.writeStream.format("delta")
    .option("checkpointLocation", "delta_tables/checkpoints/speed_predictions")
    .partitionBy("event_date")
    .outputMode("append")
    .trigger(processingTime="5 seconds")
    .start("delta_tables/speed_predictions")
)

logger.info(
    "Streaming active — CCTV→cv_vehicle_counts (from %s) | sensors→sensor_speeds + rerouting_alerts | GPS→gps_trips | weather→weather | predictions→speed_predictions (from %s)",
    TOPIC_CCTV_INFERRED,
    TOPIC_SPEED_PREDICTIONS,
)
logger.info("Stop with Ctrl+C.")

spark.streams.awaitAnyTermination()
