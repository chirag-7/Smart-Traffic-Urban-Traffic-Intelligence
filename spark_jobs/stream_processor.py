import os
import sys
import base64
import io
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, from_json, lit
from pyspark.sql.types import (
    FloatType,
    IntegerType,
    StringType,
    StructField,
    StructType,
)

load_dotenv()

# Ensure PySpark workers use this venv on Windows
os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

CONGESTION_THRESHOLD_MPH = 20.0

# Detect Kafka address for Spark
# When running locally: localhost:9092
# When running in Docker: kafka:29092 (internal network)
import socket

try:
    socket.getaddrinfo("kafka", 29092, socket.AF_UNSPEC, socket.SOCK_STREAM)
    KAFKA_SPARK = "kafka:29092"
    print("[Config] Using internal Docker Kafka address: kafka:29092")
except (socket.gaierror, socket.timeout):
    KAFKA_SPARK = "localhost:9092"
    print("[Config] Using local Kafka address: localhost:9092")

# Initialize Spark Session
print("[Stream Processor] Initializing Spark...")
spark = (
    SparkSession.builder
    .appName("SmartCityTrafficStreaming")
    .master("local[4]")
    .config("spark.jars.packages", "org.apache.spark:spark-sql-kafka-0-10_2.12:3.4.0,io.delta:delta-core_2.12:2.4.0")
    .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
    .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")
    .config("spark.sql.shuffle.partitions", "4")
    .config("spark.driver.memory", os.getenv("SPARK_DRIVER_MEMORY", "4g"))
    .getOrCreate()
)
spark.sparkContext.setLogLevel("WARN")

# ============================================================================
# BRANCH A: COMPUTER VISION — YOLOv8 Vehicle Detection
# ============================================================================

cctv_schema = StructType([
    StructField("camera_id", StringType()),
    StructField("frame_id", StringType()),
    StructField("timestamp", FloatType()),
    StructField("frame_b64", StringType()),
])

try:
    from pyspark.sql.functions import pandas_udf
    
    @pandas_udf("integer")
    def yolo_vehicle_count(frame_b64_series: pd.Series) -> pd.Series:
        """
        Pandas UDF: Runs YOLOv8 on base64-encoded CCTV frames.
        Model loads once per executor process (module-level cache).
        """
        from ultralytics import YOLO
        from PIL import Image

        import builtins

        if not hasattr(builtins, "_yolo_model"):
            yolo_path = Path("models/yolo_traffic_model/weights/best.pt")
            if yolo_path.exists():
                builtins._yolo_model = YOLO(str(yolo_path))
                print(f"[YOLOv8 UDF] Loaded model from {yolo_path}")
            else:
                print(f"[YOLOv8 UDF] Model not found at {yolo_path}, returning 0 counts")
                builtins._yolo_model = None

        model = builtins._yolo_model
        counts = []

        for frame_b64 in frame_b64_series:
            try:
                if model is None:
                    counts.append(0)
                    continue

                img_bytes = base64.b64decode(frame_b64)
                img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                results = model(img, verbose=False, conf=0.3)
                count = len(results[0].boxes) if results and len(results) > 0 else 0
                counts.append(count)
            except Exception as e:
                print(f"[YOLOv8 UDF] Error processing frame: {e}")
                counts.append(0)

        return pd.Series(counts)

    print("[Stream Processor] YOLOv8 UDF registered")
    has_yolo_udf = True
except Exception as e:
    print(f"[Stream Processor] Warning: Could not register YOLOv8 UDF: {e}")
    has_yolo_udf = False

print("[Stream Processor] Reading CCTV stream from Kafka...")
cctv_stream = (
    spark.readStream
    .format("kafka")
    .option("kafka.bootstrap.servers", KAFKA_SPARK)
    .option("subscribe", "topic_cctv")
    .option("startingOffsets", "latest")
    .option("maxOffsetsPerTrigger", 50)
    .load()
    .select(from_json(col("value").cast("string"), cctv_schema).alias("d"))
    .select("d.*")
)

if has_yolo_udf:
    cctv_stream = cctv_stream.withColumn("vehicle_count", yolo_vehicle_count(col("frame_b64")))

cctv_stream = cctv_stream.drop("frame_b64")

cctv_query = (
    cctv_stream.writeStream
    .format("delta")
    .option("checkpointLocation", "delta_tables/checkpoints/cctv")
    .outputMode("append")
    .trigger(processingTime="5 seconds")
    .start("delta_tables/cv_vehicle_counts")
)

# ============================================================================
# BRANCH B: SENSOR SPEEDS — Congestion Detection + Rerouting Trigger
# ============================================================================

sensor_schema = StructType([
    StructField("timestamp", StringType()),
    StructField("sensor_id", StringType()),
    StructField("speed", FloatType()),
])

def detect_congestion_and_reroute(batch_df, batch_id):
    """
    Micro-batch processing function for sensor data.
    - Writes raw speeds to Delta
    - Detects congestion (speed < THRESHOLD)
    - Reads pre-computed shortest paths and writes rerouting alerts
    """
    if batch_df.count() == 0:
        print(f"[Batch {batch_id}] Empty batch, skipping")
        return

    # Write raw sensor speeds
    batch_df.write.format("delta").mode("append").save("delta_tables/sensor_speeds")

    # Detect congested sensors
    congested = batch_df.filter(col("speed") < lit(CONGESTION_THRESHOLD_MPH))
    if congested.count() > 0:
        congested_list = [r["sensor_id"] for r in congested.collect()]
        print(f"[Batch {batch_id}] CONGESTION ALERT — Sensors: {congested_list}")

        # Load shortest paths and write rerouting suggestions
        try:
            sp_df = spark.read.format("delta").load("delta_tables/graph_shortest_paths")
            reroutes = sp_df.filter(col("id").isin(congested_list))

            if reroutes.count() > 0:
                reroutes.withColumn("batch_id", lit(batch_id)).write.format("delta").mode(
                    "append"
                ).save("delta_tables/rerouting_alerts")
                print(f"[Batch {batch_id}] Rerouting alerts written for {reroutes.count()} sensors")
        except Exception as e:
            print(f"[Batch {batch_id}] Shortest paths not available yet: {e}")

print("[Stream Processor] Reading sensor stream from Kafka...")
sensor_stream = (
    spark.readStream
    .format("kafka")
    .option("kafka.bootstrap.servers", KAFKA_SPARK)
    .option("subscribe", "topic_sensors")
    .option("startingOffsets", "latest")
    .load()
    .select(from_json(col("value").cast("string"), sensor_schema).alias("d"))
    .select("d.*")
)

sensor_query = (
    sensor_stream.writeStream
    .foreachBatch(detect_congestion_and_reroute)
    .option("checkpointLocation", "delta_tables/checkpoints/sensors")
    .trigger(processingTime="5 seconds")
    .start()
)

# ============================================================================
# BRANCH C: GPS DEMAND STREAM
# ============================================================================

gps_schema = StructType([
    StructField("pickup_time", StringType()),
    StructField("dropoff_time", StringType()),
    StructField("pickup_zone", IntegerType()),
    StructField("dropoff_zone", IntegerType()),
    StructField("distance_miles", FloatType()),
])

print("[Stream Processor] Reading GPS stream from Kafka...")
gps_stream = (
    spark.readStream
    .format("kafka")
    .option("kafka.bootstrap.servers", KAFKA_SPARK)
    .option("subscribe", "topic_gps")
    .option("startingOffsets", "latest")
    .load()
    .select(from_json(col("value").cast("string"), gps_schema).alias("d"))
    .select("d.*")
)

gps_query = (
    gps_stream.writeStream
    .format("delta")
    .option("checkpointLocation", "delta_tables/checkpoints/gps")
    .outputMode("append")
    .trigger(processingTime="5 seconds")
    .start("delta_tables/gps_trips")
)

# ============================================================================
print("\n[Stream Processor] ✅ All 3 streams active:")
print("  Branch A: CCTV → YOLOv8 inference → delta_tables/cv_vehicle_counts")
print("  Branch B: Sensors → congestion detection → delta_tables/sensor_speeds + rerouting_alerts")
print("  Branch C: GPS → delta_tables/gps_trips")
print("\nPress Ctrl+C to stop.\n")

spark.streams.awaitAnyTermination()
