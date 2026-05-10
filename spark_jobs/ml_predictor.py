import os
import sys
import shutil
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.feature import StandardScaler, VectorAssembler
from pyspark.ml.regression import GBTRegressor
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, dayofweek, hour, unix_timestamp

# Ensure Spark workers use this venv Python on Windows.
os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

load_dotenv()

print("[ML] Initializing Spark Session...")
spark = (
    SparkSession.builder
    .appName("TrafficSpeedPredictor")
    .master("local[4]")
    .config("spark.jars.packages", "io.delta:delta-core_2.12:2.4.0") 
    .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
    .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")
    .config("spark.driver.memory", os.getenv("SPARK_DRIVER_MEMORY", "4g"))
    .getOrCreate()
)
spark.sparkContext.setLogLevel("WARN")

print("[ML] Loading METR-LA data via h5py...")
h5_path = Path("data/metr-la/METR-LA.h5")
if not h5_path.exists():
    raise FileNotFoundError(f"METR-LA file not found: {h5_path}")

with h5py.File(h5_path, "r") as f:
    # Sliced to 1000 timestamps (~207,000 rows) to prevent memory crashes
    values = f["df/block0_values"][:1000]   
    timestamps = f["df/axis1"][:1000]
    sensors = f["df/block0_items"][:]

n_timestamps, n_sensors = values.shape
print(f"[ML] Raw matrix shape: {n_timestamps} x {n_sensors}")

# Robust timestamp parsing for METR-LA's specific HDF5 format
ts_series = pd.Series(timestamps)
if pd.api.types.is_numeric_dtype(ts_series):
    ts_series = pd.to_datetime(ts_series, unit='ns')
else:
    ts_series = pd.to_datetime(ts_series.str.decode('utf-8') if hasattr(ts_series.str, 'decode') else ts_series)

sensor_str = np.array([
    s.decode("utf-8") if isinstance(s, (bytes, bytearray)) else str(s) for s in sensors
])

metr_long = pd.DataFrame({
    "timestamp": np.repeat(ts_series.values, n_sensors),
    "sensor_id": np.tile(sensor_str, n_timestamps),
    "speed": values.reshape(-1),
})

# Drop NAs
metr_long = metr_long.dropna(subset=["timestamp", "speed"])
print(f"[ML] Long records after NA drop: {len(metr_long):,}")

# --- NEW FIX: Bypass the PyArrow/Windows DateTime crash ---
# 1. Convert Pandas datetime to a simple string
metr_long["timestamp"] = metr_long["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")

# 2. Disable PyArrow optimization for this specific conversion
spark.conf.set("spark.sql.execution.arrow.pyspark.enabled", "false")

# 3. Create the Spark DataFrame safely
df = spark.createDataFrame(metr_long)

# 4. Have Spark convert the string back into a native Timestamp
df = df.withColumn("timestamp", col("timestamp").cast("timestamp"))
# ----------------------------------------------------------

# Feature Engineering
print("[ML] Engineering time features...")
df = df.withColumn("hour", hour(col("timestamp")))
df = df.withColumn("day_of_week", dayofweek(col("timestamp")))
df = df.withColumn("ts_numeric", unix_timestamp(col("timestamp")))

feature_cols = ["hour", "day_of_week", "ts_numeric"]

assembler = VectorAssembler(inputCols=feature_cols, outputCol="raw_features")
scaler = StandardScaler(inputCol="raw_features", outputCol="scaled_features", withStd=True, withMean=True)
gbt = GBTRegressor(featuresCol="scaled_features", labelCol="speed", maxIter=20, maxDepth=5)

pipeline = Pipeline(stages=[assembler, scaler, gbt])

print("[ML] Splitting data (80/20)...")
train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)

print("[ML] Training GBTRegressor Pipeline (This may take a minute)...")
model = pipeline.fit(train_df)

print("[ML] Making predictions on test set...")
predictions = model.transform(test_df)

evaluator = RegressionEvaluator(labelCol="speed", predictionCol="prediction")
rmse = evaluator.evaluate(predictions, {evaluator.metricName: "rmse"})
r2 = evaluator.evaluate(predictions, {evaluator.metricName: "r2"})

print(f"\n[ML] ✅ Training Complete!")
print(f"[ML] RMSE: {rmse:.2f}")
print(f"[ML] R2:   {r2:.2f}")

model_out = Path("models/speed_prediction_pipeline")
if model_out.exists():
    shutil.rmtree(model_out)
print(f"[ML] Saving pipeline to {model_out}...")
model.write().overwrite().save(str(model_out))

delta_out = Path("delta_tables/speed_predictions")
print(f"[ML] Saving sample predictions to Delta at {delta_out}...")
predictions.select("sensor_id", "timestamp", "speed", "prediction") \
    .write \
    .format("delta") \
    .mode("overwrite") \
    .save(str(delta_out))

print("[ML] All Tasks Finished Successfully.")