"""
Offline speed prediction with Spark MLlib on METR-LA.

Reads a slice of ``data/metr-la/METR-LA.h5`` via h5py (avoids heavy pandas HDF paths on Windows),
engineers time features, trains a GBT regressor pipeline, saves the model under
``models/speed_prediction_pipeline``, and writes predictions to ``delta_tables/speed_predictions``
for the dashboard.

Training uses the first 1000 timestamps by default to bound memory; increase in code for fuller fits.
"""

from __future__ import annotations

import logging
import os
import shutil
import sys
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

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

logger.info("Starting Spark session...")
spark = (
    SparkSession.builder.appName("TrafficSpeedPredictor")
    .master("local[4]")
    .config("spark.jars.packages", "io.delta:delta-core_2.12:2.4.0")
    .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
    .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")
    .config("spark.driver.memory", os.getenv("SPARK_DRIVER_MEMORY", "4g"))
    .getOrCreate()
)
spark.sparkContext.setLogLevel("WARN")

h5_path = Path("data/metr-la/METR-LA.h5")
if not h5_path.exists():
    raise FileNotFoundError(f"METR-LA file not found: {h5_path}")

with h5py.File(h5_path, "r") as f:
    values = f["df/block0_values"][:1000]
    timestamps = f["df/axis1"][:1000]
    sensors = f["df/block0_items"][:]

n_timestamps, n_sensors = values.shape
logger.info("Loaded slice %s x %s", n_timestamps, n_sensors)

ts_series = pd.Series(timestamps)
if pd.api.types.is_numeric_dtype(ts_series):
    ts_series = pd.to_datetime(ts_series, unit="ns")
else:
    ts_series = pd.to_datetime(
        ts_series.str.decode("utf-8") if hasattr(ts_series.str, "decode") else ts_series
    )

sensor_str = np.array(
    [s.decode("utf-8") if isinstance(s, (bytes, bytearray)) else str(s) for s in sensors]
)

metr_long = pd.DataFrame(
    {
        "timestamp": np.repeat(ts_series.values, n_sensors),
        "sensor_id": np.tile(sensor_str, n_timestamps),
        "speed": values.reshape(-1),
    }
)
metr_long = metr_long.dropna(subset=["timestamp", "speed"])
logger.info("Long-format rows: %s", f"{len(metr_long):,}")

metr_long["timestamp"] = metr_long["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")
spark.conf.set("spark.sql.execution.arrow.pyspark.enabled", "false")

df = spark.createDataFrame(metr_long)
df = df.withColumn("timestamp", col("timestamp").cast("timestamp"))

df = (
    df.withColumn("hour", hour(col("timestamp")))
    .withColumn("day_of_week", dayofweek(col("timestamp")))
    .withColumn("ts_numeric", unix_timestamp(col("timestamp")))
)

assembler = VectorAssembler(
    inputCols=["hour", "day_of_week", "ts_numeric"], outputCol="raw_features"
)
scaler = StandardScaler(
    inputCol="raw_features", outputCol="scaled_features", withStd=True, withMean=True
)
gbt = GBTRegressor(featuresCol="scaled_features", labelCol="speed", maxIter=20, maxDepth=5)
pipeline = Pipeline(stages=[assembler, scaler, gbt])

train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)
logger.info("Training GBT pipeline...")
model = pipeline.fit(train_df)

predictions = model.transform(test_df)
evaluator = RegressionEvaluator(labelCol="speed", predictionCol="prediction")
rmse = evaluator.evaluate(predictions, {evaluator.metricName: "rmse"})
r2 = evaluator.evaluate(predictions, {evaluator.metricName: "r2"})
logger.info("Metrics — RMSE: %.4f  R²: %.4f", rmse, r2)

model_out = Path("models/speed_prediction_pipeline")
if model_out.exists():
    shutil.rmtree(model_out)
model.write().overwrite().save(str(model_out))
logger.info("Saved model to %s", model_out)

delta_out = Path("delta_tables/speed_predictions")
predictions.select("sensor_id", "timestamp", "speed", "prediction").write.format("delta").mode(
    "overwrite"
).save(str(delta_out))
logger.info("Saved predictions Delta table to %s", delta_out)

spark.stop()
