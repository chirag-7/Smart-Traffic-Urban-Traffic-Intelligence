"""
Streamlit dashboard for Delta tables produced by the streaming and batch jobs.

Reads from ``../delta_tables`` via a local Spark session with the Delta Lake package.
Auto-refreshes on an interval; use “Refresh Now” for immediate reload.

Tables consumed: ``cv_vehicle_counts``, ``sensor_speeds``, ``speed_predictions``,
``rerouting_alerts``, ``gps_trips``, ``weather``.
"""

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(
    page_title="Smart City Traffic Intelligence",
    page_icon="🚦",
    layout="wide",
)

DELTA_BASE = Path(__file__).resolve().parent.parent / "delta_tables"


@st.cache_resource(show_spinner="Connecting to Spark...")
def get_spark():
    from pyspark.sql import SparkSession

    spark = (
        SparkSession.builder.appName("TrafficDashboard")
        .master("local[*]")
        .config("spark.jars.packages", "io.delta:delta-core_2.12:2.4.0")
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
        .config(
            "spark.sql.catalog.spark_catalog",
            "org.apache.spark.sql.delta.catalog.DeltaCatalog",
        )
        .config("spark.driver.memory", "2g")
        .config("spark.sql.shuffle.partitions", "4")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")
    return spark


def read_delta(spark, table_name: str, limit: int = 500):
    from pyspark.sql.functions import col

    path = DELTA_BASE / table_name
    try:
        df = spark.read.format("delta").load(str(path))
        if "timestamp" in df.columns:
            df = df.withColumn("timestamp", col("timestamp").cast("string"))
        return df.limit(limit).toPandas()
    except Exception as exc:
        st.warning(f"Could not read {table_name}: {exc}")
        return None


st.title("Smart City Traffic Intelligence")
st.caption(
    "Kafka → Spark Streaming → Delta Lake; CCTV counts from YOLOv8; "
    "congestion-triggered rerouting using precomputed graph paths."
)

spark = get_spark()

col1, col2, col3, col4 = st.columns(4)

cv_df = read_delta(spark, "cv_vehicle_counts")
sensor_df = read_delta(spark, "sensor_speeds")
pred_df = read_delta(spark, "speed_predictions")
alert_df = read_delta(spark, "rerouting_alerts")
weather_df = read_delta(spark, "weather", limit=200)

with col1:
    total_vehicles = int(cv_df["vehicle_count"].sum()) if cv_df is not None else 0
    st.metric("Vehicles detected (stream)", f"{total_vehicles:,}")

with col2:
    avg_speed = sensor_df["speed"].mean() if sensor_df is not None else 0
    st.metric("Avg network speed (mph)", f"{avg_speed:.1f}")

with col3:
    congested = int((sensor_df["speed"] < 20).sum()) if sensor_df is not None else 0
    st.metric("Congested readings (<20 mph)", congested)

with col4:
    if weather_df is not None and len(weather_df) > 0:
        try:
            weather_df = weather_df.copy()
            weather_df["timestamp"] = pd.to_numeric(weather_df["timestamp"], errors="coerce")
            latest = weather_df.sort_values("timestamp", ascending=False).iloc[0]
            temp_c = latest.get("temp_c")
            cond = latest.get("weather_main") or latest.get("weather_description")
            st.metric("Latest temp (°C)", f"{temp_c:.1f}" if temp_c is not None else "n/a")
            st.caption(f"{latest.get('city', '')} — {cond}")
        except Exception:
            st.info("Weather data present but could not parse latest row.")
    else:
        st.info("No weather rows yet.")

    alerts = len(alert_df) if alert_df is not None else 0
    st.metric("Rerouting alert rows", alerts)

st.divider()

col_left, col_right = st.columns(2)

with col_left:
    st.subheader("Sensor speeds (mean, sample)")
    if sensor_df is not None and len(sensor_df) > 0:
        speed_by_sensor = (
            sensor_df.groupby("sensor_id")["speed"].mean().head(20).sort_values(ascending=False)
        )
        st.bar_chart(speed_by_sensor)
    else:
        st.info("Waiting for sensor Delta data.")

with col_right:
    st.subheader("Vehicle counts by camera")
    if cv_df is not None and len(cv_df) > 0:
        by_cam = cv_df.groupby("camera_id")["vehicle_count"].sum().sort_values(ascending=False)
        st.bar_chart(by_cam)
    else:
        st.info("Waiting for CCTV Delta data.")

st.divider()

col_ml, col_alert = st.columns(2)

with col_ml:
    st.subheader("Speed predictions vs actual")
    if pred_df is not None and len(pred_df) > 0:
        pred_df = pred_df.copy()
        pred_df["speed"] = pd.to_numeric(pred_df["speed"], errors="coerce")
        pred_df["prediction"] = pd.to_numeric(pred_df["prediction"], errors="coerce")
        pred_df = pred_df.dropna(subset=["speed", "prediction"])
        if len(pred_df) > 0:
            pred_df["error"] = (pred_df["speed"] - pred_df["prediction"]).abs()
            st.dataframe(
                pred_df[["sensor_id", "speed", "prediction", "error"]].head(30),
                use_container_width=True,
            )
            st.metric("Mean absolute error (mph)", f"{pred_df['error'].mean():.2f}")
        else:
            st.info("No valid numeric predictions.")
    else:
        st.info("Run ``spark_jobs/ml_predictor.py`` first.")

with col_alert:
    st.subheader("Rerouting alerts")
    if alert_df is not None and len(alert_df) > 0:
        st.dataframe(alert_df.head(20), use_container_width=True)
    else:
        st.info("No rerouting rows yet.")

st.divider()

st.subheader("Data explorer")
explorer_tab1, explorer_tab2, explorer_tab3 = st.tabs(["Sensor speeds", "Vehicle counts", "GPS trips"])

with explorer_tab1:
    if sensor_df is not None:
        st.write("Records (limited): ", len(sensor_df))
        st.dataframe(sensor_df.head(50), use_container_width=True)
    else:
        st.info("No sensor data.")

with explorer_tab2:
    if cv_df is not None:
        st.write("Records (limited): ", len(cv_df))
        st.dataframe(cv_df.head(50), use_container_width=True)
    else:
        st.info("No CCTV-derived data.")

with explorer_tab3:
    gps_df = read_delta(spark, "gps_trips")
    if gps_df is not None:
        st.write("Records (limited): ", len(gps_df))
        st.dataframe(gps_df.head(50), use_container_width=True)
    else:
        st.info("No GPS data.")

st.divider()

st.caption(f"Last updated: {pd.Timestamp.now():%Y-%m-%d %H:%M:%S}")
col_refresh_a, col_refresh_b = st.columns([2, 1])
with col_refresh_a:
    refresh_rate = st.slider("Auto-refresh interval (seconds)", 5, 60, 10)
with col_refresh_b:
    if st.button("Refresh now"):
        st.rerun()

time.sleep(refresh_rate)
st.rerun()
