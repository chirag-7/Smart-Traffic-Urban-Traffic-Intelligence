import os
import sys
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

DELTA_BASE = Path(__file__).parent.parent / "delta_tables"


@st.cache_resource(show_spinner="Connecting to Spark...")
def get_spark():
    """Initialize Spark session for Delta Lake reading."""
    from pyspark.sql import SparkSession

    spark = (
        SparkSession.builder
        .appName("TrafficDashboard")
        .master("local[*]")
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
        .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")
        .config("spark.driver.memory", "2g")
        .config("spark.sql.shuffle.partitions", "4")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")
    return spark


def read_delta(spark, table_name: str, limit: int = 500) -> pd.DataFrame:
    """Read a Delta table into pandas DataFrame."""
    path = DELTA_BASE / table_name
    try:
        df = spark.read.format("delta").load(str(path))
        return df.limit(limit).toPandas()
    except Exception as e:
        st.warning(f"Could not read {table_name}: {e}")
        return None


# ============================================================================
# HEADER
# ============================================================================

st.title("🚦 Smart City Traffic Intelligence System")
st.caption("Real-time vehicle detection, speed prediction, and congestion-triggered rerouting")
st.caption("Infrastructure: Kafka + Spark Streaming + YOLOv8 + Delta Lake + NetworkX Graph Analytics")

spark = get_spark()

# ============================================================================
# ROW 1: KPI METRICS
# ============================================================================

col1, col2, col3, col4 = st.columns(4)

cv_df = read_delta(spark, "cv_vehicle_counts")
sensor_df = read_delta(spark, "sensor_speeds")
pred_df = read_delta(spark, "speed_predictions")
alert_df = read_delta(spark, "rerouting_alerts")

with col1:
    total_vehicles = int(cv_df["vehicle_count"].sum()) if cv_df is not None else 0
    st.metric("🚗 Vehicles Detected", f"{total_vehicles:,}")

with col2:
    avg_speed = sensor_df["speed"].mean() if sensor_df is not None else 0
    st.metric("⚡ Avg Network Speed", f"{avg_speed:.1f} mph")

with col3:
    congested = int((sensor_df["speed"] < 20).sum()) if sensor_df is not None else 0
    st.metric("🔴 Congested Readings", congested, delta="threshold: <20 mph")

with col4:
    alerts = len(alert_df) if alert_df is not None else 0
    st.metric("🔀 Rerouting Alerts", alerts)

st.divider()

# ============================================================================
# ROW 2: CHARTS — Speeds & Vehicles
# ============================================================================

col_left, col_right = st.columns(2)

with col_left:
    st.subheader("📡 Sensor Speed Distribution")
    if sensor_df is not None and len(sensor_df) > 0:
        speed_by_sensor = sensor_df.groupby("sensor_id")["speed"].mean().head(20).sort_values(ascending=False)
        st.bar_chart(speed_by_sensor)
    else:
        st.info("Waiting for sensor data from Kafka...")

with col_right:
    st.subheader("📷 Vehicle Counts by Camera")
    if cv_df is not None and len(cv_df) > 0:
        vehicles_by_cam = cv_df.groupby("camera_id")["vehicle_count"].sum().sort_values(ascending=False)
        st.bar_chart(vehicles_by_cam)
    else:
        st.info("Waiting for CCTV stream data...")

st.divider()

# ============================================================================
# ROW 3: ML PREDICTIONS & ALERTS
# ============================================================================

col_ml, col_alert = st.columns(2)

with col_ml:
    st.subheader("🤖 Speed Predictions vs Actual")
    if pred_df is not None and len(pred_df) > 0:
        # Ensure 'speed' and 'prediction' are numeric
        pred_df["speed"] = pd.to_numeric(pred_df["speed"], errors="coerce")
        pred_df["prediction"] = pd.to_numeric(pred_df["prediction"], errors="coerce")
        pred_df = pred_df.dropna(subset=["speed", "prediction"])
        
        if len(pred_df) > 0:
            pred_df["error"] = abs(pred_df["speed"] - pred_df["prediction"])
            sample = pred_df[["sensor_id", "speed", "prediction", "error"]].head(30)
            st.dataframe(sample, use_container_width=True)
            st.metric("Mean Abs. Error (mph)", f"{pred_df['error'].mean():.2f}")
        else:
            st.info("No valid predictions yet.")
    else:
        st.info("Run spark_jobs/ml_predictor.py to generate predictions.")

with col_alert:
    st.subheader("🔀 Rerouting Alerts")
    if alert_df is not None and len(alert_df) > 0:
        st.dataframe(alert_df.head(20), use_container_width=True)
    else:
        st.info("No congestion alerts yet.")

st.divider()

# ============================================================================
# ROW 4: RAW DATA EXPLORER
# ============================================================================

st.subheader("📊 Data Explorer")
explorer_tab1, explorer_tab2, explorer_tab3 = st.tabs(["Sensor Speeds", "Vehicle Counts", "GPS Trips"])

with explorer_tab1:
    if sensor_df is not None:
        st.write(f"**Total records:** {len(sensor_df)}")
        st.dataframe(sensor_df.head(50), use_container_width=True)
    else:
        st.info("No sensor data yet.")

with explorer_tab2:
    if cv_df is not None:
        st.write(f"**Total records:** {len(cv_df)}")
        st.dataframe(cv_df.head(50), use_container_width=True)
    else:
        st.info("No CCTV data yet.")

with explorer_tab3:
    gps_df = read_delta(spark, "gps_trips")
    if gps_df is not None:
        st.write(f"**Total records:** {len(gps_df)}")
        st.dataframe(gps_df.head(50), use_container_width=True)
    else:
        st.info("No GPS data yet.")

st.divider()

# ============================================================================
# FOOTER & AUTO-REFRESH
# ============================================================================

st.caption(f"Last updated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
col_refresh_a, col_refresh_b = st.columns([2, 1])
with col_refresh_a:
    refresh_rate = st.slider("Auto-refresh interval (seconds)", 5, 60, 10)
with col_refresh_b:
    if st.button("🔄 Refresh Now"):
        st.rerun()

time.sleep(refresh_rate)
st.rerun()
