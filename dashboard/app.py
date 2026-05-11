"""
Smart Traffic — Streamlit monitoring dashboard.

Reads Delta Lake tables via the Rust-based ``deltalake`` package (no JVM).
Heavy widgets (CCTV grid, congestion map) use ``@st.fragment`` for independent
refresh cycles without whole-page flicker.  Annotated CCTV frames are fetched
by URL from MinIO; Delta only stores the pointer.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pydeck as pdk
import streamlit as st
from deltalake import DeltaTable
from dotenv import load_dotenv
from streamlit_autorefresh import st_autorefresh

load_dotenv()

st.set_page_config(
    page_title="Smart City Traffic Intelligence",
    page_icon="🚦",
    layout="wide",
)

DELTA_BASE = Path(__file__).resolve().parent.parent / "delta_tables"
MINIO_BASE = os.getenv("MINIO_PUBLIC_BASE_URL", "http://localhost:9000/cctv-frames")
CONGESTION_THRESHOLD = float(os.getenv("CONGESTION_THRESHOLD_MPH", "20.0"))


# ---------------------------------------------------------------------------
# Delta readers (cached, no Spark)
# ---------------------------------------------------------------------------

@st.cache_data(ttl=5, show_spinner=False)
def read_table(name: str, limit: int = 500, sort_col: str | None = "timestamp") -> pd.DataFrame:
    path = DELTA_BASE / name
    if not path.exists():
        return pd.DataFrame()
    try:
        df = DeltaTable(str(path)).to_pandas()
    except Exception as exc:  # noqa: BLE001
        # Surface the error in the UI instead of silently returning empty,
        # which made deltalake/pyarrow compatibility bugs invisible.
        st.warning(f"Could not read `delta_tables/{name}`: {type(exc).__name__}: {exc}")
        return pd.DataFrame()
    if df.empty:
        return df
    if sort_col and sort_col in df.columns:
        df = df.sort_values(sort_col, ascending=False)
    return df.head(limit)


def to_image_url(frame_url: str) -> str:
    """Frame URLs are stored in MinIO; rewrite if MINIO_PUBLIC_BASE_URL changed."""
    if not frame_url:
        return ""
    if MINIO_BASE in frame_url:
        return frame_url
    try:
        suffix = frame_url.split("/cctv-frames/", 1)[1]
        return f"{MINIO_BASE.rstrip('/')}/{suffix}"
    except IndexError:
        return frame_url


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

st_autorefresh(interval=5_000, key="top_metrics_refresh")

st.title("🚦 Smart City Traffic Intelligence")
st.caption(
    "Pure event-driven pipeline: Kafka → YOLO/ONNX workers → MinIO/Delta. "
    "Dashboard reads Delta via deltalake-rs (no JVM)."
)


# ---------------------------------------------------------------------------
# Metrics strip (always live, refreshes every 5 s)
# ---------------------------------------------------------------------------

cv_df = read_table("cv_vehicle_counts", limit=2000, sort_col="timestamp")
sensors_df = read_table("sensor_speeds", limit=5000, sort_col="timestamp")
weather_df = read_table("weather", limit=200, sort_col="timestamp")

col_m1, col_m2, col_m3, col_m4 = st.columns(4)

with col_m1:
    if not cv_df.empty and "unique_count_5min" in cv_df.columns:
        latest_per_camera = cv_df.sort_values("timestamp").groupby("camera_id").tail(1)
        total = int(latest_per_camera["unique_count_5min"].sum())
        active_cams = int(latest_per_camera["camera_id"].nunique())
        st.metric("Unique vehicles (5-min window)", f"{total:,}",
                  delta=f"{active_cams} active camera(s)")
    else:
        st.metric("Unique vehicles (5-min window)", "—")
        st.caption("No CCTV data yet — start cctv_producer.py.")

with col_m2:
    if not sensors_df.empty and "speed" in sensors_df.columns:
        avg = float(sensors_df["speed"].mean())
        st.metric("Avg network speed", f"{avg:.1f} mph")
    else:
        st.metric("Avg network speed", "—")

with col_m3:
    if not sensors_df.empty and "speed" in sensors_df.columns:
        congested_pct = float((sensors_df["speed"] < CONGESTION_THRESHOLD).mean() * 100)
        st.metric(f"Congested (<{int(CONGESTION_THRESHOLD)} mph)", f"{congested_pct:.1f}%")
    else:
        st.metric("Congested", "—")

with col_m4:
    if not weather_df.empty and "temp_c" in weather_df.columns:
        latest = weather_df.iloc[0]
        temp = latest.get("temp_c")
        cond = latest.get("weather_main") or latest.get("weather_description") or "—"
        if pd.notna(temp):
            st.metric(f"{latest.get('city','—')} weather", f"{float(temp):.1f}°C")
            st.caption(str(cond))
        else:
            st.metric("Weather", "—")
    else:
        st.metric("Weather", "—")

st.divider()


# ---------------------------------------------------------------------------
# Tabs — heavy widgets each isolated in a fragment
# ---------------------------------------------------------------------------

tab_live, tab_map, tab_ml, tab_alerts, tab_explore = st.tabs(
    ["📷 Live CCTV", "🗺️ Congestion Map", "📈 ML Predictions", "🚨 Alerts", "🔎 Explorer"]
)


# ---------- Live CCTV grid ----------

@st.fragment(run_every=2)
def cctv_grid():
    cv = read_table("cv_vehicle_counts", limit=200, sort_col="timestamp")
    if cv.empty:
        st.info(
            "No CCTV inferences yet. Start the producer with "
            "`python producers/cctv_producer.py --fps 2 --loop --source-layout detrac` "
            "and the YOLO worker will populate this view."
        )
        return

    latest_per_camera = cv.sort_values("timestamp").groupby("camera_id").tail(1)
    latest_per_camera = latest_per_camera.sort_values("timestamp", ascending=False)
    cameras = latest_per_camera.to_dict("records")

    cols = st.columns(min(len(cameras), 4) or 1)
    for i, row in enumerate(cameras[:8]):
        with cols[i % len(cols)]:
            url = to_image_url(row.get("frame_url", ""))
            cap_count = row.get("unique_count_5min", 0) or 0
            session_count = row.get("unique_count_session", 0) or 0
            cap = f"{row.get('camera_id', '?')} — {cap_count} veh (5m), {session_count} total"
            if url:
                try:
                    st.image(url, caption=cap, width="stretch")
                except Exception:  # noqa: BLE001
                    st.warning(f"Could not load {url}")
            else:
                st.warning(f"No frame URL for {row.get('camera_id')}")


with tab_live:
    st.subheader("Most recent annotated frame per camera")
    st.caption("Click an image to view full size. Refreshes every 2 seconds.")
    cctv_grid()


# ---------- Congestion map ----------

@st.fragment(run_every=10)
def congestion_map():
    sensors = read_table("sensor_speeds", limit=20_000, sort_col="timestamp")
    metadata = read_table("sensor_metadata", limit=500, sort_col=None)
    if sensors.empty or metadata.empty:
        st.info(
            "Map needs both sensor_speeds and sensor_metadata. "
            "Run `python spark_jobs/load_sensor_metadata.py` once and "
            "start sensor_producer.py."
        )
        return

    # Most recent reading per sensor.
    latest = sensors.sort_values("timestamp").groupby("sensor_id").tail(1)
    merged = latest.merge(metadata, on="sensor_id", how="inner")
    if merged.empty:
        st.info("Sensor IDs in stream don't match sensor_metadata. "
                "Did you run load_sensor_metadata.py with a real METR-LA CSV?")
        return

    # Speed → colour: red < threshold, yellow < 40, green ≥ 40.
    def speed_color(speed: float) -> list[int]:
        if speed < CONGESTION_THRESHOLD:
            return [220, 50, 50, 200]
        if speed < 40:
            return [240, 200, 60, 200]
        return [80, 200, 100, 200]

    merged["color"] = merged["speed"].apply(speed_color)
    merged["radius"] = (80 - merged["speed"].clip(0, 80)) * 4 + 50

    view = pdk.ViewState(
        latitude=float(merged["lat"].mean()),
        longitude=float(merged["lon"].mean()),
        zoom=10,
        pitch=0,
    )
    layer = pdk.Layer(
        "ScatterplotLayer",
        data=merged,
        get_position=["lon", "lat"],
        get_fill_color="color",
        get_radius="radius",
        pickable=True,
        opacity=0.85,
    )
    st.pydeck_chart(pdk.Deck(
        layers=[layer],
        initial_view_state=view,
        tooltip={"text": "{sensor_id}\nSpeed: {speed} mph"},
    ))
    st.caption(
        f"🟢 ≥40 mph  •  🟡 20-40 mph  •  🔴 <{int(CONGESTION_THRESHOLD)} mph  "
        f"— circle radius scales inversely with speed."
    )


with tab_map:
    st.subheader("Sensor network — current speed by location")
    congestion_map()


# ---------- ML predictions (scatter + residual histogram) ----------

@st.fragment(run_every=10)
def ml_predictions():
    pred = read_table("speed_predictions", limit=5000, sort_col="processed_at")
    if pred.empty:
        st.info(
            "No predictions yet. Run `python ml/train_speed_model.py` to train the "
            "model, then start the ml-predictor-worker:\n"
            "`docker compose up -d --force-recreate ml-predictor-worker`."
        )
        return

    pred = pred.dropna(subset=["predicted_speed"])
    if "actual_speed" in pred.columns:
        pred = pred.dropna(subset=["actual_speed"])

    if pred.empty or "actual_speed" not in pred.columns:
        st.info("Predictions are flowing but no actual speeds yet to compare against.")
        st.dataframe(pred.head(50), width="stretch")
        return

    residual = pred["actual_speed"] - pred["predicted_speed"]
    rmse = float(np.sqrt(np.mean(residual ** 2)))
    mae = float(np.mean(np.abs(residual)))

    c1, c2 = st.columns([1, 1])
    with c1:
        st.metric("Live RMSE (last 5000)", f"{rmse:.2f} mph")
    with c2:
        st.metric("Live MAE", f"{mae:.2f} mph")

    col_scatter, col_resid = st.columns(2)
    with col_scatter:
        st.markdown("**Actual vs Predicted**")
        scatter_df = pred[["actual_speed", "predicted_speed"]].copy()
        scatter_df.columns = ["actual", "predicted"]
        st.scatter_chart(scatter_df, x="actual", y="predicted", height=300)
    with col_resid:
        st.markdown("**Residuals (actual − predicted)**")
        hist_df = residual.to_frame(name="residual")
        st.bar_chart(np.histogram(hist_df["residual"].values, bins=40)[0], height=300)

    st.markdown("**Latest 20 predictions**")
    st.dataframe(
        pred[["sensor_id", "timestamp", "actual_speed", "predicted_speed", "model_version"]].head(20),
        width="stretch",
    )


with tab_ml:
    st.subheader("LightGBM (ONNX) speed predictions vs actual")
    ml_predictions()


# ---------- Alerts feed ----------

@st.fragment(run_every=5)
def alerts_feed():
    alerts = read_table("rerouting_alerts", limit=20, sort_col="batch_id")
    if alerts.empty:
        st.info("No congestion alerts yet. Alerts fire when sensor speed dips below "
                f"{int(CONGESTION_THRESHOLD)} mph and a landmark distance is known.")
        return
    for _, row in alerts.iterrows():
        speed = row.get("speed")
        distances = row.get("distances")
        sid = row.get("sensor_id", "?")
        sidx = row.get("id", "?")
        dist_str = (
            ", ".join(f"node {k} → {v} hops" for k, v in (distances or {}).items())
            if isinstance(distances, dict)
            else str(distances)
        )
        st.error(
            f"🚨 Sensor **{sid}** (vertex {sidx}) at "
            f"**{float(speed):.1f} mph** — reroute landmarks: {dist_str}"
        )


with tab_alerts:
    st.subheader("Live rerouting alerts")
    alerts_feed()


# ---------- Raw data explorer ----------

with tab_explore:
    st.subheader("Raw Delta tables")
    table_name = st.selectbox(
        "Table",
        options=[
            "sensor_speeds", "cv_vehicle_counts", "gps_trips", "weather",
            "speed_predictions", "rerouting_alerts", "sensor_metadata",
            "dlq_sensors", "pipeline_latency",
        ],
    )
    limit = st.slider("Rows", min_value=20, max_value=2000, value=200, step=20)
    df = read_table(table_name, limit=limit, sort_col=None)
    if df.empty:
        st.warning(f"Table `{table_name}` is empty or missing.")
    else:
        st.write(f"Showing **{len(df)}** rows from `{table_name}`")
        st.dataframe(df, width="stretch")


st.divider()
st.caption(
    f"Last metrics refresh: {datetime.now(timezone.utc).isoformat(timespec='seconds')}  •  "
    f"Auto-refresh: 5 s (metrics) / 10 s (heavy widgets via @st.fragment)"
)
