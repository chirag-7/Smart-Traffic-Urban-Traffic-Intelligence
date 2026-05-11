"""
One-shot batch job: load METR-LA sensor lat/lon into delta_tables/sensor_metadata.

Resolution order:
  1. ``data/metr-la/graph_sensor_locations.csv`` (or sensor_graph subdir) —
     the canonical file distributed with METR-LA that contains real
     latitude/longitude per sensor_id.
  2. If the CSV is missing but ``METR-LA.h5`` is present, read the REAL
     sensor_ids out of the H5 file and lay them out on a synthetic grid
     around downtown LA. This guarantees the IDs still match the live
     sensor_speeds stream (so the dashboard map joins correctly), with only
     the lat/lon being approximate. This is the recommended path when you
     don't have the CSV but do have the H5.
  3. If even the H5 is missing, fall back to placeholder ``synth_000…``
     IDs — the map will render but won't join with any real stream.

Idempotent: writes with mode="overwrite". Run once after graph_analytics.py.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from pyspark.sql.types import (
    DoubleType, IntegerType, StringType, StructField, StructType,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "spark_jobs"))
from _common import build_spark, silence_chatty_loggers  # noqa: E402

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("load_sensor_metadata")

CSV_CANDIDATES = [
    REPO_ROOT / "data" / "metr-la" / "graph_sensor_locations.csv",
    REPO_ROOT / "data" / "metr-la" / "sensor_graph" / "graph_sensor_locations.csv",
]
H5_PATH = REPO_ROOT / "data" / "metr-la" / "METR-LA.h5"
OUT_PATH = REPO_ROOT / "delta_tables" / "sensor_metadata"

# Downtown LA bounding box for synthetic grid (real METR-LA sensors are on
# I-10, I-110, etc. in this area).
LA_LAT_LO, LA_LAT_HI = 33.95, 34.20
LA_LON_LO, LA_LON_HI = -118.50, -118.10


def load_csv() -> pd.DataFrame | None:
    for p in CSV_CANDIDATES:
        if not p.exists():
            continue
        logger.info("Reading real sensor metadata from %s", p)
        df = pd.read_csv(p)
        colmap = {c.lower(): c for c in df.columns}
        sensor_col = colmap.get("sensor_id") or colmap.get("id") or df.columns[0]
        lat_col = colmap.get("latitude") or colmap.get("lat")
        lon_col = colmap.get("longitude") or colmap.get("lng") or colmap.get("lon")
        if not lat_col or not lon_col:
            logger.warning("CSV %s missing lat/lon columns: %s", p, df.columns.tolist())
            continue
        out = df[[sensor_col, lat_col, lon_col]].copy()
        out.columns = ["sensor_id", "lat", "lon"]
        out["sensor_id"] = out["sensor_id"].astype(str)
        out["sensor_index"] = np.arange(len(out), dtype=np.int32)
        return out
    return None


def real_ids_synthetic_grid(h5_path: Path) -> pd.DataFrame | None:
    """Real sensor_ids from METR-LA.h5 laid out on a synthetic LA grid.

    Returned IDs match the stream exactly, so the dashboard's join works;
    only lat/lon are approximate. This is the recommended fallback.
    """
    if not h5_path.exists():
        return None
    logger.info(
        "No CSV at %s — reading real sensor_ids from %s and synthesising LA grid lat/lon.",
        [str(p) for p in CSV_CANDIDATES], h5_path,
    )
    with h5py.File(h5_path, "r") as f:
        sensors = f["df/block0_items"][:]
    sensor_ids = [
        s.decode("utf-8") if isinstance(s, (bytes, bytearray)) else str(s) for s in sensors
    ]
    n = len(sensor_ids)
    side = int(np.ceil(np.sqrt(n)))
    lats = np.linspace(LA_LAT_LO, LA_LAT_HI, side)
    lons = np.linspace(LA_LON_LO, LA_LON_HI, side)
    rows = []
    for i, sid in enumerate(sensor_ids):
        r, c = divmod(i, side)
        rows.append({
            "sensor_id": sid,
            "sensor_index": i,
            "lat": float(lats[r]),
            "lon": float(lons[c]),
        })
    return pd.DataFrame(rows)


def placeholder_grid(n: int = 207) -> pd.DataFrame:
    logger.warning(
        "Neither CSV nor %s found — generating placeholder synth_NNN IDs. The "
        "dashboard map will render but WILL NOT join with the live sensor stream.",
        H5_PATH,
    )
    side = int(np.ceil(np.sqrt(n)))
    lats = np.linspace(LA_LAT_LO, LA_LAT_HI, side)
    lons = np.linspace(LA_LON_LO, LA_LON_HI, side)
    rows = []
    for i in range(n):
        r, c = divmod(i, side)
        rows.append({
            "sensor_id": f"synth_{i:03d}",
            "sensor_index": i,
            "lat": float(lats[r]),
            "lon": float(lons[c]),
        })
    return pd.DataFrame(rows)


def main() -> None:
    spark = build_spark("LoadSensorMetadata", cores=1, driver_memory="1g")
    silence_chatty_loggers(spark)

    df = load_csv()
    if df is None:
        df = real_ids_synthetic_grid(H5_PATH)
    if df is None:
        df = placeholder_grid()

    schema = StructType([
        StructField("sensor_id", StringType(), False),
        StructField("sensor_index", IntegerType(), True),
        StructField("lat", DoubleType(), False),
        StructField("lon", DoubleType(), False),
    ])

    spark_df = spark.createDataFrame(df, schema)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    spark_df.write.format("delta").mode("overwrite").save(str(OUT_PATH))

    logger.info("Wrote %s rows to %s", len(df), OUT_PATH)
    spark.stop()


if __name__ == "__main__":
    main()
