"""
Phase 3: nightly maintenance for Delta Lake tables.

Runs OPTIMIZE (compacts many small Parquet files into fewer big ones) and
VACUUM (deletes files older than the retention threshold that are no longer
referenced by any Delta version). Without this, partitioned high-volume
tables degrade in read latency over weeks.

Run on a schedule (e.g. Windows Task Scheduler at 03:00):
    python spark_jobs/maintenance.py

CLI overrides:
    --retention-hours 168   # default 168 (7 days), Delta minimum is 168
    --skip-vacuum
    --tables sensor_speeds,cv_vehicle_counts
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "spark_jobs"))
from _common import build_spark, silence_chatty_loggers  # noqa: E402

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("maintenance")

DEFAULT_TABLES = [
    "sensor_speeds",
    "cv_vehicle_counts",
    "gps_trips",
    "speed_predictions",
    "weather",
    "rerouting_alerts",
    "dlq_sensors",
    "pipeline_latency",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Delta OPTIMIZE + VACUUM maintenance.")
    p.add_argument("--retention-hours", type=int, default=168,
                   help="VACUUM retention (Delta default min is 168 = 7 days).")
    p.add_argument("--skip-vacuum", action="store_true",
                   help="Skip VACUUM (run OPTIMIZE only).")
    p.add_argument("--tables", type=str, default=",".join(DEFAULT_TABLES),
                   help="Comma-separated list of Delta table names under delta_tables/.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    tables = [t.strip() for t in args.tables.split(",") if t.strip()]
    base = REPO_ROOT / "delta_tables"

    spark = build_spark("DeltaMaintenance", cores=2, driver_memory="2g")
    silence_chatty_loggers(spark)

    if args.retention_hours < 168 and not args.skip_vacuum:
        logger.warning(
            "Retention %s h is below Delta's 168 h default. Setting "
            "spark.databricks.delta.retentionDurationCheck.enabled=false.",
            args.retention_hours,
        )
        spark.conf.set("spark.databricks.delta.retentionDurationCheck.enabled", "false")

    for name in tables:
        path = base / name
        if not path.exists():
            logger.info("Skip — table not present: %s", path)
            continue
        try:
            logger.info("OPTIMIZE delta.`%s`", path)
            spark.sql(f"OPTIMIZE delta.`{path}`").show(truncate=False)
        except Exception as exc:  # noqa: BLE001
            logger.warning("OPTIMIZE failed for %s: %s", name, exc)
            continue

        if args.skip_vacuum:
            continue
        try:
            logger.info("VACUUM delta.`%s` RETAIN %s HOURS", path, args.retention_hours)
            spark.sql(
                f"VACUUM delta.`{path}` RETAIN {args.retention_hours} HOURS"
            ).show(truncate=False)
        except Exception as exc:  # noqa: BLE001
            logger.warning("VACUUM failed for %s: %s", name, exc)

    logger.info("Maintenance run complete.")
    spark.stop()


if __name__ == "__main__":
    main()
