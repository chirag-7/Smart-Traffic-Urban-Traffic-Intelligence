"""
Fine-tune Ultralytics YOLOv8 on UA-DETRAC data prepared under ``data/ua-detrac/yolo_format``.

Writes checkpoints to ``models/yolo_traffic_model/``. The streaming job expects weights at
``models/yolo_traffic_model/weights/best.pt``.

Resume behavior: if ``weights/last.pt`` exists, training continues from those weights without
``resume=True`` (avoids optimizer/dataset metadata mismatches from Ultralytics).
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

from ultralytics import YOLO

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

DATASET_YAML = "data/ua-detrac/yolo_format/dataset.yaml"
OUTPUT_PROJECT = Path("models")
OUTPUT_NAME = "yolo_traffic_model"
OUTPUT_DIR = OUTPUT_PROJECT / OUTPUT_NAME
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
LAST_WEIGHTS = OUTPUT_DIR / "weights" / "last.pt"

if not Path(DATASET_YAML).exists():
    logger.error("Missing dataset: %s", DATASET_YAML)
    raise SystemExit(1)

if LAST_WEIGHTS.exists():
    logger.info("Continuing from %s", LAST_WEIGHTS)
    model = YOLO(str(LAST_WEIGHTS))
else:
    logger.info("Starting from yolov8n.pt")
    model = YOLO("yolov8n.pt")

model.train(
    data=DATASET_YAML,
    epochs=50,
    imgsz=640,
    batch=16,
    patience=10,
    project=str(OUTPUT_PROJECT),
    name=OUTPUT_NAME,
    exist_ok=True,
    save=True,
    verbose=True,
    plots=True,
)

best = OUTPUT_DIR / "weights" / "best.pt"
if best.exists():
    shutil.copy2(best, OUTPUT_DIR / "best.pt")
    logger.info("Best weights: %s", best)
else:
    logger.error("Expected weights not found: %s", best)
