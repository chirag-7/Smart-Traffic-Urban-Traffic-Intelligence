"""
Publish JPEG frames as base64 JSON to Kafka ``topic_cctv``.

Default image root: ``data/ua-detrac/DETRAC-Images/<sequence>/*.jpg``.
Alternatively you can point ingestion at ``data/ua-detrac/yolo_format/images/train`` by changing
``IMAGE_ROOT`` below to match your local dataset layout.
"""

from __future__ import annotations

import base64
import json
import logging
import os
import time
from pathlib import Path

from dotenv import load_dotenv
from kafka import KafkaProducer

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

KAFKA_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")

# Use yolo_format train images if DETRAC-Images is absent (common after conversion-only workflows)
IMAGE_ROOT = Path("data/ua-detrac/DETRAC-Images")
if not IMAGE_ROOT.exists():
    IMAGE_ROOT = Path("data/ua-detrac/yolo_format/images/train")

producer = KafkaProducer(
    bootstrap_servers=KAFKA_SERVERS,
    value_serializer=lambda v: json.dumps(v).encode("utf-8"),
    max_request_size=5_242_880,
)

if not IMAGE_ROOT.exists():
    logger.error("No images under %s — download UA-DETRAC or run conversion.", IMAGE_ROOT)
    raise SystemExit(1)

frame_count = 0
camera_dirs = sorted(d for d in IMAGE_ROOT.iterdir() if d.is_dir())

if camera_dirs:
    logger.info("Kafka %s — %s sequence folders under %s", KAFKA_SERVERS, len(camera_dirs), IMAGE_ROOT)
    for cam in camera_dirs[:5]:
        for frame_path in sorted(cam.glob("*.jpg"))[:50]:
            with open(frame_path, "rb") as fh:
                b64 = base64.b64encode(fh.read()).decode("utf-8")
            producer.send(
                "topic_cctv",
                value={
                    "camera_id": cam.name,
                    "frame_id": frame_path.name,
                    "timestamp": time.time(),
                    "frame_b64": b64,
                },
            )
            frame_count += 1
            if frame_count % 25 == 0:
                logger.info("Sent %s frames…", frame_count)
            time.sleep(0.04)
else:
    logger.info("Kafka %s — flat JPEGs under %s", KAFKA_SERVERS, IMAGE_ROOT)
    for frame_path in sorted(IMAGE_ROOT.glob("*.jpg"))[:250]:
        with open(frame_path, "rb") as fh:
            b64 = base64.b64encode(fh.read()).decode("utf-8")
        stem = frame_path.stem
        producer.send(
            "topic_cctv",
            value={
                "camera_id": stem.split("_")[0] if "_" in stem else "camera_0",
                "frame_id": frame_path.name,
                "timestamp": time.time(),
                "frame_b64": b64,
            },
        )
        frame_count += 1
        if frame_count % 25 == 0:
            logger.info("Sent %s frames…", frame_count)
        time.sleep(0.04)

producer.flush()
logger.info("Finished — %s frames.", frame_count)
