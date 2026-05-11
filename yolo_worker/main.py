"""
CPU YOLO Kafka worker for smart-traffic.

Flow per message:
  1. Consume frame from topic_cctv (sticky-partitioned by camera_id).
  2. Reject stale frames (age > YOLO_FRAME_MAX_AGE_SEC) to keep latency bounded.
  3. Decode base64 JPEG and run YOLOv8 + ByteTrack on a per-camera model instance
     so tracker state is consistent across frames of the same camera.
  4. Upload annotated JPEG to MinIO bucket `cctv-frames` under {camera_id}/{frame_id}.
  5. Publish a compact JSON message to topic_cctv_inferred (no image bytes).

Metrics exposed on /metrics port YOLO_WORKER_METRICS_PORT (default 9100):
  yolo_frames_processed_total{camera_id}
  yolo_frames_skipped_total{reason}
  yolo_inference_seconds{camera_id}
  yolo_minio_upload_seconds
  yolo_unique_count_5min{camera_id}
  yolo_unique_count_session{camera_id}
  yolo_active_cameras
"""

from __future__ import annotations

import base64
import io
import json
import logging
import os
import signal
import socket
import sys
import time
from collections import defaultdict, deque
from pathlib import Path
from typing import Deque, Dict, Tuple

import boto3
import numpy as np
from botocore.client import Config as BotoConfig
from confluent_kafka import Consumer, KafkaError, KafkaException, Producer
from jsonschema import Draft202012Validator
from PIL import Image
from prometheus_client import Counter, Gauge, Histogram, start_http_server
from ultralytics import YOLO

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s yolo-worker %(message)s",
)
logger = logging.getLogger("yolo-worker")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

WORKER_ID = os.getenv("HOSTNAME", socket.gethostname())

KAFKA_BROKERS = os.getenv("KAFKA_BROKERS", "kafka:29092")
INPUT_TOPIC = os.getenv("INPUT_TOPIC", "topic_cctv")
OUTPUT_TOPIC = os.getenv("OUTPUT_TOPIC", "topic_cctv_inferred")
CONSUMER_GROUP = os.getenv("CONSUMER_GROUP", "yolo-workers")

MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "minio:9000")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "minioadmin")
MINIO_BUCKET = os.getenv("MINIO_BUCKET", "cctv-frames")
MINIO_PUBLIC_BASE_URL = os.getenv("MINIO_PUBLIC_BASE_URL", "http://localhost:9000/cctv-frames")

MODEL_PATH = os.getenv("YOLO_MODEL_PATH", "weights/best.pt")
IMGSZ = int(os.getenv("YOLO_IMGSZ", "480"))
CONF = float(os.getenv("YOLO_CONF", "0.3"))
FRAME_MAX_AGE_SEC = float(os.getenv("YOLO_FRAME_MAX_AGE_SEC", "10"))
METRICS_PORT = int(os.getenv("YOLO_WORKER_METRICS_PORT", "9100"))
MODEL_VERSION = os.getenv("YOLO_MODEL_VERSION", "yolo_traffic_model_v1")
UNIQUE_WINDOW_SEC = float(os.getenv("YOLO_UNIQUE_WINDOW_SEC", "300"))  # 5 minutes
SCHEMA_DIR = Path(os.getenv("SCHEMA_DIR", "/app/schemas"))

# ---------------------------------------------------------------------------
# Prometheus metrics
# ---------------------------------------------------------------------------

FRAMES_PROCESSED = Counter(
    "yolo_frames_processed_total",
    "Frames successfully inferred and published.",
    labelnames=("camera_id",),
)
FRAMES_SKIPPED = Counter(
    "yolo_frames_skipped_total",
    "Frames dropped before inference.",
    labelnames=("reason",),
)
INFERENCE_SECONDS = Histogram(
    "yolo_inference_seconds",
    "Time spent in model.track() per frame.",
    labelnames=("camera_id",),
    buckets=(0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0),
)
MINIO_UPLOAD_SECONDS = Histogram(
    "yolo_minio_upload_seconds",
    "Time spent uploading annotated JPEG to MinIO.",
    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0),
)
UNIQUE_5MIN = Gauge(
    "yolo_unique_count_5min",
    "Distinct ByteTrack IDs seen in the last 5 minutes for this camera.",
    labelnames=("camera_id",),
)
UNIQUE_SESSION = Gauge(
    "yolo_unique_count_session",
    "Distinct ByteTrack IDs seen since worker start for this camera.",
    labelnames=("camera_id",),
)
ACTIVE_CAMERAS = Gauge(
    "yolo_active_cameras",
    "Number of distinct camera_id streams owned by this worker.",
)

# ---------------------------------------------------------------------------
# Schema validation
# ---------------------------------------------------------------------------

def _load_validator(name: str) -> Draft202012Validator:
    path = SCHEMA_DIR / name
    with path.open("r", encoding="utf-8") as fh:
        return Draft202012Validator(json.load(fh))


try:
    INPUT_VALIDATOR = _load_validator("cctv_raw.json")
    OUTPUT_VALIDATOR = _load_validator("cctv_inferred.json")
except FileNotFoundError as exc:
    logger.error("Schema file missing at %s: %s", SCHEMA_DIR, exc)
    sys.exit(2)

# ---------------------------------------------------------------------------
# Per-camera state
# ---------------------------------------------------------------------------

_models: Dict[str, YOLO] = {}
_session_seen: Dict[str, set] = defaultdict(set)
_window: Dict[str, Deque[Tuple[int, float]]] = defaultdict(lambda: deque(maxlen=20_000))


def get_model(camera_id: str) -> YOLO:
    """Per-camera YOLO instance so model.track() keeps tracker state isolated."""
    if camera_id not in _models:
        logger.info("Loading YOLO model for camera_id=%s (resident set will grow ~150MB).", camera_id)
        _models[camera_id] = YOLO(MODEL_PATH)
        ACTIVE_CAMERAS.set(len(_models))
    return _models[camera_id]


def update_counts(camera_id: str, track_ids: list[int], now: float) -> Tuple[int, int]:
    """Maintain session-distinct + 5-minute sliding-window distinct counts.

    Returns ``(unique_5min, unique_session)``.
    """
    win = _window[camera_id]
    cutoff = now - UNIQUE_WINDOW_SEC
    while win and win[0][1] < cutoff:
        win.popleft()

    session = _session_seen[camera_id]
    for tid in track_ids:
        win.append((int(tid), now))
        session.add(int(tid))

    unique_5min = len({tid for tid, _ in win})
    unique_session = len(session)

    UNIQUE_5MIN.labels(camera_id=camera_id).set(unique_5min)
    UNIQUE_SESSION.labels(camera_id=camera_id).set(unique_session)
    return unique_5min, unique_session


# ---------------------------------------------------------------------------
# MinIO client
# ---------------------------------------------------------------------------

def make_s3_client():
    endpoint = MINIO_ENDPOINT
    if not endpoint.startswith(("http://", "https://")):
        endpoint = f"http://{endpoint}"
    return boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=MINIO_ACCESS_KEY,
        aws_secret_access_key=MINIO_SECRET_KEY,
        config=BotoConfig(signature_version="s3v4", retries={"max_attempts": 3}),
        region_name="us-east-1",
    )


def upload_jpeg(s3, key: str, jpeg_bytes: bytes) -> str:
    t0 = time.time()
    s3.put_object(
        Bucket=MINIO_BUCKET,
        Key=key,
        Body=jpeg_bytes,
        ContentType="image/jpeg",
    )
    MINIO_UPLOAD_SECONDS.observe(time.time() - t0)
    return f"{MINIO_PUBLIC_BASE_URL.rstrip('/')}/{key}"


# ---------------------------------------------------------------------------
# Kafka factories
# ---------------------------------------------------------------------------

def make_consumer() -> Consumer:
    return Consumer({
        "bootstrap.servers": KAFKA_BROKERS,
        "group.id": CONSUMER_GROUP,
        "auto.offset.reset": "latest",
        "enable.auto.commit": False,
        "session.timeout.ms": 30_000,
        "max.poll.interval.ms": 600_000,
        "fetch.message.max.bytes": 10 * 1024 * 1024,
        "client.id": f"yolo-worker-{WORKER_ID}",
    })


def make_producer() -> Producer:
    return Producer({
        "bootstrap.servers": KAFKA_BROKERS,
        "linger.ms": 20,
        "batch.size": 32 * 1024,
        "compression.type": "lz4",
        "acks": "all",
        "enable.idempotence": True,
        "client.id": f"yolo-worker-{WORKER_ID}",
    })


# ---------------------------------------------------------------------------
# Inference pipeline
# ---------------------------------------------------------------------------

def process_message(raw: bytes, s3, producer: Producer) -> None:
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        FRAMES_SKIPPED.labels(reason="json_decode_error").inc()
        return

    errors = list(INPUT_VALIDATOR.iter_errors(data))
    if errors:
        FRAMES_SKIPPED.labels(reason="schema_invalid").inc()
        logger.warning(
            "Schema-invalid input frame_id=%s: %s",
            data.get("frame_id", "?"),
            errors[0].message,
        )
        return

    now = time.time()
    if now - float(data["timestamp"]) > FRAME_MAX_AGE_SEC:
        FRAMES_SKIPPED.labels(reason="stale").inc()
        return

    camera_id = data["camera_id"]
    frame_id = data["frame_id"]

    try:
        img_bytes = base64.b64decode(data["frame_b64"])
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        img_np = np.array(img)
    except Exception as exc:  # noqa: BLE001
        FRAMES_SKIPPED.labels(reason="decode_error").inc()
        logger.warning("Decode error frame_id=%s: %s", frame_id, exc)
        return

    model = get_model(camera_id)

    t0 = time.time()
    try:
        results = model.track(
            img_np,
            conf=CONF,
            imgsz=IMGSZ,
            persist=True,
            verbose=False,
            tracker="bytetrack.yaml",
        )
    except Exception as exc:  # noqa: BLE001
        FRAMES_SKIPPED.labels(reason="inference_error").inc()
        logger.warning("Inference error camera=%s frame=%s: %s", camera_id, frame_id, exc)
        return
    inference_ms = (time.time() - t0) * 1000.0
    INFERENCE_SECONDS.labels(camera_id=camera_id).observe(inference_ms / 1000.0)

    res = results[0]
    if res.boxes is not None and res.boxes.id is not None:
        track_ids = [int(t) for t in res.boxes.id.tolist()]
    else:
        track_ids = []

    unique_5min, unique_session = update_counts(camera_id, track_ids, now)

    annotated_np = res.plot()
    annotated_img = Image.fromarray(annotated_np[..., ::-1])  # BGR -> RGB
    buf = io.BytesIO()
    annotated_img.save(buf, format="JPEG", quality=80, optimize=True)
    jpeg_bytes = buf.getvalue()

    safe_camera = camera_id.replace("/", "_")
    safe_frame = frame_id.replace("/", "_")
    key = f"{safe_camera}/{safe_frame}"
    if not key.lower().endswith(".jpg"):
        key += ".jpg"

    try:
        frame_url = upload_jpeg(s3, key, jpeg_bytes)
    except Exception as exc:  # noqa: BLE001
        FRAMES_SKIPPED.labels(reason="s3_error").inc()
        logger.warning("MinIO upload failed camera=%s frame=%s: %s", camera_id, frame_id, exc)
        return

    out = {
        "camera_id": camera_id,
        "frame_id": frame_id,
        "timestamp": float(data["timestamp"]),
        "unique_count_5min": int(unique_5min),
        "unique_count_session": int(unique_session),
        "track_ids": track_ids,
        "frame_url": frame_url,
        "inference_ms": float(inference_ms),
        "model_version": MODEL_VERSION,
        "worker_id": WORKER_ID,
        "processed_at": time.time(),
    }

    out_errors = list(OUTPUT_VALIDATOR.iter_errors(out))
    if out_errors:
        FRAMES_SKIPPED.labels(reason="output_invalid").inc()
        logger.error("Output schema-invalid for frame_id=%s: %s", frame_id, out_errors[0].message)
        return

    producer.produce(
        OUTPUT_TOPIC,
        key=camera_id.encode("utf-8"),
        value=json.dumps(out).encode("utf-8"),
    )
    FRAMES_PROCESSED.labels(camera_id=camera_id).inc()


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

_running = True


def _handle_signal(signum, _frame):  # noqa: ANN001
    global _running
    logger.info("Received signal %s — shutting down gracefully.", signum)
    _running = False


def main() -> None:
    if not Path(MODEL_PATH).exists():
        logger.error(
            "YOLO weights not found at %s. Mount your trained model at this path "
            "(see docker-compose volume).",
            MODEL_PATH,
        )
        sys.exit(1)

    logger.info(
        "yolo-worker starting | bootstrap=%s in=%s out=%s group=%s imgsz=%s conf=%s metrics_port=%s",
        KAFKA_BROKERS, INPUT_TOPIC, OUTPUT_TOPIC, CONSUMER_GROUP, IMGSZ, CONF, METRICS_PORT,
    )

    start_http_server(METRICS_PORT)
    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    s3 = make_s3_client()
    consumer = make_consumer()
    producer = make_producer()
    consumer.subscribe([INPUT_TOPIC])

    msg_count = 0
    last_commit = time.time()
    try:
        while _running:
            msg = consumer.poll(timeout=1.0)
            if msg is None:
                producer.poll(0)
                if time.time() - last_commit > 5.0:
                    consumer.commit(asynchronous=True)
                    last_commit = time.time()
                continue
            if msg.error():
                if msg.error().code() == KafkaError._PARTITION_EOF:
                    continue
                raise KafkaException(msg.error())

            process_message(msg.value(), s3, producer)
            msg_count += 1
            if msg_count % 50 == 0:
                producer.poll(0)
                consumer.commit(asynchronous=True)
                last_commit = time.time()
    finally:
        logger.info("Flushing producer (timeout 10s) and closing consumer…")
        producer.flush(10.0)
        try:
            consumer.commit(asynchronous=False)
        except KafkaException:
            pass
        consumer.close()
        logger.info("yolo-worker stopped. Processed %s messages.", msg_count)


if __name__ == "__main__":
    main()
