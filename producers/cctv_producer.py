"""
Publish JPEG frames as base64 JSON to Kafka ``topic_cctv``.

Sticky partitioning by ``camera_id`` so all frames of one camera land on the
same partition (and therefore the same yolo-worker), keeping ByteTrack IDs
consistent across frames.

Source layouts:
  - detrac : ``data/ua-detrac/DETRAC-Images/<sequence>/img*.jpg``
            (or ``data/ua-detrac/yolo_format/images/train/*.jpg`` as fallback)
  - flat   : flat directory of ``*.jpg`` with ``camera_<id>_*.jpg`` filenames
  - video  : OpenCV-readable video files in ``--source PATH``

CLI:
    python producers/cctv_producer.py --fps 2 --loop --source-layout detrac
"""

from __future__ import annotations

import argparse
import base64
import json
import logging
import os
import signal
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from jsonschema import Draft202012Validator
from kafka import KafkaProducer

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent
SCHEMA_PATH = REPO_ROOT / "schemas" / "cctv_raw.json"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="CCTV frame producer for smart-traffic.")
    p.add_argument("--fps", type=float, default=float(os.getenv("CCTV_REPLAY_FPS", "2")),
                   help="Replay rate in frames per second across all cameras (default 2).")
    p.add_argument("--cameras", type=int, default=5,
                   help="Number of distinct camera sequences to replay (default 5).")
    p.add_argument("--frames-per-camera", type=int,
                   default=int(os.getenv("CCTV_FRAMES_PER_CAMERA", "500")),
                   help="Frames per camera per pass (default 500).")
    p.add_argument("--loop", action="store_true",
                   default=os.getenv("CCTV_LOOP", "false").lower() == "true",
                   help="Loop indefinitely (default false).")
    p.add_argument("--source-layout",
                   choices=("detrac", "flat", "video"),
                   default=os.getenv("CCTV_SOURCE_LAYOUT", "detrac"),
                   help="Layout of the on-disk source data.")
    p.add_argument("--source", type=str, default=None,
                   help="Override source root path.")
    return p.parse_args()


def load_validator() -> Draft202012Validator:
    with SCHEMA_PATH.open("r", encoding="utf-8") as fh:
        return Draft202012Validator(json.load(fh))


def resolve_detrac_root() -> Path:
    """Prefer DETRAC-Images; fall back to yolo_format/images/train."""
    primary = REPO_ROOT / "data" / "ua-detrac" / "DETRAC-Images"
    if primary.exists():
        return primary
    fallback = REPO_ROOT / "data" / "ua-detrac" / "yolo_format" / "images" / "train"
    if fallback.exists():
        return fallback
    raise SystemExit(f"No CCTV images at {primary} or {fallback}.")


def detrac_iter(root: Path, n_cameras: int, frames_per_camera: int):
    """Yield (camera_id, frame_path) ordered (camera, frame#) for sticky-partition friendly send."""
    camera_dirs = sorted(d for d in root.iterdir() if d.is_dir())
    if not camera_dirs:
        # treat as flat layout when there are no subdirectories
        for fp in sorted(root.glob("*.jpg"))[:frames_per_camera * n_cameras]:
            stem = fp.stem
            camera_id = stem.split("_")[0] if "_" in stem else "camera_0"
            yield camera_id, fp
        return

    for cam in camera_dirs[:n_cameras]:
        for fp in sorted(cam.glob("*.jpg"))[:frames_per_camera]:
            yield cam.name, fp


def flat_iter(root: Path, frames_per_camera: int, n_cameras: int):
    files = sorted(root.glob("*.jpg"))[: frames_per_camera * n_cameras]
    for fp in files:
        stem = fp.stem
        camera_id = stem.split("_")[0] if "_" in stem else "camera_0"
        yield camera_id, fp


def video_iter(path: Path, n_cameras: int, frames_per_camera: int):
    """One camera per video file (round-robin). Imported lazily so OpenCV is optional."""
    import cv2  # type: ignore
    videos = sorted(path.glob("*.mp4")) if path.is_dir() else [path]
    if not videos:
        raise SystemExit(f"No videos under {path}")
    for cam_idx, vid in enumerate(videos[:n_cameras]):
        camera_id = f"camera_{cam_idx}"
        cap = cv2.VideoCapture(str(vid))
        frame_num = 0
        while frame_num < frames_per_camera:
            ok, frame = cap.read()
            if not ok:
                break
            ok2, jpeg = cv2.imencode(".jpg", frame)
            if not ok2:
                continue
            yield camera_id, jpeg.tobytes(), f"{vid.stem}_{frame_num:06d}.jpg"
            frame_num += 1
        cap.release()


_running = True


def _handle_signal(signum, _frame):  # noqa: ANN001
    global _running
    logger.info("Received signal %s — finishing current frame and flushing…", signum)
    _running = False


def main() -> None:
    args = parse_args()
    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    validator = load_validator()
    sleep_per_frame = 1.0 / max(args.fps, 0.1)

    bootstrap = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
    topic = os.getenv("TOPIC_CCTV", "topic_cctv")
    producer = KafkaProducer(
        bootstrap_servers=bootstrap,
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
        key_serializer=lambda k: k.encode("utf-8") if isinstance(k, str) else k,
        linger_ms=20,
        batch_size=64 * 1024,
        max_request_size=10 * 1024 * 1024,
        acks="all",
        compression_type="lz4",
    )
    logger.info(
        "CCTV producer | kafka=%s topic=%s fps=%s cameras=%s frames/cam=%s layout=%s loop=%s",
        bootstrap, topic, args.fps, args.cameras, args.frames_per_camera, args.source_layout, args.loop,
    )

    if args.source_layout == "detrac":
        root = Path(args.source) if args.source else resolve_detrac_root()
        sources = lambda: detrac_iter(root, args.cameras, args.frames_per_camera)  # noqa: E731
    elif args.source_layout == "flat":
        root = Path(args.source) if args.source else (REPO_ROOT / "data" / "ua-detrac" / "yolo_format" / "images" / "train")
        sources = lambda: flat_iter(root, args.frames_per_camera, args.cameras)  # noqa: E731
    else:
        root = Path(args.source) if args.source else (REPO_ROOT / "data" / "video")
        sources = lambda: video_iter(root, args.cameras, args.frames_per_camera)  # noqa: E731

    frame_count = 0
    pass_count = 0
    try:
        while _running:
            pass_count += 1
            for item in sources():
                if not _running:
                    break

                if args.source_layout == "video":
                    camera_id, jpeg_bytes, frame_id = item
                    b64 = base64.b64encode(jpeg_bytes).decode("utf-8")
                else:
                    camera_id, fp = item
                    frame_id = fp.name
                    with fp.open("rb") as fh:
                        b64 = base64.b64encode(fh.read()).decode("utf-8")

                msg = {
                    "camera_id": camera_id,
                    "frame_id": frame_id,
                    "timestamp": time.time(),
                    "frame_b64": b64,
                    "ingested_at": time.time(),
                    "producer_id": "cctv_producer",
                }
                errors = list(validator.iter_errors(msg))
                if errors:
                    logger.warning("Skip schema-invalid msg frame=%s: %s", frame_id, errors[0].message)
                    continue

                producer.send(topic, key=camera_id, value=msg)
                frame_count += 1
                if frame_count % 25 == 0:
                    logger.info("Sent %s frames (pass %s)…", frame_count, pass_count)

                time.sleep(sleep_per_frame)

            if not args.loop:
                break

    except Exception as exc:  # noqa: BLE001
        logger.exception("Producer error: %s", exc)
        sys.exit(1)
    finally:
        logger.info("Flushing producer (timeout 10s)…")
        producer.flush(timeout=10)
        producer.close(timeout=5)
        logger.info("CCTV producer finished — %s frames over %s pass(es).", frame_count, pass_count)


if __name__ == "__main__":
    main()
