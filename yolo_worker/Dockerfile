# CPU-only YOLO Kafka worker for smart-traffic.
#
# Consumes topic_cctv (base64 JPEG payloads), runs YOLOv8 + ByteTrack per
# camera_id, uploads annotated JPEGs to MinIO, and publishes a compact JSON
# message to topic_cctv_inferred. Exposes Prometheus metrics on :9100.

FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

# OS deps for opencv-python-headless + healthcheck curl.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        curl \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first (cache-friendly).
COPY requirements.txt ./
RUN pip install --upgrade pip && pip install -r requirements.txt

# Application code.
COPY main.py ./

# Schemas (./schemas) and weights (./models/yolo_traffic_model/weights) are
# bind-mounted at runtime via docker-compose so they can be updated without
# a rebuild. If either is missing the worker logs a clear error and exits 1.

EXPOSE 9100

HEALTHCHECK --interval=15s --timeout=5s --retries=5 --start-period=20s \
    CMD curl -fsS http://localhost:9100/metrics > /dev/null || exit 1

CMD ["python", "-u", "main.py"]
