# Online speed-prediction Kafka worker.
#
# Consumes topic_sensors, maintains a per-sensor sliding window in memory,
# runs ONNX inference (LightGBM-converted model) and publishes predictions
# to topic_speed_predictions. Exposes Prometheus metrics on :9101.

FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt ./
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY main.py ./

# Schemas (/app/schemas) and ONNX model (/app/models) are bind-mounted at
# runtime so they can be updated without rebuilding the image.

EXPOSE 9101

HEALTHCHECK --interval=15s --timeout=5s --retries=5 --start-period=20s \
    CMD curl -fsS http://localhost:9101/metrics > /dev/null || exit 1

CMD ["python", "-u", "main.py"]
