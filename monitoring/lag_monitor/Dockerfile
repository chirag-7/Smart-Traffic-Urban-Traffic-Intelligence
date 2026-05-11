# Polls Kafka consumer-group lag every POLL_SECONDS and exposes it as
# Prometheus metrics on :9102. Tiny image: librdkafka via confluent-kafka
# Python wheel + prometheus_client.

FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt ./
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY main.py ./

EXPOSE 9102

HEALTHCHECK --interval=20s --timeout=5s --retries=5 --start-period=15s \
    CMD curl -fsS http://localhost:9102/metrics > /dev/null || exit 1

CMD ["python", "-u", "main.py"]
