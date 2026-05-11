"""
Poll OpenWeatherMap current weather and publish JSON to Kafka (default ``topic_weather``).

Environment (``.env``): ``OPENWEATHER_API_KEY``, ``OPENWEATHER_CITY``, optional ``WEATHER_TOPIC``,
``WEATHER_POLL_SECONDS``.

Phase 3 hardening:
  - jsonschema validation against schemas/weather.json before publishing.
  - SIGINT/SIGTERM handler flushes in-flight messages on shutdown.
"""

from __future__ import annotations

import json
import logging
import os
import signal
import time
from datetime import datetime, timezone
from pathlib import Path

import requests
from dotenv import load_dotenv
from jsonschema import Draft202012Validator
from kafka import KafkaProducer

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent
SCHEMA_PATH = REPO_ROOT / "schemas" / "weather.json"

KAFKA_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY", "").strip()
OPENWEATHER_CITY = os.getenv("OPENWEATHER_CITY", "Los Angeles").strip()
WEATHER_TOPIC = os.getenv("WEATHER_TOPIC", "topic_weather")
WEATHER_POLL_SECONDS = int(os.getenv("WEATHER_POLL_SECONDS", "300"))


def fetch_weather(city: str, api_key: str) -> dict:
    url = "https://api.openweathermap.org/data/2.5/weather"
    params = {"q": city, "appid": api_key, "units": "metric"}
    response = requests.get(url, params=params, timeout=20)
    response.raise_for_status()
    return response.json()


def build_message(payload: dict) -> dict:
    weather = payload.get("weather", [{}])[0]
    main = payload.get("main", {})
    wind = payload.get("wind", {})
    clouds = payload.get("clouds", {})

    return {
        "city": payload.get("name", OPENWEATHER_CITY),
        "country": payload.get("sys", {}).get("country", ""),
        "observed_at": datetime.now(timezone.utc).isoformat(),
        "temp_c": main.get("temp"),
        "feels_like_c": main.get("feels_like"),
        "temp_min_c": main.get("temp_min"),
        "temp_max_c": main.get("temp_max"),
        "humidity": main.get("humidity"),
        "pressure_hpa": main.get("pressure"),
        "wind_speed_mps": wind.get("speed"),
        "wind_deg": wind.get("deg"),
        "clouds_pct": clouds.get("all"),
        "weather_main": weather.get("main"),
        "weather_description": weather.get("description"),
        "weather_id": weather.get("id"),
        "timestamp": time.time(),
        "ingested_at": time.time(),
    }


_running = True


def _handle_signal(signum, _frame):  # noqa: ANN001
    global _running
    logger.info("Received signal %s — flushing producer and stopping.", signum)
    _running = False


def main() -> None:
    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    logger.info(
        "Kafka %s | topic=%s | city=%s | poll=%ss",
        KAFKA_SERVERS,
        WEATHER_TOPIC,
        OPENWEATHER_CITY,
        WEATHER_POLL_SECONDS,
    )

    if not OPENWEATHER_API_KEY:
        logger.error("Set OPENWEATHER_API_KEY in .env")
        return

    with SCHEMA_PATH.open("r", encoding="utf-8") as fh:
        validator = Draft202012Validator(json.load(fh))

    producer = KafkaProducer(
        bootstrap_servers=KAFKA_SERVERS,
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
        linger_ms=10,
        batch_size=16384,
        acks="all",
    )

    message_count = 0
    invalid = 0
    try:
        while _running:
            try:
                payload = fetch_weather(OPENWEATHER_CITY, OPENWEATHER_API_KEY)
                message = build_message(payload)
                errors = list(validator.iter_errors(message))
                if errors:
                    invalid += 1
                    logger.warning("Skip schema-invalid weather row: %s", errors[0].message)
                else:
                    producer.send(WEATHER_TOPIC, value=message)
                    producer.flush()
                    message_count += 1
                    logger.info(
                        "%s | %.1f°C | %s",
                        message["city"],
                        message["temp_c"] or 0.0,
                        message["weather_main"],
                    )
            except requests.RequestException as exc:
                logger.warning("HTTP error: %s", exc)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Unexpected: %s", exc)

            # Wait but check _running every second so SIGINT exits promptly.
            for _ in range(WEATHER_POLL_SECONDS):
                if not _running:
                    break
                time.sleep(1)
    finally:
        logger.info("Flushing producer (timeout 10s)…")
        producer.flush(timeout=10)
        producer.close(timeout=5)
        logger.info("Total sent=%s invalid=%s", message_count, invalid)


if __name__ == "__main__":
    main()
