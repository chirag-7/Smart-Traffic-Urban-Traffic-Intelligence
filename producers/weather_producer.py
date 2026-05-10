import json
import os
import time
from datetime import datetime, timezone

import requests
from dotenv import load_dotenv
from kafka import KafkaProducer


load_dotenv()

KAFKA_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY", "").strip()
OPENWEATHER_CITY = os.getenv("OPENWEATHER_CITY", "Los Angeles").strip()
WEATHER_TOPIC = os.getenv("WEATHER_TOPIC", "topic_weather")
WEATHER_POLL_SECONDS = int(os.getenv("WEATHER_POLL_SECONDS", "300"))


def fetch_weather(city: str, api_key: str) -> dict:
    """Fetch current weather from OpenWeatherMap."""
    url = "https://api.openweathermap.org/data/2.5/weather"
    params = {
        "q": city,
        "appid": api_key,
        "units": "metric",
    }
    response = requests.get(url, params=params, timeout=20)
    response.raise_for_status()
    return response.json()


def build_message(payload: dict) -> dict:
    """Normalize OpenWeather response into a compact Kafka message."""
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
    }


def main():
    print("[Weather Producer] Starting...")
    print(f"[Weather Producer] Kafka servers: {KAFKA_SERVERS}")
    print(f"[Weather Producer] Topic: {WEATHER_TOPIC}")
    print(f"[Weather Producer] City: {OPENWEATHER_CITY}")
    print(f"[Weather Producer] Poll interval: {WEATHER_POLL_SECONDS} seconds")

    if not OPENWEATHER_API_KEY:
        print("[Weather Producer] ✗ OPENWEATHER_API_KEY is missing in .env")
        return

    producer = KafkaProducer(
        bootstrap_servers=KAFKA_SERVERS,
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
        linger_ms=10,
        batch_size=16384,
    )

    message_count = 0

    try:
        while True:
            try:
                payload = fetch_weather(OPENWEATHER_CITY, OPENWEATHER_API_KEY)
                message = build_message(payload)
                producer.send(WEATHER_TOPIC, value=message)
                producer.flush()
                message_count += 1

                print(
                    f"[Weather Producer] Sent {message_count} — "
                    f"{message['city']} | {message['temp_c']}°C | "
                    f"{message['weather_main']} | humidity {message['humidity']}%"
                )
            except requests.RequestException as exc:
                print(f"[Weather Producer] Weather API error: {exc}")
            except Exception as exc:
                print(f"[Weather Producer] Unexpected error: {exc}")

            time.sleep(WEATHER_POLL_SECONDS)

    except KeyboardInterrupt:
        print("\n[Weather Producer] Stopping...")
    finally:
        producer.flush()
        print(f"[Weather Producer] Done. Total messages: {message_count}")


if __name__ == "__main__":
    main()