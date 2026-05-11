"""Validate each Kafka topic schema with a known-good and known-bad payload."""

from __future__ import annotations

import time

import pytest
from jsonschema import Draft202012Validator

# ---- Known-good payloads (must validate) -----------------------------------

GOOD_CCTV_RAW = {
    "camera_id": "MVI_20011",
    "frame_id": "img00001.jpg",
    "timestamp": time.time(),
    "frame_b64": "/9j/4AAQSkZJRgABAQAAAQABAAD",
}

GOOD_CCTV_INFERRED = {
    "camera_id": "MVI_20011",
    "frame_id": "img00037.jpg",
    "timestamp": time.time(),
    "unique_count_5min": 7,
    "unique_count_session": 12,
    "track_ids": [3, 5, 7, 11, 12, 18, 21],
    "frame_url": "http://localhost:9000/cctv-frames/MVI_20011/img00037.jpg",
    "inference_ms": 142.3,
    "model_version": "yolo_traffic_model_v1",
}

GOOD_SENSORS = {
    "timestamp": "2022-01-01 00:05:00",
    "sensor_id": "773869",
    "sensor_index": 42,
    "speed": 65.4,
}

GOOD_GPS = {
    "pickup_time": "2022-01-01 00:05:00",
    "dropoff_time": "2022-01-01 00:20:00",
    "pickup_zone": 142,
    "dropoff_zone": 161,
    "distance_miles": 3.14,
}

GOOD_WEATHER = {
    "city": "Los Angeles",
    "observed_at": "2026-05-11T20:00:00+00:00",
    "timestamp": time.time(),
    "temp_c": 22.5,
    "humidity": 50,
}

GOOD_PRED = {
    "sensor_id": "773869",
    "timestamp": "2022-01-01 00:05:00",
    "predicted_speed": 60.2,
    "model_version": "lgbm_speed_v1",
}


# ---- Bad payloads (must fail) ----------------------------------------------

BAD_CCTV_RAW_MISSING_FIELD = {
    "camera_id": "MVI_20011",
    # frame_id missing
    "timestamp": time.time(),
    "frame_b64": "/9j/4AAQ",
}

BAD_SENSORS_OUT_OF_RANGE = {
    "timestamp": "2022-01-01 00:05:00",
    "sensor_id": "773869",
    "sensor_index": 42,
    "speed": 9999.0,  # > 150 max
}

BAD_GPS_NEGATIVE_DISTANCE = {
    "pickup_time": "2022-01-01 00:05:00",
    "dropoff_time": "2022-01-01 00:20:00",
    "pickup_zone": 142,
    "dropoff_zone": 161,
    "distance_miles": -1.0,
}


@pytest.mark.parametrize("schema_name, payload", [
    ("cctv_raw.json", GOOD_CCTV_RAW),
    ("cctv_inferred.json", GOOD_CCTV_INFERRED),
    ("sensors.json", GOOD_SENSORS),
    ("gps.json", GOOD_GPS),
    ("weather.json", GOOD_WEATHER),
    ("speed_predictions.json", GOOD_PRED),
])
def test_valid_payloads(load_schema, schema_name, payload):
    validator = Draft202012Validator(load_schema(schema_name))
    errors = list(validator.iter_errors(payload))
    assert not errors, f"{schema_name} rejected a known-good payload: {[e.message for e in errors]}"


@pytest.mark.parametrize("schema_name, payload", [
    ("cctv_raw.json", BAD_CCTV_RAW_MISSING_FIELD),
    ("sensors.json", BAD_SENSORS_OUT_OF_RANGE),
    ("gps.json", BAD_GPS_NEGATIVE_DISTANCE),
])
def test_invalid_payloads(load_schema, schema_name, payload):
    validator = Draft202012Validator(load_schema(schema_name))
    errors = list(validator.iter_errors(payload))
    assert errors, f"{schema_name} accepted a known-bad payload"
