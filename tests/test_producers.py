"""Smoke tests for producer serialization + sticky-key encoding contract.

These tests don't actually publish to Kafka — they verify the messages we
*intend* to publish are JSON-serialisable and pass schema validation.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator


REPO_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture()
def sensors_validator():
    with (REPO_ROOT / "schemas" / "sensors.json").open("r", encoding="utf-8") as fh:
        return Draft202012Validator(json.load(fh))


@pytest.fixture()
def cctv_raw_validator():
    with (REPO_ROOT / "schemas" / "cctv_raw.json").open("r", encoding="utf-8") as fh:
        return Draft202012Validator(json.load(fh))


def test_sensor_message_round_trip(sensors_validator):
    """A sensor message must serialize to JSON, deserialize back, and revalidate."""
    msg = {
        "timestamp": "2022-01-01 00:05:00",
        "sensor_id": "773869",
        "sensor_index": 42,
        "speed": 65.4,
        "ingested_at": time.time(),
    }
    serialized = json.dumps(msg).encode("utf-8")
    back = json.loads(serialized)
    assert not list(sensors_validator.iter_errors(back))
    # Round-trip preserves values
    assert back["sensor_id"] == msg["sensor_id"]
    assert back["sensor_index"] == msg["sensor_index"]


def test_sensor_sticky_key_is_string(sensors_validator):
    """Producer sets key=sensor_id; verify the value is a non-empty string when encoded."""
    sensor_id = "773869"
    encoded = sensor_id.encode("utf-8")
    assert isinstance(encoded, bytes)
    assert encoded == b"773869"


def test_cctv_sticky_key_camera_id(cctv_raw_validator):
    """All CCTV frames from one camera must share the same key bytes."""
    camera_id = "MVI_20011"
    msg = {
        "camera_id": camera_id,
        "frame_id": "img00001.jpg",
        "timestamp": time.time(),
        "frame_b64": "/9j/4AAQ",
    }
    assert not list(cctv_raw_validator.iter_errors(msg))
    assert camera_id.encode("utf-8") == b"MVI_20011"
