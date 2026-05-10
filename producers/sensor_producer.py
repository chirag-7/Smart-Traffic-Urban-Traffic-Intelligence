import json
import time
import os
import numpy as np
import h5py
from kafka import KafkaProducer
from dotenv import load_dotenv

load_dotenv()
KAFKA_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")

print("[Sensor Producer] Starting...")
print(f"[Sensor Producer] Kafka servers: {KAFKA_SERVERS}")

producer = KafkaProducer(
    bootstrap_servers=KAFKA_SERVERS,
    value_serializer=lambda v: json.dumps(v).encode('utf-8'),
    linger_ms=10,
    batch_size=16384
)

print("[Sensor Producer] Loading METR-LA data...")
with h5py.File('data/metr-la/METR-LA.h5', 'r') as f:
    # Extract data from HDF5 structure
    values = f['df/block0_values'][:]  # Shape: (34272, 207)
    timestamps = f['df/axis1'][:]      # Timestamps
    sensors = f['df/block0_items'][:]  # Sensor IDs
    
    print(f"[Sensor Producer] Loaded {values.shape} — {values.shape[0]} timestamps × {values.shape[1]} sensors")
    
    row_count = 0
    for t_idx, timestamp in enumerate(timestamps):
        for s_idx, sensor_id in enumerate(sensors):
            speed_value = values[t_idx, s_idx]
            if np.isnan(speed_value):
                continue
            # sensor_index matches METR-LA adjacency row/column order (graph vertex id as string "0".."n-1")
            message = {
                "timestamp": str(timestamp),
                "sensor_id": sensor_id.decode('utf-8') if isinstance(sensor_id, bytes) else str(sensor_id),
                "sensor_index": int(s_idx),
                "speed": float(speed_value),
            }
            producer.send('topic_sensors', value=message)
            row_count += 1
        
        # Print progress every 100 rows
        if row_count % 100 == 0:
            print(f"[Sensor Producer] Sent {row_count:,} messages...")
        
        time.sleep(0.01)

producer.flush()
print(f"[Sensor Producer] Done. Total messages: {row_count:,}")
