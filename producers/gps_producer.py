import json
import time
import os
import pandas as pd
from kafka import KafkaProducer
from dotenv import load_dotenv

load_dotenv()
KAFKA_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")

print("[GPS Producer] Starting...")
print(f"[GPS Producer] Kafka servers: {KAFKA_SERVERS}")

producer = KafkaProducer(
    bootstrap_servers=KAFKA_SERVERS,
    value_serializer=lambda v: json.dumps(v).encode('utf-8'),
    linger_ms=10,
    batch_size=16384
)

print("[GPS Producer] Loading NYC Taxi data...")
df = pd.read_parquet(
    'data/nyc-taxi/yellow_tripdata_2022-01.parquet',
    columns=['tpep_pickup_datetime', 'tpep_dropoff_datetime',
             'PULocationID', 'DOLocationID', 'trip_distance']
)
print(f"[GPS Producer] Loaded {len(df):,} trips")

row_count = 0
for _, row in df.iterrows():
    message = {
        "pickup_time": str(row['tpep_pickup_datetime']),
        "dropoff_time": str(row['tpep_dropoff_datetime']),
        "pickup_zone": int(row['PULocationID']),
        "dropoff_zone": int(row['DOLocationID']),
        "distance_miles": float(row['trip_distance'])
    }
    producer.send('topic_gps', value=message)
    row_count += 1
    
    # Print progress every 50 rows
    if row_count % 50 == 0:
        print(f"[GPS Producer] Sent {row_count:,} messages...")
    
    time.sleep(0.005)

producer.flush()
print(f"[GPS Producer] Done. Total messages: {row_count:,}")
