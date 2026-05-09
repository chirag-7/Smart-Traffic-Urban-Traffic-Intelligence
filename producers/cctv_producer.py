import json
import time
import os
import base64
from pathlib import Path
from kafka import KafkaProducer
from dotenv import load_dotenv

load_dotenv()
KAFKA_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")

print("[CCTV Producer] Starting...")
print(f"[CCTV Producer] Kafka servers: {KAFKA_SERVERS}")

producer = KafkaProducer(
    bootstrap_servers=KAFKA_SERVERS,
    value_serializer=lambda v: json.dumps(v).encode('utf-8'),
    max_request_size=5_242_880   # 5 MB
)

YOLO_IMAGES = Path("data/ua-detrac/yolo_format/images/train")

# Check if UA-DETRAC images exist
if YOLO_IMAGES.exists():
    cameras = sorted([d for d in YOLO_IMAGES.iterdir() if d.is_dir()])[:5]
    print(f"[CCTV Producer] Found {len(cameras)} camera folders")
    
    frame_count = 0
    try:
        for cam in cameras:
            frame_files = sorted(list(cam.glob("*.jpg")))[:50]  # 50 frames per camera
            for frame_path in frame_files:
                with open(frame_path, 'rb') as f:
                    frame_b64 = base64.b64encode(f.read()).decode('utf-8')
                
                camera_id = cam.name
                message = {
                    "camera_id": camera_id,
                    "frame_id": frame_path.name,
                    "timestamp": time.time(),
                    "frame_b64": frame_b64
                }
                producer.send('topic_cctv', value=message)
                frame_count += 1
                
                if frame_count % 10 == 0:
                    print(f"[CCTV Producer] Sent {frame_count} frames...")
                
                time.sleep(0.04)  # 25fps
        
        producer.flush()
        print(f"[CCTV Producer] Done. Total frames: {frame_count}")
    except Exception as e:
        print(f"[CCTV Producer] Error: {e}")
else:
    print(f"[CCTV Producer] ⚠️  UA-DETRAC data not found at {YOLO_IMAGES}")
    print("[CCTV Producer] Please download and extract UA-DETRAC first")
    print("[CCTV Producer] For now, skipping CCTV producer")
