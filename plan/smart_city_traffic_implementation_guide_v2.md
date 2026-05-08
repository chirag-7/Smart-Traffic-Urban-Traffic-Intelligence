# Smart City Traffic Intelligence System
## Complete Implementation Guide v2 — Windows + Docker Desktop

> **Stack:** Windows + Docker Desktop | Python 3.10 | Intermediate Level  
> **v2 fixes:** Java setup moved to prerequisites, Kafka Docker networking fixed, UA-DETRAC conversion script added, `.env` file added, `requirements.txt` added, congestion-triggered rerouting added, version matrix clarified.

---

## Table of Contents

1. [Prerequisites & Environment Setup](#1-prerequisites--environment-setup)
2. [Project Folder Structure](#2-project-folder-structure)
3. [Configuration Files](#3-configuration-files)
4. [Sprint 1 — Infrastructure & Data Preparation](#4-sprint-1--infrastructure--data-preparation)
5. [Sprint 2 — Modeling & Analytics](#5-sprint-2--modeling--analytics)
6. [Sprint 3 — Integration & UDFs](#6-sprint-3--integration--udfs)
7. [Sprint 4 — UI & Final Polish](#7-sprint-4--ui--final-polish)
8. [Running the Full System](#8-running-the-full-system)
9. [Troubleshooting](#9-troubleshooting)
10. [Sprint Checklists](#10-sprint-checklists)

---

## 1. Prerequisites & Environment Setup

### 1.1 Install Required Tools (in this exact order)

| Tool | Version | Why | Download |
|------|---------|-----|----------|
| **Java 11 (JDK)** | 11 LTS | PySpark requires Java — install FIRST | https://adoptium.net |
| Docker Desktop | Latest | Runs Kafka + Spark containers | https://www.docker.com/products/docker-desktop |
| Python | **3.10.x** | Must be 3.10 — PySpark 3.5 + Delta have issues on 3.11/3.12 | https://www.python.org/downloads/ |
| Git | Latest | Version control | https://git-scm.com/download/win |
| VS Code | Latest | Editor | https://code.visualstudio.com/ |

> ⚠️ **Java first, always.** PySpark will silently fail with cryptic errors if Java isn't found. Install it before Python packages.

---

### 1.2 Set JAVA_HOME After Installing Java

After installing Java 11 from Adoptium, open PowerShell **as Administrator** and run:

```powershell
# Find your Java install path (look for jdk-11.x.x folder)
ls "C:\Program Files\Eclipse Adoptium\"

# Set JAVA_HOME — replace x.x with your actual version numbers
[System.Environment]::SetEnvironmentVariable("JAVA_HOME", "C:\Program Files\Eclipse Adoptium\jdk-11.0.23.9-hotspot", "Machine")

# Also add Java bin to PATH
$oldPath = [System.Environment]::GetEnvironmentVariable("Path", "Machine")
[System.Environment]::SetEnvironmentVariable("Path", "$oldPath;C:\Program Files\Eclipse Adoptium\jdk-11.0.23.9-hotspot\bin", "Machine")
```

**Close and reopen PowerShell**, then verify:

```powershell
java -version
# Expected: openjdk version "11.0.x" ...
```

---

### 1.3 Docker Desktop Settings

Open Docker Desktop → Settings → Resources and set:

- **Memory:** 8 GB minimum (10 GB recommended)
- **CPUs:** 4 minimum
- **Disk image size:** 60 GB (datasets are large)

Verify Docker works:

```powershell
docker --version
docker compose version
docker run hello-world
```

---

### 1.4 Create Project and Virtual Environment

```powershell
# Create project folder on a drive with enough space (datasets = ~8GB)
mkdir C:\Projects\smart-traffic
cd C:\Projects\smart-traffic

# Create virtual environment with Python 3.10 specifically
py -3.10 -m venv venv

# Activate
.\venv\Scripts\Activate.ps1

# If blocked by execution policy:
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
# Then try activating again
```

---

### 1.5 Install Python Dependencies (pinned versions — do not change)

> ⚠️ **Version matrix matters.** PySpark 3.5, Delta Lake 2.4, and their Scala JARs must match exactly. Wrong versions cause silent crashes at runtime.

```powershell
pip install `
  pyspark==3.5.0 `
  delta-spark==2.4.0 `
  kafka-python==2.0.2 `
  ultralytics==8.0.200 `
  streamlit==1.29.0 `
  pandas==2.1.3 `
  pyarrow==14.0.1 `
  numpy==1.26.2 `
  h5py==3.10.0 `
  pillow==10.1.0 `
  pyyaml==6.0.1 `
  python-dotenv==1.0.0 `
  requests==2.31.0
```

Save these as `requirements.txt` (see Section 3).

---

## 2. Project Folder Structure

```
smart-traffic/
│
├── .env                          # ← API keys and config (never commit this)
├── .gitignore                    # ← Excludes .env, data/, venv/
├── requirements.txt              # ← Pinned dependencies
├── docker-compose.yml            # ← All services
│
├── data/                         # ← Raw datasets (add to .gitignore)
│   ├── metr-la/
│   │   ├── metr-la.h5
│   │   └── adj_mat.npy
│   ├── nyc-taxi/
│   │   └── yellow_tripdata_2022-01.parquet
│   └── ua-detrac/
│       ├── DETRAC-train-data/    # Raw images
│       └── yolo_format/          # ← Converted by prep script
│           ├── images/
│           │   ├── train/
│           │   └── val/
│           ├── labels/
│           │   ├── train/
│           │   └── val/
│           └── dataset.yaml
│
├── producers/
│   ├── sensor_producer.py
│   ├── gps_producer.py
│   └── cctv_producer.py
│
├── spark_jobs/
│   ├── stream_processor.py       # Main streaming pipeline
│   ├── graph_analytics.py        # GraphFrames + PageRank
│   └── ml_predictor.py           # GBT model training
│
├── scripts/
│   └── convert_detrac_to_yolo.py # ← NEW: dataset conversion
│
├── models/
│   └── (YOLOv8 weights go here after training)
│
├── delta_tables/                 # Auto-created at runtime
│
└── dashboard/
    └── app.py
```

Create all folders at once:

```powershell
mkdir data\metr-la, data\nyc-taxi, data\ua-detrac\yolo_format\images\train, `
      data\ua-detrac\yolo_format\images\val, `
      data\ua-detrac\yolo_format\labels\train, `
      data\ua-detrac\yolo_format\labels\val, `
      producers, spark_jobs, scripts, models, delta_tables, dashboard
```

---

## 3. Configuration Files

### 3.1 `.env` — All Secrets and Paths in One Place

Create `.env` in your project root:

```env
# OpenWeatherMap
OPENWEATHER_API_KEY=your_api_key_here
OPENWEATHER_CITY=Los Angeles

# Kafka
KAFKA_BOOTSTRAP_SERVERS=localhost:9092

# Paths (use forward slashes even on Windows)
DATA_DIR=./data
DELTA_DIR=./delta_tables
MODELS_DIR=./models

# Spark tuning
SPARK_DRIVER_MEMORY=4g
SPARK_EXECUTOR_MEMORY=4g
```

> 🔒 **Never commit `.env` to Git.** It contains your API key.

---

### 3.2 `.gitignore`

```gitignore
# Environment
.env
venv/

# Data (too large for Git)
data/
delta_tables/

# Python
__pycache__/
*.pyc
*.pyo

# Models (large binary files)
models/*.pt
models/gbt_traffic_model/

# Jupyter
.ipynb_checkpoints/

# OS
.DS_Store
Thumbs.db
```

---

### 3.3 `requirements.txt`

```txt
pyspark==3.5.0
delta-spark==2.4.0
kafka-python==2.0.2
ultralytics==8.0.200
streamlit==1.29.0
pandas==2.1.3
pyarrow==14.0.1
numpy==1.26.2
h5py==3.10.0
pillow==10.1.0
pyyaml==6.0.1
python-dotenv==1.0.0
requests==2.31.0
```

Install from it anytime:

```powershell
pip install -r requirements.txt
```

---

### 3.4 `docker-compose.yml`

> ⚠️ **Critical fix from v1:** Kafka needs TWO listener configs — one for containers talking to each other (internal, using the Docker service name `kafka:29092`) and one for your host machine (`localhost:9092`). Without this, Spark containers cannot reach Kafka.

```yaml
version: '3.8'

services:

  zookeeper:
    image: confluentinc/cp-zookeeper:7.5.0
    container_name: zookeeper
    environment:
      ZOOKEEPER_CLIENT_PORT: 2181
      ZOOKEEPER_TICK_TIME: 2000
    ports:
      - "2181:2181"
    healthcheck:
      test: ["CMD", "nc", "-z", "localhost", "2181"]
      interval: 10s
      timeout: 5s
      retries: 5

  kafka:
    image: confluentinc/cp-kafka:7.5.0
    container_name: kafka
    depends_on:
      zookeeper:
        condition: service_healthy
    ports:
      - "9092:9092"     # Host access (your Python producers)
      - "29092:29092"   # Internal Docker access (Spark containers)
    environment:
      KAFKA_BROKER_ID: 1
      KAFKA_ZOOKEEPER_CONNECT: zookeeper:2181
      # Two listeners: PLAINTEXT for host, PLAINTEXT_INTERNAL for containers
      KAFKA_LISTENER_SECURITY_PROTOCOL_MAP: PLAINTEXT:PLAINTEXT,PLAINTEXT_INTERNAL:PLAINTEXT
      KAFKA_ADVERTISED_LISTENERS: PLAINTEXT://localhost:9092,PLAINTEXT_INTERNAL://kafka:29092
      KAFKA_LISTENERS: PLAINTEXT://0.0.0.0:9092,PLAINTEXT_INTERNAL://0.0.0.0:29092
      KAFKA_INTER_BROKER_LISTENER_NAME: PLAINTEXT_INTERNAL
      KAFKA_OFFSETS_TOPIC_REPLICATION_FACTOR: 1
      KAFKA_AUTO_CREATE_TOPICS_ENABLE: "true"
    healthcheck:
      test: ["CMD", "kafka-topics", "--bootstrap-server", "localhost:9092", "--list"]
      interval: 15s
      timeout: 10s
      retries: 10

  spark-master:
    image: bitnami/spark:3.5.0
    container_name: spark-master
    environment:
      - SPARK_MODE=master
      - SPARK_RPC_AUTHENTICATION_ENABLED=no
      - SPARK_RPC_ENCRYPTION_ENABLED=no
    ports:
      - "8080:8080"
      - "7077:7077"
    volumes:
      - ./spark_jobs:/opt/spark_jobs
      - ./delta_tables:/opt/delta_tables
      - ./models:/opt/models
      - ./data:/opt/data
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8080"]
      interval: 15s
      timeout: 5s
      retries: 5

  spark-worker:
    image: bitnami/spark:3.5.0
    container_name: spark-worker
    depends_on:
      spark-master:
        condition: service_healthy
    environment:
      - SPARK_MODE=worker
      - SPARK_MASTER_URL=spark://spark-master:7077
      - SPARK_WORKER_MEMORY=4G
      - SPARK_WORKER_CORES=2
    volumes:
      - ./spark_jobs:/opt/spark_jobs
      - ./delta_tables:/opt/delta_tables
      - ./models:/opt/models
      - ./data:/opt/data
```

---

## 4. Sprint 1 — Infrastructure & Data Preparation

### 4.1 Start Docker Services

```powershell
cd C:\Projects\smart-traffic
docker compose up -d

# Wait ~30 seconds, then verify all 4 containers are healthy
docker compose ps
```

Expected output — all show `healthy` or `running`:

```
NAME           STATUS
zookeeper      running (healthy)
kafka          running (healthy)
spark-master   running (healthy)
spark-worker   running
```

Check Spark UI: http://localhost:8080 — you should see 1 worker registered.

---

### 4.2 Dataset Downloads

#### Dataset 1: METR-LA

```python
# scripts/download_metrla.py
import urllib.request, zipfile, os

os.makedirs("data/metr-la", exist_ok=True)

print("Downloading METR-LA (~50MB)...")
urllib.request.urlretrieve(
    "https://zenodo.org/record/5724774/files/METR-LA.zip",
    "data/metr-la/METR-LA.zip"
)
with zipfile.ZipFile("data/metr-la/METR-LA.zip", 'r') as z:
    z.extractall("data/metr-la/")
print("Done. Files:", os.listdir("data/metr-la/"))
```

Run it:

```powershell
python scripts/download_metrla.py
```

You should see `metr-la.h5` and `adj_mat.npy` inside `data/metr-la/`.

---

#### Dataset 2: NYC TLC Taxi (1 month — ~400MB)

```python
# scripts/download_nyctaxi.py
import urllib.request, os

os.makedirs("data/nyc-taxi", exist_ok=True)

print("Downloading NYC Taxi Jan 2022 (~400MB)...")
urllib.request.urlretrieve(
    "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2022-01.parquet",
    "data/nyc-taxi/yellow_tripdata_2022-01.parquet"
)
print("Done.")
```

```powershell
python scripts/download_nyctaxi.py
```

> 💡 The proposal mentions 396 million rows over 5 years. For a local demo, 1 month (~3 million rows) is enough to demonstrate Spark's distributed joins at scale.

---

#### Dataset 3: UA-DETRAC (Manual Download — ~6.7 GB)

This requires free registration, so it must be done manually:

```
1. Go to: https://detrac-db.rit.albany.edu/
2. Register for a free account and verify your email
3. Download these two files:
   - DETRAC-train-data.zip       (~6.7 GB) → save to data/ua-detrac/
   - DETRAC-Train-Annotations-XML.zip → save to data/ua-detrac/
4. Extract both:
```

```powershell
# In PowerShell, after downloading
Expand-Archive -Path "data\ua-detrac\DETRAC-train-data.zip" -DestinationPath "data\ua-detrac\DETRAC-train-data"
Expand-Archive -Path "data\ua-detrac\DETRAC-Train-Annotations-XML.zip" -DestinationPath "data\ua-detrac\DETRAC-Train-Annotations-XML"
```

---

#### Dataset 4: OpenWeatherMap API Key

```
1. Go to: https://openweathermap.org/api
2. Sign up (free tier is enough)
3. Navigate to: Profile → My API Keys
4. Copy your key
5. Paste it into .env:  OPENWEATHER_API_KEY=your_key_here
```

---

### 4.3 Convert UA-DETRAC to YOLO Format

> ⚠️ **This step was missing in v1.** UA-DETRAC ships with XML annotations in its own format. YOLOv8 requires a specific folder structure with `.txt` label files. This script does the conversion.

Create `scripts/convert_detrac_to_yolo.py`:

```python
"""
Converts UA-DETRAC XML annotations to YOLOv8 format.

UA-DETRAC class mapping:
  car=0, bus=1, van=2, others=3

Output structure:
  data/ua-detrac/yolo_format/
    images/train/ + images/val/
    labels/train/ + labels/val/
    dataset.yaml
"""

import os
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path
import random

# Paths
DETRAC_IMAGES = Path("data/ua-detrac/DETRAC-train-data/Insight-MVT_Annotation_Train")
DETRAC_ANNOTATIONS = Path("data/ua-detrac/DETRAC-Train-Annotations-XML/DETRAC-Train-Annotations-XML")
YOLO_OUT = Path("data/ua-detrac/yolo_format")
VAL_SPLIT = 0.2  # 20% validation
MAX_SEQUENCES = 10  # Use 10 camera sequences for local demo (full = 60)

CLASS_MAP = {"car": 0, "bus": 1, "van": 2, "others": 3}

def convert_bbox(img_w, img_h, left, top, width, height):
    """Convert absolute bbox to YOLO normalized xywh format."""
    x_center = (left + width / 2) / img_w
    y_center = (top + height / 2) / img_h
    w_norm = width / img_w
    h_norm = height / img_h
    return x_center, y_center, w_norm, h_norm

def process_sequence(seq_folder, xml_file, split):
    """Process one camera sequence."""
    if not xml_file.exists():
        return 0

    tree = ET.parse(xml_file)
    root = tree.getroot()

    # Get image dimensions from first frame
    img_width = int(root.attrib.get('width', 960))
    img_height = int(root.attrib.get('height', 540))

    frames_processed = 0
    for frame in root.findall('frame'):
        frame_num = int(frame.attrib['num'])
        frame_filename = f"img{frame_num:05d}.jpg"
        src_img = seq_folder / frame_filename

        if not src_img.exists():
            continue

        # Copy image
        dst_img = YOLO_OUT / "images" / split / f"{seq_folder.name}_{frame_filename}"
        shutil.copy2(src_img, dst_img)

        # Write YOLO label file
        dst_label = YOLO_OUT / "labels" / split / f"{seq_folder.name}_{frame_filename.replace('.jpg', '.txt')}"
        label_lines = []

        target_list = frame.find('target_list')
        if target_list is not None:
            for target in target_list.findall('target'):
                box = target.find('box')
                attr = target.find('attribute')
                if box is None or attr is None:
                    continue

                vehicle_type = attr.attrib.get('vehicle_type', 'others').lower()
                class_id = CLASS_MAP.get(vehicle_type, 3)

                left = float(box.attrib['left'])
                top = float(box.attrib['top'])
                width = float(box.attrib['width'])
                height = float(box.attrib['height'])

                x, y, w, h = convert_bbox(img_width, img_height, left, top, width, height)
                # Clamp to [0, 1]
                x, y, w, h = [max(0, min(1, v)) for v in [x, y, w, h]]
                label_lines.append(f"{class_id} {x:.6f} {y:.6f} {w:.6f} {h:.6f}")

        with open(dst_label, 'w') as f:
            f.write('\n'.join(label_lines))

        frames_processed += 1

    return frames_processed

def main():
    # Create output dirs
    for split in ['train', 'val']:
        (YOLO_OUT / 'images' / split).mkdir(parents=True, exist_ok=True)
        (YOLO_OUT / 'labels' / split).mkdir(parents=True, exist_ok=True)

    # Collect all sequences
    sequences = sorted([d for d in DETRAC_IMAGES.iterdir() if d.is_dir()])[:MAX_SEQUENCES]
    random.shuffle(sequences)
    val_count = max(1, int(len(sequences) * VAL_SPLIT))
    val_seqs = set(s.name for s in sequences[:val_count])

    total_frames = 0
    for seq in sequences:
        xml_file = DETRAC_ANNOTATIONS / f"{seq.name}_v3.xml"
        split = 'val' if seq.name in val_seqs else 'train'
        n = process_sequence(seq, xml_file, split)
        print(f"  [{split}] {seq.name}: {n} frames")
        total_frames += n

    # Write dataset.yaml
    yaml_content = f"""path: {YOLO_OUT.resolve().as_posix()}
train: images/train
val: images/val
nc: 4
names: ['car', 'bus', 'van', 'others']
"""
    with open(YOLO_OUT / 'dataset.yaml', 'w') as f:
        f.write(yaml_content)

    print(f"\nDone. Total frames converted: {total_frames}")
    print(f"Dataset YAML written to: {YOLO_OUT / 'dataset.yaml'}")

if __name__ == "__main__":
    main()
```

Run it:

```powershell
python scripts/convert_detrac_to_yolo.py
```

This will take 5–10 minutes and process ~10,000 frames from 10 camera sequences.

---

### 4.4 Create Kafka Topics

```powershell
# Wait for Kafka to be fully healthy before running this
docker exec -it kafka kafka-topics --create --topic topic_sensors --bootstrap-server localhost:9092 --partitions 3 --replication-factor 1

docker exec -it kafka kafka-topics --create --topic topic_gps --bootstrap-server localhost:9092 --partitions 3 --replication-factor 1

docker exec -it kafka kafka-topics --create --topic topic_cctv --bootstrap-server localhost:9092 --partitions 3 --replication-factor 1

# Verify all 3 topics exist
docker exec -it kafka kafka-topics --list --bootstrap-server localhost:9092
```

---

### 4.5 Kafka Producers

All producers load config from `.env` using `python-dotenv`.

#### `producers/sensor_producer.py`

```python
import json, time, os
import pandas as pd
import numpy as np
from kafka import KafkaProducer
from dotenv import load_dotenv

load_dotenv()
KAFKA_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")

producer = KafkaProducer(
    bootstrap_servers=KAFKA_SERVERS,
    value_serializer=lambda v: json.dumps(v).encode('utf-8'),
    linger_ms=10,          # Batch messages for efficiency
    batch_size=16384
)

df = pd.read_hdf('data/metr-la/metr-la.h5', key='df')
print(f"[Sensor Producer] Loaded {df.shape} — streaming to Kafka...")

for timestamp, row in df.iterrows():
    for sensor_id, speed_value in row.items():
        if pd.isna(speed_value):
            continue
        message = {
            "timestamp": str(timestamp),
            "sensor_id": str(sensor_id),
            "speed": float(speed_value)
        }
        producer.send('topic_sensors', value=message)
    time.sleep(0.1)

producer.flush()
print("[Sensor Producer] Done.")
```

#### `producers/gps_producer.py`

```python
import json, time, os
import pandas as pd
from kafka import KafkaProducer
from dotenv import load_dotenv

load_dotenv()
KAFKA_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")

producer = KafkaProducer(
    bootstrap_servers=KAFKA_SERVERS,
    value_serializer=lambda v: json.dumps(v).encode('utf-8'),
    linger_ms=10,
    batch_size=16384
)

df = pd.read_parquet(
    'data/nyc-taxi/yellow_tripdata_2022-01.parquet',
    columns=['tpep_pickup_datetime', 'tpep_dropoff_datetime',
             'PULocationID', 'DOLocationID', 'trip_distance']
)
print(f"[GPS Producer] Loaded {len(df):,} trips — streaming to Kafka...")

for _, row in df.iterrows():
    message = {
        "pickup_time":  str(row['tpep_pickup_datetime']),
        "dropoff_time": str(row['tpep_dropoff_datetime']),
        "pickup_zone":  int(row['PULocationID']),
        "dropoff_zone": int(row['DOLocationID']),
        "distance_miles": float(row['trip_distance'])
    }
    producer.send('topic_gps', value=message)
    time.sleep(0.005)

producer.flush()
print("[GPS Producer] Done.")
```

#### `producers/cctv_producer.py`

```python
import json, time, os, base64
from pathlib import Path
from kafka import KafkaProducer
from dotenv import load_dotenv

load_dotenv()
KAFKA_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")

producer = KafkaProducer(
    bootstrap_servers=KAFKA_SERVERS,
    value_serializer=lambda v: json.dumps(v).encode('utf-8'),
    max_request_size=5_242_880   # 5 MB — frames are large
)

YOLO_IMAGES = Path("data/ua-detrac/yolo_format/images/train")
cameras = sorted(YOLO_IMAGES.iterdir())[:5]  # 5 cameras for demo

print(f"[CCTV Producer] Streaming from {len(cameras)} cameras...")

for frame_path in (f for cam in cameras for f in sorted(cam.parent.glob(f"{cam.name.split('_')[0]}*"))[:50]):
    with open(frame_path, 'rb') as f:
        frame_b64 = base64.b64encode(f.read()).decode('utf-8')

    # Camera ID = first part of filename before underscore
    camera_id = frame_path.stem.split('_img')[0]

    message = {
        "camera_id": camera_id,
        "frame_id": frame_path.name,
        "timestamp": time.time(),
        "frame_b64": frame_b64
    }
    producer.send('topic_cctv', value=message)
    time.sleep(0.04)  # Simulate 25fps

producer.flush()
print("[CCTV Producer] Done.")
```

---

## 5. Sprint 2 — Modeling & Analytics

### 5.1 Fine-Tune YOLOv8 on UA-DETRAC

```python
# spark_jobs/train_yolo.py
import os
from ultralytics import YOLO
from dotenv import load_dotenv

load_dotenv()
MODELS_DIR = os.getenv("MODELS_DIR", "./models")
DATA_YAML = "data/ua-detrac/yolo_format/dataset.yaml"

# Start from pretrained YOLOv8 nano weights (auto-downloads ~6MB)
model = YOLO('yolov8n.pt')

print("[YOLOv8] Starting fine-tuning on UA-DETRAC...")
results = model.train(
    data=DATA_YAML,
    epochs=10,          # Increase to 50 for better accuracy
    imgsz=640,
    batch=8,            # Lower to 4 if you run out of RAM
    device='cpu',       # Change to device=0 if you have an NVIDIA GPU
    project=MODELS_DIR,
    name='yolov8_traffic',
    exist_ok=True,
    patience=5          # Stop early if no improvement for 5 epochs
)

best_weights = f"{MODELS_DIR}/yolov8_traffic/weights/best.pt"
print(f"[YOLOv8] Training done. Best weights: {best_weights}")
print(f"[YOLOv8] mAP50: {results.results_dict.get('metrics/mAP50(B)', 'N/A')}")
```

```powershell
python spark_jobs/train_yolo.py
```

> ⏱️ Estimated time on CPU: 2–4 hours for 10 epochs on 10,000 frames. Run overnight or use `epochs=3` for a quick smoke test.

---

### 5.2 Build the GraphFrames Road Network

Download the GraphFrames JAR first:

```powershell
# Create jars directory and download the JAR
mkdir spark_jobs\jars
```

```python
# scripts/download_graphframes_jar.py
import urllib.request, os

os.makedirs("spark_jobs/jars", exist_ok=True)
url = "https://repos.spark-packages.org/graphframes/graphframes/0.8.3-spark3.5-s_2.12/graphframes-0.8.3-spark3.5-s_2.12.jar"
print("Downloading GraphFrames JAR...")
urllib.request.urlretrieve(url, "spark_jobs/jars/graphframes-0.8.3-spark3.5-s_2.12.jar")
print("Done.")
```

```powershell
python scripts/download_graphframes_jar.py
```

#### `spark_jobs/graph_analytics.py`

```python
import os
import numpy as np
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from dotenv import load_dotenv

load_dotenv()

spark = SparkSession.builder \
    .appName("TrafficGraphAnalytics") \
    .config("spark.jars", "spark_jobs/jars/graphframes-0.8.3-spark3.5-s_2.12.jar") \
    .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension") \
    .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog") \
    .config("spark.driver.memory", os.getenv("SPARK_DRIVER_MEMORY", "4g")) \
    .getOrCreate()

spark.sparkContext.setLogLevel("ERROR")

# Load METR-LA adjacency matrix
adj_matrix = np.load('data/metr-la/adj_mat.npy')   # Shape: (207, 207)
n_sensors = adj_matrix.shape[0]

# Build graph vertices (road sensors)
vertices_data = [{"id": str(i), "sensor_id": i} for i in range(n_sensors)]
vertices_df = spark.createDataFrame(vertices_data)

# Build graph edges (connections between adjacent sensors)
edges_data = [
    {"src": str(i), "dst": str(j), "weight": float(adj_matrix[i][j])}
    for i in range(n_sensors)
    for j in range(n_sensors)
    if adj_matrix[i][j] > 0 and i != j
]
edges_df = spark.createDataFrame(edges_data)

print(f"[Graph] Vertices: {vertices_df.count()}, Edges: {len(edges_data)}")

from graphframes import GraphFrame
g = GraphFrame(vertices_df, edges_df)

# PageRank — find most critical road nodes
pr = g.pageRank(resetProbability=0.15, maxIter=10)
print("[Graph] Top 10 most critical nodes (PageRank):")
pr.vertices.orderBy(col("pagerank").desc()).limit(10).show()

# Shortest Paths from all nodes to landmark sensors 0, 50, 100
sp = g.shortestPaths(landmarks=["0", "50", "100"])

# Save to Delta Lake
pr.vertices.write.format("delta").mode("overwrite").save("delta_tables/graph_pagerank")
sp.select("id", "distances").write.format("delta").mode("overwrite").save("delta_tables/graph_shortest_paths")

print("[Graph] Results saved to Delta Lake.")
spark.stop()
```

---

### 5.3 Train the Spark MLlib Speed Prediction Model

#### `spark_jobs/ml_predictor.py`

```python
import os
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, hour, dayofweek, unix_timestamp
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.regression import GBTRegressor
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml import Pipeline
from dotenv import load_dotenv

load_dotenv()

spark = SparkSession.builder \
    .appName("TrafficSpeedPredictor") \
    .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension") \
    .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog") \
    .config("spark.driver.memory", os.getenv("SPARK_DRIVER_MEMORY", "4g")) \
    .getOrCreate()

spark.sparkContext.setLogLevel("ERROR")

# Load METR-LA and reshape to long format
print("[ML] Loading METR-LA dataset...")
metr_pd = pd.read_hdf('data/metr-la/metr-la.h5', key='df')
metr_long = (
    metr_pd.reset_index()
    .melt(id_vars='index', var_name='sensor_id', value_name='speed')
    .rename(columns={'index': 'timestamp'})
    .dropna()
)
metr_long['timestamp'] = pd.to_datetime(metr_long['timestamp'])

df = spark.createDataFrame(metr_long)
print(f"[ML] Loaded {df.count():,} records across 207 sensors.")

# Feature Engineering
df = df \
    .withColumn("hour",        hour(col("timestamp"))) \
    .withColumn("day_of_week", dayofweek(col("timestamp"))) \
    .withColumn("unix_ts",     unix_timestamp(col("timestamp"))) \
    .filter(col("speed") > 0)   # Remove zero-speed noise

feature_cols = ["hour", "day_of_week", "unix_ts"]

assembler = VectorAssembler(inputCols=feature_cols, outputCol="raw_features")
scaler    = StandardScaler(inputCol="raw_features", outputCol="features",
                           withStd=True, withMean=True)

gbt = GBTRegressor(
    labelCol="speed",
    featuresCol="features",
    maxIter=50,
    maxDepth=5,
    stepSize=0.1,
    subsamplingRate=0.8
)

pipeline = Pipeline(stages=[assembler, scaler, gbt])

train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)
print(f"[ML] Training on {train_df.count():,} rows...")

model = pipeline.fit(train_df)

predictions = model.transform(test_df)
evaluator = RegressionEvaluator(labelCol="speed", predictionCol="prediction", metricName="rmse")
rmse = evaluator.evaluate(predictions)
r2_eval = RegressionEvaluator(labelCol="speed", predictionCol="prediction", metricName="r2")
r2 = r2_eval.evaluate(predictions)

print(f"[ML] RMSE: {rmse:.4f} mph    R²: {r2:.4f}")

model.save("models/gbt_traffic_model")

predictions.select("timestamp", "sensor_id", "speed", "prediction") \
    .write.format("delta").mode("overwrite").save("delta_tables/speed_predictions")

print("[ML] Model and predictions saved.")
spark.stop()
```

---

## 6. Sprint 3 — Integration & UDFs

### 6.1 Main Streaming Pipeline with Congestion-Triggered Rerouting

> ⚠️ **Critical addition from v1:** The proposal requires the system to detect congestion and trigger shortest-path rerouting. This is now implemented as a `foreachBatch` trigger in the sensor stream.

#### `spark_jobs/stream_processor.py`

```python
import os, base64, io
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    from_json, col, pandas_udf, avg, lit
)
from pyspark.sql.types import (
    StructType, StructField, StringType, FloatType, IntegerType
)
from dotenv import load_dotenv

load_dotenv()

# ---- Kafka addresses:
# Producers (running on host) → localhost:9092
# Spark containers → kafka:29092  (internal Docker network)
KAFKA_HOST  = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")

# When this script runs INSIDE Docker, use internal address
# When running on HOST (local Python), use localhost
# Detect automatically:
import socket
try:
    socket.getaddrinfo("kafka", 29092)
    KAFKA_SPARK = "kafka:29092"   # Inside Docker
except Exception:
    KAFKA_SPARK = "localhost:9092"  # Running locally

print(f"[Config] Kafka address for Spark: {KAFKA_SPARK}")

# ---- Spark Session ----
spark = SparkSession.builder \
    .appName("SmartCityTrafficStreaming") \
    .config("spark.jars.packages",
            "org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.0,"
            "io.delta:delta-core_2.12:2.4.0") \
    .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension") \
    .config("spark.sql.catalog.spark_catalog",
            "org.apache.spark.sql.delta.catalog.DeltaCatalog") \
    .config("spark.sql.shuffle.partitions", "4") \
    .config("spark.driver.memory", os.getenv("SPARK_DRIVER_MEMORY", "4g")) \
    .getOrCreate()

spark.sparkContext.setLogLevel("ERROR")

CONGESTION_THRESHOLD_MPH = 20.0   # Below this = congested

# ===========================================================================
# BRANCH A: COMPUTER VISION — YOLOv8 via Pandas UDF
# ===========================================================================

cctv_schema = StructType([
    StructField("camera_id",  StringType()),
    StructField("frame_id",   StringType()),
    StructField("timestamp",  FloatType()),
    StructField("frame_b64",  StringType())
])

@pandas_udf("integer")
def yolo_vehicle_count(frame_b64_series: pd.Series) -> pd.Series:
    """
    Runs YOLOv8 inference on base64 encoded frames.
    Pandas UDF runs distributed across Spark executors.
    Model is loaded once per executor process (cached via module-level global).
    """
    from ultralytics import YOLO
    from PIL import Image
    import numpy as np

    # Module-level cache — model loads only once per executor JVM process
    import builtins
    if not hasattr(builtins, '_yolo_model'):
        builtins._yolo_model = YOLO("/opt/models/yolov8_traffic/weights/best.pt")

    model = builtins._yolo_model
    counts = []

    for frame_b64 in frame_b64_series:
        try:
            img_bytes = base64.b64decode(frame_b64)
            img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            results = model(img, verbose=False, conf=0.3)
            counts.append(len(results[0].boxes))
        except Exception:
            counts.append(0)

    return pd.Series(counts)

cctv_stream = (
    spark.readStream
    .format("kafka")
    .option("kafka.bootstrap.servers", KAFKA_SPARK)
    .option("subscribe", "topic_cctv")
    .option("startingOffsets", "latest")
    .option("maxOffsetsPerTrigger", 50)   # Limit to 50 frames per micro-batch
    .load()
    .select(from_json(col("value").cast("string"), cctv_schema).alias("d"))
    .select("d.*")
    .withColumn("vehicle_count", yolo_vehicle_count(col("frame_b64")))
    .drop("frame_b64")
)

cctv_query = (
    cctv_stream.writeStream
    .format("delta")
    .option("checkpointLocation", "delta_tables/checkpoints/cctv")
    .outputMode("append")
    .trigger(processingTime="5 seconds")
    .start("delta_tables/cv_vehicle_counts")
)

# ===========================================================================
# BRANCH B: SENSOR SPEEDS — with congestion detection trigger
# ===========================================================================

sensor_schema = StructType([
    StructField("timestamp", StringType()),
    StructField("sensor_id", StringType()),
    StructField("speed",     FloatType())
])

def detect_congestion_and_reroute(batch_df, batch_id):
    """
    Called for every micro-batch of sensor data.
    If any sensor reports speed below threshold, logs a rerouting alert
    and reads the pre-computed shortest paths from Delta Lake.
    """
    if batch_df.count() == 0:
        return

    # Write raw speeds to Delta
    batch_df.write.format("delta").mode("append").save("delta_tables/sensor_speeds")

    # Detect congested sensors
    congested = batch_df.filter(col("speed") < lit(CONGESTION_THRESHOLD_MPH))

    if congested.count() > 0:
        congested_sensors = [r['sensor_id'] for r in congested.collect()]
        print(f"[CONGESTION ALERT] Batch {batch_id} — Congested sensors: {congested_sensors}")

        # Load pre-computed shortest paths (from graph_analytics.py)
        try:
            sp_df = spark.read.format("delta").load("delta_tables/graph_shortest_paths")
            # Find alternative routes FROM congested sensors
            reroutes = sp_df.filter(col("id").isin(congested_sensors))

            # Write rerouting suggestions to Delta
            reroutes.withColumn("batch_id", lit(batch_id)) \
                    .write.format("delta").mode("append") \
                    .save("delta_tables/rerouting_alerts")

            print(f"[REROUTING] Alternatives written for {reroutes.count()} sensors.")
        except Exception as e:
            print(f"[REROUTING] Shortest paths not yet available: {e}")

sensor_stream = (
    spark.readStream
    .format("kafka")
    .option("kafka.bootstrap.servers", KAFKA_SPARK)
    .option("subscribe", "topic_sensors")
    .option("startingOffsets", "latest")
    .load()
    .select(from_json(col("value").cast("string"), sensor_schema).alias("d"))
    .select("d.*")
)

sensor_query = (
    sensor_stream.writeStream
    .foreachBatch(detect_congestion_and_reroute)
    .option("checkpointLocation", "delta_tables/checkpoints/sensors")
    .trigger(processingTime="5 seconds")
    .start()
)

# ===========================================================================
# BRANCH C: GPS DEMAND STREAM
# ===========================================================================

gps_schema = StructType([
    StructField("pickup_time",    StringType()),
    StructField("dropoff_time",   StringType()),
    StructField("pickup_zone",    IntegerType()),
    StructField("dropoff_zone",   IntegerType()),
    StructField("distance_miles", FloatType())
])

gps_stream = (
    spark.readStream
    .format("kafka")
    .option("kafka.bootstrap.servers", KAFKA_SPARK)
    .option("subscribe", "topic_gps")
    .option("startingOffsets", "latest")
    .load()
    .select(from_json(col("value").cast("string"), gps_schema).alias("d"))
    .select("d.*")
)

gps_query = (
    gps_stream.writeStream
    .format("delta")
    .option("checkpointLocation", "delta_tables/checkpoints/gps")
    .outputMode("append")
    .trigger(processingTime="5 seconds")
    .start("delta_tables/gps_trips")
)

# ===========================================================================
print("[Stream Processor] All 3 streams active:")
print("  Branch A: CCTV → YOLOv8 UDF → delta_tables/cv_vehicle_counts")
print("  Branch B: Sensors → congestion detection → delta_tables/sensor_speeds + rerouting_alerts")
print("  Branch C: GPS → delta_tables/gps_trips")
print("Press Ctrl+C to stop.\n")

spark.streams.awaitAnyTermination()
```

---

## 7. Sprint 4 — UI & Final Polish

### 7.1 Streamlit Dashboard

#### `dashboard/app.py`

```python
import os, sys
import streamlit as st
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(
    page_title="Smart City Traffic Intelligence",
    page_icon="🚦",
    layout="wide"
)

# Delta Lake path (relative to dashboard/ folder)
DELTA_BASE = os.path.join(os.path.dirname(__file__), '..', 'delta_tables')

@st.cache_resource(show_spinner="Connecting to Spark...")
def get_spark():
    from pyspark.sql import SparkSession
    from delta import configure_spark_with_delta_pip
    builder = SparkSession.builder \
        .appName("TrafficDashboard") \
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension") \
        .config("spark.sql.catalog.spark_catalog",
                "org.apache.spark.sql.delta.catalog.DeltaCatalog") \
        .config("spark.driver.memory", "2g") \
        .config("spark.sql.shuffle.partitions", "4")
    return configure_spark_with_delta_pip(builder).getOrCreate()

def read_delta(spark, table_name, limit=500):
    path = os.path.join(DELTA_BASE, table_name)
    try:
        return spark.read.format("delta").load(path).limit(limit).toPandas()
    except Exception:
        return None

# ---- Header ----
st.title("🚦 Smart City Traffic Intelligence System")
st.caption("UCP H-6 | Spark + Kafka + YOLOv8 + Delta Lake + GraphFrames")

spark = get_spark()

# ---- Row 1: KPI Metrics ----
col1, col2, col3, col4 = st.columns(4)

cv_df = read_delta(spark, "cv_vehicle_counts")
sensor_df = read_delta(spark, "sensor_speeds")
pred_df = read_delta(spark, "speed_predictions")
alert_df = read_delta(spark, "rerouting_alerts")

with col1:
    total_vehicles = int(cv_df['vehicle_count'].sum()) if cv_df is not None else 0
    st.metric("🚗 Vehicles Detected", f"{total_vehicles:,}")

with col2:
    avg_speed = sensor_df['speed'].mean() if sensor_df is not None else 0
    st.metric("⚡ Avg Network Speed", f"{avg_speed:.1f} mph")

with col3:
    congested = int((sensor_df['speed'] < 20).sum()) if sensor_df is not None else 0
    st.metric("🔴 Congested Readings", congested, delta=f"threshold: <20 mph")

with col4:
    alerts = len(alert_df) if alert_df is not None else 0
    st.metric("🔀 Rerouting Alerts", alerts)

st.divider()

# ---- Row 2: Charts ----
col_left, col_right = st.columns(2)

with col_left:
    st.subheader("📡 Sensor Speed Distribution")
    if sensor_df is not None:
        st.bar_chart(sensor_df.groupby('sensor_id')['speed'].mean().head(20))
    else:
        st.info("Waiting for sensor data from Kafka...")

with col_right:
    st.subheader("📷 Vehicle Counts by Camera")
    if cv_df is not None:
        st.bar_chart(cv_df.groupby('camera_id')['vehicle_count'].sum())
    else:
        st.info("Waiting for CCTV stream data...")

st.divider()

# ---- Row 3: ML Predictions + Alerts ----
col_ml, col_alert = st.columns(2)

with col_ml:
    st.subheader("🤖 Speed Predictions vs Actual")
    if pred_df is not None:
        sample = pred_df[['sensor_id', 'speed', 'prediction']].head(30)
        sample['error'] = abs(sample['speed'] - sample['prediction'])
        st.dataframe(sample, use_container_width=True)
        st.metric("Mean Abs. Error", f"{sample['error'].mean():.2f} mph")
    else:
        st.info("Run ml_predictor.py to generate predictions.")

with col_alert:
    st.subheader("🔀 Rerouting Alerts")
    if alert_df is not None and len(alert_df) > 0:
        st.dataframe(alert_df.head(20), use_container_width=True)
    else:
        st.info("No congestion alerts yet.")

# ---- Auto-refresh ----
st.divider()
refresh_rate = st.slider("Auto-refresh interval (seconds)", 5, 60, 10)
st.caption(f"Last updated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")

import time
time.sleep(refresh_rate)
st.rerun()
```

```powershell
cd C:\Projects\smart-traffic
.\venv\Scripts\Activate.ps1
streamlit run dashboard/app.py
```

Open: http://localhost:8501

---

## 8. Running the Full System

Open **7 PowerShell windows**, run each command in its own window, in order:

### Window 1 — Infrastructure (run once)
```powershell
cd C:\Projects\smart-traffic
docker compose up -d
# Wait 30 seconds for services to be healthy
docker compose ps
```

### Window 2 — Graph Analytics (run once, before streaming)
```powershell
.\venv\Scripts\Activate.ps1
python spark_jobs/graph_analytics.py
```

### Window 3 — ML Model Training (run once)
```powershell
.\venv\Scripts\Activate.ps1
python spark_jobs/ml_predictor.py
```

### Window 4 — Spark Streaming Pipeline (keep running)
```powershell
.\venv\Scripts\Activate.ps1
python spark_jobs/stream_processor.py
```

### Window 5 — Sensor Producer
```powershell
.\venv\Scripts\Activate.ps1
python producers/sensor_producer.py
```

### Window 6 — GPS Producer
```powershell
.\venv\Scripts\Activate.ps1
python producers/gps_producer.py
```

### Window 7 — CCTV Producer + Dashboard
```powershell
# Run CCTV producer
.\venv\Scripts\Activate.ps1
python producers/cctv_producer.py
```

```powershell
# Then in a new window — dashboard
.\venv\Scripts\Activate.ps1
streamlit run dashboard/app.py
```

### Expected URLs

| URL | What |
|-----|------|
| http://localhost:8080 | Spark Master — check worker is registered |
| http://localhost:8501 | Streamlit dashboard — live traffic data |

---

## 9. Troubleshooting

### "Java not found" or Spark won't start
```powershell
# Verify JAVA_HOME is set
echo $env:JAVA_HOME
java -version
# If empty, re-run the JAVA_HOME setup from Section 1.2
```

### Kafka "Connection refused" from Spark
The most common Docker networking issue. Check your `docker-compose.yml` has **both** listener configs (`KAFKA_ADVERTISED_LISTENERS` with both `PLAINTEXT://localhost:9092` and `PLAINTEXT_INTERNAL://kafka:29092`). Then restart Kafka:
```powershell
docker compose restart kafka
docker compose logs kafka --tail=20
```

### "delta-spark version conflict"
This means you changed one of the pinned versions. The exact working combo is:
- `pyspark==3.5.0` + `delta-spark==2.4.0` + `io.delta:delta-core_2.12:2.4.0` JAR

Do not mix versions.

### YOLOv8 training taking too long
```powershell
# Reduce to 3 epochs and 5 sequences for a quick smoke test
# Edit train_yolo.py: epochs=3
# Edit convert_detrac_to_yolo.py: MAX_SEQUENCES = 5
```

### Spark out of memory
Reduce in `docker-compose.yml`:
```yaml
SPARK_WORKER_MEMORY=2G
SPARK_WORKER_CORES=1
```
And in `.env`:
```env
SPARK_DRIVER_MEMORY=2g
```

### Port 8080 already in use (IIS or another service)
```powershell
# Find what's using port 8080
netstat -ano | findstr :8080
# Change Spark UI to a different port in docker-compose.yml: "8090:8080"
```

### Delta table "not found" in dashboard
Make sure the streaming pipeline (`stream_processor.py`) has been running for at least one micro-batch interval (5–10 seconds) before opening the dashboard.

---

## 10. Sprint Checklists

### Sprint 1 ✅
- [ ] Java 11 installed, `JAVA_HOME` set, `java -version` works
- [ ] Docker Desktop running — all 4 containers show `healthy`
- [ ] Spark UI at http://localhost:8080 shows 1 worker
- [ ] `data/metr-la/metr-la.h5` and `adj_mat.npy` exist
- [ ] `data/nyc-taxi/yellow_tripdata_2022-01.parquet` exists
- [ ] UA-DETRAC extracted, `convert_detrac_to_yolo.py` ran, `dataset.yaml` created
- [ ] OpenWeatherMap API key in `.env`
- [ ] 3 Kafka topics confirmed with `--list`
- [ ] All 3 producers stream without errors

### Sprint 2 ✅
- [ ] YOLOv8 training complete — `models/yolov8_traffic/weights/best.pt` exists
- [ ] `graph_analytics.py` ran — `delta_tables/graph_pagerank` and `delta_tables/graph_shortest_paths` exist
- [ ] `ml_predictor.py` ran — RMSE and R² printed, `delta_tables/speed_predictions` exists

### Sprint 3 ✅
- [ ] `stream_processor.py` running with all 3 branches active
- [ ] Data flowing into `delta_tables/cv_vehicle_counts`, `sensor_speeds`, `gps_trips`
- [ ] Congestion alert printed in console when speed < 20 mph
- [ ] `delta_tables/rerouting_alerts` populated

### Sprint 4 ✅
- [ ] Streamlit dashboard shows all 4 KPI metrics
- [ ] Speed chart, camera chart, predictions table, and alerts table all visible
- [ ] Auto-refresh working
- [ ] Full end-to-end demo recorded for report

---

*Smart City Traffic Intelligence System | UCP H-6 | Implementation Guide v2*
