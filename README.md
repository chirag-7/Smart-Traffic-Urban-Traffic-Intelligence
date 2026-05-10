# Smart Traffic — Smart City Traffic Intelligence

End-to-end demo pipeline that combines **Apache Kafka**, **Apache Spark Structured Streaming**, **Delta Lake**, **Ultralytics YOLOv8**, and **NetworkX** graph analytics to simulate how a city might ingest multimodal traffic data, detect congestion, surface rerouting hints from a road-sensor graph, and visualize everything in **Streamlit**.

This repository targets **Windows** with **Docker Desktop** (Kafka/Zookeeper; optional Spark containers) and **Python 3.10** — see `requirements.txt` and `plan/smart_city_traffic_implementation_guide_v2.md` for environment specifics.

---

## What this project does

| Layer | Role |
|--------|------|
| **Producers** | Push synthetic/realistic feeds into Kafka: freeway speeds (METR-LA), taxi trips (NYC TLC parquet), CCTV frames (UA-DETRAC JPEGs as base64), and optional weather (OpenWeatherMap). |
| **Streaming** | `spark_jobs/stream_processor.py` consumes four topics, parses JSON, runs **YOLOv8** in a pandas UDF on CCTV batches, appends sensor/GPS/weather rows to Delta, and on low speed joins **precomputed shortest paths** to append rerouting alerts. |
| **Batch analytics** | `graph_analytics.py` builds a directed graph from METR-LA adjacency, runs PageRank and shortest paths, writes Delta tables used by streaming. `ml_predictor.py` trains a GBT model on a time slice of speeds and writes predictions to Delta. |
| **Training** | `train_yolo.py` fine-tunes YOLOv8 on converted UA-DETRAC labels. |
| **Dashboard** | `dashboard/app.py` reads Delta tables with Spark and charts KPIs and samples. |

---

## Repository layout

```
smart-traffic/
├── producers/           # Kafka publishers
├── spark_jobs/          # Streaming + batch Spark programs
├── scripts/             # Dataset conversion & optional JAR download
├── dashboard/           # Streamlit UI
├── data/                # Local datasets (gitignored)
├── delta_tables/        # Runtime Delta output (gitignored)
├── models/              # YOLO weights & MLlib pipeline (mostly gitignored)
├── plan/                # Detailed implementation guide (PDF-style markdown)
├── docker-compose.yml   # Zookeeper, Kafka, Spark images
├── requirements.txt
└── README.md
```

---

## Prerequisites

- **Java 11** (`JAVA_HOME`) for Spark.
- **Docker Desktop** — run Kafka stack per `docker-compose.yml`. Spark **master/worker** containers are optional for this repo’s default workflow (Spark jobs use `local[*]` / `local[4]` on the host).
- **Python 3.10** virtualenv and `pip install -r requirements.txt`.

Create `.env` at the repo root (never commit it). Typical variables:

```env
OPENWEATHER_API_KEY=...
OPENWEATHER_CITY=Los Angeles
KAFKA_BOOTSTRAP_SERVERS=localhost:9092
DATA_DIR=./data
DELTA_DIR=./delta_tables
MODELS_DIR=./models
SPARK_DRIVER_MEMORY=4g
```

---

## Data you need locally

Paths are conventional; adjust if your files differ.

| Dataset | Purpose | Expected location |
|---------|---------|-------------------|
| METR-LA speeds | Sensors + ML + graph | `data/metr-la/METR-LA.h5`, `data/metr-la/adj_mat.npy` (or `adj_METR-LA.pkl`) |
| NYC TLC | GPS producer | `data/nyc-taxi/yellow_tripdata_2022-01.parquet` |
| UA-DETRAC | YOLO training + CCTV producer | Images under `data/ua-detrac/DETRAC-Images/…` or flat JPEGs under `data/ua-detrac/yolo_format/images/train` after conversion |
| OpenWeather | Weather producer | API key in `.env` |

Convert UA-DETRAC XML → YOLO layout:

```powershell
python scripts/convert_yolo_fast.py
python spark_jobs/train_yolo.py
```

---

## Kafka topics

Create explicitly or rely on broker auto-create:

- `topic_sensors` — JSON with `timestamp`, `sensor_id`, **`sensor_index`** (0-based column index aligned with adjacency matrix), `speed`.
- `topic_gps` — pickup/dropoff zones and distance.
- `topic_cctv` — `camera_id`, `frame_id`, `timestamp`, `frame_b64`.
- `topic_weather` — normalized OpenWeather JSON from `weather_producer.py`.

The **`sensor_index`** field is required for meaningful **rerouting**: graph vertices are labeled `"0"` … `"n-1"` to match matrix order.

---

## Typical run order

1. **Infrastructure:** `docker compose up -d` (wait until Kafka is healthy).
2. **One-time batch:**  
   `python spark_jobs/graph_analytics.py`  
   `python spark_jobs/ml_predictor.py`
3. **Streaming (keep running):**  
   `python spark_jobs/stream_processor.py`
4. **Producers** (separate terminals as needed):  
   `python producers/sensor_producer.py`  
   `python producers/gps_producer.py`  
   `python producers/cctv_producer.py`  
   `python producers/weather_producer.py`
5. **UI:**  
   `streamlit run dashboard/app.py` → [http://localhost:8501](http://localhost:8501)

Spark UI (if using compose Spark image with published port): host port **8090** maps to container 8080 in this repo’s `docker-compose.yml`.

---

## Important implementation notes

1. **YOLO in streaming** — Weights path is `models/yolo_traffic_model/weights/best.pt`. If missing, the UDF falls back to zero counts after logging a warning at executor startup (check Spark logs). Train first or copy weights into place.

2. **CCTV producer image roots** — Prefers `DETRAC-Images` sequence folders; if absent, uses flat `yolo_format/images/train` JPEGs.

3. **ML predictor slice** — Uses the first **1000** timestamps from METR-LA HDF5 to limit RAM; extend `[:1000]` in `ml_predictor.py` for a heavier experiment.

4. **GraphFrames JAR** — `scripts/download_graphframes_jar.py` is optional; graph jobs use **NetworkX**, not GraphFrames.

5. **Checkpoints** — Under `delta_tables/checkpoints/`. Delete carefully if you change Kafka message schemas (e.g. adding `sensor_index`).

---

## Troubleshooting (short)

| Symptom | Check |
|---------|--------|
| Kafka connection errors | `KAFKA_BOOTSTRAP_SERVERS`, Docker listeners (`localhost:9092` vs `kafka:29092`). |
| Empty rerouting alerts | Run `graph_analytics.py` first; ensure producers send **`sensor_index`**. |
| Delta read errors in Streamlit | Same Delta JAR version as streaming (`delta-core_2.12:2.4.0` via `spark.jars.packages`). |
| PySpark / Java failures | Java 11 on `PATH` and `JAVA_HOME`. |

Full narrative: **`plan/smart_city_traffic_implementation_guide_v2.md`**.

---

## License / attribution

Built as an educational smart-city traffic stack. METR-LA, NYC TLC, and UA-DETRAC have their own dataset licenses — cite them in academic or commercial reuse.
