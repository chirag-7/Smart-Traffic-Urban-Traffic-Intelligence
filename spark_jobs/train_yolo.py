from ultralytics import YOLO
from pathlib import Path
import shutil

print("[YOLOv8] Starting vehicle detection model training...")

# ==========================================
# PATHS
# ==========================================
DATASET_YAML = "data/ua-detrac/yolo_format/dataset.yaml"

OUTPUT_PROJECT = Path("models")
OUTPUT_NAME = "yolo_traffic_model"
OUTPUT_DIR = OUTPUT_PROJECT / OUTPUT_NAME
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

LAST_WEIGHTS = OUTPUT_DIR / "weights" / "last.pt"

print(f"[YOLOv8] Dataset: {DATASET_YAML}")
print(f"[YOLOv8] Output: {OUTPUT_DIR}")

# ==========================================
# CHECK DATASET EXISTS
# ==========================================
if not Path(DATASET_YAML).exists():
    print(f"[YOLOv8] ✗ ERROR: dataset.yaml not found at {DATASET_YAML}")
    exit(1)

# ==========================================
# TRAINING LOGIC
# ==========================================
if LAST_WEIGHTS.exists():
    print(f"\n[YOLOv8] Found existing weights at {LAST_WEIGHTS}")
    print("[YOLOv8] Loading weights and continuing training...")
    print("[YOLOv8] NOTE: Using manual resume (not resume=True) to avoid optimizer mismatch bug.")

    # Load weights only — do NOT use resume=True
    # resume=True re-reads dataset/optimizer from checkpoint metadata which can
    # point to the wrong dataset and cause an optimizer state dict crash.
    model = YOLO(str(LAST_WEIGHTS))

    results = model.train(
        data=DATASET_YAML,      # always explicitly pass YOUR dataset
        epochs=50,
        imgsz=640,
        batch=16,
        patience=10,
        project=str(OUTPUT_PROJECT),  # models/
        name=OUTPUT_NAME,             # yolo_traffic_model → models/yolo_traffic_model/
        exist_ok=True,                # reuse folder, append to results.csv
        save=True,
        verbose=True,
        plots=True
    )

else:
    print("\n[YOLOv8] No existing weights found. Starting fresh training.")
    print("[YOLOv8] Loading YOLOv8n (nano) base model...")

    model = YOLO('yolov8n.pt')

    print("[YOLOv8] Training parameters:")
    print(f"  • Epochs:     50")
    print(f"  • Batch size: 16")
    print(f"  • Image size: 640x640")
    print(f"  • Dataset:    70/15/15 train/val/test split")

    results = model.train(
        data=DATASET_YAML,
        epochs=50,
        imgsz=640,
        batch=16,
        patience=10,
        project=str(OUTPUT_PROJECT),
        name=OUTPUT_NAME,
        exist_ok=True,
        save=True,
        verbose=True,
        plots=True
    )

# ==========================================
# SAVE FINAL MODEL
# ==========================================
print("\n[YOLOv8] ✅ Training complete!")

model_path = OUTPUT_DIR / "weights" / "best.pt"
if model_path.exists():
    print(f"[YOLOv8] ✅ Best model saved: {model_path}")
    final_model = OUTPUT_DIR / "best.pt"
    shutil.copy2(model_path, final_model)
    print(f"[YOLOv8] ✅ Model copied to: {final_model}")
else:
    print(f"[YOLOv8] ✗ Model not found at expected location: {model_path}")

print("\n[YOLOv8] Training Results:")
print(f"  • Results saved to: {OUTPUT_DIR}")
print(f"  • Use this model in Task 14 for real-time vehicle detection")
print(f"\n[YOLOv8] ✅ Task 11 Complete!")