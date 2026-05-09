"""
Task 11: Train YOLOv8 Vehicle Detection Model
==============================================
Why: Train a YOLOv8 model on UA-DETRAC dataset to detect vehicles in CCTV footage.
This model will be used in Task 14 for real-time vehicle counting and detection.

Dataset: 138,252 images, 100% annotated with bounding boxes
Model: YOLOv8n (nano) - fast inference, suitable for real-time processing
"""

from ultralytics import YOLO
from pathlib import Path
import os

print("[YOLOv8] Starting vehicle detection model training...")

# Paths
DATASET_YAML = "data/ua-detrac/yolo_format/dataset.yaml"
OUTPUT_DIR = Path("models/yolo_traffic_model")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print(f"[YOLOv8] Dataset: {DATASET_YAML}")
print(f"[YOLOv8] Output: {OUTPUT_DIR}")

# Check dataset exists
if not Path(DATASET_YAML).exists():
    print(f"[YOLOv8] ✗ ERROR: dataset.yaml not found at {DATASET_YAML}")
    exit(1)

# Load YOLOv8 nano model (lightweight, suitable for real-time)
# Why nano: Fast inference speed for real-time CCTV processing
print("[YOLOv8] Loading YOLOv8n (nano) model...")
model = YOLO('yolov8n.pt')

print("[YOLOv8] Starting training...")
print("[YOLOv8] Training parameters:")
print(f"  • Epochs: 50")
print(f"  • Batch size: 16")
print(f"  • Image size: 640x640")
print(f"  • Device: GPU (if available)")
print(f"  • Dataset: 70/15/15 train/val/test split")

# Train the model
# Why these parameters:
# - epochs=50: Good balance between accuracy and training time
# - batch_size=16: Fits in GPU memory, reasonable convergence
# - imgsz=640: Standard YOLO resolution
# - device=0: Use first GPU if available
results = model.train(
    data=DATASET_YAML,
    epochs=50,
    imgsz=640,
    batch=16,
    patience=10,  # Early stopping after 10 epochs without improvement
    device=0,  # Use GPU 0 (or CPU if not available)
    project=str(OUTPUT_DIR),
    name='train',
    save=True,
    verbose=True,
    plots=True  # Generate training plots
)

print("\n[YOLOv8] ✅ Training complete!")

# Save final model
model_path = OUTPUT_DIR / "train" / "weights" / "best.pt"
if model_path.exists():
    print(f"[YOLOv8] ✅ Best model saved: {model_path}")
    
    # Copy to standard location
    final_model = OUTPUT_DIR / "best.pt"
    import shutil
    shutil.copy2(model_path, final_model)
    print(f"[YOLOv8] ✅ Model copied to: {final_model}")
else:
    print(f"[YOLOv8] ✗ Model not found at expected location")

print("\n[YOLOv8] Training Results:")
print(f"  • Results saved to: {OUTPUT_DIR}")
print(f"  • Use this model in Task 14 for real-time vehicle detection")
print(f"\n[YOLOv8] ✅ Task 11 Complete!")
