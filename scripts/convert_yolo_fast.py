import xml.etree.ElementTree as ET
from pathlib import Path
import random
import shutil

IMAGES_DIR = Path("data/ua-detrac/DETRAC-Images")
ANNOTATIONS_DIR_TRAIN = Path("data/ua-detrac/DETRAC-Train-Annotations-XML")
ANNOTATIONS_DIR_TEST = Path("data/ua-detrac/DETRAC-Test-Annotations-XML")
OUTPUT_DIR = Path("data/ua-detrac/yolo_format")

print("[UA-DETRAC] Starting conversion (FIXED)...")

# Create directories
for split in ["train", "val", "test"]:
    (OUTPUT_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)

# Get annotation files from both train and test folders
xml_files = []
if ANNOTATIONS_DIR_TRAIN.exists():
    xml_files.extend(ANNOTATIONS_DIR_TRAIN.glob("*.xml"))
if ANNOTATIONS_DIR_TEST.exists():
    xml_files.extend(ANNOTATIONS_DIR_TEST.glob("*.xml"))

xml_files = sorted(list(set(xml_files)))
print(f"[UA-DETRAC] Found {len(xml_files)} XML files (train + test)")

# Split 70/15/15
random.seed(42)
random.shuffle(xml_files)
train_xmls = xml_files[:int(0.7*len(xml_files))]
val_xmls = xml_files[int(0.7*len(xml_files)):int(0.85*len(xml_files))]
test_xmls = xml_files[int(0.85*len(xml_files)):]

total_images = 0
total_labels = 0

for split_name, xml_list in [("train", train_xmls), ("val", val_xmls), ("test", test_xmls)]:
    print(f"\n[UA-DETRAC] Processing {split_name}...")
    
    for file_idx, xml_file in enumerate(xml_list, 1):
        seq_name = xml_file.stem
        seq_dir = IMAGES_DIR / seq_name
        
        if not seq_dir.exists():
            continue
        
        print(f"  [{file_idx}/{len(xml_list)}] {seq_name}...", end="", flush=True)
        
        try:
            # Parse XML
            tree = ET.parse(xml_file)
            root = tree.getroot()
            
            # FIX #2: Read frame dimensions dynamically from XML root attributes
            # Standard UA-DETRAC resolution: 960×540 pixels (per research paper)
            frame_width = int(root.attrib.get('width', 960))
            frame_height = int(root.attrib.get('height', 540))
            
            seq_labels = 0
            
            # Process each frame
            for frame in root.findall('.//frame'):
                frame_num = int(frame.get("num"))
                
                # Image filename
                img_file = seq_dir / f"img{frame_num:05d}.jpg"
                
                if not img_file.exists():
                    continue
                
                label_lines = []
                
                # Extract boxes for this frame
                for target in frame.findall('.//target'):
                    bbox = target.find('box')
                    if bbox is None:
                        continue
                    
                    # Get box coordinates (left, top, width, height)
                    left = float(bbox.get("left"))
                    top = float(bbox.get("top"))
                    width = float(bbox.get("width"))
                    height = float(bbox.get("height"))
                    
                    # Convert to YOLO format (normalized center_x, center_y, width, height)
                    center_x = (left + width / 2) / frame_width
                    center_y = (top + height / 2) / frame_height
                    norm_w = width / frame_width
                    norm_h = height / frame_height
                    
                    # Clamp values to [0, 1]
                    center_x = max(0, min(1, center_x))
                    center_y = max(0, min(1, center_y))
                    norm_w = max(0, min(1, norm_w))
                    norm_h = max(0, min(1, norm_h))
                    
                    label_lines.append(f"0 {center_x:.6f} {center_y:.6f} {norm_w:.6f} {norm_h:.6f}")
                
                # Save if has annotations
                if label_lines:
                    # FIX #1: Prepend sequence name to ensure unique filenames globally
                    unique_img_name = f"{seq_name}_img{frame_num:05d}.jpg"
                    unique_lbl_name = f"{seq_name}_img{frame_num:05d}.txt"
                    
                    out_img = OUTPUT_DIR / "images" / split_name / unique_img_name
                    out_lbl = OUTPUT_DIR / "labels" / split_name / unique_lbl_name
                    
                    if not out_img.exists():
                        shutil.copy2(img_file, out_img)
                    
                    with open(out_lbl, 'w') as f:
                        f.write('\n'.join(label_lines))
                    
                    seq_labels += 1
                    total_images += 1
            
            print(f" ✓ ({seq_labels})")
            total_labels += seq_labels
        
        except Exception as e:
            print(f" ✗ ({str(e)[:30]})")

print(f"\n[UA-DETRAC] ✅ Done!")
print(f"[UA-DETRAC] Total: {total_images} images with {total_labels} annotated")

# Create dataset.yaml
yaml_content = """path: data/ua-detrac/yolo_format
train: images/train
val: images/val
test: images/test

nc: 1
names: ['vehicle']
"""

with open(OUTPUT_DIR / "dataset.yaml", 'w') as f:
    f.write(yaml_content)

print(f"[UA-DETRAC] ✅ dataset.yaml created")
print(f"\n[UA-DETRAC] Fixes Applied:")
print(f"  ✓ Unique filenames (sequence prefix)")
print(f"  ✓ Dynamic resolution from XML")
print(f"  ✓ Single vehicle class for density counting")
