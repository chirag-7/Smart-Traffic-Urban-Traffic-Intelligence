"""
Convert UA-DETRAC XML annotations to Ultralytics YOLO layout under ``data/ua-detrac/yolo_format``.

Expects:
  - ``data/ua-detrac/DETRAC-Images/<sequence>/img*.jpg``
  - XML files under ``DETRAC-Train-Annotations-XML`` / ``DETRAC-Test-Annotations-XML``

Produces train/val/test splits (70/15/15), single class ``vehicle`` (class id 0), and ``dataset.yaml``.
"""

from __future__ import annotations

import logging
import random
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

IMAGES_DIR = Path("data/ua-detrac/DETRAC-Images")
ANNOTATIONS_DIR_TRAIN = Path("data/ua-detrac/DETRAC-Train-Annotations-XML")
ANNOTATIONS_DIR_TEST = Path("data/ua-detrac/DETRAC-Test-Annotations-XML")
OUTPUT_DIR = Path("data/ua-detrac/yolo_format")

for split in ["train", "val", "test"]:
    (OUTPUT_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)

xml_files = []
if ANNOTATIONS_DIR_TRAIN.exists():
    xml_files.extend(ANNOTATIONS_DIR_TRAIN.glob("*.xml"))
if ANNOTATIONS_DIR_TEST.exists():
    xml_files.extend(ANNOTATIONS_DIR_TEST.glob("*.xml"))

xml_files = sorted(set(xml_files))
logger.info("%s annotation XML files", len(xml_files))

random.seed(42)
random.shuffle(xml_files)
n = len(xml_files)
train_xmls = xml_files[: int(0.7 * n)]
val_xmls = xml_files[int(0.7 * n) : int(0.85 * n)]
test_xmls = xml_files[int(0.85 * n) :]

total_images = 0

for split_name, xml_list in [("train", train_xmls), ("val", val_xmls), ("test", test_xmls)]:
    logger.info("Split %s — %s files", split_name, len(xml_list))

    for xml_file in xml_list:
        seq_name = xml_file.stem
        seq_dir = IMAGES_DIR / seq_name
        if not seq_dir.exists():
            continue

        try:
            tree = ET.parse(xml_file)
            root = tree.getroot()
            frame_width = int(root.attrib.get("width", 960))
            frame_height = int(root.attrib.get("height", 540))

            for frame in root.findall(".//frame"):
                frame_num = int(frame.get("num"))
                img_file = seq_dir / f"img{frame_num:05d}.jpg"
                if not img_file.exists():
                    continue

                label_lines = []
                for target in frame.findall(".//target"):
                    bbox = target.find("box")
                    if bbox is None:
                        continue
                    left = float(bbox.get("left"))
                    top = float(bbox.get("top"))
                    width = float(bbox.get("width"))
                    height = float(bbox.get("height"))
                    cx = (left + width / 2) / frame_width
                    cy = (top + height / 2) / frame_height
                    nw = width / frame_width
                    nh = height / frame_height
                    cx, cy, nw, nh = [max(0, min(1, v)) for v in (cx, cy, nw, nh)]
                    label_lines.append(f"0 {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")

                if not label_lines:
                    continue

                unique_img_name = f"{seq_name}_img{frame_num:05d}.jpg"
                unique_lbl_name = f"{seq_name}_img{frame_num:05d}.txt"
                out_img = OUTPUT_DIR / "images" / split_name / unique_img_name
                out_lbl = OUTPUT_DIR / "labels" / split_name / unique_lbl_name

                if not out_img.exists():
                    shutil.copy2(img_file, out_img)

                with open(out_lbl, "w", encoding="utf-8") as fh:
                    fh.write("\n".join(label_lines))

                total_images += 1

        except Exception as exc:
            logger.warning("Sequence %s: %s", seq_name, exc)

yaml_content = """path: data/ua-detrac/yolo_format
train: images/train
val: images/val
test: images/test

nc: 1
names: ['vehicle']
"""

with open(OUTPUT_DIR / "dataset.yaml", "w", encoding="utf-8") as fh:
    fh.write(yaml_content)

logger.info("Wrote %s labeled frames and dataset.yaml under %s", total_images, OUTPUT_DIR)
