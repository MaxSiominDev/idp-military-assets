import csv
import json
import sys
from pathlib import Path
from urllib.parse import unquote

import numpy as np
from PIL import Image
from pycocotools import mask as coco_mask

sys.stdout.reconfigure(encoding="utf-8")

BASE             = Path(__file__).parent.parent
DATA_DIR         = BASE / "dataset"
PROCESSED_DIR    = BASE / "dataset_processed"
ANNOTATIONS_FILE = BASE / "annotations.json"

CLASSES = ["gun", "spg", "ifv", "uav", "armored_vehicle", "apc", "infantry", "mlrs", "tank"]

ORIG_FOLDER = {
    "ifv":             "bmp",
    "uav":             "bpla",
    "armored_vehicle": "bronemashina",
    "apc":             "btr",
    "infantry":        "pehota",
    "mlrs":            "rszo",
    "tank":            "tank",
}

FOLDER_TO_CLASS = {v: k for k, v in ORIG_FOLDER.items()}

with open(BASE / "artillery_split.json") as f:
    _split = json.load(f)
ARTILLERY_SPLIT: dict[str, str] = {name: cls for cls in ("gun", "spg") for name in _split[cls]}

IMG_SIZE = (224, 224)

print("processing images...")
for dst_cls in ("gun", "spg"):
    (PROCESSED_DIR / dst_cls).mkdir(parents=True, exist_ok=True)
for p in (DATA_DIR / "artillery").glob("*.jpg"):
    dst_cls = ARTILLERY_SPLIT.get(p.name)
    if dst_cls is None:
        continue
    img = Image.open(p).convert("RGB").resize(IMG_SIZE)
    img.save(PROCESSED_DIR / dst_cls / p.name)
print(f"  {'gun':<18} {len(list((PROCESSED_DIR / 'gun').glob('*.jpg')))} images")
print(f"  {'spg':<18} {len(list((PROCESSED_DIR / 'spg').glob('*.jpg')))} images")

for c in CLASSES:
    if c in ("gun", "spg"):
        continue
    src_dir = DATA_DIR / c
    dst_dir = PROCESSED_DIR / c
    dst_dir.mkdir(parents=True, exist_ok=True)
    images = list(src_dir.glob("*.jpg"))
    for p in images:
        img = Image.open(p).convert("RGB").resize(IMG_SIZE)
        img.save(dst_dir / p.name)
    print(f"  {c:<18} {len(images)} images")

print("processing masks...")
with open(ANNOTATIONS_FILE) as f:
    data = json.load(f)

img_info: dict[int, dict] = {img["id"]: img for img in data["images"]}

skipped = 0
for ann in data["annotations"]:
    info    = img_info[ann["image_id"]]
    fname   = Path(info["file_name"]).name
    prefix  = fname.split(" ")[0] if " " in fname else fname.split("(")[0].strip()

    if prefix == "artilleriya":
        class_name = ARTILLERY_SPLIT.get(fname)
    else:
        class_name = FOLDER_TO_CLASS.get(prefix) or FOLDER_TO_CLASS.get(Path(info["file_name"]).parent.name)

    if class_name is None:
        skipped += 1
        continue

    rle          = ann["segmentation"]
    binary_mask  = coco_mask.decode(rle)
    mask_img     = Image.fromarray((binary_mask * 255).astype(np.uint8)).resize(IMG_SIZE, Image.NEAREST)

    masks_dir = PROCESSED_DIR / class_name / "masks"
    masks_dir.mkdir(exist_ok=True)
    stem = Path(info["file_name"]).stem
    mask_img.save(masks_dir / f"{stem}.png")

    base_img    = Image.open(PROCESSED_DIR / class_name / fname).convert("RGBA")
    overlay     = Image.new("RGBA", IMG_SIZE, (0, 0, 0, 0))
    overlay_arr = np.array(overlay)
    overlay_arr[np.array(mask_img) > 0] = [255, 0, 0, 120]
    overlaid    = Image.alpha_composite(base_img, Image.fromarray(overlay_arr)).convert("RGB")

    overlaid_dir = PROCESSED_DIR / class_name / "overlaid"
    overlaid_dir.mkdir(exist_ok=True)
    overlaid.save(overlaid_dir / f"{stem}.jpg")

counts = {c: len(list((PROCESSED_DIR / c / "masks").glob("*.png"))) for c in CLASSES}
for c, n in counts.items():
    print(f"  {c:<18} {n} masks")
if skipped:
    print(f"  skipped: {skipped}")

print("processing bounding boxes...")

BB_FILE = DATA_DIR / "bb.csv"
BB_OUT  = PROCESSED_DIR / "annotations_bb.csv"

orig_sizes: dict[str, tuple[int, int]] = {}
for p in (DATA_DIR / "artillery").glob("*.jpg"):
    orig_sizes[p.name] = Image.open(p).size
for c in CLASSES:
    if c in ("gun", "spg"):
        continue
    for p in (DATA_DIR / c).glob("*.jpg"):
        orig_sizes[p.name] = Image.open(p).size

with open(BB_FILE, newline="", encoding="utf-8") as fin, \
     open(BB_OUT,  "w", newline="", encoding="utf-8") as fout:
    reader = csv.DictReader(fin)
    writer = csv.DictWriter(fout, fieldnames=["image", "class", "xmin", "ymin", "xmax", "ymax"])
    writer.writeheader()
    for row in reader:
        fname = unquote(row["image"])
        if fname not in orig_sizes:
            continue
        orig_w, orig_h = orig_sizes[fname]
        sx = IMG_SIZE[0] / orig_w
        sy = IMG_SIZE[1] / orig_h
        folder_name = row["label"]
        if folder_name == "artilleriya":
            class_name = ARTILLERY_SPLIT.get(fname)
        else:
            class_name = FOLDER_TO_CLASS.get(folder_name, folder_name)
        if class_name is None:
            continue
        writer.writerow({
            "image":  fname,
            "class":  class_name,
            "xmin":   round(float(row["xmin"]) * sx, 4),
            "ymin":   round(float(row["ymin"]) * sy, 4),
            "xmax":   round(float(row["xmax"]) * sx, 4),
            "ymax":   round(float(row["ymax"]) * sy, 4),
        })

print(f"  saved {BB_OUT}")
print("done")
