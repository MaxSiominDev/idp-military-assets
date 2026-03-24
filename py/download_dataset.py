import os
import shutil
from pathlib import Path

DATASET  = "gon213/war-tech-v2-0-by-gontech"
DEST_DIR = Path(__file__).parent.parent / "dataset"

ORIG_FOLDER = {
    "artillery":       "artilleriya",
    "ifv":             "bmp",
    "uav":             "bpla",
    "armored_vehicle": "bronemashina",
    "apc":             "btr",
    "infantry":        "pehota",
    "mlrs":            "rszo",
    "tank":            "tank",
}

PREFIX_TO_CLASS = {v: k for k, v in ORIG_FOLDER.items()}

os.makedirs(DEST_DIR, exist_ok=True)
os.system(f"kaggle datasets download -d {DATASET} -p {DEST_DIR} --unzip")

nested = DEST_DIR / "war_TCHBYGON"
flat = nested / "obshaya_papk"
flat.rename(DEST_DIR / "images")
for f in nested.iterdir():
    shutil.move(str(f), str(DEST_DIR / f.name))
nested.rmdir()

images_dir = DEST_DIR / "images"
for img in images_dir.glob("*.jpg"):
    prefix = img.stem.split(" ")[0] if " " in img.stem else img.stem.split("(")[0].strip()
    class_name = PREFIX_TO_CLASS.get(prefix)
    if class_name:
        class_dir = DEST_DIR / class_name
        class_dir.mkdir(exist_ok=True)
        shutil.move(str(img), str(class_dir / img.name))
images_dir.rmdir()

old_csv = DEST_DIR / "war_tech_gont-export.csv"
old_csv.rename(DEST_DIR / "bb.csv")

print(f"{DEST_DIR}")
