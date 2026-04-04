from pathlib import Path

import yaml
from ultralytics import YOLO

RUN = "yolov8s_1"

BASE       = Path(__file__).parent.parent
DETECT_DIR = BASE / "dataset_detection"
run_dir    = BASE / "weights" / RUN
weights    = run_dir / "best.pt"

cfg = yaml.safe_load((DETECT_DIR / "data.yaml").read_text())
cfg["path"] = str(DETECT_DIR)
tmp_yaml = run_dir / "data_eval.yaml"
tmp_yaml.write_text(yaml.dump(cfg), encoding="utf-8")

model   = YOLO(str(weights))
metrics = model.val(data=str(tmp_yaml), split="val")

map50    = metrics.box.map50
map5095  = metrics.box.map

lines: list[str] = [
    f"# Detection Evaluation\n",
    f"## {RUN}\n",
    f"**mAP@0.5: {map50*100:.1f}%**  ",
    f"**mAP@0.5:0.95: {map5095*100:.1f}%**\n",
    "| Class | AP@0.5 |",
    "|-------|--------|",
]

for i, name in enumerate(metrics.names.values()):
    ap = float(metrics.box.ap50[i]) if i < len(metrics.box.ap50) else 0.0
    lines.append(f"| {name} | {ap*100:.1f}% |")
    print(f"  {name:<18} AP@0.5: {ap*100:.1f}%")

print(f"\nmAP@0.5: {map50*100:.1f}%  mAP@0.5:0.95: {map5095*100:.1f}%")

(run_dir / "evaluate.md").write_text("\n".join(lines), encoding="utf-8")
print(f"saved -> weights/{RUN}/evaluate.md")
