from pathlib import Path

import yaml
from ultralytics import YOLO

RUN = "yolov8s_seg_1"

BASE       = Path(__file__).parent.parent
SEG_DIR    = BASE / "dataset_segmentation"
run_dir    = BASE / "weights" / RUN
weights    = run_dir / "best.pt"

cfg = yaml.safe_load((SEG_DIR / "data.yaml").read_text())
cfg["path"] = str(SEG_DIR)
tmp_yaml = run_dir / "data_eval.yaml"
tmp_yaml.write_text(yaml.dump(cfg), encoding="utf-8")

model   = YOLO(str(weights))
metrics = model.val(data=str(tmp_yaml), split="val")

mask50   = metrics.seg.map50
mask5095 = metrics.seg.map

lines: list[str] = [
    f"# Segmentation Evaluation\n",
    f"## {RUN}\n",
    f"**Mask mAP@0.5: {mask50*100:.1f}%**  ",
    f"**Mask mAP@0.5:0.95: {mask5095*100:.1f}%**\n",
    "| Class | Mask AP@0.5 |",
    "|-------|-------------|",
]

for i, name in enumerate(metrics.names.values()):
    ap = float(metrics.seg.ap50[i]) if i < len(metrics.seg.ap50) else 0.0
    lines.append(f"| {name} | {ap*100:.1f}% |")
    print(f"  {name:<18} Mask AP@0.5: {ap*100:.1f}%")

print(f"\nMask mAP@0.5: {mask50*100:.1f}%  mAP@0.5:0.95: {mask5095*100:.1f}%")

(run_dir / "evaluate.md").write_text("\n".join(lines), encoding="utf-8")
print(f"saved -> weights/{RUN}/evaluate.md")
