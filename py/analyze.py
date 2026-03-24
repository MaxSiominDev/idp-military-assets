import sys
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageOps, ImageEnhance

sys.stdout.reconfigure(encoding="utf-8")

DATA_DIR = Path("../dataset/war_TCHBYGON/root")
PROCESSED_DIR = Path("../dataset_processed")

CLASSES = ["artillery", "ifv", "uav", "armored_vehicle", "apc", "infantry", "mlrs", "tank"]

LABELS = {
    "artillery":       "Artillery",
    "ifv":             "IFV",
    "uav":             "UAV",
    "armored_vehicle": "Armored Vehicle",
    "apc":             "APC",
    "infantry":        "Infantry",
    "mlrs":            "MLRS",
    "tank":            "Tank",
}

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

IMG_SIZE = (224, 224)
random.seed(69 + 67 + 228 + 420)

counts = {c: len(list((PROCESSED_DIR / c).glob("*.jpg"))) for c in CLASSES}
total = sum(counts.values())

print("\ndataset overview")
print(f"total images: {total}, classes: {len(CLASSES)}\n")
print(f"{'folder':<18} {'label':<20} {'count':>6}")
print("-" * 46)
for c, n in counts.items():
    print(f"{c:<18} {LABELS[c]:<20} {n:>6}")
print(f"\ntotal: {total}")

ws, hs = [], []
for c in CLASSES:
    for p in list((DATA_DIR / ORIG_FOLDER[c]).glob("*.jpg"))[:10]:
        w, h = Image.open(p).size
        ws.append(w)
        hs.append(h)

print(f"\nresolution sample:")
print(f"  w: {min(ws)}-{max(ws)} px, avg {np.mean(ws):.0f}")
print(f"  h: {min(hs)}-{max(hs)} px, avg {np.mean(hs):.0f}")
print(f"  -> resized to {IMG_SIZE[0]}x{IMG_SIZE[1]} for training\n")


fig, axes = plt.subplots(2, 4, figsize=(16, 8))
fig.suptitle("Sample Images per Class", fontsize=16, fontweight="bold")

for ax, c in zip(axes.flat, CLASSES):
    p = random.choice(list((PROCESSED_DIR / c).glob("*.jpg")))
    ax.imshow(Image.open(p))
    ax.set_title(LABELS[c], fontsize=11)
    ax.axis("off")

plt.tight_layout()
plt.savefig("plot_1_samples.png", dpi=120)
plt.show()
print("saved plot_1_samples.png")


lbls = [LABELS[c] for c in CLASSES]
vals = [counts[c] for c in CLASSES]
colors = [plt.cm.get_cmap("tab10")(i / len(CLASSES)) for i in range(len(CLASSES))]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Class Distribution", fontsize=14, fontweight="bold")

ax = axes[0]
bars = ax.bar(lbls, vals, color=colors)
ax.set_title("count per class")
ax.set_ylabel("images")
ax.tick_params(axis="x", rotation=30)
for bar, v in zip(bars, vals):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
            str(v), ha="center", va="bottom", fontsize=9)

axes[1].pie(vals, labels=lbls, colors=colors, autopct="%1.1f%%", startangle=140)
axes[1].set_title("share")

plt.tight_layout()
plt.savefig("plot_2_class_balance.png", dpi=120)
plt.show()
print("saved plot_2_class_balance.png")

big = max(counts, key=counts.get)
small = min(counts, key=counts.get)
ratio = counts[big] / counts[small]
print(f"\nbiggest: {LABELS[big]} ({counts[big]}), smallest: {LABELS[small]} ({counts[small]})")
print(f"imbalance ratio: {ratio:.2f}x ({'moderate' if ratio < 4 else 'severe'})\n")


sample_cls = "tank"
src_orig = random.choice(list((DATA_DIR / ORIG_FOLDER[sample_cls]).glob("*.jpg")))
orig_raw = Image.open(src_orig).convert("RGB")
proc_path = PROCESSED_DIR / sample_cls / src_orig.name
proc_img = Image.open(proc_path).convert("RGB").resize(IMG_SIZE)

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
fig.suptitle("Preprocessing: Original vs Processed", fontsize=13, fontweight="bold")
axes[0].imshow(orig_raw)
axes[0].set_title(f"original  {orig_raw.size[0]}×{orig_raw.size[1]}", fontsize=10)
axes[0].axis("off")
axes[1].imshow(proc_img)
axes[1].set_title(f"processed  {IMG_SIZE[0]}×{IMG_SIZE[1]}", fontsize=10)
axes[1].axis("off")
plt.tight_layout()
plt.savefig("plot_3_preprocess.png", dpi=120)
plt.show()
print("saved plot_3_preprocess.png\n")

src = random.choice(list((PROCESSED_DIR / "tank").glob("*.jpg")))
orig = Image.open(src).convert("RGB")

aug_variants = [
    ("original",        orig),
    ("hflip",           ImageOps.mirror(orig)),
    ("rotate 15",       orig.rotate(15)),
    ("brightness 1.4",  ImageEnhance.Brightness(orig).enhance(1.4)),
    ("contrast 1.3",    ImageEnhance.Contrast(orig).enhance(1.3)),
]

fig, axes = plt.subplots(1, len(aug_variants), figsize=(18, 4))
fig.suptitle("Augmentation Examples", fontsize=13, fontweight="bold")
for ax, (title, img) in zip(axes, aug_variants):
    ax.imshow(img)
    ax.set_title(title, fontsize=10)
    ax.axis("off")
plt.tight_layout()
plt.savefig("plot_4_augmentation.png", dpi=120)
plt.show()
print("saved plot_4_augmentation.png\n")

print("potential use cases: classification, detection, segmentation")
print()
print("known issues:")
print(f" - class imbalance {ratio:.2f}x, fix with weighted loss or oversampling")
print("  - low source resolution (~275x183), some detail is lost after resize")
print("  - data from osint/video frames, expect compression artifacts")
print("  - ifv/apc/armored_vehicle look similar, might hurt accuracy")
print()
print("done. plots: plot_1_samples.png, plot_2_class_balance.png,")
print("            plot_3_preprocess.png, plot_4_augmentation.png")
