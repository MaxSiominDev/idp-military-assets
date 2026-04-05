from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

RUN = "efficientnet_b0_2"

BASE          = Path(__file__).parent.parent
PROCESSED_DIR = BASE / "dataset_classification"

CLASSES     = ["gun", "spg", "ifv", "uav", "armored_vehicle", "apc", "infantry", "mlrs", "tank"]
NUM_CLASSES = len(CLASSES)
BATCH_SIZE  = 32
TRAIN_RATIO = 0.7
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


def build_simple_cnn() -> nn.Module:
    return nn.Sequential(
        nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Flatten(),
        nn.Linear(128 * 28 * 28, 256), nn.ReLU(), nn.Dropout(0.5),
        nn.Linear(256, NUM_CLASSES),
    )

def build_resnet50() -> nn.Module:
    m = models.resnet50(weights=None)
    m.fc = nn.Linear(m.fc.in_features, NUM_CLASSES)
    return m

def build_efficientnet_b0() -> nn.Module:
    m = models.efficientnet_b0(weights=None)
    m.classifier[1] = nn.Linear(m.classifier[1].in_features, NUM_CLASSES)
    return m

def build_efficientnet_b1() -> nn.Module:
    m = models.efficientnet_b1(weights=None)
    m.classifier[1] = nn.Linear(m.classifier[1].in_features, NUM_CLASSES)
    return m

def build_efficientnet_b2() -> nn.Module:
    m = models.efficientnet_b2(weights=None)
    m.classifier[1] = nn.Linear(m.classifier[1].in_features, NUM_CLASSES)
    return m


BUILDERS = {
    "simple_cnn":      build_simple_cnn,
    "resnet50":        build_resnet50,
    "efficientnet_b0": build_efficientnet_b0,
    "efficientnet_b1": build_efficientnet_b1,
    "efficientnet_b2": build_efficientnet_b2,
}

run_dir = BASE / "weights" / RUN

test_tf = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

only_root_jpg = lambda p: Path(p).parent.parent == PROCESSED_DIR and p.endswith(".jpg")
full    = datasets.ImageFolder(PROCESSED_DIR, transform=test_tf, is_valid_file=only_root_jpg)
CLASSES = full.classes
NUM_CLASSES = len(CLASSES)
n_train = int(len(full) * TRAIN_RATIO)
indices = torch.randperm(len(full), generator=torch.Generator().manual_seed(67)).tolist()
test_set    = torch.utils.data.Subset(full, indices[n_train:])
test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

print(f"run: {RUN}  test samples: {len(test_set)}  device: {DEVICE}\n")

lines: list[str] = ["# Test Evaluation\n"]

for weights_path in sorted(run_dir.glob("*.pt")):
    stem = weights_path.stem
    arch = next((k for k in BUILDERS if stem == k or stem.startswith(k + "_")), None)
    if arch is None:
        print(f"{stem}: unknown architecture, skipping")
        continue
    name = stem

    model = BUILDERS[arch]()
    model.load_state_dict(torch.load(weights_path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

    correct = 0
    per_class_correct = [0] * NUM_CLASSES
    per_class_total   = [0] * NUM_CLASSES

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            preds = model(images).argmax(1)
            correct += (preds == labels).sum().item()
            for p, l in zip(preds.tolist(), labels.tolist()):
                per_class_total[l]   += 1
                per_class_correct[l] += int(p == l)

    acc = correct / len(test_set) * 100
    print(f"{name:<20} acc: {acc:.1f}%")

    lines.append(f"## {name}\n")
    lines.append(f"Overall accuracy: **{acc:.1f}%**\n")
    lines.append("| Class | Correct | Total | Accuracy |")
    lines.append("|-------|---------|-------|----------|")
    for i, cls in enumerate(CLASSES):
        t = per_class_total[i]
        c = per_class_correct[i]
        lines.append(f"| {cls} | {c} | {t} | {c/t*100:.1f}% |" if t else f"| {cls} | 0 | 0 | — |")
    lines.append("")

(run_dir / "evaluate.md").write_text("\n".join(lines), encoding="utf-8")
print(f"\nsaved -> weights/{RUN}/evaluate.md")
