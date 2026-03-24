# Training Results — comparison

## Parameters

| Parameter | Value |
|-----------|-------|
| Image size | 224×224 |
| Batch size | 32 |
| Epochs | 20 |
| Learning rate | 0.001 (CosineAnnealingLR) |
| Weight decay | 0 |
| Optimizer | Adam |
| Loss | CrossEntropyLoss |
| Train/test split | 70/30 |
| Seed | 67 |
| Classes | 9 (gun, spg, ifv, uav, armored_vehicle, apc, infantry, mlrs, tank) |

**Augmentation (train only):** RandomHorizontalFlip, RandomRotation(15°), ColorJitter(brightness=0.3, contrast=0.3)
**Normalization:** ImageNet mean/std

---

## Results

| Model | Train acc | Test acc | Разрыв | Время/epoch |
|-------|-----------|----------|--------|-------------|
| Simple CNN | 46.6% | 42.5% | −4.1% | ~7s |
| ResNet50 | 99.6% | 81.4% | −18.2% | ~18s |
| EfficientNet-B0 | 99.0% | 82.9% | −16.1% | ~10s |
| EfficientNet-B2 | 99.4% | 82.7% | −16.7% | ~14s |

---

## Simple CNN

| Epoch | Loss | Train acc |
|-------|------|-----------|
| 1 | 2.3933 | 17.6% |
| 5 | 1.8391 | 30.0% |
| 10 | 1.6647 | 40.0% |
| 15 | 1.5305 | 44.3% |
| 20 | 1.4710 | 46.6% |

---

## ResNet50

| Epoch | Loss | Train acc |
|-------|------|-----------|
| 1 | 1.2969 | 56.8% |
| 5 | 0.4918 | 85.1% |
| 10 | 0.1735 | 94.4% |
| 15 | 0.0282 | 99.3% |
| 20 | 0.0183 | 99.6% |

---

## EfficientNet-B0

| Epoch | Loss | Train acc |
|-------|------|-----------|
| 1 | 1.2133 | 59.3% |
| 5 | 0.3118 | 89.3% |
| 10 | 0.0940 | 96.7% |
| 15 | 0.0279 | 99.0% |
| 20 | 0.0243 | 99.0% |

---

## EfficientNet-B2

| Epoch | Loss | Train acc |
|-------|------|-----------|
| 1 | 1.1846 | 61.5% |
| 5 | 0.2978 | 89.7% |
| 10 | 0.0938 | 96.7% |
| 15 | 0.0367 | 98.7% |
| 20 | 0.0202 | 99.4% |

---

## Выводы

EfficientNet-B0 — лучший результат на тесте (82.9%) при минимальном времени обучения среди претренированных моделей.
EfficientNet-B2 даёт чуть хуже (82.7%) и в 1.4x медленнее — не оправдывает размер.
ResNet50 сильно переобучился (разрыв 18.2%) — худший из претренированных.
Simple CNN слабый по всем классам, особенно gun (13.3%), spg (29.7%), infantry (2.4%).

Класс ifv имеет всего 13 тест-сэмплов — слишком мало для надёжной оценки.
