# Training Results — efficientnet_b0_1

## Цель

Сравнение влияния отдельных параметров на точность EfficientNet-B0.
Baseline: comparison/efficientnet_b0 — **82.9%** test accuracy.
Каждый прогон меняет ровно один параметр относительно baseline.

## Общие параметры

| Parameter | Value |
|-----------|-------|
| Model | EfficientNet-B0 |
| Image size | 224×224 |
| Batch size | 32 |
| Train/test split | 70/30 |
| Seed | 67 |
| Classes | 9 |
| Scheduler | CosineAnnealingLR |

---

## Результаты

| Прогон | Что изменено | Test acc | Δ от baseline |
|--------|-------------|----------|---------------|
| baseline (comparison) | — | 82.9% | — |
| `lr_3e4` | LR: 1e-3 → **3e-4** | 82.7% | −0.2% |
| `aug_strong` | + RandomGrayscale + RandomErasing | 82.1% | −0.8% |
| `epochs_40` | Epochs: 20 → **40** | 81.7% | −1.2% |
| `freeze` | Backbone заморожен первые 5 эпох | 81.7% | −1.2% |
| `wd` | Weight decay: 0 → **1e-4** | 81.7% | −1.2% |

---

## Выводы

Ни один из параметров не улучшил baseline. Возможные причины:
- Датасет небольшой (~750 train images), поэтому тонкая настройка мало помогает
- Scheduler уже был в baseline, что сглаживает эффект от изменения LR
- `gun` класс стабильно слабый (~70-77%) — мало данных (56 изображений всего)

**Наименьший вред** от снижения LR до 3e-4 (−0.2%).
**Наибольший вред** от weight decay, freeze и 40 эпох (−1.2% каждый).

---

## Per-class (лучший прогон: lr_3e4)

| Class | Correct | Total | Accuracy |
|-------|---------|-------|----------|
| apc | 58 | 83 | 69.9% |
| armored_vehicle | 78 | 101 | 77.2% |
| gun | 10 | 13 | 76.9% |
| ifv | 114 | 138 | 82.6% |
| infantry | 41 | 55 | 74.5% |
| mlrs | 107 | 116 | 92.2% |
| spg | 32 | 41 | 78.0% |
| tank | 119 | 135 | 88.1% |
| uav | 73 | 82 | 89.0% |
