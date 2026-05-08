# CASE 3B — PlantVillage Fine-tune on 5 Tomato Classes (→ M3)
## MobileNetV3-Small | 5 Tomato Classes | Edge Plant/Leaf Classification

---

## Overview

Fine-tune the plant-domain backbone produced in Case 3A (`pretrained_backbone.pth`) on the 5 Tomato disease classes. Uses **two-phase progressive unfreezing** with substantially lower learning rates than Case 2 — the backbone already contains plant-domain features, so less correction is needed (less domain shift than ImageNet→leaf).

**Input:** `outputs/models/pretrained_backbone.pth` (from Case 3A)
**Output model:** `M3.pth`

---

## 1. Input

### Model Initialisation
```python
import torchvision.models as models

# Load architecture
model = models.mobilenet_v3_small(weights=None)

# Load plant-domain backbone from Case 3A
backbone_state = torch.load('outputs/models/pretrained_backbone.pth')
model.features.load_state_dict(backbone_state)

# Replace classifier head for 5 Tomato classes
model.classifier[3] = nn.Linear(1024, 5)
# classifier[3] is randomly initialised; classifier[0–2] retain ImageNet->PlantVillage weights

print("Backbone loaded from Case 3A. Classifier head reset for 5 classes.")
```

### Data Paths

| Role | Path | Notes |
|---|---|---|
| Training data | `data/02_tomato_5cls/initial_train/` | 5 class folders |
| Validation data | `data/02_tomato_5cls/val/` | Used for early stopping |
| Test data | `data/02_tomato_5cls/test/` | **Fixed. Evaluate AFTER both phases.** |

### Class Folders
```
initial_train/
├── Tomato___Bacterial_spot/     ← T1 (~798 images)
├── Tomato___Early_blight/       ← T2 (~375 images)
├── Tomato___Late_blight/        ← T3 (~716 images)
├── Tomato___Leaf_Mold/          ← T4 (~357 images)
└── Tomato___healthy/            ← T5 (~597 images)
```

---

## 2. Class Imbalance — Class Weighting

Same strategy as Cases 1 and 2.

```python
class_counts = torch.tensor([798., 375., 716., 357., 597.])
class_weights = 1.0 / class_counts
class_weights = class_weights / class_weights.sum() * len(class_counts)
class_weights = class_weights.to(device)

criterion = nn.CrossEntropyLoss(weight=class_weights)
```

> Update counts from `data/data_stats.csv`. Cap at 3× minimum weight.

---

## 3. Data Augmentation

### Training Transforms
```python
train_transforms = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.3),
    transforms.RandomRotation(degrees=20),
    transforms.ColorJitter(
        brightness=0.2,
        contrast=0.2,
        saturation=0.2,
        hue=0.04        # Smaller hue shift — tomato diseases have distinctive colour signatures
    ),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])
```

### Validation / Test Transforms
```python
val_test_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])
```

### Augmentation Notes for Case 3B

- **Milder than Cases 1 and 3A** — the backbone already encodes plant-domain features. Aggressive augmentation is less critical and could interfere with fine-tuning at the very low LRs used.
- **Smaller hue jitter (0.04)** — Tomato diseases have specific colour signatures (orange-brown bacterial spot, yellow-green leaf mold, dark lesions in late blight). Larger hue shifts risk blurring these within-class colour cues.
- **No RandomGrayscale** — colour is highly informative for these tomato disease classes. The backbone was already built with varied colour augmentation in Case 3A.

---

## 4. Two-Phase Training

### Phase i — Late Layers + Classifier Head (10–15 epochs)

**Goal:** Adapt G3 (disease-discriminative features) and the classifier head from 26-class PlantVillage patterns to 5 specific tomato diseases. G1 and G2 are frozen — they already have excellent plant-domain features from Case 3A.

#### Layer Configuration

| Group | Layers | Status | Learning Rate |
|---|---|---|---|
| G1 Early | features[0–3] | ❄ Frozen | 0 |
| G2 Mid | features[4–8] | ❄ Frozen | 0 |
| G3 Late | features[9–12] | ~ Partial | 5e-5 (very low) |
| G4 Head | classifier[0–3] | ✅ Training | 5e-4 |

**Why LRs are 2–5× lower than Case 2:**
The backbone already contains plant-domain features from PlantVillage pre-training. Even G3 already has leaf-level disease representations — it only needs small adjustments from 26-class generic disease patterns to 5 specific tomato classes. Using Case 2's LRs (1e-4 for G3, 1e-3 for head) here would over-update the backbone.

#### Setup
```python
# Freeze G1 and G2
for param in model.features[:9].parameters():
    param.requires_grad = False

# G3 unfrozen at very low LR
for param in model.features[9:].parameters():
    param.requires_grad = True

# Verify
frozen_count = sum(1 for p in model.features[:9].parameters() if not p.requires_grad)
print(f"Frozen params in G1+G2: {frozen_count} parameter groups")

optimizer_ph1 = torch.optim.Adam([
    {'params': model.features[9:].parameters(), 'lr': 5e-5},
    {'params': model.classifier.parameters(),   'lr': 5e-4}
], weight_decay=1e-5)   # Lower weight decay — backbone already well-regularised

scheduler_ph1 = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer_ph1, mode='max', patience=3, factor=0.5, min_lr=1e-7
)
```

#### Hyperparameters — Phase i

| Parameter | Value |
|---|---|
| Epochs | 10–15 |
| Batch size | 32 |
| G3 LR | 5e-5 |
| Head LR | 5e-4 |
| Early stopping patience | 5 epochs |
| Min epochs before stopping | 5 |

---

### Phase ii — Unfreeze Group 2 (Optional, 5–10 epochs)

**Goal:** Allow Group 2 (plant mid-level features) to make very small tomato-specific adjustments — e.g., the distinctive venation patterns of tomato leaves vs. corn, grape, apple leaves seen in pre-training.

**This phase is optional.** Run Phase i first and evaluate on val. If val macro-F1 has converged and the model already generalises well across all 5 classes, Phase ii can be skipped. Run Phase ii only if there is a persistent class-level performance gap (e.g., T4 Leaf Mold consistently underperforms).

#### Layer Configuration

| Group | Layers | Status | Learning Rate |
|---|---|---|---|
| G1 Early | features[0–3] | ❄ Frozen | 0 (ALWAYS frozen) |
| G2 Mid | features[4–8] | ~ Partial | 1e-5 (very very low) |
| G3 Late | features[9–12] | ~ Partial | 5e-5 |
| G4 Head | classifier[0–3] | ✅ Training | 5e-4 |

#### Setup
```python
# Unfreeze G2 at very low LR
for param in model.features[4:9].parameters():
    param.requires_grad = True

# G1 stays frozen — NEVER update G1 in Case 3B
for param in model.features[:4].parameters():
    param.requires_grad = False

optimizer_ph2 = torch.optim.Adam([
    {'params': model.features[4:9].parameters(), 'lr': 1e-5},   # G2 — very slow
    {'params': model.features[9:].parameters(),  'lr': 5e-5},   # G3
    {'params': model.classifier.parameters(),    'lr': 5e-4}    # Head
], weight_decay=1e-5)

scheduler_ph2 = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer_ph2, mode='max', patience=5, factor=0.5, min_lr=1e-7
)
```

#### Hyperparameters — Phase ii

| Parameter | Value |
|---|---|
| Epochs | 5–10 |
| Batch size | 32 |
| G2 LR | 1e-5 |
| G3 LR | 5e-5 |
| Head LR | 5e-4 |
| Early stopping patience | 5 epochs |
| Min epochs before stopping | 3 |

**G1 (features[0–3]) stays frozen in ALL phases of Case 3B without exception.** The pre-trained backbone already has excellent low-level plant features. There is zero benefit to updating these with 5-class tomato data.

---

## 5. Training Loop

Use the same `train_phase()` function defined in Case 2, applied twice:

```python
# Phase i
model, ph1_best_f1 = train_phase(
    model, train_loader, val_loader, optimizer_ph1, scheduler_ph1, criterion,
    "3B-Phase-i", max_epochs=15, patience=5, min_epochs=5
)
torch.save(model.state_dict(), 'outputs/models/M3_phase1.pth')
print(f"Phase i best val F1: {ph1_best_f1:.4f}")

# Decision: run Phase ii?
# If ph1_best_f1 >= 0.90 and per-class F1 is balanced → skip Phase ii
# Otherwise → proceed

# Phase ii (optional)
# (set up optimizer_ph2 as above)
model, ph2_best_f1 = train_phase(
    model, train_loader, val_loader, optimizer_ph2, scheduler_ph2, criterion,
    "3B-Phase-ii", max_epochs=10, patience=5, min_epochs=3
)
```

---

## 6. Output

### Saved Artefacts

| File | Path | Contents |
|---|---|---|
| M3 Phase i checkpoint | `outputs/models/M3_phase1.pth` | After Phase i |
| M3 final model | `outputs/models/M3.pth` | After Phase ii (or Phase i if Phase ii skipped) |
| Training log | `outputs/logs/case3b_train_log.csv` | epoch, phase, train_loss, val_f1, lr |
| Val F1 curve | `outputs/plots/case3b_val_f1.png` | |

```python
# Save final M3
torch.save(model.state_dict(), 'outputs/models/M3.pth')
```

### Test Evaluation

```python
# Test path: data/02_tomato_5cls/test/
# Run evaluate() on M3 after training completes
```

**Expected:** M3 should be the strongest of the three base models (M1, M2, M3), benefiting from both plant-domain pre-training and tomato-specific fine-tuning. Particular strength expected on minority classes T2 and T4.

---

## 7. Comparison — Case 3B vs Case 2 LR Strategy

| Aspect | Case 2 (ImageNet FT) | Case 3B (PlantVillage FT) | Reason |
|---|---|---|---|
| G3 Phase i LR | frozen | 5e-5 | PlantVillage G3 already has plant disease features — small adjustment only |
| G3 Phase ii LR | 1e-4 | 5e-5 (same) | Maintain consistency — already at task-relevant scale |
| Head Phase i LR | 1e-3 | 5e-4 | Less head correction needed — 26-class classification already learned class-discriminative patterns |
| G2 unfreezing | Never | Optional at 1e-5 | PlantVillage G2 already has plant mid-features; may need slight tomato-specific adjustment |
| G1 unfreezing | Never | Never | Universal features — no benefit from updating |

---

## 8. Key Rules

1. **Always load `pretrained_backbone.pth` — never re-train from scratch.** If the backbone file is missing, run Case 3A first.
2. **G1 (features[0–3]) is ALWAYS frozen** in all phases of Case 3B. No exception.
3. **Phase ii is optional** — only run if Phase i shows persistent per-class F1 imbalance on val.
4. **Never increase LR between phases.** Phase ii LRs must be ≤ Phase i LRs.
5. **Load Phase i best weights before Phase ii** — start Phase ii from the best Phase i checkpoint.
6. **Replay buffer sampling** — after M3 training completes, sample the replay buffer from `initial_train/`. See `DATA_README.md` Section 3.

---

*Cross-reference: `DATA_README.md` § Section 3 | `LayerStrategy.md` § Section 2 — Case 3B | `CASE3A_PlantVillage_Pretrain.md` | `CL_Implementation_Flow.md`*
