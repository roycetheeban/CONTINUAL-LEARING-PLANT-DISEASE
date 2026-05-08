# CASE 3A — PlantVillage 26-Class Pre-training (→ Backbone)
## MobileNetV3-Small | 26 Non-Tomato Classes | Edge Plant/Leaf Classification

---

## Overview

Train MobileNetV3-Small from **ImageNet pre-trained weights** on 26 non-Tomato PlantVillage classes. The goal is **not** final classification performance — it is to produce a high-quality plant-domain backbone (`pretrained_backbone.pth`) that will be fine-tuned on 5 Tomato classes in Case 3B. The 26-class classifier head is discarded after this stage.

This is the most reusable artefact in the entire pipeline.

**Output:** `pretrained_backbone.pth` — state dict of `model.features` only.

---

## 1. Input

### Model Initialisation
```python
import torchvision.models as models
model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
# Replace classifier head for 26 classes
model.classifier[3] = nn.Linear(1024, 26)
```

### Data Paths

| Role | Path | Notes |
|---|---|---|
| Training data | `data/01_pretrain_26cls/train/` | 26 class folders |
| Test data | `data/01_pretrain_26cls/test/` | Used to verify backbone quality — not used in CL |

> **No validation split for pre-training.** Pre-training is not hyperparameter-sensitive at this stage. The test set is used only to confirm backbone quality before Case 3B fine-tuning proceeds.

### Class Folders — 26 Non-Tomato Classes
```
train/
├── Apple___Apple_scab/              (~504 images in train)
├── Apple___Black_rot/               (~497 images)
├── Apple___healthy/                 (~1,316 images)
├── Blueberry___healthy/             (~1,202 images)
├── Cherry___Powdery_mildew/         (~842 images)
├── Cherry___healthy/                (~683 images)
├── Corn___Cercospora_leaf_spot/     (~410 images)
├── Corn___Common_rust/              (~954 images)
├── Corn___Northern_Leaf_Blight/     (~788 images)
├── Corn___healthy/                  (~930 images)
├── Grape___Black_rot/               (~944 images)
├── Grape___Esca_(Black_Measles)/    (~1,106 images)
├── Grape___Leaf_blight/             (~861 images)
├── Grape___healthy/                 (~338 images)
├── Orange___Haunglongbing/          (~4,406 images)
├── Peach___Bacterial_spot/          (~1,838 images)
├── Peach___healthy/                 (~288 images)
├── Pepper_bell___Bacterial_spot/    (~798 images)
├── Pepper_bell___healthy/           (~1,182 images)
├── Potato___Early_blight/           (~800 images)
├── Potato___Late_blight/            (~800 images)
├── Raspberry___healthy/             (~297 images)
├── Soybean___healthy/               (~4,072 images)
├── Squash___Powdery_mildew/         (~1,468 images)
├── Strawberry___Leaf_scorch/        (~887 images)
└── Strawberry___healthy/            (~365 images)
```

> Verify exact counts from `data/data_stats.csv` before training. The figures above are 80% of total class counts (train split).

---

## 2. Class Imbalance — Class Weighting

The 26-class dataset has severe imbalance — up to **15:1 ratio** between largest (Orange Haunglongbing ~4,406) and smallest (Grape healthy ~338, Peach healthy ~288) classes.

**Solution:** Capped inverse-frequency class weighting with a **3× cap** to prevent extreme overrepresentation of rare classes destabilising training.

```python
import torch
import numpy as np

# Approximate train counts — update from data_stats.csv
class_counts_26 = torch.tensor([
    504., 497., 1316., 1202., 842., 683., 410., 954., 788., 930.,
    944., 1106., 861., 338., 4406., 1838., 288., 798., 1182., 800.,
    800., 297., 4072., 1468., 887., 365.
])

# Inverse frequency weights
class_weights = 1.0 / class_counts_26
class_weights = class_weights / class_weights.sum() * len(class_counts_26)

# Cap at 3× the minimum weight (prevents Orange/Soybean majority from being down-weighted too much)
min_weight = class_weights.min()
class_weights = torch.clamp(class_weights, max=3.0 * min_weight)

# Re-normalise after capping
class_weights = class_weights / class_weights.sum() * len(class_counts_26)

class_weights = class_weights.to(device)
criterion = nn.CrossEntropyLoss(weight=class_weights)
```

**Effect of the 3× cap:** Without capping, Orange (4,406 images) would receive an extremely low weight and Peach healthy (288 images) an extremely high weight — which would make the model focus almost entirely on the smallest classes at the expense of overall backbone quality. The cap ensures even the largest class retains meaningful gradient contribution.

---

## 3. Data Augmentation

Pre-training on 26 classes across multiple crop species benefits significantly from augmentation for generalisation across diverse leaf morphologies.

### Training Transforms
```python
train_transforms = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.3),
    transforms.RandomRotation(degrees=30),
    transforms.ColorJitter(
        brightness=0.3,
        contrast=0.3,
        saturation=0.3,
        hue=0.05
    ),
    transforms.RandomGrayscale(p=0.05),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])
```

### Test Transforms
```python
test_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])
```

### Augmentation Notes

- **Same aggressive augmentation as Case 1** — the 26-class dataset is large enough to support this and the model needs to generalise across multiple crop species.
- The diverse crop species (Apple, Grape, Orange, Corn, Soybean, etc.) already provide natural variation, but augmentation covers lighting and camera angle variation present in real-world field deployment.
- **RandomGrayscale 5%** — retained here to build backbone robustness. The backbone will later be fine-tuned on tomato disease where colour is important, but the backbone itself should not be colour-dependent.

---

## 4. Hyperparameters

### Layer Configuration

All layers train. This is full supervised pre-training, not fine-tuning.

| Group | Layers | Status | Learning Rate |
|---|---|---|---|
| G1 Early | features[0–3] | ✅ Training | 1e-3 |
| G2 Mid | features[4–8] | ✅ Training | 1e-3 |
| G3 Late | features[9–12] | ✅ Training | 1e-3 |
| G4 Head | classifier[0–3] (26-class) | ✅ Training | 1e-3 |

Starting from ImageNet weights means convergence is faster than Case 1, and the 26-class dataset (~25,490 training images) is large enough to support full fine-tuning without overfitting.

### Optimiser & Scheduler

```python
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='max',        # Monitor test accuracy
    patience=5,
    factor=0.5,
    min_lr=1e-6
)
```

### Training Settings

| Parameter | Value | Notes |
|---|---|---|
| Epochs (max) | 50 | Usually converges in 30–40 |
| Batch size | 64 | Larger dataset supports larger batch |
| Early stopping patience | 10 epochs | Stop if test acc does not improve |
| Min epochs before stopping | 15 | Ensure backbone has converged sufficiently |
| Image input size | 224×224 | |
| Weight decay | 1e-4 | |

> **Early stopping monitors test accuracy here** (not val F1, since there is no val split for pre-training). The test set is used only as a backbone quality check — it is not used in any downstream CL evaluation, so monitoring it during pre-training does not cause leakage.

---

## 5. Training Loop

```python
def train_case3a(model, train_loader, test_loader, optimizer, scheduler, criterion,
                 max_epochs=50, patience=10, min_epochs=15, device='cuda'):
    
    best_test_acc = 0.0
    best_backbone_state = None
    no_improve_count = 0

    for epoch in range(max_epochs):
        model.train()
        train_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Evaluate on test set (backbone quality check only)
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                preds = model(inputs).argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        test_acc = correct / total
        scheduler.step(test_acc)

        print(f"Epoch {epoch+1:02d} | Train Loss: {train_loss/len(train_loader):.4f} | Test Acc (26cls): {test_acc:.4f}")

        if test_acc > best_test_acc:
            best_test_acc = test_acc
            best_backbone_state = copy.deepcopy(model.features.state_dict())
            no_improve_count = 0
        else:
            no_improve_count += 1
            if epoch >= min_epochs and no_improve_count >= patience:
                print(f"Early stopping at epoch {epoch+1}. Best 26-cls test acc: {best_test_acc:.4f}")
                break

    return best_backbone_state
```

---

## 6. Output

### Critical: Save Backbone Only

```python
# Save the backbone state dict — NOT the full model
best_backbone_state = train_case3a(...)
torch.save(best_backbone_state, 'outputs/models/pretrained_backbone.pth')

# Verify save
backbone_check = models.mobilenet_v3_small(weights=None)
backbone_check.features.load_state_dict(torch.load('outputs/models/pretrained_backbone.pth'))
print("Backbone saved and reloadable.")
```

> **Only `model.features` is saved — NOT `model.classifier`.** The 26-class classifier head is discarded. `pretrained_backbone.pth` is the only artefact that matters from this stage.

### Saved Artefacts

| File | Path | Contents |
|---|---|---|
| Backbone only | `outputs/models/pretrained_backbone.pth` | `model.features` state dict — 26-class head discarded |
| Full model (optional) | `outputs/models/case3a_full.pth` | Full model for debugging backbone quality only |
| Training log | `outputs/logs/case3a_train_log.csv` | epoch, train_loss, test_acc per epoch |

### Backbone Quality Gate

Before proceeding to Case 3B, verify the backbone quality on the 26-class test set:

```
Minimum acceptable 26-class test accuracy: 75%
Target: 80%+
```

If test accuracy is below 75%, do not proceed to Case 3B. Investigate class weight configuration and training stability first.

---

## 7. Key Notes

- **The 26-class classifier head is disposable.** Its quality does not directly matter — what matters is that training it forced the backbone to learn plant-domain features.
- **Do not use the full 26-class model for any CL experiment.** Only `pretrained_backbone.pth` is passed to Case 3B.
- **Orange (5,507 total) and Soybean (5,090 total) dominate the dataset** — the 3× cap on class weights is critical here. Without it, the backbone could become biased toward these two classes' leaf morphology.
- **No replay buffer needed** for Case 3A — this is not a CL stage.

---

*Cross-reference: `DATA_README.md` § Section 2 | `LayerStrategy.md` § Section 2 — Case 3A | `CASE3B_PlantVillage_Finetune.md`*
