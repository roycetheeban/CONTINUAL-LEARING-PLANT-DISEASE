# CASE 1 — Train From Scratch (→ M1)
## MobileNetV3-Small | 5 Tomato Classes | Edge Plant/Leaf Classification

---

## Overview

Train MobileNetV3-Small from **random weight initialisation** (no pre-trained weights). All layers train simultaneously with uniform learning rate. This is the baseline model M1 — intentionally the weakest starting point to understand the benefit of pre-training in Cases 2 and 3.

**Output model:** `M1.pth`

---

## 1. Input

### Model Initialisation
```python
import torchvision.models as models
model = models.mobilenet_v3_small(weights=None)   # No pre-trained weights
# Replace classifier head for 5 classes
model.classifier[3] = nn.Linear(1024, 5)
```

### Data Paths

| Role | Path | Notes |
|---|---|---|
| Training data | `data/02_tomato_5cls/initial_train/` | 5 class folders |
| Validation data | `data/02_tomato_5cls/val/` | Used for early stopping |
| Test data | `data/02_tomato_5cls/test/` | **Fixed. Evaluate AFTER training only.** |

### Class Folders (same structure in all splits)
```
initial_train/
├── Tomato___Bacterial_spot/     ← T1 (~798 images)
├── Tomato___Early_blight/       ← T2 (~375 images)
├── Tomato___Late_blight/        ← T3 (~716 images)
├── Tomato___Leaf_Mold/          ← T4 (~357 images)
└── Tomato___healthy/            ← T5 (~597 images)
```

### Class ID Mapping
```python
class_to_idx = {
    "Tomato___Bacterial_spot": 0,   # T1
    "Tomato___Early_blight":   1,   # T2
    "Tomato___Late_blight":    2,   # T3
    "Tomato___Leaf_Mold":      3,   # T4
    "Tomato___healthy":        4    # T5
}
```

---

## 2. Class Imbalance — Class Weighting

**Problem:** The 5 classes have unequal image counts. T2 (~375) and T4 (~357) have roughly half the images of T1 (~798). Training without correction biases the model toward the majority classes.

**Solution:** Inverse-frequency class weights applied to the CrossEntropy loss.

```python
import torch
import numpy as np

# Counts from initial_train/ (update these from data_stats.csv)
class_counts = torch.tensor([798., 375., 716., 357., 597.])

# Inverse frequency weights — minority classes get higher weight
class_weights = 1.0 / class_counts
class_weights = class_weights / class_weights.sum() * len(class_counts)  # normalise to mean=1

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class_weights = class_weights.to(device)

criterion = nn.CrossEntropyLoss(weight=class_weights)
```

**Expected weights (approximate):**

| Class | Count | Approx. Weight |
|---|---|---|
| T1 Bacterial_spot | 798 | 0.62 |
| T2 Early_blight | 375 | 1.32 |
| T3 Late_blight | 716 | 0.69 |
| T4 Leaf_Mold | 357 | 1.39 |
| T5 healthy | 597 | 0.83 |

> **Cap:** If any weight exceeds 3× the minimum weight, cap it at 3× minimum. This prevents extreme minority classes from dominating gradient updates when counts are very skewed (not critical here, but good practice for forward compatibility with CL cycles).

---

## 3. Data Augmentation

Since Case 1 trains from scratch with no pre-trained weights and a small dataset (~2,843 images), augmentation is critical to prevent overfitting and improve generalisation.

### Training Transforms
```python
from torchvision import transforms

train_transforms = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomCrop(224),                    # Random crop after resize
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.3),          # Leaf orientation can be any
    transforms.RandomRotation(degrees=30),          # Rotational invariance for leaves
    transforms.ColorJitter(
        brightness=0.3,
        contrast=0.3,
        saturation=0.3,
        hue=0.05                                   # Small hue shift — diseases have colour signatures
    ),
    transforms.RandomGrayscale(p=0.05),            # Rare but helps with colour overfitting
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])  # ImageNet stats — good default
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

### Augmentation Rationale

| Augmentation | Why Applied |
|---|---|
| RandomCrop 224 from 256 | Positional invariance — lesions appear anywhere on leaf |
| RandomHorizontalFlip | Leaves have no preferred horizontal orientation |
| RandomVerticalFlip | Leaf tip can be up or down in field images |
| RandomRotation ±30° | Camera angle variation in edge deployment |
| ColorJitter | Lighting variation, camera white balance differences |
| Small hue shift (±0.05) | Preserve disease colour signatures; don't distort them |
| RandomGrayscale 5% | Prevents over-reliance on colour features alone |

> **Do NOT save augmented images to disk.** Apply augmentation on-the-fly in the DataLoader. Disk-saved augmented copies cause data leakage if they end up in val/test splits.

---

## 4. Hyperparameters

### Layer Configuration

| Group | Layers | Status | Learning Rate |
|---|---|---|---|
| G1 Early | features[0–3] | ✅ Training | 1e-3 |
| G2 Mid | features[4–8] | ✅ Training | 1e-3 |
| G3 Late | features[9–12] | ✅ Training | 1e-3 |
| G4 Head | classifier[0–3] | ✅ Training | 1e-3 |

All layers train simultaneously. No layer-wise LR differentiation — there are no pre-trained weights to protect.

### Optimiser & Scheduler

```python
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='max',        # Monitor val F1 (higher is better)
    patience=5,        # Wait 5 epochs of no improvement
    factor=0.5,        # Halve LR on plateau
    min_lr=1e-6
)
```

### Training Settings

| Parameter | Value | Notes |
|---|---|---|
| Epochs (max) | 60 | With early stopping; scratch training needs more epochs |
| Batch size | 32 | Suitable for MobileNetV3-Small on edge-class GPU |
| Early stopping patience | 10 epochs | Stop if val macro-F1 does not improve for 10 epochs |
| Min epochs before stopping | 10 | Avoid stopping on early noise |
| Image input size | 224×224 | MobileNetV3-Small native input |
| Weight decay | 1e-4 | L2 regularisation — important when training from scratch |

### Loss Function
```python
criterion = nn.CrossEntropyLoss(weight=class_weights)  # weighted as defined in Section 2
```

---

## 5. Training Loop

```python
def train_case1(model, train_loader, val_loader, optimizer, scheduler, criterion, 
                max_epochs=60, patience=10, min_epochs=10, device='cuda'):
    
    best_val_f1 = 0.0
    best_model_state = None
    no_improve_count = 0

    for epoch in range(max_epochs):
        # --- Training phase ---
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

        # --- Validation phase ---
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                preds = outputs.argmax(dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        val_f1 = f1_score(all_labels, all_preds, average='macro')
        scheduler.step(val_f1)

        print(f"Epoch {epoch+1:02d} | Train Loss: {train_loss/len(train_loader):.4f} | Val F1: {val_f1:.4f}")

        # --- Early stopping ---
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_model_state = copy.deepcopy(model.state_dict())
            no_improve_count = 0
        else:
            no_improve_count += 1
            if epoch >= min_epochs and no_improve_count >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    model.load_state_dict(best_model_state)
    return model
```

---

## 6. Output

### Saved Artefacts

| File | Path | Contents |
|---|---|---|
| M1 model | `outputs/models/M1.pth` | Full model state dict (features + classifier) |
| Training log | `outputs/logs/case1_train_log.csv` | epoch, train_loss, val_f1, lr per epoch |
| Val F1 curve | `outputs/plots/case1_val_f1.png` | Validation F1 over epochs |

```python
# Save M1
torch.save(model.state_dict(), 'outputs/models/M1.pth')

# Save training log
import pandas as pd
log_df = pd.DataFrame(log_records)  # list of dicts per epoch
log_df.to_csv('outputs/logs/case1_train_log.csv', index=False)
```

### Evaluation on Test Set

Run **after training completes** using the fixed test set. This is the Part 1 baseline for M1.

```python
def evaluate(model, test_loader, class_names, device='cuda'):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    from sklearn.metrics import classification_report, confusion_matrix
    print(classification_report(all_labels, all_preds, target_names=class_names))
    return classification_report(all_labels, all_preds, target_names=class_names, output_dict=True)

# Test data path: data/02_tomato_5cls/test/
```

### Expected Output Format — Test Evaluation

```
Classification Report — M1 (Scratch) — Part 1 Baseline
                             precision  recall  f1-score  support
Tomato___Bacterial_spot         x.xx    x.xx      x.xx      319
Tomato___Early_blight           x.xx    x.xx      x.xx      113
Tomato___Late_blight            x.xx    x.xx      x.xx      286
Tomato___Leaf_Mold              x.xx    x.xx      x.xx      107
Tomato___healthy                x.xx    x.xx      x.xx      239
macro avg                       x.xx    x.xx      x.xx     1064
```

---

## 7. Key Notes

- **Expect lower accuracy than M2 and M3.** Training from scratch on ~2,843 images is hard for MobileNetV3-Small. This model exists to quantify the benefit of pre-training, not to be the best performer.
- **Overfitting risk is highest here.** Monitor the gap between train loss and val F1 carefully. If val F1 plateaus early while train loss continues to drop, the model is overfitting — weight decay and augmentation are the primary defences.
- **No backbone saving** — M1's backbone is not reused by any other Case. Only M1.pth is needed downstream (for CL experiments).
- **Replay buffer sampling** — after training completes, sample the replay buffer from `initial_train/`. See `DATA_README.md` Section 3 for buffer sizing rules.

---

*Cross-reference: `DATA_README.md` § Section 3 | `LayerStrategy.md` § Section 2 — Case 1 | `CL_Implementation_Flow.md`*
