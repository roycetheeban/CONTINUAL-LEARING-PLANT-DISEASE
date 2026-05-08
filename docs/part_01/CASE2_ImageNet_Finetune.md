# CASE 2 — ImageNet Pre-trained Fine-tune (→ M2)
## MobileNetV3-Small | 5 Tomato Classes | Edge Plant/Leaf Classification

---

## Overview

Fine-tune a MobileNetV3-Small model pre-trained on ImageNet. Uses **two-phase progressive unfreezing**: first train only the classifier head, then unfreeze the late feature layers (Group 3) for disease-specific adaptation. The early and mid backbone layers remain frozen throughout — ImageNet low/mid-level features transfer well to leaf imagery.

**Output model:** `M2.pth`

---

## 1. Input

### Model Initialisation
```python
import torchvision.models as models
model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
# Replace classifier head for 5 classes
model.classifier[3] = nn.Linear(1024, 5)
# New classifier[3] is randomly initialised by default
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

Same class weighting strategy as Case 1. Apply inverse-frequency weights to CrossEntropy loss.

```python
class_counts = torch.tensor([798., 375., 716., 357., 597.])
class_weights = 1.0 / class_counts
class_weights = class_weights / class_weights.sum() * len(class_counts)
class_weights = class_weights.to(device)

criterion = nn.CrossEntropyLoss(weight=class_weights)
```

> Cap any weight at 3× the minimum weight. Update counts from `data/data_stats.csv` before training.

---

## 3. Data Augmentation

ImageNet pre-training provides a strong initialisation, reducing overfitting risk compared to Case 1. However, augmentation is still important — the ImageNet domain (natural objects, animals) differs from leaf disease imagery.

### Training Transforms
```python
train_transforms = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.3),
    transforms.RandomRotation(degrees=20),         # Slightly less rotation than Case 1
    transforms.ColorJitter(
        brightness=0.25,
        contrast=0.25,
        saturation=0.25,
        hue=0.04
    ),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])  # ImageNet stats — match pre-training
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

### Augmentation Notes for Case 2

- Use **ImageNet normalisation stats** — the pre-trained backbone expects this normalisation. Using different stats degrades the pre-trained feature quality immediately.
- Augmentation is slightly milder than Case 1 (lower rotation, lower jitter) because the ImageNet backbone already generalises well. Aggressive augmentation can destroy pre-trained feature patterns in Phase ii when G3 is unfrozen.
- **No RandomGrayscale** for Case 2 — the ImageNet backbone has learned colour-based features that are valuable for leaf disease; do not randomly discard colour information.

---

## 4. Two-Phase Training

### Phase i — Classifier Head Only (10–15 epochs)

**Goal:** Adapt the 1000-class ImageNet head to 5-class tomato disease classification without disturbing the pre-trained backbone.

#### Layer Configuration

| Group | Layers | Status | Learning Rate |
|---|---|---|---|
| G1 Early | features[0–3] | ❄ Frozen | 0 |
| G2 Mid | features[4–8] | ❄ Frozen | 0 |
| G3 Late | features[9–12] | ❄ Frozen | 0 |
| G4 Head | classifier[0–3] | ✅ Training | 1e-3 |

#### Setup
```python
# Freeze entire backbone
for param in model.features.parameters():
    param.requires_grad = False

# Confirm only classifier is trainable
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Trainable params Phase i: {trainable:,}")   # Should be ~1.05M (classifier only)

optimizer_ph1 = torch.optim.Adam(model.classifier.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler_ph1 = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer_ph1, mode='max', patience=3, factor=0.5, min_lr=1e-6
)
```

#### Hyperparameters — Phase i

| Parameter | Value |
|---|---|
| Epochs | 10–15 (stop when val F1 plateaus) |
| Batch size | 32 |
| LR | 1e-3 |
| Early stopping patience | 5 epochs |
| Min epochs before stopping | 5 |

#### Phase i Completion Criteria
- Val macro-F1 has plateaued for 5 consecutive epochs, OR
- 15 epochs reached

Save intermediate checkpoint:
```python
torch.save(model.state_dict(), 'outputs/models/M2_phase1.pth')
```

---

### Phase ii — Unfreeze Group 3 (10 more epochs)

**Goal:** Adapt the disease-discriminative feature layers (G3) from generic ImageNet object features to leaf disease-specific features.

**Do NOT unfreeze G1 or G2.** The 5-class tomato dataset is too small for full backbone fine-tuning and will overfit.

#### Layer Configuration

| Group | Layers | Status | Learning Rate |
|---|---|---|---|
| G1 Early | features[0–3] | ❄ Frozen | 0 |
| G2 Mid | features[4–8] | ❄ Frozen | 0 |
| G3 Late | features[9–12] | ~ Partial | 1e-4 (0.1× head LR) |
| G4 Head | classifier[0–3] | ✅ Training | 1e-3 |

#### Setup
```python
# Unfreeze Group 3 only
for param in model.features[9:].parameters():
    param.requires_grad = True

# Verify Group 1 and 2 remain frozen
for param in model.features[:9].parameters():
    assert not param.requires_grad, "G1/G2 should remain frozen!"

optimizer_ph2 = torch.optim.Adam([
    {'params': model.features[9:].parameters(), 'lr': 1e-4},
    {'params': model.classifier.parameters(),   'lr': 1e-3}
], weight_decay=1e-4)

scheduler_ph2 = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer_ph2, mode='max', patience=5, factor=0.5, min_lr=1e-6
)
```

#### Hyperparameters — Phase ii

| Parameter | Value |
|---|---|
| Epochs | 10–15 |
| Batch size | 32 |
| G3 LR | 1e-4 |
| Head LR | 1e-3 |
| Early stopping patience | 5 epochs |
| Min epochs before stopping | 5 |

#### Why 0.1× LR for G3

The ImageNet backbone's G3 features are already useful (they detect generic object boundaries, textures at the right scale). Updating at 1e-4 instead of 1e-3 makes small corrections toward leaf disease patterns without catastrophically disrupting the pre-trained representation. A full 1e-3 on G3 would erase ImageNet features too aggressively given the small dataset size.

---

## 5. Training Loop

```python
def train_phase(model, train_loader, val_loader, optimizer, scheduler, criterion,
                phase_name, max_epochs=15, patience=5, min_epochs=5, device='cuda'):
    
    best_val_f1 = 0.0
    best_model_state = None
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

        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                preds = model(inputs).argmax(dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        val_f1 = f1_score(all_labels, all_preds, average='macro')
        scheduler.step(val_f1)

        print(f"[{phase_name}] Epoch {epoch+1:02d} | Train Loss: {train_loss/len(train_loader):.4f} | Val F1: {val_f1:.4f}")

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
    return model, best_val_f1

# Run Phase i
model, ph1_best_f1 = train_phase(model, train_loader, val_loader, optimizer_ph1, 
                                  scheduler_ph1, criterion, "Phase-i", max_epochs=15)
torch.save(model.state_dict(), 'outputs/models/M2_phase1.pth')

# Setup Phase ii (unfreeze G3 as above), then run
model, ph2_best_f1 = train_phase(model, train_loader, val_loader, optimizer_ph2, 
                                  scheduler_ph2, criterion, "Phase-ii", max_epochs=15)
```

---

## 6. Output

### Saved Artefacts

| File | Path | Contents |
|---|---|---|
| M2 Phase i checkpoint | `outputs/models/M2_phase1.pth` | After Phase i completion |
| M2 final model | `outputs/models/M2.pth` | After Phase ii — this is the CL starting model |
| Training log | `outputs/logs/case2_train_log.csv` | epoch, phase, train_loss, val_f1, lr per epoch |
| Val F1 curve | `outputs/plots/case2_val_f1.png` | Both phases on one plot |

```python
# Save final M2
torch.save(model.state_dict(), 'outputs/models/M2.pth')
```

### Test Evaluation

```python
# Test path: data/02_tomato_5cls/test/
# Run evaluate() function (same as Case 1) on M2 after Phase ii completes
```

**Expected:** M2 should outperform M1 (scratch) significantly on macro-F1, especially on the minority classes T2 and T4 where limited training data most benefits from pre-trained features.

---

## 7. Key Rules

1. **Never unfreeze G1 (features[0–3]) or G2 (features[4–8]) in any phase of Case 2.** The 5-class tomato dataset is too small — full backbone fine-tuning will overfit.
2. **Always use ImageNet normalisation stats** (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) — the pre-trained backbone expects them.
3. **G3 LR must be 0.1× the head LR in Phase ii.** Equal LR would update G3 too aggressively with this small dataset.
4. **Load Phase i best weights before starting Phase ii** — always start Phase ii from the best Phase i checkpoint, not the final epoch.
5. **Replay buffer sampling** — after M2 training completes, sample the replay buffer from `initial_train/`. See `DATA_README.md` Section 3.

---

*Cross-reference: `DATA_README.md` § Section 3 | `LayerStrategy.md` § Section 2 — Case 2 | `CL_Implementation_Flow.md`*
