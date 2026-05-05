# 🌿 Continual Learning Research Journal
## Edge Plant/Leaf Classification — MobileNetV3 (PlantVillage Dataset)

> **Research Stage:** Continual Learning Model Training (Pipeline, Image Collection, Segmentation, and GAN stages already complete)
> **Model:** MobileNetV3-Small (Edge-Compatible)
> **Dataset:** PlantVillage (54,306 images / 38 classes / 14 crop species)
> **Last Updated:** 2025

---

## 📋 TABLE OF CONTENTS
1. [Dataset Analysis](#1-dataset-analysis)
2. [Data Division Strategy](#2-data-division-strategy)
3. [Experiment Structure Overview](#3-experiment-structure-overview)
4. [Part 1 — Initial Training (3 Cases)](#4-part-1--initial-training-3-cases)
5. [Part 2 — Continual Learning Scenarios](#5-part-2--continual-learning-scenarios)
6. [Evaluation Protocol](#6-evaluation-protocol)
7. [Outputs & Artifacts to Store](#7-outputs--artifacts-to-store)
8. [Folder Structure](#8-folder-structure)
9. [To-Do Checklist](#9-to-do-checklist)

---

## 1. Dataset Analysis

### 1.1 PlantVillage Overview

| Property | Value |
|---|---|
| Total Images | 54,306 |
| Total Classes | 38 |
| Crop Species | 14 |
| Diseases Covered | 26 |
| Image Format | JPEG / RGB |
| Image Versions | Color, Grayscale, Segmented |
| Pre-existing Augmentation | No (original = unaugmented; augmented version has ~87K via offline augmentation) |
| Recommended Split | 80% Train / 20% Test (leaf-grouped to prevent data leakage) |
| Source | PlantVillage / HuggingFace: `mohanty/PlantVillage` |

> ⚠️ **Note on Augmentation:** The raw PlantVillage dataset from GitHub/HuggingFace is **NOT pre-augmented**. The Kaggle "New Plant Diseases Dataset" (~87K images) IS an offline-augmented version. We will use the **original unaugmented** dataset and apply our own augmentation pipeline per training case.

---

### 1.2 Full Class Distribution (All 38 Classes)

| # | Class Label | Crop | Condition | Approx. Images |
|---|---|---|---|---|
| 1 | Apple___Apple_scab | Apple | Fungal | 630 |
| 2 | Apple___Black_rot | Apple | Fungal | 621 |
| 3 | Apple___Cedar_apple_rust | Apple | Fungal | 275 |
| 4 | Apple___healthy | Apple | Healthy | 1,645 |
| 5 | Blueberry___healthy | Blueberry | Healthy | 1,502 |
| 6 | Cherry___Powdery_mildew | Cherry | Fungal | 1,052 |
| 7 | Cherry___healthy | Cherry | Healthy | 854 |
| 8 | Corn___Cercospora_leaf_spot | Corn | Fungal | 513 |
| 9 | Corn___Common_rust | Corn | Fungal | 1,192 |
| 10 | Corn___Northern_Leaf_Blight | Corn | Fungal | 985 |
| 11 | Corn___healthy | Corn | Healthy | 1,162 |
| 12 | Grape___Black_rot | Grape | Fungal | 1,180 |
| 13 | Grape___Esca_(Black_Measles) | Grape | Fungal | 1,383 |
| 14 | Grape___Leaf_blight | Grape | Bacterial | 1,076 |
| 15 | Grape___healthy | Grape | Healthy | 423 |
| 16 | Orange___Haunglongbing | Orange | Bacterial | 5,507 |
| 17 | Peach___Bacterial_spot | Peach | Bacterial | 2,297 |
| 18 | Peach___healthy | Peach | Healthy | 360 |
| 19 | Pepper_bell___Bacterial_spot | Bell Pepper | Bacterial | 997 |
| 20 | Pepper_bell___healthy | Bell Pepper | Healthy | 1,478 |
| 21 | Potato___Early_blight | Potato | Fungal | 1,000 |
| 22 | Potato___Late_blight | Potato | Oomycete | 1,000 |
| 23 | Potato___healthy | Potato | Healthy | 152 |
| 24 | Raspberry___healthy | Raspberry | Healthy | 371 |
| 25 | Soybean___healthy | Soybean | Healthy | 5,090 |
| 26 | Squash___Powdery_mildew | Squash | Fungal | 1,835 |
| 27 | Strawberry___Leaf_scorch | Strawberry | Fungal | 1,109 |
| 28 | Strawberry___healthy | Strawberry | Healthy | 456 |
| **29** | **Tomato___Bacterial_spot** | **Tomato** | **Bacterial** | **2,127** |
| **30** | **Tomato___Early_blight** | **Tomato** | **Fungal** | **1,000** |
| **31** | **Tomato___Late_blight** | **Tomato** | **Oomycete** | **1,909** |
| **32** | **Tomato___Leaf_Mold** | **Tomato** | **Fungal** | **952** |
| 33 | Tomato___Septoria_leaf_spot | Tomato | Fungal | 1,771 |
| 34 | Tomato___Spider_mites | Tomato | Mite | 1,676 |
| 35 | Tomato___Target_Spot | Tomato | Fungal | 1,404 |
| 36 | Tomato___Tomato_Yellow_Leaf_Curl_Virus | Tomato | Viral | 5,357 |
| 37 | Tomato___Tomato_mosaic_virus | Tomato | Viral | 373 |
| **38** | **Tomato___healthy** | **Tomato** | **Healthy** | **1,591** |
| | **TOTAL** | | | **~54,306** |

> **Bold rows** = Selected 5 Tomato classes for main experiment.

---

## 2. Data Division Strategy

### 2.1 Class Assignments

#### 🔵 Pre-training Pool (Case 3 — PlantVillage 26 Classes)
Used for: **Case 3 PlantVillage pre-training backbone** (non-tomato classes)

Classes 1–28 (all non-tomato) = 28 classes available → Select **26 classes** (exclude 2 smallest: Potato___healthy (152 imgs) and Grape___healthy (423 imgs) to reduce class imbalance risk, or as decided by researcher).

**Selected 26 Pre-training Classes:**
Classes: #1–12, #14–27 (i.e., 26 non-tomato classes with sufficient samples)

> ✏️ *Finalize selection locally — document which 26 are chosen before training.*

#### 🟠 Main Experiment Classes — 5 Tomato Classes (Initial)

| ID | Class | Images |
|---|---|---|
| T1 | Tomato___Bacterial_spot | ~2,127 |
| T2 | Tomato___Early_blight | ~1,000 |
| T3 | Tomato___Late_blight | ~1,909 |
| T4 | Tomato___Leaf_Mold | ~952 |
| T5 | Tomato___healthy | ~1,591 |
| | **Subtotal** | **~7,579** |

#### 🟢 CL Category B — 2 New Classes Added (5 → 7 Classes)

| ID | Class | Images |
|---|---|---|
| T6 | Tomato___Septoria_leaf_spot | ~1,771 |
| T7 | Tomato___Spider_mites | ~1,676 |
| | **New Class Subtotal** | **~3,447** |

---

### 2.2 Train / Validation / Test Splits

#### For the 5 Tomato Classes (Initial Training + CL Cycles)

Use **leaf-grouped stratified split** to prevent data leakage:

| Split | Ratio | Purpose |
|---|---|---|
| Train | 70% | Model learning |
| Validation | 15% | Hyperparameter tuning / early stopping |
| Test (Old) | 15% | Held-out evaluation — **NEVER used during training** |

> **CL Cycle Data:** Divide Train further:
> - **Cycle 1 Stream:** 50% of training data (initial model training)
> - **Cycle 2 Stream:** Remaining 50% (simulates new data arriving later)
> - This simulates a real-world continual data stream.

#### Replay Buffer (for EWC / Experience Replay methods)
- Keep **10–15% of Cycle 1 training samples** as replay buffer
- Store in `replay_buffer/` folder
- **Do not use test or validation samples in replay buffer**

#### For Category B New Classes (T6, T7)
- Same 70/15/15 split applied independently
- New class test samples = part of the "New Class Test Set"

---

### 2.3 Data Preparation Checklist (Local — Done Before Training)

> ✅ Complete these **locally before starting practicals**

- [ ] Download PlantVillage color images from HuggingFace (`mohanty/PlantVillage`)
- [ ] Verify image counts per class match table above
- [ ] Extract 5 tomato folders → `data/tomato_5cls/`
- [ ] Extract 26 non-tomato folders → `data/pretrain_26cls/`
- [ ] Extract 2 new tomato folders → `data/tomato_new_2cls/`
- [ ] Apply leaf-grouped stratified split (use `leaf_id` from HuggingFace metadata)
- [ ] Create: `train_cycle1/`, `train_cycle2/`, `val/`, `test_old/`, `test_new/`
- [ ] Create `replay_buffer/` from Cycle 1 train (10–15% per class)
- [ ] Save split metadata as `split_info.json` (image paths + labels + split assignment)
- [ ] Verify no image overlap across splits
- [ ] Log class counts per split in `data_stats.csv`

---

## 3. Experiment Structure Overview

```
PlantVillage (38 Classes)
       │
       ├── 26 Non-Tomato Classes ──────────────────→ CASE 3 Pre-training Backbone
       │
       └── 5 Tomato Classes ──────────────────────→ PART 1: Initial Training (M1, M2, M3)
                │
                ├─────────────────────────────────→ PART 2A: Same Classes CL (5 classes)
                │                                    (New Data Stream, 2 Cycles)
                │
                └──── + 2 New Tomato Classes ──────→ PART 2B: New Class CL (5 → 7 classes)
                                                     (2 Cycles)
                                                          │
                                                          └──→ Edge Deployment
                                                               (Pruning + Quantization)
```

---

## 4. Part 1 — Initial Training (3 Cases)

> **Goal:** Produce 3 base models M1, M2, M3 trained on 5 Tomato classes

### 4.1 Common Training Configuration

| Parameter | Value |
|---|---|
| Model Architecture | MobileNetV3-Small |
| Input Size | 224 × 224 × 3 |
| Output Classes | 5 (Tomato classes) |
| Loss Function | Cross-Entropy |
| Optimizer | Adam |
| Batch Size | 32 |
| Max Epochs | 50 (with early stopping, patience=10) |
| LR Schedule | ReduceLROnPlateau |
| Augmentation | Random flip, rotation (±15°), color jitter, random crop |

---

### 4.2 Case 1 — Train From Scratch (→ M1)

**Description:** Train MobileNetV3-Small from **random initialization** on 5 Tomato classes only.

**Steps:**
1. Initialize MobileNetV3-Small with random weights
2. Replace classifier head → 5 output neurons
3. Train on `train_cycle1/` of 5 Tomato classes
4. Validate on `val/`
5. Save best model checkpoint

**Practical:**
```python
model = mobilenet_v3_small(pretrained=False)
model.classifier[-1] = nn.Linear(1024, 5)
# Train all layers from scratch
```

**Output to Store:**
- `models/M1_scratch/best_model.pth`
- `models/M1_scratch/training_log.csv` (epoch, train_loss, val_loss, val_acc)
- `models/M1_scratch/config.json`

---

### 4.3 Case 2 — ImageNet Pretrained Fine-tuning (→ M2)

**Description:** Load MobileNetV3-Small pretrained on ImageNet. **Freeze backbone, fine-tune only classifier layers** on 5 Tomato classes.

**Steps:**
1. Load `mobilenet_v3_small(pretrained=True)` (ImageNet weights)
2. Freeze all layers except `classifier`
3. Replace classifier head → 5 output neurons
4. Train classifier only for 10 epochs
5. Optionally unfreeze last 2 conv blocks, train with lower LR (1e-4) for 10 more epochs
6. Save best model checkpoint

**Practical:**
```python
model = mobilenet_v3_small(pretrained=True)
for param in model.features.parameters():
    param.requires_grad = False
model.classifier[-1] = nn.Linear(1024, 5)
# Train classifier head only
```

**Output to Store:**
- `models/M2_imagenet/best_model.pth`
- `models/M2_imagenet/training_log.csv`
- `models/M2_imagenet/config.json`

---

### 4.4 Case 3 — PlantVillage Pretrained Fine-tuning (→ M3)

**Description:** First pre-train MobileNetV3 on 26 non-tomato PlantVillage classes, then fine-tune on 5 Tomato classes.

**Sub-steps:**

**Phase A — Pre-training on 26 Classes:**
1. Load `mobilenet_v3_small(pretrained=True)` (ImageNet)
2. Replace classifier → 26 outputs
3. Train on `data/pretrain_26cls/` with same config
4. Save as `models/M3_pretrain_26cls/pretrained_backbone.pth`

**Phase B — Fine-tuning on 5 Tomato Classes:**
1. Load pre-trained backbone from Phase A
2. Replace classifier → 5 outputs
3. Freeze backbone, fine-tune classifier
4. Optionally unfreeze last 2 blocks at lower LR
5. Save best model

**Output to Store:**
- `models/M3_plantvillage/pretrained_backbone.pth` ← **critical — reusable**
- `models/M3_plantvillage/best_model.pth`
- `models/M3_plantvillage/training_log_phase_a.csv`
- `models/M3_plantvillage/training_log_phase_b.csv`
- `models/M3_plantvillage/config.json`

---

### 4.5 Part 1 Baseline Evaluation (Before CL)

After producing M1, M2, M3 — evaluate all three on the held-out test set **before any CL updates**.

| Model | Test Set | Metrics |
|---|---|---|
| M1, M2, M3 | `test_old/` (5 Tomato classes) | Accuracy, F1 (macro), Per-class F1, Confusion Matrix |

**Output to Store:**
- `results/part1_baseline/baseline_results.csv`
- `results/part1_baseline/confusion_matrices/` (3 × .png)

---

## 5. Part 2 — Continual Learning Scenarios

> **Each CL scenario runs independently for all 3 models (M1, M2, M3)**
> **2 Cycles of CL per scenario**

---

### 5.1 Category A — Same Classes, New Data Stream

**Setup:**
- Classes remain the same 5 Tomato classes
- Cycle 1: Model trained on `train_cycle1/` (already done in Part 1)
- Cycle 2: New data = `train_cycle2/` (simulates new images arriving)
- Test: `test_old/` (same 5 class test set)

#### A0 — Baseline Test (Before CL Update)
- Evaluate M1/M2/M3 on A-Test set (= `test_old/`)
- Record Acc, F1 per class → stability reference point

#### Method A1 — EWC (Elastic Weight Consolidation)

**Concept:** Penalize changes to weights that were important for old tasks (measured via Fisher Information Matrix).

**Steps per Cycle:**
1. Load model from previous cycle
2. **Compute Fisher Information Matrix (FIM)** on old training data (or replay buffer)
3. Identify important weights → store as `ewc_fisher_{cycle}.pkl`
4. Train on new cycle data with EWC penalty added to loss:
   `L_total = L_CE + λ × Σ F_i(θ_i - θ*_i)²`
5. Recommended λ: start with 1000, tune via validation
6. Evaluate on `test_old/`

**Hyperparameters to tune:** `λ` (EWC regularization strength), LR, epochs

**Output to Store per Cycle:**
- `results/cat_a/ewc/{model}/cycle_{n}/model.pth`
- `results/cat_a/ewc/{model}/cycle_{n}/fisher_matrix.pkl` ← **critical**
- `results/cat_a/ewc/{model}/cycle_{n}/optimal_params.pkl` ← θ* (old weights)
- `results/cat_a/ewc/{model}/cycle_{n}/metrics.json`

---

#### Method A2 — Experience Replay

**Concept:** Maintain a small replay buffer of old class samples. Mix old + new samples during update.

**Steps per Cycle:**
1. Load model from previous cycle
2. Sample from `replay_buffer/` (10–15% of old training data per class, balanced)
3. Combine replay samples + new cycle data → mixed training set
4. Standard cross-entropy training on mixed set
5. Update replay buffer after training (add portion of new cycle data)

**Replay Buffer Strategy:** Random sampling per class (can try reservoir sampling for Cycle 2)

**Output to Store per Cycle:**
- `results/cat_a/replay/{model}/cycle_{n}/model.pth`
- `results/cat_a/replay/{model}/cycle_{n}/replay_buffer_manifest.json` (which images are in buffer)
- `results/cat_a/replay/{model}/cycle_{n}/metrics.json`

---

#### Method A3 — Parameter Isolation (Progressive / Expansion)

**Concept:** Freeze old weights. Add new parameters for new data. Prevents catastrophic forgetting by design.

**Steps per Cycle:**
1. Load model from previous cycle
2. **Freeze all existing parameters** (old backbone + classifier)
3. Add small adapter layers (e.g., a lightweight adapter block per conv block)
   OR expand classifier with additional neurons and mask old outputs
4. Train only new parameters on new cycle data
5. At inference: route through both old + new parameter paths

**Note:** This results in a progressively larger model each cycle — measure model size growth.

**Output to Store per Cycle:**
- `results/cat_a/param_isolation/{model}/cycle_{n}/model_expanded.pth`
- `results/cat_a/param_isolation/{model}/cycle_{n}/frozen_mask.pkl` (which params are frozen)
- `results/cat_a/param_isolation/{model}/cycle_{n}/model_size_mb.txt`
- `results/cat_a/param_isolation/{model}/cycle_{n}/metrics.json`

---

#### Method A_Base — Naive Fine-tuning (No CL Protection)

**Concept:** Plain fine-tuning on new data with no CL strategy. Used as **lower bound** to show catastrophic forgetting.

**Steps per Cycle:**
1. Fine-tune model directly on new cycle data only (no replay, no penalty)
2. Evaluate on `test_old/`
3. Expect accuracy on old classes to drop — document this as forgetting evidence

**Output to Store per Cycle:**
- `results/cat_a/naive_ft/{model}/cycle_{n}/model.pth`
- `results/cat_a/naive_ft/{model}/cycle_{n}/metrics.json`

---

#### A4 — After-Update Test (Category A)

After each CL method × each cycle, evaluate:

| Metric | Description |
|---|---|
| Accuracy | Overall accuracy on `test_old/` |
| Macro F1 | F1 across all 5 classes |
| Backward Transfer (BT) | Acc drop vs. baseline → measures forgetting |
| Stability Score | BT across cycles → lower = more stable |
| Per-class F1 | Individual class performance |

**Store:** `results/cat_a/summary_table.csv` (all models × all methods × all cycles)

---

### 5.2 Category B — New Class Introduced (5 → 7 Classes)

**Setup:**
- Cycle 1: 5 Tomato classes
- After Cycle 1: Introduce T6 (Septoria_leaf_spot) + T7 (Spider_mites)
- Cycle 2: Train CL update with 7 classes
- Test Sets:
  - `test_old/` = 5 original classes (tests old knowledge retention)
  - `test_new/` = T6 + T7 only (tests new class acquisition)

#### B0 — Baseline Test (Before CL Update)
- Evaluate M1/M2/M3 on B-Test set (includes new classes T6, T7)
- Model has never seen T6, T7 → expect near-0 on new classes
- Record to quantify starting point

---

#### Method B1 — EWC + New Class Head

**Steps:**
1. Load model (5-class classifier)
2. Compute FIM on old data → store fisher matrix
3. Expand classifier head: 5 → 7 outputs
4. Train on combined data (old replay + new class data) with EWC penalty on old weights
5. New class neurons initialized randomly; old neurons penalized via EWC

**Output to Store per Cycle:**
- `results/cat_b/ewc/{model}/cycle_{n}/model.pth`
- `results/cat_b/ewc/{model}/cycle_{n}/fisher_matrix.pkl`
- `results/cat_b/ewc/{model}/cycle_{n}/optimal_params.pkl`
- `results/cat_b/ewc/{model}/cycle_{n}/metrics_old.json`
- `results/cat_b/ewc/{model}/cycle_{n}/metrics_new.json`

---

#### Method B2 — Experience Replay + New Class Training

**Steps:**
1. Load model (5-class classifier)
2. Expand classifier: 5 → 7
3. Mix: replay buffer (old classes) + full new class data
4. Standard cross-entropy training on mixed set
5. Update replay buffer to include samples from T6, T7

**Output to Store per Cycle:**
- `results/cat_b/replay/{model}/cycle_{n}/model.pth`
- `results/cat_b/replay/{model}/cycle_{n}/replay_buffer_manifest.json`
- `results/cat_b/replay/{model}/cycle_{n}/metrics_old.json`
- `results/cat_b/replay/{model}/cycle_{n}/metrics_new.json`

---

#### Method B3 — Parameter Isolation (Zero Forgetting)

**Concept:** Freeze entire old model. Add a dedicated new branch for T6, T7 only. Theoretically zero forgetting on old classes.

**Steps:**
1. Load model (5-class, fully frozen)
2. Add lightweight parallel branch (small MobileNetV3 feature extractor or adapter)
3. Train new branch only on T6 + T7 data
4. At inference: combine both branches (e.g., argmax across both outputs with class routing)
5. Measure model size increase

**Output to Store:**
- `results/cat_b/param_isolation/{model}/cycle_{n}/base_model_frozen.pth`
- `results/cat_b/param_isolation/{model}/cycle_{n}/new_branch.pth`
- `results/cat_b/param_isolation/{model}/cycle_{n}/model_size_mb.txt`
- `results/cat_b/param_isolation/{model}/cycle_{n}/metrics_old.json`
- `results/cat_b/param_isolation/{model}/cycle_{n}/metrics_new.json`

---

#### Method B_Base — Naive Fine-tuning (No CL)

Same as A_Base but with expanded 7-class head. Expect severe forgetting of old classes.

---

#### B4 — After-Update Test (Category B)

| Metric | Old Classes (T1–T5) | New Classes (T6, T7) |
|---|---|---|
| Accuracy | ✅ | ✅ |
| Macro F1 | ✅ | ✅ |
| Forgetting Score | `Acc_before - Acc_after` | N/A |
| Forward Transfer | N/A | Acc on new class vs. random baseline |
| Per-class F1 | ✅ | ✅ |

**Store:** `results/cat_b/summary_table.csv`

---

## 6. Evaluation Protocol

### 6.1 Metrics Reference

| Metric | Formula | Measures |
|---|---|---|
| Accuracy | Correct / Total | Overall performance |
| Macro F1 | Mean F1 per class | Balanced performance |
| Backward Transfer (BT) | `Acc_now - Acc_before_CL` | Catastrophic forgetting |
| Forward Transfer (FT) | `Acc_new_after - Acc_random` | New class acquisition |
| Stability | `1 - |BT|` | Knowledge retention |
| Plasticity | FT on new classes | Adaptability |

### 6.2 Test Timing — Before & After

For every CL method, evaluate at these checkpoints:

| Checkpoint | When | Test Set Used |
|---|---|---|
| Pre-CL Baseline | Before any CL update | `test_old/` |
| Post-Cycle-1 | After Cycle 1 CL update | `test_old/` |
| Post-Cycle-2 | After Cycle 2 CL update | `test_old/` + `test_new/` (Cat B) |

### 6.3 Comparison Matrix

For the final paper comparison:

| Model | Method | Cat | Cycle | Old Acc | Old F1 | New Acc | New F1 | BT | Model Size |
|---|---|---|---|---|---|---|---|---|---|
| M1 | EWC | A | 1 | | | — | — | | |
| M1 | Replay | A | 1 | | | — | — | | |
| M1 | Isolation | A | 1 | | | — | — | | |
| M1 | Naive FT | A | 1 | | | — | — | | |
| ... | ... | ... | ... | | | | | | |

---

## 7. Outputs & Artifacts to Store

### 7.1 Model Weights

```
models/
├── M1_scratch/
│   └── best_model.pth              ← State dict, all layers
├── M2_imagenet/
│   └── best_model.pth
└── M3_plantvillage/
    ├── pretrained_backbone.pth     ← Pre-trained on 26 classes (REUSABLE)
    └── best_model.pth              ← Fine-tuned on 5 Tomato classes
```

### 7.2 CL Method Artifacts

```
results/
├── cat_a/
│   ├── ewc/{M1,M2,M3}/cycle_{1,2}/
│   │   ├── model.pth
│   │   ├── fisher_matrix.pkl       ← Fisher Information Matrix
│   │   ├── optimal_params.pkl      ← θ* (weights before update)
│   │   └── metrics.json
│   ├── replay/{M1,M2,M3}/cycle_{1,2}/
│   │   ├── model.pth
│   │   ├── replay_buffer_manifest.json
│   │   └── metrics.json
│   ├── param_isolation/{M1,M2,M3}/cycle_{1,2}/
│   │   ├── model_expanded.pth
│   │   ├── frozen_mask.pkl
│   │   ├── model_size_mb.txt
│   │   └── metrics.json
│   ├── naive_ft/{M1,M2,M3}/cycle_{1,2}/
│   │   ├── model.pth
│   │   └── metrics.json
│   └── summary_table.csv
│
└── cat_b/
    ├── ewc/ ...
    ├── replay/ ...
    ├── param_isolation/ ...
    ├── naive_ft/ ...
    └── summary_table.csv
```

### 7.3 Data Artifacts

```
data/
├── split_info.json                 ← All image paths + leaf_id + split assignment
├── data_stats.csv                  ← Class counts per split
├── replay_buffer/                  ← 10-15% old class images (FROZEN, never updated during training)
│   └── {class}/
├── tomato_5cls/
│   ├── train_cycle1/
│   ├── train_cycle2/
│   ├── val/
│   └── test_old/
└── tomato_new_2cls/
    ├── train/
    └── test_new/
```

### 7.4 Training Logs

```
logs/
├── part1_training_{M1,M2,M3}.csv      ← epoch, loss, val_acc per epoch
├── cat_a_{method}_{model}_cycle{n}.csv
└── cat_b_{method}_{model}_cycle{n}.csv
```

### 7.5 Evaluation Reports

```
results/
├── part1_baseline/
│   ├── baseline_results.csv
│   └── confusion_matrices/
├── cat_a/summary_table.csv
├── cat_b/summary_table.csv
└── final_comparison_table.csv       ← Master results table for paper
```

### 7.6 Edge Deployment Artifacts

```
deployment/
├── pruned_models/
│   └── {model}_{method}/pruned.pth  ← After pruning
├── quantized_models/
│   └── {model}_{method}/quantized.onnx  ← INT8 / FP16
└── edge_eval_results.csv            ← Acc drop vs. speed/size gain
```

---

## 8. Folder Structure

```
cl_research/
├── data/                    ← Dataset splits (prepared locally)
├── models/                  ← Base model weights (M1, M2, M3)
├── results/                 ← All experiment results
│   ├── part1_baseline/
│   ├── cat_a/
│   └── cat_b/
├── logs/                    ← Training logs
├── deployment/              ← Edge deployment artifacts
├── notebooks/               ← Jupyter notebooks for analysis/plots
│   ├── 01_data_analysis.ipynb
│   ├── 02_part1_training.ipynb
│   ├── 03_cat_a_cl.ipynb
│   ├── 04_cat_b_cl.ipynb
│   └── 05_results_analysis.ipynb
├── src/                     ← Python modules
│   ├── train.py
│   ├── evaluate.py
│   ├── ewc.py
│   ├── replay.py
│   ├── param_isolation.py
│   └── utils.py
├── configs/                 ← YAML/JSON configs per experiment
└── README.md
```

---

## 9. To-Do Checklist

### 📂 Phase 0 — Data Preparation (Local)

- [ ] Download PlantVillage color dataset from HuggingFace
- [ ] Run image count verification per class (compare with table above)
- [ ] Note whether dataset is already augmented or raw
- [ ] Extract and organize 5 Tomato class folders → `data/tomato_5cls/`
- [ ] Extract 26 non-Tomato class folders → `data/pretrain_26cls/`
- [ ] Extract 2 new Tomato class folders → `data/tomato_new_2cls/`
- [ ] Apply leaf-grouped stratified 70/15/15 split
- [ ] Create `train_cycle1/` (50% of train) and `train_cycle2/` (remaining 50%)
- [ ] Create `replay_buffer/` (10–15% of train_cycle1 per class)
- [ ] Save `split_info.json` with full metadata
- [ ] Save `data_stats.csv` with per-class counts per split
- [ ] Verify zero overlap between train/val/test sets
- [ ] Verify zero overlap between test_old and train_cycle1/2

---

### 🏗️ Phase 1 — Environment Setup

- [ ] Set up Python environment (Python 3.10+)
- [ ] Install: `torch`, `torchvision`, `timm`, `scikit-learn`, `pandas`, `matplotlib`, `seaborn`, `tqdm`
- [ ] Install: `onnx`, `onnxruntime` (for deployment)
- [ ] Verify GPU availability
- [ ] Create project folder structure as defined above
- [ ] Write `src/utils.py` (dataloaders, metrics computation, logging)
- [ ] Write `src/evaluate.py` (Acc, F1, confusion matrix, forgetting score)
- [ ] Write `configs/` YAML files for each experiment

---

### 🎯 Phase 2 — Part 1: Initial Training

- [ ] **Case 1 (M1):** Train MobileNetV3-Small from scratch on 5 Tomato classes
  - [ ] Implement training loop
  - [ ] Train on `train_cycle1/`
  - [ ] Validate on `val/`
  - [ ] Save `models/M1_scratch/best_model.pth`
  - [ ] Save training log

- [ ] **Case 2 (M2):** Fine-tune ImageNet pretrained MobileNetV3
  - [ ] Load ImageNet weights
  - [ ] Freeze backbone, train classifier head
  - [ ] (Optional) Unfreeze last 2 blocks
  - [ ] Save `models/M2_imagenet/best_model.pth`

- [ ] **Case 3 (M3):** PlantVillage pretrained → fine-tune
  - [ ] Phase A: Pre-train on 26 non-tomato classes
  - [ ] Save `models/M3_plantvillage/pretrained_backbone.pth`
  - [ ] Phase B: Fine-tune on 5 Tomato classes
  - [ ] Save `models/M3_plantvillage/best_model.pth`

- [ ] **Baseline Evaluation (All 3 Models)**
  - [ ] Evaluate M1, M2, M3 on `test_old/`
  - [ ] Save `results/part1_baseline/baseline_results.csv`
  - [ ] Plot confusion matrices

---

### 🔁 Phase 3 — Category A: Same Class CL

> Run for M1, M2, M3 × 2 Cycles

- [ ] Implement `src/ewc.py` (Fisher matrix, EWC loss)
- [ ] Implement `src/replay.py` (replay buffer sampling, mixed training)
- [ ] Implement `src/param_isolation.py` (freeze + adapter expansion)
- [ ] Write `src/train_cl.py` for CL training loop

**A0 — Baselines:**
- [ ] Record A0 baseline for M1
- [ ] Record A0 baseline for M2
- [ ] Record A0 baseline for M3

**A1 — EWC:**
- [ ] Run EWC Cycle 1 for M1, M2, M3
- [ ] Store fisher matrices + optimal params
- [ ] Evaluate on `test_old/` (Cycle 1)
- [ ] Run EWC Cycle 2 for M1, M2, M3
- [ ] Evaluate on `test_old/` (Cycle 2)

**A2 — Experience Replay:**
- [ ] Run Replay Cycle 1 for M1, M2, M3
- [ ] Store replay buffer manifest
- [ ] Evaluate (Cycle 1)
- [ ] Run Replay Cycle 2 for M1, M2, M3
- [ ] Evaluate (Cycle 2)

**A3 — Parameter Isolation:**
- [ ] Run Isolation Cycle 1 for M1, M2, M3
- [ ] Store frozen masks
- [ ] Evaluate (Cycle 1)
- [ ] Run Isolation Cycle 2 for M1, M2, M3
- [ ] Evaluate (Cycle 2)

**A_Base — Naive Fine-tuning:**
- [ ] Run Naive FT Cycle 1 for M1, M2, M3
- [ ] Run Naive FT Cycle 2 for M1, M2, M3

**A4 — After-Update Evaluation:**
- [ ] Compile `results/cat_a/summary_table.csv`
- [ ] Plot: Accuracy vs Cycle per method per model
- [ ] Plot: Forgetting score comparison

---

### 🆕 Phase 4 — Category B: New Class CL

> Run for M1, M2, M3 × 2 Cycles

**B0 — Baselines:**
- [ ] Evaluate M1, M2, M3 on B-Test set (all 7 classes) before CL
- [ ] Note near-0 performance on T6, T7 as starting point

**B1 — EWC + Head Expansion:**
- [ ] Expand classifier 5 → 7
- [ ] Run EWC Cycle 1 for M1, M2, M3
- [ ] Run EWC Cycle 2
- [ ] Evaluate on `test_old/` + `test_new/`

**B2 — Experience Replay + New Classes:**
- [ ] Run Replay Cycle 1 for M1, M2, M3
- [ ] Update replay buffer to include T6, T7 samples
- [ ] Run Cycle 2
- [ ] Evaluate

**B3 — Parameter Isolation (Zero Forgetting):**
- [ ] Freeze old model entirely
- [ ] Build new branch for T6, T7
- [ ] Train new branch
- [ ] Combine at inference (routing)
- [ ] Measure model size growth
- [ ] Evaluate

**B_Base — Naive Fine-tuning:**
- [ ] Run with expanded head, no CL
- [ ] Document forgetting evidence

**B4 — After-Update Evaluation:**
- [ ] Compile `results/cat_b/summary_table.csv`
- [ ] Plot: Old class retention vs new class acquisition (plasticity-stability tradeoff)
- [ ] Plot: Forgetting score vs method

---

### 📱 Phase 5 — Edge Deployment

- [ ] Select best CL model per category (based on BT + F1)
- [ ] Apply **post-training quantization** (INT8 or FP16)
- [ ] Apply **pruning** (structured pruning, 20–40% target sparsity)
- [ ] Export to ONNX format
- [ ] Benchmark on edge device or CPU:
  - [ ] Inference time (ms)
  - [ ] Throughput (FPS)
  - [ ] Model size (MB)
  - [ ] Accuracy vs. uncompressed model (accuracy drop)
- [ ] Save results to `deployment/edge_eval_results.csv`

---

### 📊 Phase 6 — Analysis & Paper Writing

- [ ] Compile final comparison table `results/final_comparison_table.csv`
- [ ] Generate all figures:
  - [ ] Dataset distribution bar chart
  - [ ] Training curves (loss/acc) per model
  - [ ] CL comparison: Acc vs Cycle (Cat A)
  - [ ] Plasticity-Stability tradeoff chart (Cat B)
  - [ ] Forgetting score heatmap (methods × models)
  - [ ] Edge deployment: Accuracy vs Model Size scatter plot
- [ ] Write CL section for research paper
- [ ] Cite: Kirkpatrick et al. (EWC, 2017), Rolnick et al. (Experience Replay), Rusu et al. (Progressive Nets)
- [ ] Verify all model weights and data artifacts are saved and backed up

---

## 📎 References

- Mohanty et al. (2016). *Using Deep Learning for Image-Based Plant Disease Detection.* Frontiers in Plant Science.
- Kirkpatrick et al. (2017). *Overcoming Catastrophic Forgetting in Neural Networks.* PNAS. ← EWC
- Rolnick et al. (2019). *Experience Replay for Continual Learning.* NeurIPS. ← Replay
- Rusu et al. (2016). *Progressive Neural Networks.* ← Parameter Isolation
- Howard et al. (2019). *Searching for MobileNetV3.* ICCV. ← Model Architecture
- Khan et al. (2023). *Plant Disease Detection Model for Edge Computing Devices.* Frontiers in Plant Science. ← MobileNetV3 + PlantVillage edge baseline

---

*This journal documents the continual learning phase of the research. Previous phases (image collection, segmentation pipeline, GAN augmentation) are documented in separate research journal entries.*
