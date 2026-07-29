# LEAFSENSE — Offline Jetson Phase (Edge Product)

**Version:** 1.1 (v1.1: added Model Runtime Strategy — all-TensorRT serving, ONNX as interchange, DeepStream skipped)
**Date:** July 2026
**Status:** Idea Finalized — Implementation In Progress (tested on Jetson Orin Nano)
**Supersedes:** Fixed-camera / cloud-first edge design in [SYSTEM_OVERVIEW.md](SYSTEM_OVERVIEW.md) where they conflict

> "Ship the intelligence with the device — the greenhouse should not need the internet to be smart."

---

## 1. Product Concept

The core commercial unit is a **Jetson Orin Nano shipped with a pre-trained, plant-specific model variant already installed** (initial variant: **Turmeric**, 5 classes). The device is fully operational **without any internet connection**:

- Captures images automatically on a rotating camera
- Classifies leaf diseases on-device
- Tracks **class-count differentiation** (the primary output signal) within a day and across days
- Correlates disease trends with CO₂, temperature, humidity, and soil moisture
- Provides **rule-based insights offline**
- Continuously improves itself through **on-device incremental retraining (EWC)**

One device = one plant type. The model variant is swappable, so the same hardware supports any plant for which a trained variant exists.

### Initial plant variant: Turmeric — 5 classes

| # | Class |
|---|---|
| 1 | Healthy |
| 2 | Leaf Blotch |
| 3 | Leaf Spot |
| 4 | Dry |
| 5 | Aphid disease |

---

## 2. Hardware Setup

| Component | Detail |
|---|---|
| Compute | NVIDIA Jetson Orin Nano (development directly in on-device VS Code) |
| Camera | Motorized mount, **360° rotation in 45° steps → 8 positions per capture set** |
| Sensors | CO₂, Temperature, Humidity, Soil Moisture (currently simulated by a CSV with 1 reading/second during development) |
| Storage | Local storage for retrain buffer, relabel queue, results DB, model versions |
| Display / UI | Streamlit dashboard served locally |

---

## 3. Capture Strategy

### 3.1 Rotation and schedule

```
One capture SET  =  8 images (camera rotates 360° in 45° steps, low overlap)

Per day          =  2 sets  (Cycle A: morning, Cycle B: evening)
                 =  16 images/day

Extra            =  Manual "Capture Now" button on the dashboard
                    (instant set outside the periodic schedule)
```

### 3.2 Why 2 sets per day

- Cycle A vs Cycle B → **intra-day class-count differentiation** (did disease counts move within the day?)
- Day-over-day sets → **trend lines per disease class**
- Sensor readings are matched to each capture timestamp, so environmental context is attached to every count.

### 3.3 On leaf double-counting

Adjacent 45° views may show the same leaf twice. This is **accepted by design**:

- Overlap is kept low by camera placement/setup.
- The primary signal is the **change** in counts, not the absolute count — a small constant bias cancels out in differencing.
- The setup (rotation step, camera distance, position masking) is **adjustable per farm** so the installer can tune overlap to the greenhouse layout.

---

## 4. Edge ML Pipeline (Image Path)

Five edge-optimized models make up the system: **YOLOv8n (detection), YOLOv8n-seg (segmentation), Lightweight GAN (optional occlusion recovery), MobileNetV3-Small (classification), Decision Tree (environmental risk)**. All were chosen specifically for edge deployment and incremental-learning compatibility.

```
Input image (any size, camera or manual upload)
      │
      ▼
[1] Resize to 640×640 — aspect ratio preserved, black padding
      │
      ▼
[2] YOLOv8n DETECTION
      → bounding box per leaf on the full-plant image
      │
      ▼
[3] Crop leaves from boxes
      → QUALITY FILTER: only crops ≥ 124×124 continue
        (guarantees enough pixels for the downstream stages)
      │
      ▼
[4] YOLOv8n SEGMENTATION
      → per-leaf mask; background noise removed, filled black
      │
      ▼
[5] Resize to 200×200 → pad to 224×224 (black)
      │
      ▼
[6] LIGHTWEIGHT GAN  ── OPTIONAL STAGE ──
      → reconstructs occluded leaf portions, mask added back
        onto the original crop so a full leaf structure reaches
        the classifier
      ⚠ Status: performs well on internet datasets, NOT yet
        validated on real farm data → shipped disabled by
        default until it proves out; pipeline runs 5→7 directly.
      │
      ▼
[7] MobileNetV3-Small CLASSIFIER
      → final disease class + confidence per leaf
      │
      ▼
Output: per-class leaf counts for the capture set
        e.g. { healthy: 41, blotch: 3, leafspot: 5, dry: 2, aphid: 0 }
```

---

## 5. Sensor Pipeline (Environmental Path)

```
CO₂ / Temperature / Humidity / Soil Moisture
(readings matched to each image-capture timestamp;
 dev phase: dummy CSV @ 1 reading/second)
      │
      ▼
Decision Tree — Environmental Risk Prediction
      │
      ▼
Risk level output (rule-interpretable, edge-cheap)
```

## 6. Label-Level Fusion

The classification counts and the decision-tree risk output are combined by **label-level fusion** into the final prediction shown on the dashboard:

```
Class counts (image path)  ┐
                           ├─► Label-level fusion ─► Final output
Risk level (sensor path)   ┘      (e.g. "leaf spot rising + high humidity
                                   → elevated fungal risk")
```

---

## 7. Confidence Routing & Data for Retraining

Every classified leaf is routed by prediction confidence:

```
Prediction confidence
      │
 ┌────┴─────────────────────────┐
 │ HIGH confidence              │ LOW confidence
 ▼                              ▼
Auto-labeled and saved         Held in a separate relabel queue
into the on-device             (exported as an Excel/simple sheet)
RETRAIN BUFFER                        │
(stored per class,                    ▼
 separately)               Sent to developer WHEN a connection
                           is available → relabeled IN THE CLOUD
                           → corrected labels sent back to the
                           Jetson → joins the NEXT retraining cycle
```

- Low-confidence samples are **never** trained on unreviewed — this prevents model drift from wrong self-labels.
- **Future plan:** replace the human relabeler with an **LLM teacher module** — low-confidence samples pushed to remote storage, labeled by the LLM, retrieved back for training.

---

## 8. On-Device Incremental Retraining (Monthly)

Retraining runs **on the Jetson itself** — a lightweight fine-tune with **EWC (Elastic Weight Consolidation)**, roughly **once a month** (also manually triggerable from the dashboard).

```
Monthly trigger (or dashboard button)
      │
      ▼
Train on retrain buffer (auto-labeled high-conf samples
+ any returned relabeled samples), with EWC preserving
weights important to previously learned knowledge
      │
      ▼
Evaluate NEW model on the FIXED common test set
(held out since initial training, never changed —
 detects catastrophic forgetting on old data)
      │
      ▼
Metric comparison: new model vs current base model
      │
 ┌────┴────────┐
 │ Better      │ Worse / regressed
 ▼             ▼
DEPLOY new    KEEP base model,
version       retain data for next cycle
```

---

## 9. Model Training Strategy (How the Shipped Model Was Built)

### Phase 1 — From-scratch pretraining
- MobileNetV3-Small architecture trained **from scratch on PlantVillage** (not ImageNet weights).
- Rationale: plant-domain features from the start → easier adaptation to new diseases later, and empirically better than ImageNet fine-tuning for this domain.

### Phase 2 — Transfer to Turmeric (two-stage fine-tuning)
- Start from the Phase 1 model; freeze base (or allow partial tuning).
- New 5-class classification head, fine-tuned in **2 stages** on the turmeric dataset (~400 images/class target).

### Phase 3 — Incremental Learning with EWC (hyperparameter tuning)
- Turmeric data is scarce (~200 images/class), so incremental-learning hyperparameters are tuned on the **cassava dataset** as a stand-in.
- Data split for the experiments:
  - **60%** → Phase 2 (initial fine-tune) training
  - **10%** → fixed overall test set (unchanged across ALL iterations — the catastrophic-forgetting probe)
  - **Remainder** → incremental batches of **50 / 100 / 200** images
- Multiple incremental iterations are run per batch size; accuracy on the fixed test set is tracked each iteration to find the **optimal batch size and hyperparameters** before locking the on-device recipe.

---

## 10. Model Runtime Strategy — All TensorRT in Production

**Decision:** on the Jetson, **every neural model serves as a TensorRT engine** (FP16). ONNX and PyTorch files exist on the device but never do the daily inference.

| Component | Serves as | Also kept on device | Why |
|---|---|---|---|
| Detector (YOLOv8n) | TensorRT `.engine` | — | Frozen, never retrained on device |
| Segmenter (YOLOv8n-seg) | TensorRT `.engine` | — | Frozen |
| GAN (optional, off by default) | TensorRT `.engine` | — | Frozen |
| **Classifier (MobileNetV3-Small)** | **TensorRT `.engine`** | **PyTorch weights + EWC Fisher state** | Monthly on-device retraining needs trainable weights |
| Risk tree (decision tree) | sklearn `.pkl` | — | Tiny, no GPU needed |

### Role of each format

- **TensorRT** — the production runtime. ~2× ONNX Runtime speed on Orin Nano with FP16; INT8 later with a calibration set from real greenhouse images.
- **ONNX** — the interchange format, never the production runtime. It is (1) what the repo ships (`.engine` files are hardware- and TRT-version-specific, so they are **always built on the device** via `jetson/scripts/build_engines.sh`, never committed or copied from a PC) and (2) the dev runtime on a PC (`runtime: onnx` in `config.yaml`), where the identical pipeline code runs without NVIDIA tooling.
- **PyTorch** — kept **only for the classifier**, as the trainable weights the monthly EWC cycle starts from.
- **DeepStream** — **deliberately skipped.** It targets continuous multi-stream video (RTSP, 30 fps); our workload is 16 stills/day with custom logic between every stage (≥124px crop filter, padding, optional GAN, confidence routing) that would fight DeepStream's rigid `nvinfer` pipeline. It uses TensorRT underneath anyway, so it adds nothing here. Folder placeholders remain in case of a future pivot to live-video monitoring.

### Retraining fits the same runtime (monthly cycle, classifier only)

```
Monthly trigger
  → EWC fine-tune classifier (PyTorch, on-device, checkpointed)
  → export classifier.onnx                          (seconds)
  → trtexec → classifier_candidate.engine           (~1–2 min on Orin Nano)
  → evaluate candidate on the FIXED test set
    ⚠ evaluated AS AN ENGINE, through the same TRT runtime that will serve it —
      FP16 conversion can shift accuracy slightly; the gate must measure reality
  → metrics better than current model?
      yes → atomic file swap + bump manifest version   (previous engine kept as instant rollback)
      no  → keep current engine, retain data for next cycle
```

The conversion overhead the retraining adds (~2 minutes, once a month) is the trade for ~2× faster inference on every capture, every day.

---

## 11. Streamlit Dashboard (Local, Offline)

| Feature | Description |
|---|---|
| Classification details | Latest per-class leaf counts and per-leaf results |
| Disease trend graph | **5 colored lines**, one per class, daily leaf counts — shows at a glance which disease is increasing |
| Risk prediction | Decision-tree environmental risk + fused final output |
| **Retrain button** | Manually triggers the incremental retraining cycle |
| **Capture Now button** | Triggers an immediate capture set outside the periodic schedule |
| Insights (rule-based) | Offline, predefined logic-based recommendations from count changes + sensor readings |

---

## 12. What the Offline Phase Includes (Free Tier Scope)

| Included offline (free) | Not included (see [FULL_COMBINED_SYSTEM.md](FULL_COMBINED_SYSTEM.md)) |
|---|---|
| Full capture → classify → count pipeline | RAG chatbot (server-side, subscription) |
| Class-count trend tracking + sensor correlation | Knowledge-base docs (plant + Sri Lanka specific) |
| Rule-based (predefined logic) insights | Daily insight-provider service |
| On-device monthly EWC retraining | LLM teacher relabeling |
| Streamlit dashboard | |
| Relabel-queue sync + maintenance/updates when a connection is available | |

---

## 13. Future Work

- **GAN maturation** — validate the occlusion-recovery GAN on real farm data, then enable by default.
- **LLM teacher module** — automated relabeling of low-confidence samples via remote storage.
- **More plant variants** — the pipeline is plant-agnostic; only the classifier variant (and KB docs, in the paid tier) change per plant.
- Fully automated end-to-end incremental pipeline: no manual image processing — device separates, classifies, buffers, retrains, and self-gates deployment continuously.
