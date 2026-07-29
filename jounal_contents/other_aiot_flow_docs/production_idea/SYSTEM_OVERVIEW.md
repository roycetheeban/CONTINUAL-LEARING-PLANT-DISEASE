# AIoT Plant Monitoring System — Simple Overview

> "Build for one plant, design for many"

---

## What Is This System?

An intelligent plant monitoring platform that uses a camera + sensors placed at a farm to automatically detect plant diseases, track health over time, and alert farmers before problems get worse.

- Initial target plant: **Turmeric**
- Designed to expand to any plant type with minimal changes
- Works with or without internet (edge + cloud hybrid)

---

## System Layers at a Glance

```
┌──────────────────────────────────────────────────────┐
│               Mobile / Web App                       │
│         (farmer views results & alerts)              │
└──────────────────────┬───────────────────────────────┘
                       │
┌──────────────────────▼───────────────────────────────┐
│                  Cloud Layer                         │
│   API Gateway → Microservices → Databases            │
└──────────────────────┬───────────────────────────────┘
                       │  sync (when online)
┌──────────────────────▼───────────────────────────────┐
│              Edge Device (at farm)                   │
│   Camera + Sensors → Local AI → Local Storage        │
└──────────────────────────────────────────────────────┘
```

---

## 1. Edge Device — The Starting Point

The edge device sits physically at the farm. It is the **primary data source** for the entire system.

```
┌─────────────────────────────────────────────────────┐
│                  Edge Device                        │
│                                                     │
│  ┌─────────────┐   ┌──────────────┐                │
│  │   Camera    │   │   Sensors    │                │
│  │  (fixed     │   │  Temp / Hum  │                │
│  │   position) │   │  Soil / CO₂  │                │
│  └──────┬──────┘   └──────┬───────┘                │
│         │                 │                         │
│  ┌──────▼─────────────────▼───────┐                │
│  │        Edge Controller         │                │
│  └──────┬─────────────────────────┘                │
│         │                                           │
│  ┌──────▼──────┐  ┌─────────────┐                  │
│  │  ML Module  │  │Local Storage│                  │
│  │ (optional)  │  │ (100 images │                  │
│  └──────┬──────┘  │  + sensor   │                  │
│         │         │   buffer)   │                  │
│  ┌──────▼──────┐  └─────────────┘                  │
│  │Sync Service │──────────────────► Cloud           │
│  └─────────────┘  (when online)                    │
└─────────────────────────────────────────────────────┘
```

### What the edge device does

| Component | Job |
|---|---|
| Camera | Captures plant images at fixed position, 2× per day |
| Sensors | Reads temperature, humidity, soil moisture, CO₂ every 1 minute |
| Edge Controller | Orchestrates all modules |
| ML Module | Runs AI locally (optional, when offline) |
| Local Storage | Buffers up to 100 images + sensor data when offline |
| Sync Service | Uploads buffered data to cloud when internet is available |
| Local API | Lets mobile app connect directly to device (no internet needed) |

### Two operation modes

```
Cloud Mode (default):
  Edge captures → sends to cloud → cloud runs AI → results back

Edge Mode (offline):
  Edge captures → runs AI locally → stores results → syncs later
```

---

## 2. Image Capture — Why 2× Per Day?

The camera captures **one image in the morning and one in the evening**, every day.

```
  6:00 AM – 9:00 AM          4:00 PM – 7:00 PM
  ┌─────────────┐             ┌─────────────┐
  │   Morning   │             │   Evening   │
  │   Capture   │             │   Capture   │
  └──────┬──────┘             └──────┬──────┘
         │                           │
         └──────────┬────────────────┘
                    ▼
           Sent to ML Pipeline
```

**Why morning and evening specifically?**

- Morning light gives consistent, shadow-free images for baseline
- Evening captures any disease progression that developed during the day
- Two data points per day is enough to track daily trends without overloading storage
- Matches natural disease progression cycles in plants like turmeric

### Camera Position Detection

Because the camera must stay in the same position every day for accurate trend comparison, the system automatically checks if it has moved.

```
Day 1 Morning Image
       │
       ▼
  Stored as Reference
       │
       ▼
Day 1 Evening Image ──► Compare features (ORB matching)
                              │
                    ┌─────────▼──────────┐
                    │ Similarity ≥ 70%?  │
                    └─────────┬──────────┘
                         Yes  │  No
                         ▼         ▼
                    Position OK   Alert farmer
                                  "Camera may have moved"
                                       │
                              ┌────────▼────────┐
                              │ Reset analytics │
                              │ or Ignore       │
                              └─────────────────┘
```

Each morning image becomes the reference for the next comparison. This keeps the baseline fresh and accounts for natural plant growth.

---

## 3. The ML Pipeline — How Disease Is Detected

Every image goes through a **3-stage AI pipeline**:

```
Input Image (full plant view)
         │
         ▼
┌─────────────────────┐
│  Stage 1: Detection │  YOLOv8n
│  Find all leaves    │  → bounding boxes around each leaf
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ Stage 2: Segmentation│  YOLOv8n-seg
│ Isolate each leaf   │  → precise leaf shape (mask)
└──────────┬──────────┘
           │
           ▼
┌──────────────────────┐
│ Stage 3: Classification│  MobileNetV3
│ Diagnose each leaf   │  → disease class per leaf
└──────────┬───────────┘
           │
           ▼
  Final Output:
  {
    "healthy": 8,
    "leaf_spot": 3,
    "leaf_blotch": 1
  }
  Total leaves: 12
```

**Why 3 separate models instead of one?**

- Each model is small enough to run on edge hardware (total ~27 MB)
- Any stage can be updated independently without retraining everything
- Easier to debug — if detection is wrong, segmentation and classification are not affected
- Modular design supports adding new plant types by only retraining the classification stage

---

## 4. Model Tiers — Personalization Without Sacrificing Privacy

The system uses a **3-tier model strategy** so every farmer gets the most accurate model available for their situation:

```
Tier 1: Global Base Model
  ├── Trained on all plants from all users (anonymous)
  ├── High generalization, lower specialization
  └── Used as fallback

        ↓ transfer learning

Tier 2: Plant-Specific Model  (default)
  ├── Trained on all turmeric data (anonymous, opt-in)
  ├── Optimized for turmeric diseases specifically
  └── Used for all turmeric users

        ↓ fine-tuning

Tier 3: Farm-Specific Model  (optional, premium)
  ├── Trained only on this farmer's own data
  ├── Adapts to local soil, climate, and conditions
  └── Requires 500+ images to activate
```

The system automatically picks the best available tier for each prediction. If Tier 3 fails or has low confidence, it falls back to Tier 2, then Tier 1.

---

## 5. Model Update Frequency & Why

```
Tier 1 (Global)     ──► Every 3 months
Tier 2 (Plant)      ──► Every month  (after data cleanup cycle)
Tier 3 (Farm)       ──► Every month  (if 500+ new images available)
```

**Why monthly for Tier 2 and 3?**

- New disease patterns and seasonal variations appear over time
- Farmers' data accumulates enough volume monthly to make retraining worthwhile
- Monthly aligns with the data cleanup cycle — old images are deleted after retraining, keeping storage costs low
- More frequent retraining would be expensive and provide diminishing returns for a 2×/day capture schedule

**Why quarterly for Tier 1?**

- Global model needs large, diverse datasets to improve meaningfully
- Quarterly gives enough time to accumulate data from many farms across different regions and seasons

---

## 6. Risk Analysis — Combining Vision + Environment

After each image is processed, the system combines the ML results with sensor data to compute a **risk level**:

```
  ML Results              Sensor Data (last 24h)
  ┌──────────┐            ┌──────────────────────┐
  │ leaf     │            │ Temperature          │
  │ counts   │            │ Humidity             │
  │ per class│            │ Soil Moisture        │
  └────┬─────┘            │ CO₂ levels           │
       │                  └──────────┬───────────┘
       └──────────┬──────────────────┘
                  ▼
         Risk Analysis Service
                  │
                  ▼
        ┌─────────────────┐
        │  Risk Level     │
        │  Low / Medium / │
        │  High / Critical│
        └────────┬────────┘
                 │
        ┌────────▼────────┐
        │ Recommendations │
        │ + 7-day forecast│
        └─────────────────┘
```

Example: High humidity + increasing leaf spot count → elevated risk → system recommends fungicide check before it becomes critical.

---

## 7. Other Services — How the Rest Works

### Data Flow (Cloud Mode)

```
Mobile App
    │ upload image
    ▼
API Gateway ──► authenticates, routes request
    │
    ▼
Data Ingestion Service ──► stores image in S3, metadata in DB
    │
    ▼
ML Inference Service ──► runs 3-stage pipeline
    │
    ▼
Risk Analysis Service ──► computes risk score
    │
    ├──► Analytics Service ──► updates trends & dashboard
    │
    └──► Notification Service ──► sends alert if risk is high
                                       │
                                       ▼
                                  Mobile App
                               (push notification)
```

### All Cloud Services

| Service | What it does |
|---|---|
| API Gateway | Single entry point — auth, rate limiting, routing |
| User & Auth Service | Registration, login, JWT tokens, roles |
| Device Management | Register/monitor edge devices, manage API keys |
| Data Ingestion | Accepts images and sensor data, validates and stores |
| ML Inference | Runs the 3-stage detection pipeline |
| Risk Analysis | Combines disease data + environment → risk score |
| Analytics | Tracks health trends over days/weeks/months |
| Notification | Push, SMS, email alerts for high-risk events |
| Model Training | Monthly retraining using new collected data |
| Model Registry | Versions and deploys models to cloud and edge |
| Chatbot (RAG) | Answers farmer questions using plant knowledge + their data |
| Subscription/Billing | Manages free vs premium tier access |

---

## 8. Continual Learning — The System Gets Smarter Over Time

Every prediction the system makes is evaluated for confidence. This drives automatic improvement:

```
New Image Processed
         │
         ▼
  Confidence Score?
         │
   ┌─────┴──────┐
   │            │
  < 60%       > 90%
   │            │
   ▼            ▼
Flag for      Auto-label
manual        and add to
review        training pool
         │
         ▼
   Monthly Retraining
         │
         ▼
   Validate new model
         │
         ▼
   Canary deploy (5% users)
         │
         ▼
   Full rollout if metrics pass
```

This means the model improves continuously without requiring manual intervention for high-confidence cases, while still catching edge cases through human review.

---

## 9. Data Cleanup — Keeping Storage Costs Low

Images are large. The system handles this with an automated monthly cleanup:

```
After model retraining completes each month:

Raw Images (30 days old)  ──► Deleted  (~90% of storage freed)
Processed Results         ──► Kept for 2 years
Sensor Data               ──► Kept for 2 years (full), 5 years (aggregated)
Best training samples     ──► Kept permanently (curated dataset)
```

Users are notified 3 days before cleanup and can download their data. Premium users can extend image retention up to 1 year.

**Result: ~85% storage cost reduction** while keeping all analytics intact.

---

## 10. Multi-Tenancy — Every Farmer's Data Is Isolated

All data is tagged with `user_id` at every level. No farmer can ever see another farmer's data.

```
User A                    User B
  │                         │
  ▼                         ▼
S3: users/A/images/...    S3: users/B/images/...
DB: WHERE user_id = A     DB: WHERE user_id = B
```

Row-level security in the database enforces this automatically, even if a query accidentally omits the filter.

---

## 11. Mobile App — What the Farmer Sees

```
┌─────────────────────┐
│   Dashboard         │  Overall health score, active alerts
├─────────────────────┤
│   Capture Image     │  Manual capture + upload
├─────────────────────┤
│   Results           │  Leaf counts per disease class
├─────────────────────┤
│   Trends            │  Daily/weekly/monthly health charts
├─────────────────────┤
│   Alerts            │  Risk notifications
├─────────────────────┤
│   Chatbot           │  Ask questions about plant health
├─────────────────────┤
│   Devices           │  Manage edge devices
└─────────────────────┘
```

The app works in two modes:
- **Cloud mode** — connects to cloud API (default)
- **Edge mode** — connects directly to local edge device (no internet needed)

---

## Quick Reference

| What | Detail |
|---|---|
| Capture frequency | 2× per day (morning + evening) |
| Sensor frequency | Every 1 minute |
| ML pipeline | Detection → Segmentation → Classification |
| Models | YOLOv8n, YOLOv8n-seg, MobileNetV3 |
| Total model size | ~27 MB (fits on edge device) |
| Inference time | < 10s (edge), < 30s (cloud) |
| Model update | Monthly (Tier 2 & 3), Quarterly (Tier 1) |
| Image retention | 30 days (free), up to 1 year (premium) |
| Analytics retention | 2 years |
| Offline buffer | Up to 100 images |
| Initial plant | Turmeric (extensible to any plant) |

---

*For full technical details see the individual specification documents in this repository.*
