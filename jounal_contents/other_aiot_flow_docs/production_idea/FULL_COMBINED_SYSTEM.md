# LEAFSENSE — Full Combined System (Offline Edge + Connected Services)

**Version:** 1.0
**Date:** July 2026
**Status:** Idea Finalized
**Reads with:** [OFFLINE_JETSON_PHASE.md](OFFLINE_JETSON_PHASE.md) (the edge product this builds on)
**Supersedes:** Cloud-first framing in [SYSTEM_OVERVIEW.md](SYSTEM_OVERVIEW.md) / [PROJECT_SUMMARY.md](../PROJECT_SUMMARY.md) where they conflict — LEAFSENSE is **offline-first**; the cloud is a value-add layer, not a dependency.

---

## 1. Business Model at a Glance

The product is sold as **hardware + intelligence**: a Jetson Orin Nano pre-loaded with a plant-specific model variant (initially Turmeric, 5 classes). The device is fully functional offline. Connectivity unlocks a subscription layer.

```
┌─────────────────────────────────────────────────────────────┐
│  FREE PLAN  (offline-first — this is the shipped device)    │
│                                                             │
│  • Full edge pipeline: rotate-capture → detect → segment    │
│    → classify → per-class leaf counts                       │
│  • Class-count differentiation (intra-day + day-over-day)   │
│    correlated with CO₂ / temp / humidity / soil moisture    │
│  • Rule-based (predefined logic) insights, offline          │
│  • On-device monthly EWC incremental retraining             │
│  • Local Streamlit dashboard                                │
│                                                             │
│  Internet, when available, is used ONLY for:                │
│  • Relabel-queue sync (low-confidence samples → cloud       │
│    relabeling → corrected labels back to device)            │
│  • Maintenance / software & model updates                   │
├─────────────────────────────────────────────────────────────┤
│  SUBSCRIPTION PLAN  (requires internet — server-side)       │
│                                                             │
│  • Intelligent RAG CHATBOT grounded on:                     │
│      – Plant-specific knowledge-base docs                   │
│      – Sri Lanka–specific agronomy docs                     │
│      – THIS device's tracked data: class-count changes and  │
│        sensor-variable changes over time                    │
│  • Daily INSIGHT PROVIDER service: interprets the day's     │
│    collected data and returns concrete recommendations      │
│  • KB is per-plant: switching plant variant switches the    │
│    KB docs — the chatbot is always plant-specific           │
└─────────────────────────────────────────────────────────────┘
```

Key differentiation rule: **offline mode ≈ free**; everything intelligent-and-server-side (chatbot, KB, daily insights) is subscription. The KB and RAG never run on-device.

---

## 2. System Architecture

```
┌───────────────────────────────────────────────────────────────┐
│                        CLOUD (subscription + support)          │
│                                                                │
│  ┌──────────────┐  ┌───────────────┐  ┌─────────────────────┐ │
│  │ Relabeling   │  │ Model Registry │  │ RAG Chatbot Service │ │
│  │ Service      │  │ & Updates      │  │  KB: plant docs +   │ │
│  │ (dev labels  │  │ (maintenance)  │  │  Sri Lanka docs +   │ │
│  │  low-conf    │  └───────────────┘  │  device trend data  │ │
│  │  samples)    │  ┌───────────────┐  └─────────────────────┘ │
│  └──────────────┘  │ Subscription/  │  ┌─────────────────────┐ │
│                    │ Billing        │  │ Insight Provider    │ │
│                    └───────────────┘  │ (daily, data-driven)│ │
│                                        └─────────────────────┘ │
└───────────────▲───────────────────────────────▲───────────────┘
                │ intermittent sync             │ online only
                │ (relabel queue, updates,      │ (chat, daily
                │  trend data upload)           │  insights)
┌───────────────┴───────────────────────────────┴───────────────┐
│                 EDGE — Jetson Orin Nano (per plant)            │
│                                                                │
│  360°/45° rotating camera ──► Image pipeline                   │
│    (2 sets/day + manual)      YOLOv8n → crop/filter ≥124×124   │
│                               → YOLOv8n-seg → [GAN, optional]  │
│                               → MobileNetV3-Small              │
│  CO₂/Temp/Humidity/Soil  ──► Decision-tree risk                │
│                                                                │
│  Label-level fusion ──► class counts + risk ──► Streamlit      │
│  Confidence routing ──► retrain buffer / relabel queue         │
│  Monthly on-device EWC retrain ──► metric gate ──► deploy      │
└───────────────────────────────────────────────────────────────┘
```

Full edge detail lives in [OFFLINE_JETSON_PHASE.md](OFFLINE_JETSON_PHASE.md).

---

## 3. The Primary Signal: Class-Count Differentiation

The system's main output — and what both the rule-based insights and the subscription services are built on — is the **change in per-class leaf counts**:

1. **Intra-day:** difference between the morning and evening capture cycles.
2. **Day-over-day:** trend of each of the 5 class lines across days.
3. **Environmental correlation:** every count is timestamp-matched to CO₂, temperature, humidity, and soil moisture, so a count change is always interpretable in its environmental context.

```
Example chain the system reasons over:

  leaf_spot count:  Day1 → 3,  Day2 → 5,  Day3 → 9      (rising)
  humidity (same window):  sustained > 85%              (high)
  soil moisture:           high                          
        │
        ▼
  Rule-based (free):   "Fungal-favorable conditions + leaf spot rising
                        → inspect and consider fungicide"
  Chatbot/Insights ($): grounded answer citing the KB docs for turmeric
                        leaf spot management under Sri Lankan conditions,
                        referencing THIS farm's actual trend
```

---

## 4. Data Flows

### 4.1 Offline (default, free)
1. Rotating capture set (8 × 45°) → edge pipeline → per-class counts
2. Sensor readings matched by timestamp → decision-tree risk
3. Label-level fusion → dashboard + rule-based insights
4. High-conf predictions → retrain buffer; low-conf → relabel queue
5. Monthly: on-device EWC retrain → fixed-test-set comparison vs base model → deploy only if metrics pass

### 4.2 When connectivity is available (free)
1. Relabel queue (Excel/simple sheet) uploads to Relabeling Service
2. Developer (future: LLM teacher module) corrects labels
3. Corrected labels return to the Jetson → join the **next** retraining cycle
4. Software/model maintenance updates pulled from Model Registry

### 4.3 Online subscription
1. Device trend data (class-count history + sensor history) syncs to the cloud
2. **Chatbot:** farmer asks questions → RAG over plant KB + Sri Lanka docs + the farm's own trend data → grounded, farm-specific answers
3. **Insight Provider:** runs daily over the collected data → pushes concrete recommendations

---

## 5. Model & Training Summary (reference)

| Item | Choice |
|---|---|
| Detection | YOLOv8n |
| Segmentation | YOLOv8n-seg |
| Occlusion recovery | Lightweight GAN — **optional**, off by default until validated on real farm data |
| Classification | MobileNetV3-Small — trained from scratch on PlantVillage, 2-stage fine-tune to turmeric |
| Environmental risk | Decision tree |
| Continual learning | EWC incremental fine-tuning, on-device, ~monthly, metric-gated deployment |
| Hyperparameter tuning | Cassava dataset stand-in; 60/10/rest split; 50/100/200 batch-size search; fixed test set for catastrophic-forgetting checks |

---

## 6. Extensibility

- **New plants:** same hardware and pipeline; ship a new classifier variant + (for subscribers) the matching plant KB docs. "Build for one plant, design for many" still holds.
- **Per-farm tuning:** rotation overlap, capture times, and rule thresholds adjustable at installation.
- **LLM teacher module (future):** low-confidence samples auto-labeled by an LLM in remote storage, closing the relabel loop without a human.
- **GAN enablement (future):** switched on once validated on real greenhouse data.

---

## 7. Version History

| Version | Date | Changes |
|---|---|---|
| 1.0 | July 2026 | Initial combined-system idea: offline-first Jetson edge product + free/subscription split + server-side RAG chatbot & daily insight provider |
