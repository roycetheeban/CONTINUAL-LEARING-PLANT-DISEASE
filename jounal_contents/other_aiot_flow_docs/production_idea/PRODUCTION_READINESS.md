# LEAFSENSE — Production Readiness (Optimization, Infrastructure & Deployment)

**Version:** 1.0
**Date:** July 2026
**Status:** Recommendations Accepted — Guides Implementation
**Reads with:** [OFFLINE_JETSON_PHASE.md](OFFLINE_JETSON_PHASE.md) · [FULL_COMBINED_SYSTEM.md](FULL_COMBINED_SYSTEM.md)
**Scope:** How to take the current repo (edge/ + cloud/ microservices scaffold + docker-compose) to a production-level system.

> Two guiding corrections over the naive path:
> **1. No heavy infrastructure on the Jetson.** **2. No Kafka until the fleet actually demands it.**

---

## 1. Edge (Jetson Orin Nano) — Lightweight & Model-Optimized

### 1.1 Model optimization (single biggest win: TensorRT)

| Step | Action |
|---|---|
| Detection / Segmentation | Export YOLOv8n and YOLOv8n-seg to TensorRT engines: `model.export(format="engine", half=True)` — FP16 roughly **doubles throughput** on Orin Nano |
| Classifier | MobileNetV3-Small → ONNX → TensorRT (FP16) |
| Later | INT8 quantization using a calibration set built from **our own greenhouse images** |
| Runtime rules | Load all models **once at startup**, keep warm — never load per capture. Batch **all leaf crops from one image** through the classifier in a single forward pass (not leaf-by-leaf) |

### 1.2 On-device storage: SQLite + filesystem ONLY

- **One SQLite DB**: results, class counts, sensor readings, retrain-buffer metadata, relabel queue, model version history.
- **Images**: dated folder structure (`/data/images/YYYY-MM-DD/cycleA/pos_3.jpg`) with a retention policy so the disk never fills.
- ❌ **No Redis, no Kafka, no Postgres on the device** — one producer, few consumers; a DB server just burns RAM the models need.

### 1.3 Sensors

- Sensors on a separate MCU (e.g. ESP32) → **MQTT (Mosquitto)** on the Jetson as ingest path.
- Sensors wired directly to the Jetson → **no broker at all**, read directly.

### 1.4 Process management & resilience

- Run **capture-scheduler, inference, dashboard, sync** as separate **systemd services** — auto-restart on crash, survive greenhouse power cuts.
- Monthly EWC retraining: low-priority (`nice`), **checkpoint every epoch** (power-cut safe), deploy only through the existing metric gate.

### 1.5 Docker on the Jetson — for fleet reasons, not architecture reasons

- Use NVIDIA **l4t base images** + nvidia container runtime.
- Value appears when shipping many units: identical images per device + OTA updates.
- Fleet management at scale: evaluate **Mender** or **balena** later.
- During development: bare systemd + venv is fine.

---

## 2. Cloud — Right Pieces Already Exist; Resist Adding More

### 2.1 PostgreSQL + TimescaleDB (already in docker-compose ✅)

- Make `sensor_readings` and `class_counts` **hypertables**.
- Use `time_bucket` + **continuous aggregates** (precomputed daily counts per class per device) — the trend graphs and insight provider become cheap queries.
- Add the **pgvector extension** for RAG embeddings — **no separate vector DB** (Pinecone/Qdrant = pure extra ops at our scale).

### 2.2 Redis (already in docker-compose ✅) — three jobs

1. API caching + rate limiting at the gateway.
2. **Celery broker** for background jobs: relabel-sheet processing, daily insight-provider runs, notification fan-out.
3. **Celery Beat** for scheduling the daily jobs.

### 2.3 Kafka — deliberately SKIPPED

| Fact | Consequence |
|---|---|
| Devices are **offline by design**, sync in occasional bursts (16 images/day metadata, a relabel sheet, trend history) | This is **Celery-task territory**, not event-streaming territory |
| Kafka earns its keep at sustained high-throughput streams with multiple consumer groups (thousands of always-online devices) | Adding it now = running a KRaft cluster for a queue that's empty 99% of the time |
| Escalation path | ~1000s of always-online devices → **Redis Streams** first → Kafka only after that |

### 2.4 Missing from docker-compose — ADD these

| Service | Purpose |
|---|---|
| **MinIO** | S3-compatible object storage: images, relabel-queue files, model artifacts. Swap to real S3 in production **without code changes** (same S3 API) |
| **Mosquitto** *(only if MQTT path chosen)* | Device/sensor messaging |
| **Celery worker + beat containers** | Background jobs + scheduling |
| Model registry | MLflow is the standard, but a `model_versions` table in Postgres + artifacts in MinIO covers the `model_registry` service's needs for a long time |

### 2.5 Device ↔ cloud sync: plain HTTPS pull

Devices are intermittently connected — no always-on channel to maintain:

```
Device comes online
  → authenticates (device API key)
  → POST relabel queue + trend data
  → GET pending relabeled data + model/software updates
  → disconnects
```

---

## 3. Subscription Services — Chatbot & LLM Teacher

### 3.1 RAG Chatbot service

- Python SDK (`anthropic`), model **`claude-opus-4-8`**.
- KB chunks (plant docs + Sri Lanka docs) embedded into **pgvector**, top-k retrieval per query.
- **Prompt caching is the cost architecture:** stable KB/system content first with `cache_control: {"type": "ephemeral"}` on the last stable block; volatile content (farm's recent class-count trend, sensor summary, the user question) **after** the cache breakpoint. Cached reads ≈ **10% of normal input cost** — critical when every subscriber question re-sends the same KB context.

### 3.2 LLM Teacher (future auto-relabeling module)

- Perfect fit for the **Message Batches API** — monthly, not latency-sensitive, **50% of standard price**.
- Each low-confidence leaf image sent as a vision request with **structured outputs** (`client.messages.parse()` + Pydantic schema):

```python
class LeafLabel(BaseModel):
    label: Literal["healthy", "blotch", "leafspot", "dry", "aphid"]
    confidence: float
    reasoning: str
```

- Results are machine-readable → straight into the retrain buffer. Keep a **human spot-check on a sample** until the teacher is trusted.

### 3.3 Daily Insight Provider

Celery Beat job per subscriber → one Claude call (same cached KB prefix + that day's aggregated data) → store insight → push via **FCM** (per the notification decision in [extras.md](../../docs_new_idea/extras.md)).

---

## 4. Docker & Deployment Path

- Per-service Dockerfiles: fine as-is. Compose additions:
  - **MinIO** (+ Mosquitto if MQTT), Celery worker + beat.
  - `healthcheck` blocks + `depends_on: condition: service_healthy` so services stop racing Postgres at boot.
- **Production hosting: docker compose on a single VM first.** At 50 devices the entire cloud fits on one $20–40/month box. **Kubernetes is a year-2 decision.**
- ⚠️ **14 separately-deployed services is heavy for a small team.** Keep the folder structure, but initially **run several services in one container** (e.g. gateway + auth + device_management) and split only when load demands it.

---

## 5. Decision Summary

| Concern | Decision | Revisit when |
|---|---|---|
| Edge inference | TensorRT FP16 (INT8 later) | Never — always right |
| Edge storage | SQLite + filesystem | Never for single-device |
| Edge messaging | MQTT only if sensors on separate MCU | Hardware design final |
| Edge processes | systemd services | Containerize for fleet shipping |
| Cloud DB | Postgres + TimescaleDB hypertables + pgvector | — |
| Async jobs | Celery + Redis (broker) + Beat | — |
| Event streaming | ❌ No Kafka | 1000s of always-online devices (Redis Streams first) |
| Object storage | MinIO → S3 (same API) | — |
| Vector DB | pgvector | Massive KB / multi-tenant embedding scale |
| Model registry | Postgres table + MinIO artifacts | MLflow if experiment tracking needed |
| Device sync | HTTPS pull on connect | Always-online fleet |
| Chatbot | claude-opus-4-8 + pgvector RAG + prompt caching | — |
| LLM teacher | Batches API (50% cost) + structured outputs | — |
| Hosting | Compose on one VM | Kubernetes at scale (year 2) |
| Service count | Consolidate services per container initially | Split when load demands |

---

## 6. Version History

| Version | Date | Changes |
|---|---|---|
| 1.0 | July 2026 | Initial production-readiness plan: edge optimization (TensorRT), no-heavy-infra-on-Jetson rule, Kafka deferral, Redis/Celery job architecture, MinIO addition, chatbot/LLM-teacher stack, deployment path |
