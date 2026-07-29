# LEAFSENSE — Jetson Production Flow (Implementation Spec)

**Audience:** developer building the on-device application
**Target:** NVIDIA Jetson Orin Nano
**Status:** build spec — this document defines *what to implement*, not a record of what exists
**Background reading:** `jounal_contents/other_aiot_flow_docs/production_idea/OFFLINE_JETSON_PHASE.md` (product concept), `jounal_contents/journal/leafsense.tex` §III–V (design rationale)

---

## 0. Scope of this document

Covers the complete on-device loop:

```
capture → image pipeline → per-class counts ─┐
                                             ├─► fusion ─► action + dashboard
sensor read → risk model ────────────────────┘
                    │
                    └─► confidence routing → retrain buffer → monthly retrain → gate → swap
```

**Phase-1 focus (what to build first):** the image path end-to-end through classification, plus the retraining cycle. Sensors run on **dummy data** in this phase (§6.4) — the ESP32 hardware path is specified so it drops in later without redesign.

---

## 1. ⚠️ Critical status notes — read before coding

| # | Issue | Impact |
|---|---|---|
| 1 | **The classifier checkpoint in this repo is TOMATO 5-class, not turmeric.** `Tomato___Bacterial_spot, Early_blight, Late_blight, Leaf_Mold, healthy` | The production unit ships turmeric (Healthy / Leaf Blotch / Leaf Spot / Dry / Aphid). A turmeric classifier **does not exist yet** — that is experiment E4, not started. Build the pipeline against the tomato checkpoint, keep class names in config, swap the checkpoint when E4 completes. |
| 2 | **Detection + segmentation models are turmeric-trained**, classifier is tomato-trained | Mixed-crop pipeline during Phase 1. Functional for building/testing the plumbing; not a valid product until E4. |
| 3 | **YOLO `.pt` files require the `ultralytics` package** — architecture is pickled inside them | `pip install ultralytics` is mandatory; version must match the Jetson's torch/JetPack build. |
| 4 | **GAN generator is a bare `state_dict`** (180,338 params) — no architecture inside | Must copy the `TinyUNet` class out of `other models/code/light_weight_gan - Copy.ipynb` into a `.py` module before loading. Optional stage, disabled by default. |
| 5 | Confidence thresholds τ_high / τ_low are **not yet decided** | Provisional values in §11; treat as tunable config, validate on real data before trusting. |

---

## 2. Hardware layer

### 2.1 Compute + camera

| Item | Spec | Notes |
|---|---|---|
| Compute | Jetson Orin Nano | JetPack ships TensorRT + CUDA preinstalled |
| Camera | **USB camera** (UVC) | Phase 1 decision. Access via OpenCV `VideoCapture(index, cv2.CAP_V4L2)` |
| Mount | Motorized, 360° in 45° steps → 8 positions | Control interface TBD — abstract behind `CameraMount` (§5.2) |

### 2.2 Sensor cluster (ESP32-mediated)

Sensors do **not** connect to the Jetson directly. An ESP32 reads them and relays to the Jetson.

| Sensor | Qty | Measures | ESP32 interface | Constraints that affect design |
|---|---|---|---|---|
| **ESP32** | 1 | — | host MCU | Bridges all sensors → Jetson |
| **DHT22** | 3 | Temperature + Humidity | 1-wire digital, one GPIO each | **Max 0.5 Hz** (one reading / 2 s). Hard sensor limit. |
| **Capacitive Soil Moisture v1.2** | 4 | Soil moisture | **Analog** → ESP32 ADC | Use **ADC1 pins only** (ADC2 is unusable while WiFi is active). Non-linear ADC — needs calibration. |
| **MH-Z19B** | 1 | CO₂ | UART, 9600 baud | ~3 min warm-up from cold; ~5 s response time. |

**Three design consequences — do not ignore:**

1. **The "1 reading/second" figure from the dev CSV is not physically achievable.** DHT22 caps at 0.5 Hz. Set the production sample interval to **≥ 2 s**; recommend **5 s** to give MH-Z19B headroom. This does not harm the model — the risk features aggregate over a 120-hour window (§8), so sub-minute resolution is irrelevant.

2. **Multiple sensors of the same type need an aggregation policy.** 3× DHT22 and 4× soil probes = multi-zone sensing. The risk model consumes **one scalar per variable**. Decide and record in config: `median` (recommended — robust to a single failed probe) vs `mean`. Report per-probe values to the dashboard, but feed the aggregate to the model.

3. **Capacitive soil sensors require per-probe calibration.** Raw ADC counts are meaningless across probes. Store `dry_raw` / `wet_raw` per probe in config; convert with
   `moisture_pct = 100 × (dry_raw − raw) / (dry_raw − wet_raw)`, clamped to [0, 100].

### 2.3 ESP32 → Jetson transport

**Recommended: USB serial (CDC-ACM).** Simplest, no network dependency, matches the offline-first product promise. Appears as `/dev/ttyUSB0` or `/dev/ttyACM0`.

Alternative (WiFi/MQTT) is viable but adds a broker and contradicts offline-first; only choose it if the ESP32 must be physically distant from the Jetson.

**Wire format — one JSON object per line (NDJSON), newline-terminated:**

```json
{"ts":"2026-07-27T08:00:03.120Z","seq":19244,"temp_c":[27.4,27.1,27.9],"rh_pct":[81.2,80.7,82.0],"soil_pct":[43.1,45.8,41.0,44.2],"co2_ppm":812,"status":{"dht":[1,1,1],"soil":[1,1,1,1],"co2":1}}
```

| Field | Meaning |
|---|---|
| `ts` | ESP32 timestamp, ISO-8601 UTC. **See §6.3 on clock trust.** |
| `seq` | Monotonic counter — lets the Jetson detect dropped lines |
| `temp_c`, `rh_pct` | Arrays, one entry per DHT22, index-stable |
| `soil_pct` | Array, one per probe, **already calibrated** on the ESP32 |
| `co2_ppm` | Single integer |
| `status` | Per-sensor health: `1` ok, `0` read failure. Failed sensors send `null` in the data array. |

**Rules:**
- Never drop a field on sensor failure — send `null` and flag it in `status`. A missing key breaks parsing; an explicit `null` is handled.
- The ESP32 does calibration (§2.2 item 3) so the Jetson receives physical units only.
- Newline-delimited so the Jetson can read line-by-line without framing logic.

---

## 3. Module layout

```
app_dev/
  leafsense/
    config.py            # load + validate YAML, expose typed settings
    capture/
      camera.py          # USBCamera: open, warm, grab frame
      mount.py           # CameraMount: rotate_to(position), home()
      scheduler.py       # Cycle A/B timing, on-demand trigger
    sensors/
      reader.py          # serial listener thread → latest reading + ring buffer
      dummy.py           # Phase-1 synthetic source, same interface as reader
      aggregate.py       # multi-probe → single scalar (median/mean)
    pipeline/
      detect.py          # YOLOv8n
      segment.py         # YOLOv8n-seg
      gan.py             # TinyUNet (optional, off by default)
      classify.py        # MobileNetV3-Small
      runner.py          # orchestrates stages 1-7 for one image
    risk/
      features.py        # 120h window aggregation → 13 features
      tree.py            # decision_tree.pkl inference
    fusion/
      rules.py           # Φ lookup: (trend, risk) → action
      counts.py          # per-class counting, Δ vs previous set
    learning/
      routing.py         # confidence → buffer | queue | discard
      retrain.py         # monthly cycle
      gate.py            # ONNX → TensorRT → eval → swap/rollback
    storage/
      db.py              # SQLite schema + accessors
      paths.py           # canonical on-device paths
    app.py               # main service loop
    dashboard.py         # Streamlit UI
```

---

## 4. Configuration

Single YAML at `app_dev/config/jetson.yaml`. Every tunable lives here — no magic numbers in code.

```yaml
crop_variant: turmeric          # selects class list + model bundle
classes:                        # ⚠ swap to turmeric list when E4 lands
  - Healthy
  - Leaf_Blotch
  - Leaf_Spot
  - Dry
  - Aphid

paths:
  root: /var/lib/leafsense
  models: ${paths.root}/models
  captures: ${paths.root}/captures
  retrain_buffer: ${paths.root}/retrain_buffer
  relabel_queue: ${paths.root}/relabel_queue
  db: ${paths.root}/leafsense.db

camera:
  device_index: 0
  backend: v4l2
  resolution: [1920, 1080]
  warmup_frames: 5              # discard first N frames (auto-exposure settle)
  jpeg_quality: 95

mount:
  positions: 8
  step_degrees: 45
  settle_ms: 800                # wait after rotation before shutter — anti-blur
  home_on_start: true

schedule:
  cycle_a: "08:00"
  cycle_b: "18:00"
  timezone: Asia/Colombo

sensors:
  mode: dummy                   # dummy | serial   ← Phase 1 = dummy
  serial:
    port: /dev/ttyUSB0
    baud: 115200
    read_timeout_s: 5
  sample_interval_s: 5          # ≥2 s enforced by DHT22
  aggregation: median           # median | mean
  stale_after_s: 120            # older than this ⇒ risk = UNKNOWN

pipeline:
  letterbox_size: 640
  min_crop_px: 124
  seg_input: 640
  classifier_input: 224
  classifier_resize: 200        # resize to 200, pad to 224
  detect_conf: 0.25
  detect_iou: 0.45
  gan_enabled: false            # ships disabled — not validated on real data

models:
  detector:   ${paths.models}/best_enhanced_01_n.pt
  segmenter:  ${paths.models}/middle_leaf_segmentation_best.pt
  classifier: ${paths.models}/classifier_current.pth
  gan:        ${paths.models}/generator_final.pth
  risk_tree:  ${paths.models}/decision_tree.pkl
  engine_dir: ${paths.models}/engines

routing:
  tau_high: 0.90                # ⚠ PROVISIONAL — not validated
  tau_low:  0.50                # ⚠ PROVISIONAL — not validated

retrain:
  interval_days: 30
  min_buffer_per_class: 40      # skip cycle if any class below this
  epochs: 12
  batch_size: 32
  lr_g3: 0.00005
  lr_head: 0.0002
  freeze_through_layer: 9       # G1-G2 frozen; G3-G4 trainable
  checkpoint_every_epoch: true  # power-cut safety
```

---

## 5. Capture subsystem

### 5.1 Capture set definition

One **capture set** = 8 images, one per 45° mount position. Two sets/day (Cycle A morning, Cycle B evening) = 16 images/day, plus on-demand from the dashboard.

### 5.2 Sequence

```
for position in 0..7:
    mount.rotate_to(position)         # blocking
    sleep(mount.settle_ms)            # ← mandatory: prevents motion blur
    frame = camera.grab()             # after warmup_frames discarded
    sensor_snapshot = sensors.latest()  # timestamp-matched, §6.3
    save(frame, meta={set_id, position, ts, sensor_snapshot})
mount.home()
```

**Implementation requirements:**
- **Discard `warmup_frames`.** USB cameras deliver stale/underexposed frames on first grab; a UVC device also buffers. Grab and throw away N frames before keeping one.
- **Settle delay is not optional.** Capturing mid-vibration produces blur that silently degrades detection recall — a failure mode that is invisible unless you inspect crops.
- Store the raw full frame **and** the derived crops. Retraining consumes crops; debugging needs the original.
- If any single position fails, log it and continue — a 7-image set is still usable. Abort only if ≥4 fail.

### 5.3 Filenames

```
captures/2026-07-27/setA_20260727T080000/pos3.jpg
                                        /pos3_crop02.jpg
                                        /meta.json
```

---

## 6. Sensor subsystem

### 6.1 Reader thread (serial mode)

Runs continuously, independent of capture. Owns the serial port.

```
loop:
    line = serial.readline()            # blocking, with timeout
    reading = parse_ndjson(line)
    validate(reading)                   # ranges + status flags
    aggregated = aggregate(reading)     # §6.2
    ring_buffer.append(aggregated)      # in-memory, ~48h
    db.insert_reading(aggregated)       # durable
    latest = aggregated
```

Never let a malformed line kill the thread — catch, log, drop the line, continue. Sensor faults must degrade the system, not crash it.

### 6.2 Aggregation + validation

| Variable | Source | Aggregate | Valid range |
|---|---|---|---|
| `temp_c` | 3× DHT22 | median | −10 … 60 °C |
| `rh_pct` | 3× DHT22 | median | 0 … 100 % |
| `soil_pct` | 4× capacitive | median | 0 … 100 % |
| `co2_ppm` | 1× MH-Z19B | (single) | 300 … 5000 ppm |

Out-of-range → treat as `null` for that probe and exclude from the median. If **all** probes for a variable are invalid, that variable is `null` for the reading.

### 6.3 Timestamp matching — trust the Jetson clock

Every image carries the **most recent sensor reading**, matched at capture time.

**The ESP32 has no RTC and no NTP.** Its `ts` field is uptime-relative and will not match wall-clock. Therefore:
- The **Jetson** stamps `received_at` when it reads the line. **This is the authoritative timestamp.**
- The ESP32's `ts`/`seq` are used only to detect drops and ordering.
- At capture, attach the reading with the greatest `received_at` ≤ capture time.
- If that reading is older than `sensors.stale_after_s`, attach it but mark `sensor_stale: true` → risk state becomes **UNKNOWN**, and fusion degrades to vision-only (§10).

### 6.4 Dummy mode (Phase 1)

`sensors/dummy.py` implements the identical interface as the real reader, so `app.py` is unaware of the difference. It must be *plausible*, not random:

- Diurnal sinusoid on temperature and humidity (peak/trough at realistic hours)
- Slow AR(1) drift so consecutive readings correlate — the 120 h feature window needs realistic autocorrelation or the risk model sees nonsense
- Occasional dropout to exercise the `null` path
- Emits on the same `sample_interval_s`, generated **at capture time** so images and readings share timestamps

Reference implementation to copy: `experiments/part2_env_risk/code/simulate_sensors.py` already generates exactly this (diurnal + seasonal + AR(1) + noise + dropout).

---

## 7. Image pipeline — stage by stage

Runs per image; a capture set runs it 8×.

| # | Stage | Input | Output | Model / op |
|---|---|---|---|---|
| 1 | Letterbox | H×W×3 arbitrary | 640×640×3 | aspect preserved, **black** pad |
| 2 | Detect | 640×640×3 | N boxes | `best_enhanced_01_n.pt` (6.15 MB) |
| 3 | Crop + filter | boxes | M ≤ N crops | keep only **≥124×124 px** |
| 4 | Segment | crop | mask, bg→black | `middle_leaf_segmentation_best.pt` (6.77 MB) |
| 5 | Resize + pad | masked crop | **224×224×3** | resize→200×200, zero-pad→224 |
| 6 | GAN *(optional)* | 224×224×3 | 224×224×3 | `generator_final.pth` — **off by default** |
| 7 | Classify | 224×224×3 | class + confidence | `classifier_current.pth` (MobileNetV3-Small) |

### 7.1 Stage details that matter

**Letterbox (1)** — pad, never stretch. Distorting aspect ratio degrades detection because the model was trained on letterboxed 640×640.

**Quality filter (3)** — crops below 124×124 are discarded, not upscaled. Upscaling fabricates detail and produces confident-but-wrong classifications, which then poison the retrain buffer via confidence routing. This filter is a **data-integrity control**, not just an optimization.

**Resize→pad (5)** — resize to 200×200 then pad to 224×224 (not direct resize to 224). This preserves the leaf's apparent scale consistently with training.

**Classifier preprocessing (7)** — must match training exactly:
```python
transforms.Resize((224, 224))
transforms.ToTensor()
transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
```
Class index order is **alphabetical (ImageFolder order)** and is **not stored in the checkpoint** — it must come from config. A wrong order silently mislabels everything. Reference: `experiments/part3_fusion_eval/code/vision_infer.py`.

**GAN (6)** — ships disabled. Two reasons: it is unvalidated on real greenhouse imagery (trained on synthetic occlusions), and its `state_dict` needs the `TinyUNet` class extracted from the notebook first. Enable only after validation.

### 7.2 Set-level output

```json
{
  "set_id": "setA_20260727T080000",
  "counts": {"Healthy": 41, "Leaf_Blotch": 3, "Leaf_Spot": 5, "Dry": 2, "Aphid": 0},
  "total_leaves": 51,
  "images_ok": 8,
  "sensor_snapshot": {"temp_c": 27.4, "rh_pct": 81.2, "soil_pct": 43.1, "co2_ppm": 812},
  "sensor_stale": false
}
```

---

## 8. Risk model (sensor path)

Independent of the image path; CPU-only; ~10 µs per inference.

1. Pull the trailing **120-hour** window of readings from the DB.
2. Build the **13 features** (exact names and order fixed by the trained model):
   `temp_mean, temp_min, temp_max, temp_range, hours_temp_in_band, rh_mean, rh_max, rh_min, rh_std, hours_rh_above_85, rh_trend, soil_mean, co2_mean`
3. `decision_tree.pkl` → **Low | Medium | High**

**Window rule:** if the window has <80% coverage (sensor outage), emit **UNKNOWN** rather than a guess computed from sparse data.

> **Why 120 h:** the model was trained on a 5-day accumulation window matching the underlying infection model. A shorter window was tested during development and produced near-random accuracy. Do not shorten it.

Reference: `experiments/part2_env_risk/code/features.py`.

---

## 9. Per-class counting and Δ

Fusion consumes the **change** in counts, not absolutes — this cancels the constant double-counting bias from overlapping 45° views.

```
Δ_class = count(current set) − count(previous set of the same cycle)
```

Compare **Cycle A to Cycle A** and **B to B**, not A to B — the two differ systematically (lighting, leaf turgor). Trend classification:

| Δ (% of previous) | Trend |
|---|---|
| ≥ +20 % | `rising` |
| −20 % … +20 % | `stable` |
| ≤ −20 % | `falling` |

Guard the divide-by-zero case: when the previous count is 0, any increase is `rising`; 0→0 is `stable`.

---

## 10. Label-level fusion (Φ)

Combines the two branches into one action. Pure lookup — no learned parameters, no inference cost.

| Trend | Risk | Action |
|---|---|---|
| rising | High | **ALERT** — probable active infection |
| rising | Medium | **WATCH** — monitor closely |
| rising | Low | **OBSERVE** — likely non-environmental cause |
| stable | High | **WATCH** — conditions favourable, no symptoms yet *(early warning)* |
| stable | Low/Medium | **NORMAL** |
| falling | any | **NORMAL** — recovering |
| any | **UNKNOWN** | **fall back to vision-only** |

**Sensor-failure degradation is mandatory behaviour**, not a nicety: when risk is UNKNOWN, report the vision-only decision (`rising → ALERT`, else `NORMAL`) rather than a joint state computed from a missing input.

**Known limitation to surface in the UI:** Φ cannot distinguish genuinely-elevated humidity from coincidental humidity (e.g. just after irrigation). Both produce `stable + High → WATCH`. This was measured — it is the one scenario category where fusion fails. Show the contributing inputs alongside the action so the grower can apply context the rule cannot.

Reference: `experiments/part3_fusion_eval/code/fusion_eval.py`.

---

## 11. Confidence routing

Every classified leaf is routed by softmax confidence. This is how the device generates its own training data.

```
c ≥ τ_high  → retrain buffer, auto-labelled with predicted class
c < τ_low   → relabel queue (never trained on unreviewed)
otherwise   → discard
```

**The mid-band discard is deliberate.** Training on unreviewed mid-confidence self-labels is the primary mechanism by which self-training systems drift — errors become their own supervision and compound each cycle.

- Buffer is organized **per class**, with a cap per class (evict oldest) to bound storage.
- Relabel queue exports as a review sheet; corrected labels return and join the **next** cycle, never the current one.
- τ values in §4 are **provisional**. Calibrate against a labelled sample before relying on them — an incorrectly low τ_high poisons the buffer silently.

---

## 12. Monthly retraining cycle

Only the **classifier** retrains. Detection, segmentation, and GAN are frozen in the field.

```
1. Precheck    — every class has ≥ min_buffer_per_class samples, else skip cycle
2. Assemble    — retrain buffer + returned relabelled samples
3. Fine-tune   — G1-G2 frozen, G3-G4 trained (lr_g3, lr_head), EWC/replay
                 checkpoint every epoch  ← power-cut safe
4. Export      — candidate → ONNX
5. Compile     — ONNX → TensorRT FP16 engine  (built ON DEVICE, never shipped)
6. Gate        — evaluate the COMPILED ENGINE on the FIXED test set
7. Decide      — engine_acc > incumbent_acc ? atomic swap : keep incumbent
8. Retain      — keep previous engine for instant rollback
```

**Why the gate evaluates the engine, not the PyTorch model:** FP16 conversion can shift accuracy. Gating on the training-precision model measures something that will never serve production. This is the single most important correctness property of the cycle.

**Fixed test set** is held out at initial training and **never modified**. It is the only defence against catastrophic forgetting going undetected.

**Measured cost** (laptop RTX 4050 proxy — Jetson will differ, likely slower):

| Step | Time | Memory |
|---|---|---|
| Retrain cycle | 373 s | 156 MB VRAM |
| ONNX export | 1.5 s | 5.8 MB model |
| Classification, eager | 10.2 ms/img | |
| Classification, compiled | 3.8 ms/img | |

Reference: `experiments/part4_ondevice_proxy/`.

---

## 13. Storage (SQLite)

```sql
readings(id, received_at, esp_ts, seq, temp_c, rh_pct, soil_pct, co2_ppm,
         raw_json, valid)
capture_sets(set_id, cycle, started_at, images_ok, sensor_stale)
detections(id, set_id, position, crop_path, pred_class, confidence, routed_to)
counts(set_id, class, count)
risk(set_id, risk_level, window_coverage)
fusion(set_id, trend, risk_level, action)
model_versions(version, deployed_at, engine_path, fixed_test_acc, active)
```

Retention: raw frames age out after N days (config); crops in the retrain buffer persist until the buffer cap evicts them.

---

## 14. Performance budget (per 8-image set)

Using per-image figures: detection 12.2 ms, segmentation 5.0 ms/leaf, classification 3.8 ms/leaf (compiled).

At ~6 leaves/image: `8 × (12.2 + 6×5.0 + 6×3.8) ≈ 520 ms` plus capture and rotation overhead (~8 s dominated by `settle_ms`).

Two sets/day ⇒ the device is computationally idle >99.9% of the time. Retraining contends with nothing.

---

## 15. Failure handling

| Failure | Required behaviour |
|---|---|
| Camera not found | Retry with backoff; alert on dashboard; **do not** crash the service |
| Single position capture fails | Log, continue; abort set only if ≥4 fail |
| Serial disconnected | Reader retries; readings go stale → risk UNKNOWN → vision-only fusion |
| One DHT22 / soil probe dead | Excluded from median; system continues on remaining probes |
| MH-Z19B dead | `co2_ppm` null → feature imputed; log prominently (single point of failure, no redundancy) |
| Power cut mid-retrain | Resume from last epoch checkpoint |
| Engine build fails | Keep incumbent engine; log; retry next cycle |
| Candidate fails gate | Keep incumbent; carry data forward to next cycle |
| Disk full | Stop capture, preserve DB + buffer, alert |

---

## 16. Build order

1. `config.py`, `storage/` — foundation
2. `capture/camera.py` + `dummy` sensors — get images and plausible readings flowing
3. `pipeline/` stages 1→7, verify crop shapes at each boundary
4. `fusion/counts.py`, `risk/`, `fusion/rules.py` — close the decision loop
5. `learning/routing.py` — start accumulating training data
6. `learning/retrain.py` + `gate.py` — close the learning loop
7. `dashboard.py`
8. Swap dummy sensors → ESP32 serial; swap tomato classifier → turmeric (E4)

---

## 17. Open items

| # | Item | Blocks |
|---|---|---|
| 1 | **Turmeric classifier (E4)** not trained | Real product; pipeline can be built without it |
| 2 | Mount control interface undefined (GPIO / serial / driver) | `capture/mount.py` |
| 3 | τ_high / τ_low not calibrated | Routing quality — provisional values usable for build |
| 4 | ESP32 firmware not written | Phase 2; dummy source covers Phase 1 |
| 5 | Soil probe calibration constants not measured | Per-probe `dry_raw`/`wet_raw` needed at install |
| 6 | GAN `TinyUNet` class not extracted to `.py` | Optional stage only |
| 7 | Fixed test set for the turmeric variant not defined | Retrain gate — required before first cycle |
