# E3/E5 — On-Device Cycle Cost and Latency (Laptop-GPU Proxy)

Fills **Table X** of the LEAFSENSE journal paper. See
`jounal_contents/journal/06_EXPERIMENT_PROTOCOL.md` PART 3 for the original protocol
this follows.

> ## ⚠️ Read this first
> **The Jetson Orin Nano measurement this experiment was originally specified to
> perform has NOT been done.** No Jetson hardware was available. What is measured
> here runs on a **laptop GPU (NVIDIA RTX 4050 Laptop, 6 GB, CUDA 12.1)** — a
> discrete GPU with a completely different power envelope (35–80 W vs. ~15 W),
> memory architecture (dedicated VRAM vs. shared), and thermal behaviour than the
> embedded ARM SoC that is the actual deployment target.
>
> **Do not read the timing, RAM, or latency numbers below as deployment numbers.**
> They validate that the retrain → export → compile → accuracy-gate *pipeline*
> works and that the *methodology* is sound. Closing the gap to real Jetson numbers
> remains E3's genuinely open task.

---

## 1. What is measured here vs. cited from prior work

| Table X row | Source |
|---|---|
| Detection (YOLOv8n) | **Cited** from the prior conference paper (`\cite{ref21}`, Table IV) — not re-measured |
| Segmentation (YOLOv8n-seg) | **Cited** from the same, Table IV — not re-measured |
| Retrain cycle, ONNX export, FP16 gate, classification latency | **Measured here**, on the laptop GPU |

Citing detection/segmentation rather than re-measuring is the paper's own intended
design (`01_SCOPE_AND_COVERAGE.md` §1.2: the 3-stage edge pipeline is "cited to the
IEEE conference paper"), not a shortcut. Re-measuring them on a *third* hardware
platform would only add confusion.

**Caveat on the cited numbers**: the prior paper reports 82.1 FPS (detection) and
200 FPS (segmentation) but does not state unambiguously that these were measured
*on* Jetson hardware — its phrasing ("suitable for Jetson Nano deployment") reads as
a design claim. Separately, the raw training artifacts in this repo
(`other models/model/results_enhanced_yolov8n/performance_metrics.txt`: mAP 0.664,
10.45 ms/img) do **not** match the published Table IV numbers (mAP 0.632, 82.1 FPS),
i.e. they come from a different training run. **Resolution adopted:** cite the
*published* Table IV values, attributed plainly as "reported in [ref21]", with no
claim about which hardware produced them.

---

## 2. Retrain cycle (measured)

`code/time_retrain_cycle.py` re-runs the **real** Case 2 + Replay cycle-2 training
config (`configs/part1_case2_imagenet/continuation/cat_a/replay/cycle2/catA_replay_cycle2.yaml`)
with identical data, hyperparameters, and seed — only the output directory is
redirected, so the real `catA_replay_cycle2/checkpoints/model.pth` that E2 depends
on is never overwritten (verified: its original timestamp is unchanged).

| Metric | Value |
|---|---|
| Total wall time | **373.2 s** (~6.2 min) |
| Training wall time | 348.4 s |
| Peak VRAM | 155.6 MB |
| Process RAM (end of run) | 1245.1 MB |
| Reproduced test accuracy | **98.42%** (macro-F1 0.9827) |

The reproduced accuracy matches the original cycle-2 run exactly, which is a useful
sanity check that this is a faithful re-timing and not a differently-configured run.

Per-epoch breakdown: `outputs/retrain_epoch_log.csv`.

---

## 3. ONNX export and the "compiled artifact" (measured, with a substitution)

| Metric | Value |
|---|---|
| ONNX export time | **1.46 s** |
| ONNX model size | 5.82 MB |
| TensorRT FP16 engine | **NOT BUILT** — see below |

**TensorRT could not be installed in this environment.** The pip packages
(`tensorrt`, `tensorrt-cu12`) pull in `nvidia-cuda-runtime-cu13`, whose wheel fails
to build against this machine's CUDA 12.1 PyTorch build on Windows. This is an
environment constraint, not a code defect; the failure and its reason are recorded
in `outputs/export_compile_metrics.json` rather than silently swallowed.

Two substitute "optimized inference" paths were benchmarked instead, each labeled
for exactly what it is:

- **ONNX Runtime GPU** (`CUDAExecutionProvider`) — a genuine *compiled-graph*
  execution path (graph fusion + kernel selection). This is the closest available
  analogue to the TensorRT step in the paper's deployment design, though it is
  **not** TensorRT.
- **PyTorch eager FP16** (`model.half()`) — the simplest half-precision comparison
  point, used for the accuracy-delta gate measurement.

---

## 4. The FP16 accuracy-delta gate ★ (measured)

This is the measurement that justifies the paper's gating design — evaluating the
*compiled/converted* artifact rather than assuming it matches the training-time
candidate.

| Variant | Test accuracy | Δ vs. FP32 |
|---|---|---|
| PyTorch FP32 (candidate) | 98.505% | — |
| PyTorch FP16 | 98.593% | **+0.088 pp** |
| ONNX Runtime GPU (compiled graph) | 98.505% | **0.000 pp** |

Measured on the full real held-out tomato test set (1,137 images).

**The delta is essentially zero** — and per the protocol's explicit instruction
("Report Δ whatever it is... If it is essentially zero, report that honestly and
reframe the argument as *the gate must verify this rather than assume it*"), this is
reported as-is rather than quietly dropped. The +0.088 pp FP16 difference amounts to
a single image out of 1,137 changing its prediction, i.e. numerical noise, not a
systematic regression. The defensible claim is therefore **not** "FP16 conversion
degrades accuracy, so we gate"; it is "conversion *can* alter behaviour, the gate
measures the artifact that will actually be deployed rather than assuming
equivalence, and in this instance the measurement confirms equivalence."

---

## 5. Classification latency (measured)

Median-of-5-trials × 200 repeats each, after warm-up. A laptop GPU's clocks vary
with thermal/power state, so single-shot timings swung by several ms between runs;
median-of-trials was adopted to get numbers stable enough to report.

| Variant | Per image | Per 8-image capture set |
|---|---|---|
| PyTorch FP32 eager | 10.24 ms | 10.59 ms |
| PyTorch FP16 eager | 12.38 ms | 12.44 ms |
| **ONNX Runtime GPU (compiled)** | **3.78 ms** | **4.63 ms** |

Two honest observations:

- **PyTorch FP16 is *slower* than FP32 here.** MobileNetV3-Small is small enough
  that per-op half-precision conversion overhead outweighs any arithmetic gain at
  this batch size. This is a real result, reported rather than suppressed — it also
  reinforces §4's point that conversion effects must be measured, not assumed.
- **The compiled ONNX graph is ~2.7× faster** than PyTorch eager at identical
  accuracy, which is the meaningful argument for compiling before deployment.

---

## 6. Assumptions and limitations

- **Wrong hardware.** Everything in §2–§5 is a laptop-GPU proxy for a Jetson Orin
  Nano. Absolute times, RAM, and power characteristics do not transfer.
- **TensorRT substituted.** The paper's design specifies a TensorRT FP16 engine;
  what was actually measured is ONNX Runtime GPU + PyTorch FP16. Anywhere the paper
  describes these numbers, it must say so — never attribute them to TensorRT.
- **No power measurement.** The protocol's optional idle/peak wattage and thermal
  throttling checks (`tegrastats`) are Jetson-specific and were not performed.
- **Detection/segmentation not re-measured** — cited from `\cite{ref21}`, with the
  provenance caveat in §1.
- **Single seed (42).** No multi-run variance on the retrain cycle timing.
- **End-to-end capture-set time is a sum across sources**, not one unified trace:
  cited detect + segment times added to measured classification time, from different
  hardware contexts. Treat it as indicative only.

---

## 7. What's missing / open items

1. **The actual Jetson Orin Nano measurement** — E3's real, still-unfinished task.
   Requires the physical device: retrain wall-time, peak RAM via `tegrastats`,
   `trtexec` engine build time, thermal-throttling check, and power mode
   (`nvpmodel -q`) disclosure.
2. **A real TensorRT FP16 engine** and its accuracy delta — blocked by the CUDA
   12/13 packaging conflict described in §3; on a Jetson (JetPack ships TensorRT)
   this would not be an issue.
3. **Per-stage detection/segmentation latency measured in this repo** — the
   YOLOv8n/-seg checkpoints exist under `other models/model/` but their published
   metrics come from a different training run than the artifacts here (§1); they are
   cited, not re-derived.
4. **No multi-seed variance** on any timing number.
5. **GAN occlusion-recovery stage not benchmarked** — optional per the paper's scope
   (`01_SCOPE_AND_COVERAGE.md` §1.2), cited to prior work if needed.

---

## 8. How to reproduce

```bash
python experiments/part4_ondevice_proxy/code/time_retrain_cycle.py
python experiments/part4_ondevice_proxy/code/export_and_compile.py
python experiments/part4_ondevice_proxy/code/bench_accuracy_latency.py
```

Run with the repo's `AIOT` conda environment (Python 3.10, torch 2.5.1+cu121,
torchvision 0.20.1+cu121), plus `onnx` 1.22.0 and `onnxruntime-gpu` 1.28.0 which
this experiment adds. Note `time_retrain_cycle.py` takes ~6 minutes.

Outputs in `experiments/part4_ondevice_proxy/outputs/`:

| File | Contents |
|---|---|
| `retrain_cycle_metrics.json` | Wall time, peak VRAM/RAM, reproduced accuracy |
| `retrain_epoch_log.csv` | Per-epoch timing and validation metrics |
| `model.onnx` | Exported ONNX graph |
| `export_compile_metrics.json` | Export time/size, which compile path ran, TensorRT failure reason |
| `accuracy_latency_metrics.json` | FP32/FP16/ONNX-Runtime accuracy and latency |
| `retrain_timing/` | Full training run output (separate from the real cycle-2 artifacts) |
