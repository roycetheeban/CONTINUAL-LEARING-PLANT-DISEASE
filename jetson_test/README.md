# Jetson Orin Nano — On-Device Measurement Bundle

Self-contained. Copy this **whole folder** to the Jetson and run it there. Nothing
inside references the main repo, so nothing else needs transferring.

**Size:** ~54 MB, 4,046 files.

---

## Why this exists

Table XIII of the paper currently reports **laptop RTX 4050 proxy numbers**, because
no Jetson was available and TensorRT would not install on Windows. That forced three
compromises:

1. TensorRT FP16 engine build → reported as **`n/a`**
2. The "compiled artifact" → an **ONNX Runtime graph**, not the TensorRT engine the design specifies
3. "The retraining cycle fits the target device's envelope" → **inferred, not measured**

It also left the abstract's claim that *"a Jetson Orin Nano executes the complete
perception-to-adaptation loop on device"* unsupported by any measurement.

This bundle closes all of that.

**Not in scope:** the CL benchmark (Tables VI–XII), E1, E2, E4. Those are method
comparisons and are legitimately run on any GPU. Only *deployment-cost* claims need
the target hardware.

---

## Before you run anything

Open `config.yaml` and fill the three `FILL ME` fields:

```yaml
hardware:
  variant: "8GB"            # or "4GB"
  jetpack_version: "6.0"    # cat /etc/nv_tegra_release
  tensorrt_version: "8.6.2" # python3 -c "import tensorrt; print(tensorrt.__version__)"
```

The scripts **refuse to run** while those say `FILL ME` — they become the hardware
provenance printed in the paper, and placeholder metadata attached to real numbers is
how bad claims get published.

Thermal and power draw are **not** measured (server/headless access only). That is
recorded in the config and should not be reported.

---

## Environment

Budget half a day for this — it eats more time than the measurements do.

```bash
python3 -c "import torch; print(torch.__version__, torch.cuda.is_available())"
python3 -c "import tensorrt; print(tensorrt.__version__)"
```

- **PyTorch must be NVIDIA's JetPack wheel.** `pip install torch` pulls an x86 build that will not work.
- `torchvision` usually needs building from source, matched to the torch version.
- Step 4 needs `pip install ultralytics --no-deps` — **the `--no-deps` matters**, it stops pip replacing JetPack's torch.

Also needed: `numpy`, `pyyaml`, `pillow`, `scikit-learn`, `matplotlib`, `psutil`.

---

## Running

```bash
cd jetson_test
python3 code/run_all.py               # everything
python3 code/run_all.py --only 1 2    # critical pair only (~2 min)
python3 code/run_all.py --skip 3      # skip the long retrain
```

| Step | Script | Measures | Time |
|---|---|---|---|
| 1 | `01_export_build_trt.py` | ONNX export + **TensorRT FP16 engine build** | ~1 min |
| 2 | `02_gate_and_latency.py` | **Conversion gate** (FP32 vs TRT-FP16 accuracy) + classification latency | ~1 min |
| 3 | `03_retrain_cycle.py` | **Retraining cycle** wall-clock + peak memory | 10 min – hours |
| 4 | `04_bench_detect_seg.py` | Detection + segmentation **latency** | ~2 min |

**If Jetson time is limited, steps 1 and 2 alone fix most of the problem.**

---

## What to send back

Everything in `outputs/*.json`. Each file carries its own hardware/software
provenance block, so nothing has to be remembered separately.

```
outputs/01_export_build_trt.json
outputs/02_gate_and_latency.json
outputs/03_retrain_cycle.json
outputs/04_detect_seg_latency.json
```

---

## Two results worth watching for

**The gate delta may be non-zero.** On the laptop, ONNX Runtime gave Δ = 0.00, which
made the deployment gate look decorative and forced the paper to argue it was
*"correct by luck rather than by evidence."* Real TensorRT FP16 does layer fusion and
kernel autotuning, so it may genuinely shift accuracy. **If it does, the paper gets
stronger** — the gate becomes empirically justified. Both outcomes are reportable.
**Do not tune anything to chase one.**

**The retrain cycle may not fit.** Peak was 155.6 MB on the laptop, so 8 GB shared
memory should be ample; the 4 GB variant is tighter. If it does not fit, that is a
finding that changes the paper's claim — report it, don't work around it.

---

## Design notes

- **No silent fallbacks.** If the TensorRT build fails, step 1 writes the reason and exits non-zero. It will not substitute an ONNX Runtime graph and let a fallback number get reported as TensorRT — that is exactly what happened on the laptop and it cost a table row.
- **Nothing in the main repo is touched.** Step 3 writes to `outputs/retrain_cycle/`; the real `catA_replay_cycle2` checkpoint that E2 depends on is never overwritten.
- **The eval transform is a direct resize to 224×224, no crop** — matching what the training scripts actually do. Do not "correct" this to resize-256-then-centre-crop; that is not the transform the reported accuracies were measured under.
- **The replay manifest was rewritten.** The original carried Windows backslash paths pointing outside the bundle; the 200 cycle-1 images it references are included in `data/cl_cycle1_replay_adds/` and paths are made absolute at runtime.
- **Detection/segmentation is latency only.** No ground truth ships here, so mAP@0.5, pixel accuracy and mIoU stay cited to [ref21].

---

## Contents

```
config.yaml                     paths, hyperparameters, hardware provenance
code/
  common_jetson.py              model build, transforms, provenance stamping
  01_export_build_trt.py
  02_gate_and_latency.py
  03_retrain_cycle.py
  04_bench_detect_seg.py
  run_all.py
  train_catA_replay.py          copied verbatim from the main repo
models/
  classifier_catA_replay_cycle2.pth   deployed classifier (98.42% test acc)
  yolov8n_detect.pt
  yolov8n_seg.pt
data/
  test/                  1,137   fixed test set — the gate's evaluation set
  val/                     758   retrain-cycle validation
  cl_cycle2_stream/      1,420   cycle-2 incoming stream
  replay_buffer/           526   replay exemplars
  cl_cycle1_replay_adds/   200   cycle-1 additions the manifest references
  replay_buffer_manifest_cycle1.json
outputs/                        written on the Jetson
```
