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

## Verified target device

Surveyed on the actual unit on 2026-08-09. These are measured, not assumed.

| Field | Value | Source |
|---|---|---|
| Device | NVIDIA Jetson Orin Nano **Super** Developer Kit (Engineering Reference) | `/proc/device-tree/model` |
| Variant | 8 GB (7.4 GiB visible, shared CPU/GPU) | `free -h` |
| L4T | R36 rev 5.0 (GCID 43688277, aarch64) | `/etc/nv_tegra_release` |
| JetPack | 6.2.2+b24 | `dpkg -l nvidia-jetpack` |
| TensorRT | 10.3.0 (deb `10.3.0.30-1+cuda12.5`) | `python3 -c "import tensorrt"` |
| CUDA toolkit | 12.6 (V12.6.68) | `nvcc --version` |
| Python | 3.10.12 (system) | `python3 -V` |
| CPU | 6 cores, `schedutil` governor | `nproc` |
| Power mode | **25 W** | `nvpmodel -q` |
| Disk | 11 GB free of 56 GB | `df -h /` |

Two corrections to earlier assumptions, both of which matter for the paper:

- This is the **Super** variant (67 TOPS), not the original 40-TOPS Orin Nano. Latency
  numbers are not comparable to the older board and the paper must say "Super".
- Power **mode** is queryable after all (`nvpmodel -q` → 25 W), so it should be reported.
  Power **draw** and thermals are still not measured, and remain out of scope.

## Before you run anything

Fill the `FILL ME` fields in `config.yaml` with the surveyed values above:

```yaml
hardware:
  device_name: "NVIDIA Jetson Orin Nano Super Developer Kit"
  variant: "8GB"
  jetpack_version: "6.2.2"     # L4T R36.5.0
  tensorrt_version: "10.3.0"
  power_mode: "25W"            # nvpmodel -q
```

The scripts **refuse to run** while those say `FILL ME` — they become the hardware
provenance printed in the paper, and placeholder metadata attached to real numbers is
how bad claims get published.

---

## Environment

Budget half a day for this — it eats more time than the measurements do.

### The venv must see system site-packages

TensorRT ships as a **system apt package** at `/usr/lib/python3.10/dist-packages/tensorrt`.
It is not on PyPI for aarch64 and cannot be pip-installed. A plain `python3 -m venv`
hides it, and steps 1 and 2 then fail with `ModuleNotFoundError: tensorrt`.

```bash
python3 -m venv --system-site-packages venv     # the flag is not optional
source venv/bin/activate
python -c "import tensorrt; print(tensorrt.__version__)"   # must print 10.3.0
```

### PyTorch must be the Jetson aarch64 CUDA wheel

Install by **direct wheel URL**. Do not use `--index-url`/`--extra-index-url` with a
PyPI fallback, and do not `pip install torch` — see the trap below for why.

```bash
B=https://pypi.jetson-ai-lab.io/jp6/cu126/+f
pip install --no-deps \
  "$B/62a/1beee9f2f1470/torch-2.8.0-cp310-cp310-linux_aarch64.whl" \
  "$B/907/c4c1933789645/torchvision-0.23.0-cp310-cp310-linux_aarch64.whl"
```

Then install torch's own dependencies from PyPI (safe — torch itself is already in):

```bash
pip install filelock fsspec jinja2 networkx sympy typing-extensions
```

Available for JetPack 6 / cu126 / cp310: torch 2.8.0, 2.9.1, 2.10.0, 2.11.0 with
torchvision 0.23.0, 0.24.1, 0.25.0, 0.26.0 respectively. **Pin 2.8.0 + 0.23.0.**
Step 1 calls `torch.onnx.export(..., dynamic_axes=...)`; torch 2.9 deprecated the
TorchScript ONNX exporter and later versions move toward the dynamo exporter, which
emits a different graph for TensorRT to parse. For a published measurement the
exporter should not be a free variable.

To get URLs for a different version, browse `https://pypi.jetson-ai-lab.io/jp6/cu126/torch/`
and resolve the `../../+f/...` hrefs against **`/jp6/cu126/`** — the page 302-redirects
to `/jp6/cu126/+simple/torch/`, so resolving against the pre-redirect path 404s.

> #### ⚠️ The silent CPU-wheel trap — read this
>
> PyPI publishes a *generic* `manylinux_2_28_aarch64` torch wheel that installs
> happily on a Jetson, reports success, and has **no CUDA**. If any PyPI index is
> reachable during resolution, pip prefers it over the Jetson wheel.
>
> Tell them apart by filename and size:
>
> | | Wheel tag | Size | CUDA |
> |---|---|---|---|
> | ❌ PyPI generic | `manylinux_2_28_aarch64` | ~102 MB | none (`2.8.0+cpu`) |
> | ✅ Jetson | `linux_aarch64` | ~226 MB | 12.6 |
>
> This is dangerous here specifically because **step 3 does not fail on CPU-only
> torch.** `train_catA_replay.py` falls back silently:
> `device = torch.device("cuda" if ... and torch.cuda.is_available() else "cpu")`.
> It would report `peak_vram_mb: null` and a CPU wall-clock several times too large —
> and that number is what backs the "fits the device envelope" claim and the Lin et
> al. [30] rebuttal. Steps 1, 2 and 4 do guard on CUDA and fail loudly.

### Ultralytics (step 4 only)

```bash
pip install ultralytics --no-deps
pip install tqdm          # the only runtime dep --no-deps leaves missing here
```

**The `--no-deps` matters** — without it pip pins its own torch requirement and
replaces the Jetson wheel, and CUDA silently stops working. Re-check
`torch.cuda.is_available()` after installing it.

### Already present system-wide

Visible through `--system-site-packages`, none need reinstalling: `numpy` 1.26.4,
`pyyaml` 6.0.3, `pillow` 9.0.1, `scikit-learn` 1.7.2, `matplotlib` 3.5.1,
`psutil` 7.2.2, `onnx` 1.22.0, `opencv` 4.13.0, `pandas` 1.3.5, `scipy` 1.8.0,
`requests` 2.25.1.

(`scipy` 1.8.0 warns that it wants numpy < 1.25 against the installed 1.26.4. It is a
warning only — nothing in this bundle exercises the affected paths.)

### Verify before measuring

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
python -c "import tensorrt; print(tensorrt.__version__)"
python -c "import torchvision; print(torchvision.__version__)"
```

Expected on this unit:

```
2.8.0 True          <- must NOT say '2.8.0+cpu', must NOT say False
10.3.0
0.23.0
```

If `cuda.is_available()` is `False`, the wrong torch wheel is installed — stop and
reinstall. Every number in this bundle is meaningless without it.

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
