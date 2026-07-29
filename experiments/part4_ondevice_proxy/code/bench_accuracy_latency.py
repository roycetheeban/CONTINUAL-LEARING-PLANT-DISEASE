"""Measure the FP16 accuracy-delta gate and classification latency.

Compares the deployed classifier in FP32 against its FP16 "compiled" counterpart
(TensorRT engine if one was built, otherwise PyTorch FP16 -- see
export_compile_metrics.json for which). Reports:
  - accuracy of each on the real held-out tomato test set, and the delta between
    them (the measurement that justifies gating on the compiled artifact rather
    than the training-time candidate)
  - per-image and per-capture-set (8 images) latency for each

All numbers are measured on this laptop's GPU and are a PROXY for the Jetson Orin
Nano deployment target -- see the experiment README."""
import json
import time

import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

from common_ondevice import CONFIG_PATH, ensure_dirs, load_cfg, set_seed, OUTPUT_DIR, REPO_ROOT


def build_model(cfg, device, half=False):
    m = cfg["model"]
    model = models.mobilenet_v3_small(weights=None)
    model.classifier[3] = nn.Linear(1024, int(m["num_classes"]))
    model = model.to(device)
    state = torch.load(REPO_ROOT / m["base_checkpoint"], map_location=device, weights_only=True)
    model.load_state_dict(state, strict=True)
    model.eval()
    if half:
        model = model.half()
    return model


def build_transform(cfg):
    m = cfg["model"]
    size = int(m["image_size"])
    return transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=m["imagenet_mean"], std=m["imagenet_std"]),
    ])


def load_test_tensors(cfg, transform, device):
    """Preload the whole test set once so latency timing measures compute, not disk."""
    root = REPO_ROOT / cfg["data"]["test_dir"]
    class_names = cfg["model"]["class_names"]
    xs, ys = [], []
    for idx, cname in enumerate(class_names):
        for p in sorted((root / cname).glob("*")):
            if p.suffix.lower() in (".jpg", ".jpeg", ".png"):
                xs.append(transform(Image.open(p).convert("RGB")))
                ys.append(idx)
    x = torch.stack(xs).to(device)
    y = torch.tensor(ys, device=device)
    return x, y


@torch.no_grad()
def accuracy_of(model, x, y, half, batch=64):
    correct = 0
    for i in range(0, len(x), batch):
        xb = x[i:i + batch]
        xb = xb.half() if half else xb
        preds = torch.argmax(model(xb), dim=1)
        correct += int((preds == y[i:i + batch]).sum())
    return correct / len(x)


@torch.no_grad()
def time_latency(model, x, half, batch_size, repeats=200, trials=5):
    """Median-of-trials seconds per forward call.

    A laptop GPU's clocks vary with thermal/power state, so a single timing run
    swings by several ms between invocations. Taking the median across repeated
    trials gives a number that is stable enough to report."""
    xb = x[:batch_size]
    xb = xb.half() if half else xb
    for _ in range(20):  # warm up (discard -- first calls include lazy init)
        model(xb)
    if x.is_cuda:
        torch.cuda.synchronize()

    times = []
    for _ in range(trials):
        start = time.perf_counter()
        for _ in range(repeats):
            model(xb)
        if x.is_cuda:
            torch.cuda.synchronize()
        times.append((time.perf_counter() - start) / repeats)
    times.sort()
    return times[len(times) // 2]


def bench_onnxruntime(cfg, x, y, capture_set, repeats=200):
    """Evaluate the exported ONNX graph under ONNX Runtime GPU.

    This is a genuinely *compiled/optimized graph* execution path (graph-level
    fusion + kernel selection), so it is a closer analogue to the paper's
    TensorRT-engine step than PyTorch eager FP16 -- but it is still NOT TensorRT
    and still not Jetson. Reported as its own clearly-named row."""
    try:
        import numpy as np
        import onnxruntime as ort
    except ImportError as e:
        return {"available": False, "reason": str(e)}

    onnx_path = REPO_ROOT / cfg["export"]["onnx_path"]
    providers = ort.get_available_providers()
    provider = "CUDAExecutionProvider" if "CUDAExecutionProvider" in providers else "CPUExecutionProvider"
    try:
        sess = ort.InferenceSession(str(onnx_path), providers=[provider])
    except Exception as e:  # noqa: BLE001 - environment-dependent, must be reported not hidden
        return {"available": False, "reason": f"{type(e).__name__}: {e}", "providers": providers}

    x_np = x.cpu().numpy()
    y_np = y.cpu().numpy()
    name = sess.get_inputs()[0].name

    correct = 0
    for i in range(0, len(x_np), 64):
        logits = sess.run(None, {name: x_np[i:i + 64]})[0]
        correct += int((logits.argmax(axis=1) == y_np[i:i + 64]).sum())
    acc = correct / len(x_np)

    def timed(batch, trials=5):
        xb = x_np[:batch]
        for _ in range(20):
            sess.run(None, {name: xb})
        times = []
        for _ in range(trials):
            start = time.perf_counter()
            for _ in range(repeats):
                sess.run(None, {name: xb})
            times.append((time.perf_counter() - start) / repeats)
        times.sort()
        return times[len(times) // 2]

    return {
        "available": True,
        "provider": provider,
        "accuracy_pct": round(100 * acc, 3),
        "latency_per_image_ms": round(1000 * timed(1), 4),
        "latency_per_capture_set_ms": round(1000 * timed(capture_set), 4),
    }


def main():
    cfg = load_cfg(CONFIG_PATH)
    set_seed(cfg["seed"])
    ensure_dirs(OUTPUT_DIR)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    with open(OUTPUT_DIR / "export_compile_metrics.json", "r", encoding="utf-8") as f:
        export_metrics = json.load(f)
    compiled_path_used = export_metrics["compiled_path_used"]
    print(f"compiled artifact under test: {compiled_path_used}")

    transform = build_transform(cfg)
    x, y = load_test_tensors(cfg, transform, device)
    print(f"test images loaded: {len(x)}")

    capture_set = int(cfg["data"]["capture_set_size"])

    model_fp32 = build_model(cfg, device, half=False)
    acc_fp32 = accuracy_of(model_fp32, x, y, half=False)
    lat_fp32_1 = time_latency(model_fp32, x, half=False, batch_size=1)
    lat_fp32_set = time_latency(model_fp32, x, half=False, batch_size=capture_set)

    model_fp16 = build_model(cfg, device, half=True)
    acc_fp16 = accuracy_of(model_fp16, x, y, half=True)
    lat_fp16_1 = time_latency(model_fp16, x, half=True, batch_size=1)
    lat_fp16_set = time_latency(model_fp16, x, half=True, batch_size=capture_set)

    ort_metrics = bench_onnxruntime(cfg, x, y, capture_set)

    metrics = {
        "compiled_path_used": compiled_path_used,
        "hardware_note": "measured on this laptop's GPU, NOT a Jetson Orin Nano -- proxy only",
        "n_test_images": int(len(x)),
        "capture_set_size": capture_set,
        "onnxruntime_gpu": ort_metrics,
        "accuracy_fp32_pct": round(100 * acc_fp32, 3),
        "accuracy_fp16_pct": round(100 * acc_fp16, 3),
        "accuracy_delta_pct": round(100 * (acc_fp16 - acc_fp32), 4),
        "latency_fp32_per_image_ms": round(1000 * lat_fp32_1, 4),
        "latency_fp16_per_image_ms": round(1000 * lat_fp16_1, 4),
        "latency_fp32_per_capture_set_ms": round(1000 * lat_fp32_set, 4),
        "latency_fp16_per_capture_set_ms": round(1000 * lat_fp16_set, 4),
    }

    with open(OUTPUT_DIR / "accuracy_latency_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
