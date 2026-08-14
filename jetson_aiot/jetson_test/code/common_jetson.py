"""Shared helpers for the Jetson on-device measurement bundle.

Self-contained: imports nothing from the main repo, so this folder can be copied
to the Jetson on its own. Mirrors the model construction and evaluation transform
of the laptop-proxy scripts exactly, so the two sets of numbers are comparable.
"""
import json
import platform
import random
from pathlib import Path

import numpy as np
import yaml

BUNDLE_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = BUNDLE_ROOT / "config.yaml"
OUTPUT_DIR = BUNDLE_ROOT / "outputs"


def load_cfg(path=CONFIG_PATH) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve(rel) -> Path:
    """Bundle-relative path -> absolute. Absolute paths pass through."""
    p = Path(rel)
    return p if p.is_absolute() else (BUNDLE_ROOT / p)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


# ------------------------------------------------------------------ model
def build_model(cfg, device, half=False):
    """MobileNetV3-Small, 5-class head -- identical to the laptop-proxy build."""
    import torch
    import torch.nn as nn
    from torchvision import models

    m = cfg["model"]
    model = models.mobilenet_v3_small(weights=None)
    model.classifier[3] = nn.Linear(1024, int(m["num_classes"]))
    model = model.to(device)
    state = torch.load(resolve(m["checkpoint"]), map_location=device, weights_only=True)
    model.load_state_dict(state, strict=True)
    model.eval()
    if half:
        model = model.half()
    return model


def build_eval_transform(cfg):
    """Evaluation transform: direct resize to 224x224, NO crop.

    This matches what the training scripts actually do at eval time
    (transforms.Resize((image_size, image_size))). Do not "fix" this to a
    resize-256-then-centre-crop -- that would not be the transform the reported
    accuracies were measured under.
    """
    from torchvision import transforms

    m = cfg["model"]
    size = int(m["image_size"])
    return transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=m["imagenet_mean"], std=m["imagenet_std"]),
    ])


def load_test_tensors(cfg, transform, device):
    """Preload the whole test set once, so latency timing measures compute, not disk."""
    import torch
    from PIL import Image

    root = resolve(cfg["data"]["test_dir"])
    xs, ys = [], []
    for idx, cname in enumerate(cfg["model"]["class_names"]):
        for p in sorted((root / cname).glob("*")):
            if p.suffix.lower() in (".jpg", ".jpeg", ".png"):
                xs.append(transform(Image.open(p).convert("RGB")))
                ys.append(idx)
    if not xs:
        raise RuntimeError(f"No test images found under {root}")
    return torch.stack(xs).to(device), torch.tensor(ys, device=device)


# ------------------------------------------------------------- provenance
def env_block(cfg) -> dict:
    """Hardware + software provenance, stamped into every metrics file."""
    info = {"hardware": cfg.get("hardware", {}), "platform": platform.platform(),
            "machine": platform.machine(), "python": platform.python_version()}
    try:
        import torch
        info["torch"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["cuda"] = torch.version.cuda
            info["gpu_name"] = torch.cuda.get_device_name(0)
    except ImportError:
        info["torch"] = None
    try:
        import tensorrt
        info["tensorrt"] = tensorrt.__version__
    except ImportError:
        info["tensorrt"] = None
    return info


def write_metrics(name: str, payload: dict, cfg) -> Path:
    ensure_dir(OUTPUT_DIR)
    payload = {"env": env_block(cfg), **payload}
    out = OUTPUT_DIR / name
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"  wrote {out.relative_to(BUNDLE_ROOT)}")
    return out


def check_hardware_filled(cfg) -> None:
    """Refuse to record numbers against placeholder hardware metadata."""
    hw = cfg.get("hardware", {})
    missing = [k for k, v in hw.items() if isinstance(v, str) and "FILL ME" in v]
    if missing:
        raise SystemExit(
            f"config.yaml hardware fields still say 'FILL ME': {missing}\n"
            "Fill them in first -- these become the hardware provenance in the paper."
        )
