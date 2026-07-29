"""Shared helpers for the E3/E5 on-device-proxy experiment.
Mirrors the Part-2/Part-3 convention (set_seed / load_cfg / ensure_dirs)."""
import random
from pathlib import Path

import numpy as np
import yaml

try:
    import torch
except ImportError:
    torch = None


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


def load_cfg(path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_dirs(output_root: Path) -> Path:
    output_root.mkdir(parents=True, exist_ok=True)
    return output_root


# repo paths (this file lives in experiments/part4_ondevice_proxy/code/)
CODE_DIR = Path(__file__).resolve().parent
EXP_DIR = CODE_DIR.parent                       # experiments/part4_ondevice_proxy/
REPO_ROOT = EXP_DIR.parent.parent               # repo root
CONFIG_PATH = REPO_ROOT / "configs" / "part4_ondevice_proxy.yaml"
OUTPUT_DIR = EXP_DIR / "outputs"
