"""Shared helpers for the E1 environmental-risk experiment.
Mirrors the Part-1 convention (set_seed / ensure_dirs / RAM logging)."""
import random
from pathlib import Path

import numpy as np
import yaml

try:
    import psutil
except ImportError:
    psutil = None


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def load_cfg(path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_dirs(output_root: Path) -> Path:
    output_root.mkdir(parents=True, exist_ok=True)
    return output_root


def get_process_ram_mb():
    if psutil is None:
        return None
    return round(psutil.Process().memory_info().rss / (1024 ** 2), 2)


# repo paths (this file lives in experiments/part2_env_risk/code/)
CODE_DIR = Path(__file__).resolve().parent
EXP_DIR = CODE_DIR.parent                       # experiments/part2_env_risk/
REPO_ROOT = EXP_DIR.parent.parent               # repo root
CONFIG_PATH = REPO_ROOT / "configs" / "part2_env_risk.yaml"
OUTPUT_DIR = EXP_DIR / "outputs"
