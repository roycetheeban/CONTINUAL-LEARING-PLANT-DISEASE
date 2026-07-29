"""Shared helpers for E4 -- turmeric continual-learning validation.

Deliberately mirrors the tomato CatA pipeline (torchvision MobileNetV3-Small,
layer groups G1-G4, split learning rates) so that E4 answers the question the
paper actually asks: does the recipe selected on tomato transfer to the target
crop? Using a different backbone or head would answer a different question.

NOT built on other models/model/MobilenetV3_Phase3_EWC_Incremental_Results/ --
that checkpoint is a timm model with a custom 2-layer head (2.19M params) whose
block structure does not map onto the G1-G4 grouping. It remains a separate
reference point, not the basis for this experiment.
"""
from __future__ import annotations

import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, models, transforms

try:
    import psutil
except ImportError:
    psutil = None


# ---------------------------------------------------------------- paths
CODE_DIR = Path(__file__).resolve().parent
EXP_DIR = CODE_DIR.parent                       # experiments/part5_turmeric_cl/
REPO_ROOT = EXP_DIR.parent.parent               # repo root
CONFIG_PATH = REPO_ROOT / "configs" / "part5_turmeric_cl.yaml"
OUTPUT_DIR = EXP_DIR / "outputs"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_cfg(path=CONFIG_PATH) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_dirs(root: Path) -> dict:
    dirs = {
        "root": root,
        "checkpoints": root / "checkpoints",
        "logs": root / "logs",
        "metrics": root / "metrics",
    }
    for p in dirs.values():
        p.mkdir(parents=True, exist_ok=True)
    return dirs


def get_process_ram_mb():
    return None if psutil is None else round(psutil.Process().memory_info().rss / (1024 ** 2), 2)


# ---------------------------------------------------------------- data
def build_transforms(cfg: dict):
    size = int(cfg["data"]["image_size"])
    resize = int(cfg["data"]["resize_size"])
    train_tfms = transforms.Compose([
        transforms.Resize((resize, resize)),
        transforms.RandomCrop(size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),   # leaves have no canonical up/down
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.03),
        transforms.ToTensor(),
        transforms.Normalize(mean=cfg["data"]["mean"], std=cfg["data"]["std"]),
    ])
    eval_tfms = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=cfg["data"]["mean"], std=cfg["data"]["std"]),
    ])
    return train_tfms, eval_tfms


class PathDataset(Dataset):
    """Used for replay, where samples come from two directories with a shared
    label space rather than from one ImageFolder root."""

    def __init__(self, pairs: list[tuple[str, int]], transform):
        self.items = pairs
        self.transform = transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        path, label = self.items[idx]
        img = Image.open(path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, label


def data_root(cfg: dict) -> Path:
    return REPO_ROOT / cfg["data"]["root"]


def make_loader(cfg: dict, split: str, tfms, shuffle: bool, drop_last: bool = False,
                batch_size: int | None = None):
    ds = datasets.ImageFolder(str(data_root(cfg) / split), transform=tfms)
    loader = DataLoader(
        ds,
        batch_size=int(cfg["train"]["batch_size"]) if batch_size is None else int(batch_size),
        shuffle=shuffle,
        num_workers=int(cfg["num_workers"]),
        pin_memory=True,
        drop_last=drop_last,
    )
    return ds, loader


def pairs_from_dir(root: Path, class_to_idx: dict) -> list[tuple[str, int]]:
    out = []
    for cname, idx in class_to_idx.items():
        for p in sorted((root / cname).glob("*")):
            if p.suffix.lower() in (".jpg", ".jpeg", ".png"):
                out.append((str(p), idx))
    return out


def compute_class_weights(dataset, device):
    """Inverse-frequency weights. Turmeric is fairly balanced (1.22:1) so this is
    a mild correction, but it keeps parity with the tomato pipeline."""
    counts = np.zeros(len(dataset.classes), dtype=np.float64)
    for _, label in dataset.samples:
        counts[label] += 1
    weights = counts.sum() / (len(counts) * np.maximum(counts, 1))
    return torch.tensor(weights, dtype=torch.float32, device=device)


# ---------------------------------------------------------------- model
def build_model(cfg: dict, device, pretrained: bool = True) -> nn.Module:
    """Case 2 initialization: MobileNetV3-Small with ImageNet weights."""
    weights = models.MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.mobilenet_v3_small(weights=weights)
    model.classifier[3] = nn.Linear(1024, int(cfg["model"]["num_classes"]))
    return model.to(device)


def freeze_g1_g2(model: nn.Module) -> None:
    """G1 features[0:4] + G2 features[4:9] frozen; G3 features[9:] and G4 head train.
    Matches Table II of the paper."""
    for p in model.features[:9].parameters():
        p.requires_grad = False
    for p in model.features[9:].parameters():
        p.requires_grad = True
    for p in model.classifier.parameters():
        p.requires_grad = True


def freeze_all_features(model: nn.Module) -> None:
    """Phase i of base training: head only."""
    for p in model.features.parameters():
        p.requires_grad = False
    for p in model.classifier.parameters():
        p.requires_grad = True


# ---------------------------------------------------------------- eval
@torch.no_grad()
def evaluate(model, loader, device, with_preds: bool = False) -> dict:
    model.eval()
    y_true, y_pred, confs = [], [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        probs = torch.softmax(model(x), dim=1)
        c, pred = probs.max(dim=1)
        y_pred.extend(pred.cpu().tolist())
        y_true.extend(y.tolist())
        confs.extend(c.cpu().tolist())
    out = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "macro_precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
    }
    if with_preds:
        out.update({"y_true": y_true, "y_pred": y_pred, "confidence": confs})
    return out


def save_predictions(path: Path, loader, result: dict) -> None:
    """Persist per-image predictions so method-vs-method comparison can use a
    PAIRED test (McNemar). With ~107 test images, comparing two independent
    confidence intervals cannot resolve the differences CL methods produce;
    only the images where two models disagree carry information."""
    samples = loader.dataset.samples
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["path", "y_true", "y_pred", "confidence", "correct"])
        for (p, _), yt, yp, c in zip(samples, result["y_true"], result["y_pred"], result["confidence"]):
            w.writerow([p, yt, yp, f"{c:.6f}", int(yt == yp)])


# ---------------------------------------------------------------- EWC
def compute_fisher(model, loader, device, max_batches: int) -> tuple[dict, int]:
    """Diagonal Fisher information over trainable parameters."""
    fisher = {n: torch.zeros_like(p, device=device)
              for n, p in model.named_parameters() if p.requires_grad}
    model.eval()
    ce = nn.CrossEntropyLoss()
    used = 0
    for i, (x, y) in enumerate(loader):
        if i >= max_batches:
            break
        x, y = x.to(device), y.to(device)
        model.zero_grad(set_to_none=True)
        ce(model(x), y).backward()
        for n, p in model.named_parameters():
            if p.requires_grad and p.grad is not None:
                fisher[n] += p.grad.detach().pow(2)
        used += 1
    denom = max(used, 1)
    return {n: (v / denom).detach().cpu() for n, v in fisher.items()}, used


def build_theta_star(model) -> dict:
    return {n: p.detach().clone().cpu()
            for n, p in model.named_parameters() if p.requires_grad}


def ewc_penalty(model, fisher: dict, theta_star: dict, device) -> torch.Tensor:
    penalty = torch.tensor(0.0, device=device)
    for n, p in model.named_parameters():
        if n in fisher:
            penalty = penalty + (fisher[n].to(device) * (p - theta_star[n].to(device)).pow(2)).sum()
    return penalty


# ---------------------------------------------------------------- io
def write_json(path: Path, obj) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def write_log_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
