import copy
import csv
import json
import random
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, models, transforms

try:
    import psutil
except ImportError:
    psutil = None


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dirs(output_root: Path) -> dict:
    dirs = {
        "root": output_root,
        "checkpoints": output_root / "checkpoints",
        "logs": output_root / "logs",
        "metrics": output_root / "metrics",
        "figures": output_root / "figures",
        "reports": output_root / "reports",
        "exports": output_root / "exports",
    }
    for p in dirs.values():
        p.mkdir(parents=True, exist_ok=True)
    return dirs


def get_process_ram_mb() -> float | None:
    if psutil is None:
        return None
    return round(psutil.Process().memory_info().rss / (1024 ** 2), 2)


def get_model_size_mb(model: nn.Module) -> float:
    param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    return round(param_bytes / (1024 ** 2), 4)


def build_transforms(cfg: dict):
    image_size = int(cfg["data"]["image_size"])
    resize_size = int(cfg["data"]["resize_size"])
    train_tfms = transforms.Compose(
        [
            transforms.Resize((resize_size, resize_size)),
            transforms.RandomCrop(image_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=12),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.03),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    eval_tfms = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return train_tfms, eval_tfms


class RemapImageFolder(Dataset):
    def __init__(self, root: str, transform, class_to_global_idx: dict[str, int]):
        ds = datasets.ImageFolder(root=root)
        self.transform = transform
        self.samples = []
        for path, local_idx in ds.samples:
            class_name = ds.classes[local_idx]
            if class_name not in class_to_global_idx:
                continue
            self.samples.append((path, class_to_global_idx[class_name]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, label


def build_global_classes(cfg: dict) -> tuple[list[str], list[str], list[str]]:
    base_classes = list(cfg["classes"]["old_classes"])
    new_ds = datasets.ImageFolder(cfg["data"]["new_train_dir"])
    new_classes = [c for c in new_ds.classes if c not in base_classes]
    global_classes = base_classes + new_classes
    return global_classes, base_classes, new_classes


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device):
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            logits = model(x)
            pred = torch.argmax(logits, dim=1).cpu().numpy()
            y_pred.extend(pred.tolist())
            y_true.extend(y.numpy().tolist())
    return y_true, y_pred


def score(y_true, y_pred):
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "macro_precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
    }


def save_confusion(y_true, y_pred, labels, out_png: Path, title: str):
    # Get unique labels that actually appear in the data
    unique_labels = sorted(set(y_true + y_pred))
    
    # Create confusion matrix only for labels that exist
    cm = confusion_matrix(y_true, y_pred, labels=unique_labels)
    
    # Map back to original label names for display
    display_labels = [labels[i] if (0 <= i and i < len(labels)) else f"Class_{i}" for i in unique_labels]
    
    fig = plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation="nearest")
    plt.title(title)
    plt.colorbar()
    ticks = np.arange(len(unique_labels))
    plt.xticks(ticks, display_labels, rotation=45, ha="right")
    plt.yticks(ticks, display_labels)
    plt.ylabel("True")
    plt.xlabel("Pred")
    plt.tight_layout()
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def plot_training(log_rows: list[dict], out_png: Path, title: str):
    if not log_rows:
        return
    epochs = [int(r["epoch"]) for r in log_rows]
    train_loss = [float(r["train_loss"]) for r in log_rows]
    old_f1 = [float(r.get("val_old_macro_f1", 0.0)) for r in log_rows]
    new_f1 = [float(r.get("val_new_macro_f1", 0.0)) for r in log_rows]

    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax1.plot(epochs, train_loss, color="tab:blue", linewidth=2)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Train Loss", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")

    ax2 = ax1.twinx()
    ax2.plot(epochs, old_f1, color="tab:green", linewidth=2, label="old_f1")
    ax2.plot(epochs, new_f1, color="tab:orange", linewidth=2, label="new_f1")
    ax2.set_ylabel("Val Macro-F1", color="tab:green")
    ax2.tick_params(axis="y", labelcolor="tab:green")

    plt.title(title)
    fig.tight_layout()
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def load_base_model(base_ckpt: str, old_num_classes: int, device: torch.device):
    model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
    state = torch.load(base_ckpt, map_location=device)
    ckpt_out = old_num_classes
    if isinstance(state, dict) and "classifier.3.weight" in state:
        ckpt_out = int(state["classifier.3.weight"].shape[0])
    model.classifier[3] = nn.Linear(1024, ckpt_out)
    model.load_state_dict(state, strict=True)
    return model.to(device)


def expand_head_to_7(model: nn.Module, total_classes: int):
    old_layer = model.classifier[3]
    old_out = old_layer.out_features
    # Keep new head on same device/dtype as existing classifier to avoid cpu/cuda mismatch.
    device = old_layer.weight.device
    dtype = old_layer.weight.dtype
    new_layer = nn.Linear(old_layer.in_features, total_classes).to(device=device, dtype=dtype)
    with torch.no_grad():
        new_layer.weight[:old_out] = old_layer.weight
        new_layer.bias[:old_out] = old_layer.bias
    model.classifier[3] = new_layer


def freeze_for_catb(model: nn.Module, unfreeze_g2: bool = True):
    for p in model.features[:4].parameters():
        p.requires_grad = False
    for p in model.features[4:9].parameters():
        p.requires_grad = unfreeze_g2
    for p in model.features[9:].parameters():
        p.requires_grad = True
    for p in model.classifier.parameters():
        p.requires_grad = True


def train_epoch_mixed(
    model,
    old_loader,
    new_loader,
    device,
    criterion,
    optimizer,
    lambda_ewc=0.0,
    fisher=None,
    theta_star=None,
    old_class_count: int | None = None,
    old_head_grad_scale: float | None = None,
):
    model.train()
    old_it = iter(old_loader)
    new_it = iter(new_loader)
    steps = max(len(old_loader), len(new_loader))
    total_loss = 0.0

    for _ in range(steps):
        try:
            x_old, y_old = next(old_it)
        except StopIteration:
            old_it = iter(old_loader)
            x_old, y_old = next(old_it)
        try:
            x_new, y_new = next(new_it)
        except StopIteration:
            new_it = iter(new_loader)
            x_new, y_new = next(new_it)

        x = torch.cat([x_old, x_new], dim=0).to(device, non_blocking=True)
        y = torch.cat([y_old, y_new], dim=0).to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y)
        if lambda_ewc > 0.0 and fisher is not None and theta_star is not None:
            pen = torch.tensor(0.0, device=device)
            for n, p in model.named_parameters():
                if p.requires_grad and n in fisher:
                    if n == "classifier.3.weight":
                        old_rows = min(5, p.shape[0])
                        pen = pen + (fisher[n][:old_rows].to(device) * (p[:old_rows] - theta_star[n][:old_rows].to(device)).pow(2)).sum()
                    elif n == "classifier.3.bias":
                        old_rows = min(5, p.shape[0])
                        pen = pen + (fisher[n][:old_rows].to(device) * (p[:old_rows] - theta_star[n][:old_rows].to(device)).pow(2)).sum()
                    else:
                        pen = pen + (fisher[n].to(device) * (p - theta_star[n].to(device)).pow(2)).sum()
            loss = loss + lambda_ewc * pen
        loss.backward()
        if (
            old_class_count is not None
            and old_head_grad_scale is not None
            and 0.0 < old_head_grad_scale < 1.0
            and hasattr(model, "classifier")
            and len(model.classifier) > 3
        ):
            head = model.classifier[3]
            if getattr(head, "weight", None) is not None and head.weight.grad is not None:
                head.weight.grad[:old_class_count] *= old_head_grad_scale
            if getattr(head, "bias", None) is not None and head.bias.grad is not None:
                head.bias.grad[:old_class_count] *= old_head_grad_scale
        optimizer.step()
        total_loss += loss.item()

    return total_loss / max(steps, 1)


def compute_fisher(model, loader, device, max_batches=100):
    model.eval()
    ce = nn.CrossEntropyLoss()
    fisher = {n: torch.zeros_like(p, device=device) for n, p in model.named_parameters() if p.requires_grad}
    used = 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        model.zero_grad(set_to_none=True)
        logits = model(x)
        loss = ce(logits, y)
        loss.backward()
        for n, p in model.named_parameters():
            if p.requires_grad and p.grad is not None:
                fisher[n] += p.grad.detach().pow(2)
        used += 1
        if used >= max_batches:
            break
    for n in fisher:
        fisher[n] = (fisher[n] / max(used, 1)).detach().cpu()
    return fisher, used


def build_theta_star(model):
    return {n: p.detach().cpu().clone() for n, p in model.named_parameters() if p.requires_grad}


def save_logs(log_rows, csv_path: Path):
    if not log_rows:
        return
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(log_rows[0].keys()))
        writer.writeheader()
        writer.writerows(log_rows)


def build_eval_loaders(cfg: dict, global_classes: list[str], eval_tfms):
    cls_to_idx = {c: i for i, c in enumerate(global_classes)}
    old_test = RemapImageFolder(cfg["data"]["old_test_dir"], eval_tfms, cls_to_idx)
    new_test = RemapImageFolder(cfg["data"]["new_test_dir"], eval_tfms, cls_to_idx)
    old_val = RemapImageFolder(cfg["data"]["old_val_dir"], eval_tfms, cls_to_idx)
    new_val = RemapImageFolder(cfg["data"]["new_val_dir"], eval_tfms, cls_to_idx)
    bs = int(cfg["train"]["batch_size"])
    nw = int(cfg["num_workers"])
    return (
        DataLoader(old_test, batch_size=bs, shuffle=False, num_workers=nw, pin_memory=True),
        DataLoader(new_test, batch_size=bs, shuffle=False, num_workers=nw, pin_memory=True),
        DataLoader(old_val, batch_size=bs, shuffle=False, num_workers=nw, pin_memory=True),
        DataLoader(new_val, batch_size=bs, shuffle=False, num_workers=nw, pin_memory=True),
    )


def dump_metrics(dirs, cfg, method, cycle, global_classes, old_classes, new_classes, old_stats, new_stats, runtime, model, extra=None):
    metrics_old = {
        "cycle": cycle,
        "method": method,
        "classes": old_classes,
        **old_stats,
    }
    metrics_new = {
        "cycle": cycle,
        "method": method,
        "classes": new_classes,
        **new_stats,
    }
    with (dirs["metrics"] / "metrics_old.json").open("w", encoding="utf-8") as f:
        json.dump(metrics_old, f, indent=2)
    with (dirs["metrics"] / "metrics_new.json").open("w", encoding="utf-8") as f:
        json.dump(metrics_new, f, indent=2)

    combo = {
        "cycle": cycle,
        "method": method,
        "old": metrics_old,
        "new": metrics_new,
        "runtime": runtime,
        "model_footprint": {
            "num_parameters": int(sum(p.numel() for p in model.parameters())),
            "model_size_mb": get_model_size_mb(model),
        },
        "classes": {
            "global": global_classes,
            "old": old_classes,
            "new": new_classes,
        },
        "config": cfg,
    }
    if extra is not None:
        combo["extra"] = extra
    with (dirs["metrics"] / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(combo, f, indent=2)
