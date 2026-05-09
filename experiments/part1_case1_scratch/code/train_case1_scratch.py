import argparse
import copy
import csv
import json
import random
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import yaml
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, WeightedRandomSampler
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


def build_dataloaders(cfg: dict):
    image_size = int(cfg["data"]["image_size"])
    resize_size = int(cfg["data"]["resize_size"])
    batch_size = int(cfg["train"]["batch_size"])
    num_workers = int(cfg["num_workers"])

    train_tfms = transforms.Compose(
        [
            transforms.Resize((resize_size, resize_size)),
            transforms.RandomCrop(image_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
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

    train_ds = datasets.ImageFolder(cfg["data"]["train_dir"], transform=train_tfms)
    val_ds = datasets.ImageFolder(cfg["data"]["val_dir"], transform=eval_tfms)
    test_ds = datasets.ImageFolder(cfg["data"]["test_dir"], transform=eval_tfms)

    sampler_cfg = cfg["train"].get("weighted_sampling", {})
    use_weighted_sampler = bool(sampler_cfg.get("enabled", False))
    class_weights = None
    sample_weights = None

    if use_weighted_sampler:
        targets = train_ds.targets
        counts = np.bincount(targets, minlength=len(train_ds.classes))
        class_weights = 1.0 / np.maximum(counts, 1)
        sample_weights = [float(class_weights[t]) for t in targets]
        sampler = WeightedRandomSampler(
            weights=torch.DoubleTensor(sample_weights),
            num_samples=len(sample_weights),
            replacement=True,
        )
        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=True,
        )
    else:
        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
        )

    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return train_ds, train_loader, val_loader, test_loader, class_weights, sample_weights


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, with_preds: bool = False):
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            logits = model(x)
            pred = torch.argmax(logits, dim=1).cpu().numpy()
            y_pred.extend(pred.tolist())
            y_true.extend(y.numpy().tolist())

    out = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "macro_precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
    }
    if with_preds:
        out["y_true"] = y_true
        out["y_pred"] = y_pred
    return out


def train(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, cfg: dict, device: torch.device, log_csv: Path):
    lr = float(cfg["train"]["lr"])
    weight_decay = float(cfg["train"]["weight_decay"])
    max_epochs = int(cfg["train"]["max_epochs"])
    min_epochs = int(cfg["train"]["min_epochs"])
    patience = int(cfg["train"]["patience"])

    optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode="max", patience=3, factor=0.5, min_lr=1e-6)
    criterion = nn.CrossEntropyLoss()

    best_f1 = -1.0
    best_state = copy.deepcopy(model.state_dict())
    best_epoch = -1
    bad_epochs = 0
    rows = []
    phase_start = time.perf_counter()

    for epoch in range(1, max_epochs + 1):
        epoch_start = time.perf_counter()
        model.train()
        running_loss = 0.0
        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        train_loss = running_loss / max(len(train_loader), 1)
        val_stats = evaluate(model, val_loader, device)
        scheduler.step(val_stats["macro_f1"])

        rows.append(
            {
                "epoch": epoch,
                "epoch_time_sec": round(time.perf_counter() - epoch_start, 4),
                "train_loss": train_loss,
                "val_accuracy": val_stats["accuracy"],
                "val_macro_f1": val_stats["macro_f1"],
                "val_macro_precision": val_stats["macro_precision"],
                "val_macro_recall": val_stats["macro_recall"],
                "lr": optimizer.param_groups[0]["lr"],
            }
        )
        print(f"epoch={epoch:02d} loss={train_loss:.4f} val_f1={val_stats['macro_f1']:.4f} lr={optimizer.param_groups[0]['lr']:.6f}")

        if val_stats["macro_f1"] > best_f1:
            best_f1 = val_stats["macro_f1"]
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            bad_epochs = 0
        else:
            bad_epochs += 1

        if epoch >= min_epochs and bad_epochs >= patience:
            print("early stop: patience reached")
            break

    model.load_state_dict(best_state)

    with log_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["epoch"])
        writer.writeheader()
        if rows:
            writer.writerows(rows)

    return model, {"best_epoch": best_epoch, "best_val_macro_f1": best_f1, "train_wall_time_sec": round(time.perf_counter() - phase_start, 4)}


def save_classification_report(y_true, y_pred, class_names, out_csv: Path):
    report = classification_report(y_true, y_pred, target_names=class_names, digits=4, output_dict=True, zero_division=0)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["label", "precision", "recall", "f1-score", "support"])
        for label in class_names:
            vals = report.get(label, {})
            writer.writerow([label, vals.get("precision", ""), vals.get("recall", ""), vals.get("f1-score", ""), vals.get("support", "")])
        for agg in ["macro avg", "weighted avg"]:
            vals = report.get(agg, {})
            writer.writerow([agg, vals.get("precision", ""), vals.get("recall", ""), vals.get("f1-score", ""), vals.get("support", "")])


def save_confusion(y_true, y_pred, class_names, out_png: Path):
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    fig = plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation="nearest")
    plt.title("Case1 Scratch Test Confusion Matrix")
    plt.colorbar()
    ticks = np.arange(len(class_names))
    plt.xticks(ticks, class_names, rotation=45, ha="right")
    plt.yticks(ticks, class_names)
    plt.ylabel("True")
    plt.xlabel("Pred")
    plt.tight_layout()
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def plot_training_curves(log_csv: Path, out_png: Path):
    epochs, train_losses, val_f1s = [], [], []
    with log_csv.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            epochs.append(int(row["epoch"]))
            train_losses.append(float(row["train_loss"]))
            val_f1s.append(float(row["val_macro_f1"]))
    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax1.plot(epochs, train_losses, color="tab:blue", linewidth=2)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Train Loss", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")
    ax2 = ax1.twinx()
    ax2.plot(epochs, val_f1s, color="tab:green", linewidth=2)
    ax2.set_ylabel("Val Macro-F1", color="tab:green")
    ax2.tick_params(axis="y", labelcolor="tab:green")
    plt.title("Case1 Scratch Training Curves")
    fig.tight_layout()
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=str)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    total_start = time.perf_counter()
    set_seed(int(cfg["seed"]))
    device = torch.device("cuda" if cfg["device"] == "cuda" and torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    dirs = ensure_dirs(Path(cfg["output"]["root"]))
    train_ds, train_loader, val_loader, test_loader, class_weights, sample_weights = build_dataloaders(cfg)
    class_names = train_ds.classes

    model = models.mobilenet_v3_small(weights=None)
    model.classifier[3] = nn.Linear(1024, int(cfg["model"]["num_classes"]))
    model = model.to(device)

    model, train_summary = train(model, train_loader, val_loader, cfg, device, dirs["logs"] / "train_log.csv")
    ckpt_path = dirs["checkpoints"] / "model.pth"
    torch.save(model.state_dict(), ckpt_path)

    val_metrics = evaluate(model, val_loader, device)
    test_metrics = evaluate(model, test_loader, device, with_preds=True)
    per_class_f1 = f1_score(test_metrics["y_true"], test_metrics["y_pred"], average=None).tolist()
    per_class_precision = precision_score(test_metrics["y_true"], test_metrics["y_pred"], average=None, zero_division=0).tolist()
    per_class_recall = recall_score(test_metrics["y_true"], test_metrics["y_pred"], average=None, zero_division=0).tolist()

    metrics = {
        "method": "part1_case1_scratch",
        "class_names": class_names,
        "train_summary": train_summary,
        "sampling": {
            "weighted_sampling_enabled": bool(cfg["train"].get("weighted_sampling", {}).get("enabled", False)),
            "class_weights_inverse_freq": class_weights.tolist() if class_weights is not None else None,
            "num_sample_weights": len(sample_weights) if sample_weights is not None else None,
        },
        "runtime": {
            "total_wall_time_sec": round(time.perf_counter() - total_start, 4),
            "train_wall_time_sec": train_summary["train_wall_time_sec"],
            "peak_vram_mb": round(torch.cuda.max_memory_allocated() / (1024 ** 2), 2) if device.type == "cuda" else None,
            "process_ram_mb_end": get_process_ram_mb(),
        },
        "model_footprint": {
            "num_parameters": int(sum(p.numel() for p in model.parameters())),
            "model_size_mb": get_model_size_mb(model),
        },
        "val": {k: val_metrics[k] for k in ["accuracy", "macro_f1", "macro_precision", "macro_recall"]},
        "test": {k: test_metrics[k] for k in ["accuracy", "macro_f1", "macro_precision", "macro_recall"]},
        "test_per_class_precision": dict(zip(class_names, per_class_precision)),
        "test_per_class_recall": dict(zip(class_names, per_class_recall)),
        "test_per_class_f1": dict(zip(class_names, per_class_f1)),
        "config": cfg,
    }
    with (dirs["metrics"] / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    save_classification_report(test_metrics["y_true"], test_metrics["y_pred"], class_names, dirs["metrics"] / "classification_report.csv")
    plot_training_curves(dirs["logs"] / "train_log.csv", dirs["figures"] / "train_val_curves.png")
    save_confusion(test_metrics["y_true"], test_metrics["y_pred"], class_names, dirs["figures"] / "confusion_matrix.png")
    print(f"Saved: {ckpt_path}")
    print(f"Test macro_f1: {test_metrics['macro_f1']:.4f}")


if __name__ == "__main__":
    main()

