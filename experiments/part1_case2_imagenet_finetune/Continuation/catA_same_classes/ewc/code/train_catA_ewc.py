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
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
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


def build_dataloaders(cfg: dict):
    image_size = cfg["data"]["image_size"]
    resize_size = cfg["data"]["resize_size"]

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

    train_ds = datasets.ImageFolder(cfg["data"]["train_dir"], transform=train_tfms)
    val_ds = datasets.ImageFolder(cfg["data"]["val_dir"], transform=eval_tfms)
    test_ds = datasets.ImageFolder(cfg["data"]["test_dir"], transform=eval_tfms)
    fisher_ds = datasets.ImageFolder(cfg["data"]["fisher_dir"], transform=eval_tfms)

    batch_size = int(cfg["train"]["batch_size"])
    num_workers = int(cfg["num_workers"])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    fisher_loader = DataLoader(fisher_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)

    return train_ds, train_loader, val_loader, test_loader, fisher_loader


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


def get_model_size_mb(model: nn.Module) -> float:
    param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    return round(param_bytes / (1024 ** 2), 4)


def get_process_ram_mb() -> float | None:
    if psutil is None:
        return None
    return round(psutil.Process().memory_info().rss / (1024 ** 2), 2)


def get_trainable_named_params(model: nn.Module):
    for name, p in model.named_parameters():
        if p.requires_grad:
            yield name, p


def freeze_for_catA(model: nn.Module) -> None:
    for p in model.features[:9].parameters():
        p.requires_grad = False
    for p in model.features[9:].parameters():
        p.requires_grad = True
    for p in model.classifier.parameters():
        p.requires_grad = True


def compute_fisher(model: nn.Module, loader: DataLoader, device: torch.device, max_batches: int):
    fisher = {
        name: torch.zeros_like(param, device=device)
        for name, param in get_trainable_named_params(model)
    }

    model.eval()
    ce = nn.CrossEntropyLoss()
    used = 0
    for x, y in loader:
        if used >= max_batches:
            break
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        model.zero_grad(set_to_none=True)
        logits = model(x)
        loss = ce(logits, y)
        loss.backward()

        for name, param in get_trainable_named_params(model):
            if param.grad is not None:
                fisher[name] += param.grad.detach().pow(2)
        used += 1

    denom = max(used, 1)
    for name in fisher:
        fisher[name] = (fisher[name] / denom).detach().cpu()
    return fisher, used


def build_theta_star(model: nn.Module):
    return {
        name: param.detach().cpu().clone()
        for name, param in get_trainable_named_params(model)
    }


def ewc_penalty(model: nn.Module, fisher: dict, theta_star: dict, device: torch.device):
    penalty = torch.tensor(0.0, device=device)
    for name, param in get_trainable_named_params(model):
        f = fisher[name].to(device)
        t = theta_star[name].to(device)
        penalty = penalty + (f * (param - t).pow(2)).sum()
    return penalty


def train_with_ewc(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    fisher: dict,
    theta_star: dict,
    cfg: dict,
    device: torch.device,
    log_csv: Path,
):
    lr_g3 = float(cfg["train"]["lr_g3"])
    lr_head = float(cfg["train"]["lr_head"])
    weight_decay = float(cfg["train"]["weight_decay"])
    max_epochs = int(cfg["train"]["max_epochs"])
    min_epochs = int(cfg["train"]["min_epochs"])
    patience = int(cfg["train"]["patience"])
    lambda_ewc = float(cfg["ewc"]["lambda"])

    optimizer = Adam(
        [
            {"params": model.features[9:].parameters(), "lr": lr_g3},
            {"params": model.classifier.parameters(), "lr": lr_head},
        ],
        weight_decay=weight_decay,
    )
    scheduler = ReduceLROnPlateau(optimizer, mode="max", patience=3, factor=0.5, min_lr=1e-6)
    ce = nn.CrossEntropyLoss()

    cycle_start_val = evaluate(model, val_loader, device)
    floor_f1 = cycle_start_val["macro_f1"] * (1.0 - float(cfg["train"]["old_f1_drop_tolerance"]))

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
        running_ce = 0.0
        running_ewc = 0.0

        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss_ce = ce(logits, y)
            loss_ewc = ewc_penalty(model, fisher, theta_star, device)
            loss = loss_ce + lambda_ewc * loss_ewc
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            running_ce += loss_ce.item()
            running_ewc += loss_ewc.item()

        val_stats = evaluate(model, val_loader, device)
        scheduler.step(val_stats["macro_f1"])

        avg_loss = running_loss / max(len(train_loader), 1)
        avg_ce = running_ce / max(len(train_loader), 1)
        avg_ewc = running_ewc / max(len(train_loader), 1)
        lrs = [pg["lr"] for pg in optimizer.param_groups]

        row = {
            "epoch": epoch,
            "epoch_time_sec": round(time.perf_counter() - epoch_start, 4),
            "train_total_loss": avg_loss,
            "train_ce_loss": avg_ce,
            "train_ewc_penalty": avg_ewc,
            "val_accuracy": val_stats["accuracy"],
            "val_macro_f1": val_stats["macro_f1"],
            "val_macro_precision": val_stats["macro_precision"],
            "val_macro_recall": val_stats["macro_recall"],
            "lr_g3": lrs[0],
            "lr_head": lrs[1],
        }
        rows.append(row)

        print(
            f"epoch={epoch:02d} loss={avg_loss:.4f} ce={avg_ce:.4f} ewc={avg_ewc:.4f} "
            f"val_f1={val_stats['macro_f1']:.4f} lrs={lrs}"
        )

        if val_stats["macro_f1"] > best_f1:
            best_f1 = val_stats["macro_f1"]
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            bad_epochs = 0
        else:
            bad_epochs += 1

        if epoch >= min_epochs and val_stats["macro_f1"] < floor_f1:
            print("early stop: old-class val F1 dropped beyond tolerance")
            break

        if epoch >= min_epochs and bad_epochs >= patience:
            print("early stop: patience reached")
            break

    model.load_state_dict(best_state)

    with log_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["epoch"])
        writer.writeheader()
        if rows:
            writer.writerows(rows)

    return model, {
        "cycle_start_val": cycle_start_val,
        "best_epoch": best_epoch,
        "best_val_macro_f1": best_f1,
        "old_f1_floor": floor_f1,
        "train_wall_time_sec": round(time.perf_counter() - phase_start, 4),
    }


def plot_training_curves(log_csv: Path, out_png: Path):
    epochs, train_losses, val_f1s = [], [], []
    with log_csv.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            epochs.append(int(row["epoch"]))
            train_losses.append(float(row["train_total_loss"]))
            val_f1s.append(float(row["val_macro_f1"]))

    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax1.plot(epochs, train_losses, color="tab:blue", linewidth=2, label="train_total_loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Train Total Loss", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")

    ax2 = ax1.twinx()
    ax2.plot(epochs, val_f1s, color="tab:green", linewidth=2, label="val_macro_f1")
    ax2.set_ylabel("Val Macro-F1", color="tab:green")
    ax2.tick_params(axis="y", labelcolor="tab:green")

    plt.title("Cat A EWC Training Curves")
    fig.tight_layout()
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def save_confusion(y_true, y_pred, class_names, out_png: Path):
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    fig = plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation="nearest")
    plt.title("Cat A EWC Test Confusion Matrix")
    plt.colorbar()
    ticks = np.arange(len(class_names))
    plt.xticks(ticks, class_names, rotation=45, ha="right")
    plt.yticks(ticks, class_names)
    plt.ylabel("True")
    plt.xlabel("Pred")
    plt.tight_layout()
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

    out_root = Path(cfg["output"]["root"])
    dirs = ensure_dirs(out_root)

    train_ds, train_loader, val_loader, test_loader, fisher_loader = build_dataloaders(cfg)
    class_names = train_ds.classes

    model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
    model.classifier[3] = nn.Linear(1024, int(cfg["model"]["num_classes"]))
    model = model.to(device)

    base_ckpt = Path(cfg["model"]["base_checkpoint"])
    state = torch.load(base_ckpt, map_location=device)
    model.load_state_dict(state, strict=True)

    freeze_for_catA(model)

    fisher, used_batches = compute_fisher(
        model=model,
        loader=fisher_loader,
        device=device,
        max_batches=int(cfg["ewc"]["fisher_max_batches"]),
    )
    theta_star = build_theta_star(model)

    torch.save(fisher, dirs["metrics"] / "fisher_matrix.pt")
    torch.save(theta_star, dirs["checkpoints"] / "theta_star.pt")

    model, train_summary = train_with_ewc(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        fisher=fisher,
        theta_star=theta_star,
        cfg=cfg,
        device=device,
        log_csv=dirs["logs"] / "train_log.csv",
    )

    best_model_path = dirs["checkpoints"] / "model.pth"
    torch.save(model.state_dict(), best_model_path)

    val_metrics = evaluate(model, val_loader, device)
    test_metrics = evaluate(model, test_loader, device, with_preds=True)

    metrics = {
        "cycle": cfg["meta"]["cycle_name"],
        "method": "ewc",
        "class_names": class_names,
        "base_checkpoint": str(base_ckpt),
        "fisher_batches_used": used_batches,
        "train_summary": train_summary,
        "runtime": {
            "total_wall_time_sec": round(time.perf_counter() - total_start, 4),
            "train_wall_time_sec": train_summary["train_wall_time_sec"],
            "peak_vram_mb": round(torch.cuda.max_memory_allocated() / (1024 ** 2), 2)
            if device.type == "cuda"
            else None,
            "process_ram_mb_end": get_process_ram_mb(),
        },
        "model_footprint": {
            "num_parameters": int(sum(p.numel() for p in model.parameters())),
            "model_size_mb": get_model_size_mb(model),
        },
        "val": val_metrics,
        "test": {
            "accuracy": test_metrics["accuracy"],
            "macro_f1": test_metrics["macro_f1"],
            "macro_precision": test_metrics["macro_precision"],
            "macro_recall": test_metrics["macro_recall"],
        },
        "config": cfg,
    }
    with (dirs["metrics"] / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    plot_training_curves(dirs["logs"] / "train_log.csv", dirs["figures"] / "train_val_curves.png")
    save_confusion(
        test_metrics["y_true"],
        test_metrics["y_pred"],
        class_names,
        dirs["figures"] / "confusion_matrix.png",
    )

    print(f"Saved: {best_model_path}")
    print(f"Test macro_f1: {test_metrics['macro_f1']:.4f}")


if __name__ == "__main__":
    main()
