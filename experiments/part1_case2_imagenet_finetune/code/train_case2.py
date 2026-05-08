import argparse
import copy
import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

import matplotlib.pyplot as plt


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dirs(output_root: Path, cfg: dict) -> dict:
    dirs = {
        "root": output_root,
        "checkpoints": output_root / cfg["output"]["checkpoints_dir"],
        "logs": output_root / cfg["output"]["logs_dir"],
        "metrics": output_root / cfg["output"]["metrics_dir"],
        "figures": output_root / cfg["output"]["figures_dir"],
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
            transforms.RandomVerticalFlip(p=0.3),
            transforms.RandomRotation(degrees=20),
            transforms.ColorJitter(
                brightness=0.25, contrast=0.25, saturation=0.25, hue=0.04
            ),
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

    batch_size = cfg["train"]["batch_size"]
    num_workers = cfg["num_workers"]

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
    )
    return train_ds, val_ds, test_ds, train_loader, val_loader, test_loader


def compute_class_weights(train_ds: datasets.ImageFolder, device: torch.device) -> torch.Tensor:
    targets = np.array(train_ds.targets)
    counts = np.bincount(targets)
    inv = 1.0 / np.maximum(counts, 1)
    weights = inv / inv.sum() * len(inv)
    min_w = weights.min()
    weights = np.minimum(weights, min_w * 3.0)
    return torch.tensor(weights, dtype=torch.float32, device=device)


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device):
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            logits = model(x)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            y_pred.extend(preds.tolist())
            y_true.extend(y.numpy().tolist())
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    macro_precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
    macro_recall = recall_score(y_true, y_pred, average="macro", zero_division=0)
    per_class_f1 = f1_score(y_true, y_pred, average=None).tolist()
    per_class_precision = precision_score(y_true, y_pred, average=None, zero_division=0).tolist()
    per_class_recall = recall_score(y_true, y_pred, average=None, zero_division=0).tolist()
    return {
        "accuracy": float(acc),
        "macro_f1": float(macro_f1),
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "per_class_f1": per_class_f1,
        "per_class_precision": per_class_precision,
        "per_class_recall": per_class_recall,
        "y_true": y_true,
        "y_pred": y_pred,
    }


def train_phase(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: ReduceLROnPlateau,
    criterion: nn.Module,
    phase_name: str,
    max_epochs: int,
    min_epochs: int,
    patience: int,
    device: torch.device,
    csv_path: Path,
):
    best_f1 = -1.0
    best_epoch = -1
    best_train_loss = None
    best_val_acc = None
    best_val_precision = None
    best_val_recall = None
    best_state = None
    bad_epochs = 0
    rows = []

    for epoch in range(1, max_epochs + 1):
        model.train()
        running_loss = 0.0

        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        train_loss = running_loss / max(len(train_loader), 1)
        val_result = evaluate(model, val_loader, device)
        val_acc = val_result["accuracy"]
        val_f1 = val_result["macro_f1"]
        val_precision = val_result["macro_precision"]
        val_recall = val_result["macro_recall"]
        scheduler.step(val_f1)

        lrs = [pg["lr"] for pg in optimizer.param_groups]
        row = {
            "phase": phase_name,
            "epoch": epoch,
            "train_loss": train_loss,
            "val_accuracy": val_acc,
            "val_macro_f1": val_f1,
            "val_macro_precision": val_precision,
            "val_macro_recall": val_recall,
            "lr_groups": "|".join([f"{lr:.8f}" for lr in lrs]),
        }
        rows.append(row)
        print(
            f"[{phase_name}] epoch={epoch:02d} loss={train_loss:.4f} val_acc={val_acc:.4f} val_f1={val_f1:.4f} lrs={lrs}"
        )

        if val_f1 > best_f1:
            best_f1 = val_f1
            best_epoch = epoch
            best_train_loss = train_loss
            best_val_acc = val_acc
            best_val_precision = val_precision
            best_val_recall = val_recall
            best_state = copy.deepcopy(model.state_dict())
            bad_epochs = 0
        else:
            bad_epochs += 1
            if epoch >= min_epochs and bad_epochs >= patience:
                print(f"[{phase_name}] early stop at epoch {epoch}")
                break

    model.load_state_dict(best_state)

    file_exists = csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "phase",
                "epoch",
                "train_loss",
                "val_accuracy",
                "val_macro_f1",
                "val_macro_precision",
                "val_macro_recall",
                "lr_groups",
            ],
        )
        if not file_exists:
            writer.writeheader()
        writer.writerows(rows)

    return model, {
        "phase": phase_name,
        "best_epoch": best_epoch,
        "best_train_loss": best_train_loss,
        "best_val_accuracy": best_val_acc,
        "best_val_macro_f1": best_f1,
        "best_val_macro_precision": best_val_precision,
        "best_val_macro_recall": best_val_recall,
    }


def plot_training_curves(csv_path: Path, out_png: Path):
    phases, epochs, train_losses, val_f1s = [], [], [], []
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            phases.append(row["phase"])
            epochs.append(int(row["epoch"]))
            train_losses.append(float(row["train_loss"]))
            val_f1s.append(float(row["val_macro_f1"]))

    fig, ax1 = plt.subplots(figsize=(9, 5))
    x = np.arange(len(epochs))
    labels = [f"{p}-e{e}" for p, e in zip(phases, epochs)]

    ax1.plot(x, train_losses, label="train_loss", color="tab:blue", linewidth=2)
    ax1.set_xlabel("Phase-Epoch")
    ax1.set_ylabel("Train Loss", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")
    ax1.set_xticks(x[:: max(1, len(x)//10)])
    ax1.set_xticklabels(labels[:: max(1, len(labels)//10)], rotation=30, ha="right")

    ax2 = ax1.twinx()
    ax2.plot(x, val_f1s, label="val_macro_f1", color="tab:green", linewidth=2)
    ax2.set_ylabel("Val Macro F1", color="tab:green")
    ax2.tick_params(axis="y", labelcolor="tab:green")

    plt.title("Case2 Training/Validation Curves")
    fig.tight_layout()
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def save_confusion(y_true, y_pred, class_names, out_png: Path):
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    fig = plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation="nearest")
    plt.title("Case2 Test Confusion Matrix")
    plt.colorbar()
    ticks = np.arange(len(class_names))
    plt.xticks(ticks, class_names, rotation=45, ha="right")
    plt.yticks(ticks, class_names)
    plt.ylabel("True")
    plt.xlabel("Pred")
    plt.tight_layout()
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def compute_fisher_matrix(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    max_batches: int = 100,
):
    fisher = {}
    for name, p in model.named_parameters():
        if p.requires_grad:
            fisher[name] = torch.zeros_like(p, device=device)

    model.eval()
    used_batches = 0
    for x, y in loader:
        if used_batches >= max_batches:
            break
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        model.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()

        for name, p in model.named_parameters():
            if p.requires_grad and p.grad is not None:
                fisher[name] += p.grad.detach().pow(2)
        used_batches += 1

    denom = max(used_batches, 1)
    for name in fisher:
        fisher[name] = (fisher[name] / denom).detach().cpu()
    return fisher, used_batches


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="experiments/part1_case2_imagenet_finetune/configs/case2_config.yaml",
    )
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg["seed"])
    device = torch.device(
        "cuda" if cfg["device"] == "cuda" and torch.cuda.is_available() else "cpu"
    )
    print(f"Using device: {device}")

    output_root = Path(cfg["output"]["root"])
    dirs = ensure_dirs(output_root, cfg)

    train_ds, _, _, train_loader, val_loader, test_loader = build_dataloaders(cfg)
    class_names = train_ds.classes
    n_classes = len(class_names)
    class_weights = compute_class_weights(train_ds, device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    model = models.mobilenet_v3_small(
        weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1
    )
    model.classifier[3] = nn.Linear(1024, n_classes)
    model = model.to(device)

    log_csv = dirs["logs"] / "case2_train_log.csv"

    for p in model.features.parameters():
        p.requires_grad = False
    optimizer_p1 = Adam(
        model.classifier.parameters(),
        lr=cfg["train"]["phase1"]["lr_head"],
        weight_decay=cfg["train"]["weight_decay"],
    )
    scheduler_p1 = ReduceLROnPlateau(optimizer_p1, mode="max", patience=3, factor=0.5, min_lr=1e-6)
    model, best_p1 = train_phase(
        model,
        train_loader,
        val_loader,
        optimizer_p1,
        scheduler_p1,
        criterion,
        "phase1",
        cfg["train"]["phase1"]["max_epochs"],
        cfg["train"]["min_epochs"],
        cfg["train"]["patience"],
        device,
        log_csv,
    )
    torch.save(model.state_dict(), dirs["checkpoints"] / "M2_phase1.pth")
    print(f"Saved phase1 best: macro_f1={best_p1['best_val_macro_f1']:.4f}")

    for p in model.features[9:].parameters():
        p.requires_grad = True
    optimizer_p2 = Adam(
        [
            {"params": model.features[9:].parameters(), "lr": cfg["train"]["phase2"]["lr_g3"]},
            {"params": model.classifier.parameters(), "lr": cfg["train"]["phase2"]["lr_head"]},
        ],
        weight_decay=cfg["train"]["weight_decay"],
    )
    scheduler_p2 = ReduceLROnPlateau(optimizer_p2, mode="max", patience=5, factor=0.5, min_lr=1e-6)
    model, best_p2 = train_phase(
        model,
        train_loader,
        val_loader,
        optimizer_p2,
        scheduler_p2,
        criterion,
        "phase2",
        cfg["train"]["phase2"]["max_epochs"],
        cfg["train"]["min_epochs"],
        cfg["train"]["patience"],
        device,
        log_csv,
    )

    final_ckpt = dirs["checkpoints"] / "M2.pth"
    torch.save(model.state_dict(), final_ckpt)
    print(
        f"Saved final M2: {final_ckpt} | best_val_f1_phase2={best_p2['best_val_macro_f1']:.4f}"
    )

    test_result = evaluate(model, test_loader, device)
    metrics = {
        "case": "part1_case2_imagenet_finetune",
        "best_phase1": best_p1,
        "best_phase2": best_p2,
        "test_accuracy": test_result["accuracy"],
        "test_macro_f1": test_result["macro_f1"],
        "test_macro_precision": test_result["macro_precision"],
        "test_macro_recall": test_result["macro_recall"],
        "test_per_class_precision": dict(zip(class_names, test_result["per_class_precision"])),
        "test_per_class_recall": dict(zip(class_names, test_result["per_class_recall"])),
        "test_per_class_f1": dict(zip(class_names, test_result["per_class_f1"])),
        "classes": class_names,
    }
    with (dirs["metrics"] / "case2_test_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    report_dict = classification_report(
        test_result["y_true"],
        test_result["y_pred"],
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    report_csv_path = dirs["metrics"] / "case2_classification_report.csv"
    with report_csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["label", "precision", "recall", "f1-score", "support"])
        for label, vals in report_dict.items():
            if isinstance(vals, dict):
                writer.writerow(
                    [
                        label,
                        vals.get("precision", ""),
                        vals.get("recall", ""),
                        vals.get("f1-score", ""),
                        vals.get("support", ""),
                    ]
                )

    with (dirs["metrics"] / "case2_best_epochs_summary.json").open(
        "w", encoding="utf-8"
    ) as f:
        json.dump({"phase1": best_p1, "phase2": best_p2}, f, indent=2)

    plot_training_curves(
        csv_path=dirs["logs"] / "case2_train_log.csv",
        out_png=dirs["figures"] / "case2_train_val_curves.png",
    )

    save_confusion(
        test_result["y_true"],
        test_result["y_pred"],
        class_names,
        dirs["figures"] / "case2_test_confusion_matrix.png",
    )

    ewc_cfg = cfg.get("ewc_prep", {})
    if ewc_cfg.get("enabled", False):
        fisher, used_batches = compute_fisher_matrix(
            model=model,
            loader=train_loader,
            criterion=criterion,
            device=device,
            max_batches=int(ewc_cfg.get("fisher_max_batches", 100)),
        )
        fisher_path = dirs["metrics"] / "case2_fisher_matrix.pt"
        torch.save(fisher, fisher_path)
        print(f"Saved Fisher matrix: {fisher_path} (batches={used_batches})")

        if ewc_cfg.get("save_theta_star", True):
            theta_star = {
                name: p.detach().cpu().clone()
                for name, p in model.named_parameters()
                if p.requires_grad
            }
            theta_path = dirs["checkpoints"] / "case2_theta_star.pt"
            torch.save(theta_star, theta_path)
            print(f"Saved theta_star: {theta_path}")

    print("Saved logs, training/validation curves, metrics, and confusion matrix.")


if __name__ == "__main__":
    main()
