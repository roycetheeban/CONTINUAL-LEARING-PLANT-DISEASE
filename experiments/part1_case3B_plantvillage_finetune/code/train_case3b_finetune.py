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
        "models": output_root / "models",
        "logs": output_root / "logs",
        "results": output_root / "results",
        "plots": output_root / "plots",
        "fim": output_root / "fim",
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


def compute_capped_class_weights(targets: list[int], n_classes: int, cap: float) -> np.ndarray:
    counts = np.bincount(targets, minlength=n_classes).astype(np.float64)
    inv = 1.0 / np.maximum(counts, 1.0)
    inv = inv / inv.min()
    inv = np.minimum(inv, cap)
    return inv


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
            transforms.RandomVerticalFlip(p=0.3),
            transforms.RandomRotation(degrees=20),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.04),
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

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    fisher_loader = DataLoader(fisher_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    return train_ds, val_ds, test_ds, fisher_ds, train_loader, val_loader, test_loader, fisher_loader


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


def train_phase(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer,
    scheduler: ReduceLROnPlateau,
    criterion: nn.Module,
    phase_name: str,
    max_epochs: int,
    min_epochs: int,
    patience: int,
):
    best_f1 = -1.0
    best_state = copy.deepcopy(model.state_dict())
    best_epoch = -1
    bad_epochs = 0
    rows = []

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
        cur_lr = optimizer.param_groups[-1]["lr"]
        rows.append(
            {
                "phase": phase_name,
                "epoch": epoch,
                "epoch_time_sec": round(time.perf_counter() - epoch_start, 4),
                "train_loss": train_loss,
                "val_accuracy": val_stats["accuracy"],
                "val_macro_f1": val_stats["macro_f1"],
                "val_macro_precision": val_stats["macro_precision"],
                "val_macro_recall": val_stats["macro_recall"],
                "lr_last_group": cur_lr,
            }
        )
        print(f"[{phase_name}] epoch={epoch:02d} loss={train_loss:.4f} val_f1={val_stats['macro_f1']:.4f} lr={cur_lr:.6f}")

        if val_stats["macro_f1"] > best_f1:
            best_f1 = val_stats["macro_f1"]
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            bad_epochs = 0
        else:
            bad_epochs += 1

        if epoch >= min_epochs and bad_epochs >= patience:
            print(f"[{phase_name}] early stop: patience reached")
            break

    model.load_state_dict(best_state)
    return model, best_f1, best_epoch, rows


def compute_diagonal_fisher(model: nn.Module, loader: DataLoader, device: torch.device, max_batches: int):
    model.eval()
    fisher = {n: torch.zeros_like(p, device=device) for n, p in model.named_parameters() if p.requires_grad}
    criterion = nn.CrossEntropyLoss()
    used = 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        model.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        for n, p in model.named_parameters():
            if p.requires_grad and p.grad is not None:
                fisher[n] += p.grad.detach().pow(2)
        used += 1
        if used >= max_batches:
            break
    denom = max(used, 1)
    for n in fisher:
        fisher[n] = (fisher[n] / denom).detach().cpu()
    return fisher, used


def build_theta_star(model: nn.Module):
    return {n: p.detach().cpu().clone() for n, p in model.named_parameters() if p.requires_grad}


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
    plt.title("Case3B Test Confusion Matrix")
    plt.colorbar()
    ticks = np.arange(len(class_names))
    plt.xticks(ticks, class_names, rotation=45, ha="right")
    plt.yticks(ticks, class_names)
    plt.ylabel("True")
    plt.xlabel("Pred")
    plt.tight_layout()
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def plot_val_curve(log_csv: Path, out_png: Path):
    xs, vals, phases = [], [], []
    with log_csv.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        i = 0
        for row in reader:
            i += 1
            xs.append(i)
            vals.append(float(row["val_macro_f1"]))
            phases.append(row["phase"])
    fig = plt.figure(figsize=(9, 4.5))
    plt.plot(xs, vals, linewidth=2)
    for i in range(1, len(phases)):
        if phases[i] != phases[i - 1]:
            plt.axvline(i + 1, linestyle="--", linewidth=1)
    plt.title("Case3B Validation Macro-F1")
    plt.xlabel("Epoch (global sequence)")
    plt.ylabel("Val Macro-F1")
    plt.tight_layout()
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def set_phase1_trainable(model: nn.Module):
    for p in model.features[:9].parameters():
        p.requires_grad = False
    for p in model.features[9:].parameters():
        p.requires_grad = True
    for p in model.classifier.parameters():
        p.requires_grad = True


def set_phase2_trainable(model: nn.Module):
    for p in model.features[:4].parameters():
        p.requires_grad = False
    for p in model.features[4:].parameters():
        p.requires_grad = True
    for p in model.classifier.parameters():
        p.requires_grad = True


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
    train_ds, val_ds, test_ds, fisher_ds, train_loader, val_loader, test_loader, fisher_loader = build_dataloaders(cfg)
    class_names = train_ds.classes

    model = models.mobilenet_v3_small(weights=None)
    backbone_state = torch.load(cfg["model"]["pretrained_backbone"], map_location="cpu")
    model.features.load_state_dict(backbone_state)
    model.classifier[3] = nn.Linear(1024, int(cfg["model"]["num_classes"]))
    model = model.to(device)

    class_w = compute_capped_class_weights(train_ds.targets, len(class_names), float(cfg["train"]["class_weight_cap"]))
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_w, dtype=torch.float32, device=device))

    all_rows = []

    set_phase1_trainable(model)
    opt1 = Adam(
        [
            {"params": model.features[9:].parameters(), "lr": float(cfg["train"]["phase1"]["lr_g3"])},
            {"params": model.classifier.parameters(), "lr": float(cfg["train"]["phase1"]["lr_head"])},
        ],
        weight_decay=float(cfg["train"]["weight_decay"]),
    )
    sch1 = ReduceLROnPlateau(opt1, mode="max", patience=3, factor=0.5, min_lr=1e-7)
    model, best1, best1_epoch, rows1 = train_phase(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        optimizer=opt1,
        scheduler=sch1,
        criterion=criterion,
        phase_name="phase1",
        max_epochs=int(cfg["train"]["phase1"]["max_epochs"]),
        min_epochs=int(cfg["train"]["phase1"]["min_epochs"]),
        patience=int(cfg["train"]["phase1"]["patience"]),
    )
    torch.save(model.state_dict(), dirs["models"] / "M3_phase1.pth")
    all_rows.extend(rows1)

    ran_phase2 = False
    if bool(cfg["train"]["phase2"]["enabled"]):
        ran_phase2 = True
        set_phase2_trainable(model)
        opt2 = Adam(
            [
                {"params": model.features[4:9].parameters(), "lr": float(cfg["train"]["phase2"]["lr_g2"])},
                {"params": model.features[9:].parameters(), "lr": float(cfg["train"]["phase2"]["lr_g3"])},
                {"params": model.classifier.parameters(), "lr": float(cfg["train"]["phase2"]["lr_head"])},
            ],
            weight_decay=float(cfg["train"]["weight_decay"]),
        )
        sch2 = ReduceLROnPlateau(opt2, mode="max", patience=4, factor=0.5, min_lr=1e-7)
        model, best2, best2_epoch, rows2 = train_phase(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            optimizer=opt2,
            scheduler=sch2,
            criterion=criterion,
            phase_name="phase2",
            max_epochs=int(cfg["train"]["phase2"]["max_epochs"]),
            min_epochs=int(cfg["train"]["phase2"]["min_epochs"]),
            patience=int(cfg["train"]["phase2"]["patience"]),
        )
        all_rows.extend(rows2)
    else:
        best2, best2_epoch = None, None

    final_model_path = dirs["models"] / "M3.pth"
    torch.save(model.state_dict(), final_model_path)
    torch.save(cfg, dirs["models"] / "config_phaseB.json")
    with (dirs["models"] / "class_weights_5cls.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "class_names": class_names,
                "class_counts": np.bincount(train_ds.targets, minlength=len(class_names)).tolist(),
                "class_weights_capped_invfreq": class_w.tolist(),
                "cap": float(cfg["train"]["class_weight_cap"]),
            },
            f,
            indent=2,
        )

    log_csv = dirs["logs"] / "case3b_train_log.csv"
    with log_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()) if all_rows else ["phase", "epoch"])
        writer.writeheader()
        if all_rows:
            writer.writerows(all_rows)

    val_metrics = evaluate(model, val_loader, device)
    test_metrics = evaluate(model, test_loader, device, with_preds=True)
    save_classification_report(test_metrics["y_true"], test_metrics["y_pred"], class_names, dirs["results"] / "classification_report.csv")
    save_confusion(test_metrics["y_true"], test_metrics["y_pred"], class_names, dirs["plots"] / "case3b_confusion_matrix.png")
    plot_val_curve(log_csv, dirs["plots"] / "case3b_val_f1.png")

    fisher, used_batches = compute_diagonal_fisher(
        model=model,
        loader=fisher_loader,
        device=device,
        max_batches=int(cfg["fim"]["max_batches"]),
    )
    theta_star = build_theta_star(model)
    fisher_path = dirs["fim"] / "fisher_matrix.pt"
    theta_path = dirs["fim"] / "theta_star.pt"
    torch.save(fisher, fisher_path)
    torch.save(theta_star, theta_path)

    metrics = {
        "method": "part1_case3B_plantvillage_finetune",
        "class_names": class_names,
        "phase_summary": {
            "phase1_best_val_f1": best1,
            "phase1_best_epoch": best1_epoch,
            "phase2_enabled": bool(cfg["train"]["phase2"]["enabled"]),
            "phase2_ran": ran_phase2,
            "phase2_best_val_f1": best2,
            "phase2_best_epoch": best2_epoch,
        },
        "val": val_metrics,
        "test": {k: test_metrics[k] for k in ["accuracy", "macro_f1", "macro_precision", "macro_recall"]},
        "test_per_class_f1": dict(zip(class_names, f1_score(test_metrics["y_true"], test_metrics["y_pred"], average=None).tolist())),
        "runtime": {
            "total_wall_time_sec": round(time.perf_counter() - total_start, 4),
            "peak_vram_mb": round(torch.cuda.max_memory_allocated() / (1024 ** 2), 2) if device.type == "cuda" else None,
            "process_ram_mb_end": get_process_ram_mb(),
        },
        "model_footprint": {
            "num_parameters": int(sum(p.numel() for p in model.parameters())),
            "model_size_mb": get_model_size_mb(model),
        },
        "fim": {
            "fisher_dir": cfg["data"]["fisher_dir"],
            "fisher_max_batches": int(cfg["fim"]["max_batches"]),
            "fisher_used_batches": used_batches,
            "fisher_matrix_path": str(fisher_path),
            "theta_star_path": str(theta_path),
            "num_fisher_samples": len(fisher_ds),
        },
        "artifacts": {
            "phase1_model": str(dirs["models"] / "M3_phase1.pth"),
            "final_model": str(final_model_path),
            "train_log": str(log_csv),
        },
        "config": cfg,
    }
    with (dirs["results"] / "case3b_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    with (dirs["fim"] / "fim_metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metrics["fim"], f, indent=2)

    print(f"Saved final model: {final_model_path}")
    print(f"Saved FIM: {fisher_path}")
    print(f"Saved theta*: {theta_path}")
    print(f"Test macro_f1: {metrics['test']['macro_f1']:.4f}")


if __name__ == "__main__":
    main()
