import argparse
import copy
import csv
import json
import random
import time
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import yaml
from PIL import Image
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score
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
    }
    for p in dirs.values():
        p.mkdir(parents=True, exist_ok=True)
    return dirs


def get_model_size_mb(model: nn.Module) -> float:
    param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    return round(param_bytes / (1024 ** 2), 4)


def get_process_ram_mb() -> float | None:
    if psutil is None:
        return None
    return round(psutil.Process().memory_info().rss / (1024 ** 2), 2)


def freeze_for_catA(model: nn.Module) -> None:
    for p in model.features[:9].parameters():
        p.requires_grad = False
    for p in model.features[9:].parameters():
        p.requires_grad = True
    for p in model.classifier.parameters():
        p.requires_grad = True


class PathDataset(Dataset):
    def __init__(self, path_label_pairs: list[tuple[str, int]], transform):
        self.items = path_label_pairs
        self.transform = transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        path, label = self.items[idx]
        img = Image.open(path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, label


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


def load_manifest(path: Path) -> dict | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def stratified_sample(stream_samples: list[tuple[str, int]], class_names: list[str], n_per_class: int, seed: int):
    rng = random.Random(seed)
    grouped = defaultdict(list)
    for p, y in stream_samples:
        grouped[int(y)].append((p, y))
    picked = []
    for cid in range(len(class_names)):
        arr = grouped[cid]
        if not arr:
            continue
        k = min(n_per_class, len(arr))
        picked.extend(rng.sample(arr, k))
    return picked


def build_replay_sets(cfg: dict, class_names: list[str], cycle_name: str):
    replay_root = Path(cfg["data"]["replay_dir"])
    stream_root = Path(cfg["data"]["train_dir"])
    seed = int(cfg["seed"])
    add_per_class = int(cfg["replay"]["add_per_class_after_cycle"])

    replay_ds = datasets.ImageFolder(str(replay_root))
    stream_ds = datasets.ImageFolder(str(stream_root))
    if replay_ds.classes != class_names or stream_ds.classes != class_names:
        raise RuntimeError("Class ordering mismatch among replay/train datasets.")

    base_replay = [(replay_ds.samples[i][0], replay_ds.samples[i][1]) for i in range(len(replay_ds.samples))]
    stream_samples = [(stream_ds.samples[i][0], stream_ds.samples[i][1]) for i in range(len(stream_ds.samples))]

    prior_manifest = None
    if cfg["replay"].get("input_manifest"):
        prior_manifest = load_manifest(Path(cfg["replay"]["input_manifest"]))

    prior_added = []
    if prior_manifest is not None:
        prior_added = [(item["path"], int(item["label_idx"])) for item in prior_manifest.get("added_cycle_paths", [])]

    # Cycle-specific logical update: add samples from current stream by manifest, no folder mutation.
    newly_added = stratified_sample(stream_samples, class_names, add_per_class, seed + 101)
    effective_old = base_replay + prior_added + newly_added

    manifest = {
        "cycle": cycle_name,
        "replay_folder_immutable": True,
        "replay_root": str(replay_root.as_posix()),
        "base_replay_count": len(base_replay),
        "prior_added_count": len(prior_added),
        "newly_added_count": len(newly_added),
        "effective_old_count": len(effective_old),
        "class_names": class_names,
        "added_cycle_paths": [
            {"path": p, "label_idx": int(y), "label_name": class_names[int(y)]}
            for p, y in newly_added
        ],
    }
    return effective_old, stream_samples, manifest


def build_dataloaders(cfg: dict):
    train_tfms, eval_tfms = build_transforms(cfg)

    val_ds = datasets.ImageFolder(cfg["data"]["val_dir"], transform=eval_tfms)
    test_ds = datasets.ImageFolder(cfg["data"]["test_dir"], transform=eval_tfms)
    class_names = val_ds.classes

    old_pairs, new_pairs, manifest = build_replay_sets(cfg, class_names, cfg["meta"]["cycle_name"])

    old_ds = PathDataset(old_pairs, train_tfms)
    new_ds = PathDataset(new_pairs, train_tfms)

    old_bs = int(cfg["replay"]["old_per_batch"])
    new_bs = int(cfg["replay"]["new_per_batch"])
    num_workers = int(cfg["num_workers"])

    old_loader = DataLoader(old_ds, batch_size=old_bs, shuffle=True, num_workers=num_workers, pin_memory=True, drop_last=True)
    new_loader = DataLoader(new_ds, batch_size=new_bs, shuffle=True, num_workers=num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=int(cfg["train"]["batch_size"]), shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=int(cfg["train"]["batch_size"]), shuffle=False, num_workers=num_workers, pin_memory=True)
    return class_names, old_loader, new_loader, val_loader, test_loader, manifest, old_pairs


def next_batch(it, loader):
    try:
        return next(it), it
    except StopIteration:
        it = iter(loader)
        return next(it), it


def train_replay(model, old_loader, new_loader, val_loader, cfg, device, log_csv: Path):
    lr_g3 = float(cfg["train"]["lr_g3"])
    lr_head = float(cfg["train"]["lr_head"])
    weight_decay = float(cfg["train"]["weight_decay"])
    max_epochs = int(cfg["train"]["max_epochs"])
    min_epochs = int(cfg["train"]["min_epochs"])
    patience = int(cfg["train"]["patience"])

    optimizer = Adam(
        [
            {"params": model.features[9:].parameters(), "lr": lr_g3},
            {"params": model.classifier.parameters(), "lr": lr_head},
        ],
        weight_decay=weight_decay,
    )
    scheduler = ReduceLROnPlateau(optimizer, mode="max", patience=3, factor=0.5, min_lr=1e-6)
    ce = nn.CrossEntropyLoss()

    best_f1 = -1.0
    best_state = copy.deepcopy(model.state_dict())
    best_epoch = -1
    bad_epochs = 0
    rows = []

    steps_per_epoch = max(len(old_loader), len(new_loader))
    phase_start = time.perf_counter()
    for epoch in range(1, max_epochs + 1):
        epoch_start = time.perf_counter()
        model.train()
        running_loss = 0.0
        old_it = iter(old_loader)
        new_it = iter(new_loader)

        for _ in range(steps_per_epoch):
            (x_old, y_old), old_it = next_batch(old_it, old_loader)
            (x_new, y_new), new_it = next_batch(new_it, new_loader)

            x = torch.cat([x_old, x_new], dim=0).to(device, non_blocking=True)
            y = torch.cat([y_old, y_new], dim=0).to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = ce(logits, y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        train_loss = running_loss / max(steps_per_epoch, 1)
        val_stats = evaluate(model, val_loader, device)
        scheduler.step(val_stats["macro_f1"])
        lrs = [pg["lr"] for pg in optimizer.param_groups]

        rows.append(
            {
                "epoch": epoch,
                "epoch_time_sec": round(time.perf_counter() - epoch_start, 4),
                "steps_per_epoch": steps_per_epoch,
                "train_loss": train_loss,
                "val_accuracy": val_stats["accuracy"],
                "val_macro_f1": val_stats["macro_f1"],
                "val_macro_precision": val_stats["macro_precision"],
                "val_macro_recall": val_stats["macro_recall"],
                "lr_g3": lrs[0],
                "lr_head": lrs[1],
            }
        )

        print(f"epoch={epoch:02d} loss={train_loss:.4f} val_f1={val_stats['macro_f1']:.4f} lrs={lrs}")

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
    plt.title("Cat A Experience Replay Training Curves")
    fig.tight_layout()
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def save_confusion(y_true, y_pred, class_names, out_png: Path):
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    fig = plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation="nearest")
    plt.title("Cat A Replay Test Confusion Matrix")
    plt.colorbar()
    ticks = np.arange(len(class_names))
    plt.xticks(ticks, class_names, rotation=45, ha="right")
    plt.yticks(ticks, class_names)
    plt.ylabel("True")
    plt.xlabel("Pred")
    plt.tight_layout()
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


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
    class_names, old_loader, new_loader, val_loader, test_loader, manifest, effective_old = build_dataloaders(cfg)

    model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
    model.classifier[3] = nn.Linear(1024, int(cfg["model"]["num_classes"]))
    model = model.to(device)

    base_ckpt = Path(cfg["model"]["base_checkpoint"])
    model.load_state_dict(torch.load(base_ckpt, map_location=device), strict=True)
    freeze_for_catA(model)

    model, train_summary = train_replay(
        model=model,
        old_loader=old_loader,
        new_loader=new_loader,
        val_loader=val_loader,
        cfg=cfg,
        device=device,
        log_csv=dirs["logs"] / "train_log.csv",
    )

    best_model_path = dirs["checkpoints"] / "model.pth"
    torch.save(model.state_dict(), best_model_path)

    val_metrics = evaluate(model, val_loader, device)
    test_metrics = evaluate(model, test_loader, device, with_preds=True)
    per_class_f1 = f1_score(test_metrics["y_true"], test_metrics["y_pred"], average=None).tolist()
    per_class_precision = precision_score(test_metrics["y_true"], test_metrics["y_pred"], average=None, zero_division=0).tolist()
    per_class_recall = recall_score(test_metrics["y_true"], test_metrics["y_pred"], average=None, zero_division=0).tolist()

    manifest_out = dirs["metrics"] / "replay_buffer_manifest.json"
    with manifest_out.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    buffer_list_out = dirs["metrics"] / "buffer_image_paths.txt"
    with buffer_list_out.open("w", encoding="utf-8") as f:
        for p, _ in effective_old:
            f.write(f"{p}\n")

    metrics = {
        "cycle": cfg["meta"]["cycle_name"],
        "method": "experience_replay",
        "class_names": class_names,
        "base_checkpoint": str(base_ckpt),
        "train_summary": train_summary,
        "replay": {
            "old_per_batch": int(cfg["replay"]["old_per_batch"]),
            "new_per_batch": int(cfg["replay"]["new_per_batch"]),
            "effective_old_count": len(effective_old),
            "manifest_path": str(manifest_out),
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

    print(f"Saved: {best_model_path}")
    print(f"Test macro_f1: {test_metrics['macro_f1']:.4f}")


if __name__ == "__main__":
    main()

