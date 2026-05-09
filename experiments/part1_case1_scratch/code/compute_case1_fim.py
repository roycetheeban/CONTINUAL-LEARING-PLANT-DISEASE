import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_fisher_dir(cfg: dict, fisher_split: str, fisher_dir_arg: str | None) -> str:
    if fisher_dir_arg:
        return fisher_dir_arg
    data_cfg = cfg["data"]
    if fisher_split == "replay":
        return cfg.get("fim", {}).get("fisher_dir", data_cfg.get("fisher_dir", "data/02_tomato_5cls/replay_buffer"))
    if fisher_split == "train":
        return data_cfg["train_dir"]
    if fisher_split == "val":
        return data_cfg["val_dir"]
    raise ValueError(f"Unsupported fisher_split: {fisher_split}")


def build_loader(cfg: dict, fisher_dir: str) -> tuple[datasets.ImageFolder, DataLoader]:
    image_size = int(cfg["data"]["image_size"])
    batch_size = int(cfg["fim"]["batch_size"])
    num_workers = int(cfg["num_workers"])

    tfm = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    ds = datasets.ImageFolder(fisher_dir, transform=tfm)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    return ds, loader


def compute_diagonal_fisher(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    max_batches: int,
) -> tuple[dict, int]:
    model.eval()
    fisher = {name: torch.zeros_like(p, device=device) for name, p in model.named_parameters() if p.requires_grad}
    criterion = nn.CrossEntropyLoss()
    used_batches = 0

    for x, y in loader:
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
        if used_batches >= max_batches:
            break

    denom = max(used_batches, 1)
    for name in fisher:
        fisher[name] = (fisher[name] / denom).detach().cpu()

    return fisher, used_batches


def build_theta_star(model: nn.Module) -> dict:
    return {name: p.detach().cpu().clone() for name, p in model.named_parameters() if p.requires_grad}


def summarize_fisher(fisher: dict) -> dict:
    all_vals = []
    for v in fisher.values():
        all_vals.append(v.flatten())
    if not all_vals:
        return {"mean": None, "max": None, "min": None}
    cat = torch.cat(all_vals)
    return {
        "mean": float(cat.mean().item()),
        "max": float(cat.max().item()),
        "min": float(cat.min().item()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=str)
    parser.add_argument("--ckpt", type=str, default=None, help="Path to trained case1 checkpoint (default: <output_root>/checkpoints/model.pth)")
    parser.add_argument("--fisher-split", type=str, default="replay", choices=["replay", "train", "val"])
    parser.add_argument("--fisher-dir", type=str, default=None, help="Override fisher dataset folder")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    set_seed(int(cfg["seed"]))
    cfg.setdefault("fim", {})
    cfg["fim"].setdefault("batch_size", int(cfg["train"]["batch_size"]))
    cfg["fim"].setdefault("max_batches", 100)

    output_root = Path(cfg["output"]["root"])
    fim_dir = output_root / "fim"
    fim_dir.mkdir(parents=True, exist_ok=True)

    ckpt_path = Path(args.ckpt) if args.ckpt else (output_root / "checkpoints" / "model.pth")
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    fisher_dir = resolve_fisher_dir(cfg, args.fisher_split, args.fisher_dir)
    if not Path(fisher_dir).exists():
        raise FileNotFoundError(f"Fisher dataset dir not found: {fisher_dir}")

    device = torch.device("cuda" if cfg["device"] == "cuda" and torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Checkpoint: {ckpt_path}")
    print(f"Fisher split: {args.fisher_split}")
    print(f"Fisher dir: {fisher_dir}")

    ds, loader = build_loader(cfg, fisher_dir)
    model = models.mobilenet_v3_small(weights=None)
    model.classifier[3] = nn.Linear(1024, int(cfg["model"]["num_classes"]))
    state = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state)
    model = model.to(device)

    start = time.perf_counter()
    fisher, used_batches = compute_diagonal_fisher(
        model=model,
        loader=loader,
        device=device,
        max_batches=int(cfg["fim"]["max_batches"]),
    )
    theta_star = build_theta_star(model)
    elapsed = round(time.perf_counter() - start, 4)

    fisher_path = fim_dir / "fisher_matrix.pt"
    theta_path = fim_dir / "theta_star.pt"
    meta_path = fim_dir / "fim_metadata.json"
    torch.save(fisher, fisher_path)
    torch.save(theta_star, theta_path)

    meta = {
        "method": "part1_case1_scratch_fim_export",
        "checkpoint": str(ckpt_path),
        "fisher_split": args.fisher_split,
        "fisher_dir": fisher_dir,
        "num_samples_in_fisher_ds": len(ds),
        "batch_size": int(cfg["fim"]["batch_size"]),
        "max_batches_config": int(cfg["fim"]["max_batches"]),
        "used_batches": used_batches,
        "elapsed_sec": elapsed,
        "fisher_stats": summarize_fisher(fisher),
        "outputs": {
            "fisher_matrix": str(fisher_path),
            "theta_star": str(theta_path),
        },
    }
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"Saved Fisher: {fisher_path}")
    print(f"Saved theta*: {theta_path}")
    print(f"Saved metadata: {meta_path}")


if __name__ == "__main__":
    main()
