"""E4 step 1 -- turmeric base model (Case 2: ImageNet init).

Two-phase layer-wise schedule from Table II of the paper:
  Phase i  : G1-G4 frozen except the head. Adapts the randomly-initialised
             5-class head before any gradient reaches the pretrained backbone.
             Skipping this is the most common transfer-learning mistake -- large
             early head gradients corrupt the features you are trying to reuse.
  Phase ii : unfreeze G3 (features[9:]) at a low LR; G1-G2 stay frozen.

Produces the checkpoint every CL cycle starts from.

Usage:
    python train_base_turmeric.py --seed 42
"""
from __future__ import annotations

import argparse
import copy
import time

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from common_turmeric import (
    CONFIG_PATH, OUTPUT_DIR, REPO_ROOT, build_model, build_transforms,
    compute_class_weights, ensure_dirs, evaluate, freeze_all_features,
    freeze_g1_g2, get_process_ram_mb, load_cfg, make_loader, save_predictions,
    set_seed, write_json, write_log_csv,
)


def train_phase(model, train_loader, val_loader, optimizer, scheduler, criterion,
                device, phase: str, max_epochs: int, min_epochs: int, patience: int,
                rows: list) -> tuple[nn.Module, dict]:
    best_f1, best_state, best_epoch, bad = -1.0, copy.deepcopy(model.state_dict()), -1, 0
    t0 = time.perf_counter()

    for epoch in range(1, max_epochs + 1):
        model.train()
        running, n_batches = 0.0, 0
        for x, y in train_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            running += loss.item()
            n_batches += 1

        val = evaluate(model, val_loader, device)
        scheduler.step(val["macro_f1"])
        lrs = [pg["lr"] for pg in optimizer.param_groups]

        rows.append({
            "phase": phase, "epoch": epoch,
            "train_loss": running / max(n_batches, 1),
            "val_accuracy": val["accuracy"], "val_macro_f1": val["macro_f1"],
            "lrs": "|".join(f"{lr:.8f}" for lr in lrs),
        })
        print(f"[{phase}] epoch={epoch:02d} loss={running/max(n_batches,1):.4f} "
              f"val_acc={val['accuracy']:.4f} val_f1={val['macro_f1']:.4f}")

        if val["macro_f1"] > best_f1:
            best_f1, best_epoch, bad = val["macro_f1"], epoch, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            bad += 1

        if epoch >= min_epochs and bad >= patience:
            print(f"[{phase}] early stop at epoch {epoch}")
            break

    model.load_state_dict(best_state)
    return model, {
        "phase": phase, "best_epoch": best_epoch, "best_val_macro_f1": best_f1,
        "wall_time_sec": round(time.perf_counter() - t0, 3),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(CONFIG_PATH))
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()

    cfg = load_cfg(args.config)
    seed = args.seed if args.seed is not None else cfg["seeds"][0]
    set_seed(seed)

    device = torch.device(cfg["device"] if torch.cuda.is_available() else "cpu")
    out_root = REPO_ROOT / cfg["output"]["root"] / f"base_seed{seed}"
    dirs = ensure_dirs(out_root)
    print(f"device={device} seed={seed} -> {out_root}")

    train_tfms, eval_tfms = build_transforms(cfg)
    train_ds, train_loader = make_loader(cfg, cfg["data"]["initial_train"], train_tfms, shuffle=True)
    _, val_loader = make_loader(cfg, cfg["data"]["val"], eval_tfms, shuffle=False)
    _, test_loader = make_loader(cfg, cfg["data"]["test"], eval_tfms, shuffle=False)

    print(f"classes: {train_ds.classes}")
    print(f"initial_train={len(train_ds)} val={len(val_loader.dataset)} "
          f"test={len(test_loader.dataset)}")

    criterion = nn.CrossEntropyLoss(weight=compute_class_weights(train_ds, device))
    model = build_model(cfg, device, pretrained=True)

    rows: list[dict] = []
    total_start = time.perf_counter()

    # ---- Phase i: head only -------------------------------------------------
    freeze_all_features(model)
    opt1 = Adam(model.classifier.parameters(),
                lr=float(cfg["base"]["phase1"]["lr_head"]),
                weight_decay=float(cfg["train"]["weight_decay"]))
    sch1 = ReduceLROnPlateau(opt1, mode="max", patience=3, factor=0.5, min_lr=1e-6)
    model, best1 = train_phase(model, train_loader, val_loader, opt1, sch1, criterion,
                               device, "phase1", int(cfg["base"]["phase1"]["max_epochs"]),
                               int(cfg["train"]["min_epochs"]), int(cfg["train"]["patience"]), rows)
    torch.save(model.state_dict(), dirs["checkpoints"] / "base_phase1.pth")

    # ---- Phase ii: unfreeze G3 ---------------------------------------------
    freeze_g1_g2(model)
    opt2 = Adam([
        {"params": model.features[9:].parameters(), "lr": float(cfg["base"]["phase2"]["lr_g3"])},
        {"params": model.classifier.parameters(), "lr": float(cfg["base"]["phase2"]["lr_head"])},
    ], weight_decay=float(cfg["train"]["weight_decay"]))
    sch2 = ReduceLROnPlateau(opt2, mode="max", patience=3, factor=0.5, min_lr=1e-6)
    model, best2 = train_phase(model, train_loader, val_loader, opt2, sch2, criterion,
                               device, "phase2", int(cfg["base"]["phase2"]["max_epochs"]),
                               int(cfg["train"]["min_epochs"]), int(cfg["train"]["patience"]), rows)

    ckpt = dirs["checkpoints"] / "base_model.pth"
    torch.save(model.state_dict(), ckpt)

    # ---- evaluate on the fixed test set ------------------------------------
    test = evaluate(model, test_loader, device, with_preds=True)
    save_predictions(dirs["metrics"] / "test_predictions.csv", test_loader, test)

    write_log_csv(dirs["logs"] / "train_log.csv", rows)
    write_json(dirs["metrics"] / "metrics.json", {
        "stage": "base", "seed": seed, "classes": train_ds.classes,
        "phase1": best1, "phase2": best2,
        "test": {k: test[k] for k in ("accuracy", "macro_f1", "macro_precision", "macro_recall")},
        "runtime": {
            "total_wall_time_sec": round(time.perf_counter() - total_start, 3),
            "peak_vram_mb": (round(torch.cuda.max_memory_allocated() / 1024 ** 2, 2)
                             if device.type == "cuda" else None),
            "process_ram_mb": get_process_ram_mb(),
        },
        "counts": {"initial_train": len(train_ds), "val": len(val_loader.dataset),
                   "test": len(test_loader.dataset)},
    })

    print(f"\nSaved {ckpt}")
    print(f"BASE test: acc={test['accuracy']:.4f} macro_f1={test['macro_f1']:.4f}")


if __name__ == "__main__":
    main()
