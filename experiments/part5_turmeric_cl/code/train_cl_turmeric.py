"""E4 step 2 -- turmeric continual-learning cycles.

Runs one CL cycle with one method, starting from the previous checkpoint:

  replay : each batch is 50/50 replay-buffer samples and new-stream samples, so
           old-distribution gradients are present in every step.
  ewc    : trains on the new stream only, with a quadratic penalty anchoring
           parameters to their previous values, weighted by diagonal Fisher
           information estimated on the data the model already knows.
  naive  : new stream only, no protection. Reference condition -- NOT a valid CL
           method; included to show what forgetting looks like without it.

G1-G2 stay frozen throughout; only G3 (features[9:]) and the head adapt, at the
per-cycle learning rates from the paper's schedule.

Usage:
    python train_cl_turmeric.py --method replay --cycle 1 --seed 42
"""
from __future__ import annotations

import argparse
import copy
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from common_turmeric import (
    CONFIG_PATH, PathDataset, REPO_ROOT, build_model, build_transforms,
    build_theta_star, compute_fisher, data_root, ensure_dirs, evaluate,
    ewc_penalty, freeze_g1_g2, get_process_ram_mb, load_cfg, make_loader,
    pairs_from_dir, save_predictions, set_seed, write_json, write_log_csv,
)


def resolve_base_checkpoint(cfg, method: str, cycle: int, seed: int) -> Path:
    """Cycle 1 starts from the base model; cycle 2 continues from cycle 1 of the
    SAME method -- otherwise the cycles would not form a continual sequence."""
    root = REPO_ROOT / cfg["output"]["root"]
    if cycle == 1:
        return root / f"base_seed{seed}" / "checkpoints" / "base_model.pth"
    return root / f"{method}_cycle1_seed{seed}" / "checkpoints" / "model.pth"


def next_batch(it, loader):
    try:
        return next(it), it
    except StopIteration:
        it = iter(loader)
        return next(it), it


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(CONFIG_PATH))
    ap.add_argument("--method", required=True, choices=["replay", "ewc", "naive"])
    ap.add_argument("--cycle", required=True, type=int, choices=[1, 2])
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()

    cfg = load_cfg(args.config)
    seed = args.seed if args.seed is not None else cfg["seeds"][0]
    set_seed(seed)
    method, cycle = args.method, args.cycle

    device = torch.device(cfg["device"] if torch.cuda.is_available() else "cpu")
    cyc_cfg = cfg["cl"][f"cycle{cycle}"]
    out_root = REPO_ROOT / cfg["output"]["root"] / f"{method}_cycle{cycle}_seed{seed}"
    dirs = ensure_dirs(out_root)
    print(f"device={device} method={method} cycle={cycle} seed={seed} -> {out_root}")

    train_tfms, eval_tfms = build_transforms(cfg)
    stream_split = cfg["data"][f"cycle{cycle}"]
    # For replay the stream batch must be new_per_batch (not train.batch_size), so
    # that concatenating it with old_per_batch gives the intended 50/50 mix. Using
    # the generic batch size here silently halves the replay ratio to 1:2.
    stream_bs = int(cfg["replay"]["new_per_batch"]) if method == "replay" \
        else int(cfg["train"]["batch_size"])
    stream_ds, stream_loader = make_loader(cfg, stream_split, train_tfms,
                                           shuffle=True, drop_last=False,
                                           batch_size=stream_bs)
    _, val_loader = make_loader(cfg, cfg["data"]["val"], eval_tfms, shuffle=False)
    _, test_loader = make_loader(cfg, cfg["data"]["test"], eval_tfms, shuffle=False)
    class_to_idx = stream_ds.class_to_idx
    print(f"stream({stream_split})={len(stream_ds)} val={len(val_loader.dataset)} "
          f"test={len(test_loader.dataset)}")

    # ---- model, resumed from the correct point in the sequence --------------
    base_ckpt = resolve_base_checkpoint(cfg, method, cycle, seed)
    if not base_ckpt.exists():
        raise FileNotFoundError(
            f"required checkpoint missing: {base_ckpt}\n"
            f"(cycle 1 needs the base model; cycle 2 needs {method} cycle 1)")
    model = build_model(cfg, device, pretrained=False)
    model.load_state_dict(torch.load(base_ckpt, map_location=device), strict=True)
    freeze_g1_g2(model)
    print(f"resumed from {base_ckpt.name}")

    incumbent = evaluate(model, test_loader, device)
    print(f"incumbent test: acc={incumbent['accuracy']:.4f} f1={incumbent['macro_f1']:.4f}")

    optimizer = Adam([
        {"params": model.features[9:].parameters(), "lr": float(cyc_cfg["lr_g3"])},
        {"params": model.classifier.parameters(), "lr": float(cyc_cfg["lr_head"])},
    ], weight_decay=float(cfg["train"]["weight_decay"]))
    scheduler = ReduceLROnPlateau(optimizer, mode="max", patience=3, factor=0.5, min_lr=1e-6)
    ce = nn.CrossEntropyLoss()

    # ---- method-specific setup ---------------------------------------------
    replay_loader = None
    fisher = theta_star = None
    lambda_ewc = 0.0

    if method == "replay":
        buf_pairs = pairs_from_dir(data_root(cfg) / cfg["data"]["replay_buffer"], class_to_idx)
        replay_ds = PathDataset(buf_pairs, train_tfms)
        replay_loader = DataLoader(replay_ds, batch_size=int(cfg["replay"]["old_per_batch"]),
                                   shuffle=True, num_workers=int(cfg["num_workers"]),
                                   pin_memory=True, drop_last=True)
        print(f"replay buffer: {len(replay_ds)} images")

    elif method == "ewc":
        # Fisher is estimated on data the model ALREADY knows (initial_train for
        # cycle 1; that plus the previous stream for cycle 2) -- never on the new
        # stream, which is what the penalty is meant to let the model learn.
        prev_pairs = pairs_from_dir(data_root(cfg) / cfg["data"]["initial_train"], class_to_idx)
        if cycle == 2:
            prev_pairs += pairs_from_dir(data_root(cfg) / cfg["data"]["cycle1"], class_to_idx)
        fisher_loader = DataLoader(PathDataset(prev_pairs, eval_tfms),
                                   batch_size=int(cfg["train"]["batch_size"]), shuffle=True,
                                   num_workers=int(cfg["num_workers"]), pin_memory=True)
        lambda_ewc = float(cfg["ewc"]["lambda"] if cycle == 1 else cfg["ewc"]["lambda_cycle2"])
        fisher, used = compute_fisher(model, fisher_loader, device,
                                      int(cfg["ewc"]["fisher_max_batches"]))
        theta_star = build_theta_star(model)
        print(f"fisher over {used} batches ({len(prev_pairs)} imgs), lambda={lambda_ewc}")

    # ---- train --------------------------------------------------------------
    max_epochs = int(cyc_cfg["max_epochs"])
    min_epochs, patience = int(cfg["train"]["min_epochs"]), int(cfg["train"]["patience"])
    best_f1, best_state, best_epoch, bad = -1.0, copy.deepcopy(model.state_dict()), -1, 0
    rows: list[dict] = []
    t0 = time.perf_counter()

    steps = max(len(stream_loader), len(replay_loader)) if replay_loader else len(stream_loader)

    for epoch in range(1, max_epochs + 1):
        model.train()
        running = running_pen = 0.0
        stream_it = iter(stream_loader)
        replay_it = iter(replay_loader) if replay_loader else None

        for _ in range(steps):
            (xs, ys), stream_it = next_batch(stream_it, stream_loader)
            if replay_loader is not None:
                (xo, yo), replay_it = next_batch(replay_it, replay_loader)
                x = torch.cat([xo, xs]).to(device, non_blocking=True)
                y = torch.cat([yo, ys]).to(device, non_blocking=True)
            else:
                x, y = xs.to(device, non_blocking=True), ys.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            loss = ce(model(x), y)
            if method == "ewc":
                pen = ewc_penalty(model, fisher, theta_star, device)
                running_pen += float(pen.detach())
                loss = loss + lambda_ewc * pen
            loss.backward()
            optimizer.step()
            running += loss.item()

        val = evaluate(model, val_loader, device)
        scheduler.step(val["macro_f1"])
        rows.append({
            "epoch": epoch, "train_loss": running / max(steps, 1),
            "ewc_penalty": running_pen / max(steps, 1) if method == "ewc" else "",
            "val_accuracy": val["accuracy"], "val_macro_f1": val["macro_f1"],
            "lr_g3": optimizer.param_groups[0]["lr"],
            "lr_head": optimizer.param_groups[1]["lr"],
        })
        print(f"epoch={epoch:02d} loss={running/max(steps,1):.4f} "
              f"val_acc={val['accuracy']:.4f} val_f1={val['macro_f1']:.4f}")

        if val["macro_f1"] > best_f1:
            best_f1, best_epoch, bad = val["macro_f1"], epoch, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            bad += 1
        if epoch >= min_epochs and bad >= patience:
            print(f"early stop at epoch {epoch}")
            break

    model.load_state_dict(best_state)
    ckpt = dirs["checkpoints"] / "model.pth"
    torch.save(model.state_dict(), ckpt)

    # ---- evaluate on the FIXED test set -------------------------------------
    test = evaluate(model, test_loader, device, with_preds=True)
    save_predictions(dirs["metrics"] / "test_predictions.csv", test_loader, test)

    write_log_csv(dirs["logs"] / "train_log.csv", rows)
    write_json(dirs["metrics"] / "metrics.json", {
        "stage": f"cycle{cycle}", "method": method, "seed": seed,
        "base_checkpoint": str(base_ckpt),
        "classes": stream_ds.classes,
        "lambda_ewc": lambda_ewc if method == "ewc" else None,
        "best_epoch": best_epoch, "best_val_macro_f1": best_f1,
        "incumbent_test": {k: incumbent[k] for k in ("accuracy", "macro_f1")},
        "test": {k: test[k] for k in ("accuracy", "macro_f1", "macro_precision", "macro_recall")},
        "delta_vs_incumbent": {
            "accuracy": round(test["accuracy"] - incumbent["accuracy"], 6),
            "macro_f1": round(test["macro_f1"] - incumbent["macro_f1"], 6),
        },
        "runtime": {
            "wall_time_sec": round(time.perf_counter() - t0, 3),
            "peak_vram_mb": (round(torch.cuda.max_memory_allocated() / 1024 ** 2, 2)
                             if device.type == "cuda" else None),
            "process_ram_mb": get_process_ram_mb(),
        },
        "counts": {"stream": len(stream_ds),
                   "replay_buffer": len(replay_loader.dataset) if replay_loader else 0},
    })

    print(f"\nSaved {ckpt}")
    print(f"{method} cycle{cycle} test: acc={test['accuracy']:.4f} f1={test['macro_f1']:.4f} "
          f"(incumbent {incumbent['accuracy']:.4f})")


if __name__ == "__main__":
    main()
