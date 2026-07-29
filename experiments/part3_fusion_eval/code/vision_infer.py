"""Run the trained Case 2 + Replay (cycle 2) checkpoint over the held-out tomato
test pool once, caching each image's predicted class. Episodes are later built by
sampling from this cache, so the vision-branch "leaf count" a fusion episode sees is
the real classifier's prediction (including its ~1-2% real error rate), not the
folder ground-truth label.

Mirrors the model definition in
experiments/part1_case2_imagenet_finetune/Continuation/catA_same_classes/experience_replay/code/train_catA_replay.py
(mobilenet_v3_small, classifier[3] replaced with a 5-class head, eval transform =
Resize(224,224) -> ToTensor -> ImageNet normalize)."""
import csv
import sys
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

from common_fusion import CONFIG_PATH, ensure_dirs, load_cfg, set_seed, OUTPUT_DIR, REPO_ROOT


def build_model(cfg: dict, device: torch.device) -> nn.Module:
    m = cfg["model"]
    model = models.mobilenet_v3_small(weights=None)
    model.classifier[3] = nn.Linear(1024, int(m["num_classes"]))
    model = model.to(device)
    ckpt_path = REPO_ROOT / m["checkpoint"]
    state = torch.load(ckpt_path, map_location=device)
    missing, unexpected = model.load_state_dict(state, strict=True)  # raises on mismatch
    model.eval()
    return model


def build_transform(cfg: dict):
    m = cfg["model"]
    size = int(m["image_size"])
    return transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=m["imagenet_mean"], std=m["imagenet_std"]),
    ])


def collect_pool_images(pool_root: Path, class_names: list[str]) -> list[tuple[str, str]]:
    """Returns (path, ground_truth_class_name) pairs from the test pool."""
    items = []
    for cname in class_names:
        cdir = pool_root / cname
        if not cdir.is_dir():
            raise FileNotFoundError(f"expected class folder not found: {cdir}")
        for p in sorted(cdir.glob("*")):
            if p.suffix.lower() in (".jpg", ".jpeg", ".png"):
                items.append((str(p), cname))
    return items


@torch.no_grad()
def run_inference(model, transform, items, class_names, device, batch_size=64):
    """Returns list of dicts: path, gt_class, pred_class, correct."""
    results = []
    for start in range(0, len(items), batch_size):
        batch = items[start:start + batch_size]
        tensors = []
        for path, _ in batch:
            img = Image.open(path).convert("RGB")
            tensors.append(transform(img))
        x = torch.stack(tensors).to(device)
        logits = model(x)
        preds = torch.argmax(logits, dim=1).cpu().tolist()
        for (path, gt_class), pred_idx in zip(batch, preds):
            pred_class = class_names[pred_idx]
            results.append({
                "path": path,
                "gt_class": gt_class,
                "pred_class": pred_class,
                "correct": int(pred_class == gt_class),
            })
    return results


def main():
    cfg = load_cfg(CONFIG_PATH)
    set_seed(cfg["seed"])
    ensure_dirs(OUTPUT_DIR)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    class_names = cfg["model"]["class_names"]
    pool_root = REPO_ROOT / cfg["data"]["image_pool_root"]

    model = build_model(cfg, device)
    transform = build_transform(cfg)

    items = collect_pool_images(pool_root, class_names)
    print(f"test pool images: {len(items)}")

    results = run_inference(model, transform, items, class_names, device)

    n_correct = sum(r["correct"] for r in results)
    acc = n_correct / len(results)
    print(f"sanity check -- pool inference accuracy: {acc:.4f} ({n_correct}/{len(results)})")
    # sanity gate: this checkpoint scored 98.42% on this exact test set during
    # training-time evaluation; if this run disagrees by more than a rounding
    # amount, the class-order mapping or preprocessing is wrong -- fail loudly.
    if acc < 0.95:
        print("ERROR: pool accuracy far below the expected ~98.4% -- check class "
              "order / checkpoint path / preprocessing before trusting downstream "
              "episode construction.", file=sys.stderr)
        sys.exit(1)

    out_path = OUTPUT_DIR / "test_pool_predictions.csv"
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["path", "gt_class", "pred_class", "correct"])
        writer.writeheader()
        writer.writerows(results)
    print(f"wrote {out_path} ({len(results)} rows)")


if __name__ == "__main__":
    main()
