"""STEP 3 -- the monthly retraining cycle, measured on the target device.

Re-runs the REAL Case 2 + Replay cycle-2 configuration -- same data, same
hyperparameters, same seed -- with output redirected into this bundle. This backs
the claim the paper currently only infers: that a retrain cycle fits inside the
Jetson's envelope.

If it does NOT fit, that is a finding to report, not a failure to hide.

Note this may take considerably longer than the 373 s measured on the laptop
RTX 4050; that difference is the point of the measurement.
"""
import json
import subprocess
import sys
import time
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common_jetson import (BUNDLE_ROOT, check_hardware_filled, ensure_dir,  # noqa: E402
                           load_cfg, resolve, write_metrics)


def materialize_manifest(cfg) -> Path:
    """Rewrite the shipped manifest's bundle-relative POSIX paths to absolute.

    The training script consumes item["path"] directly as a filesystem path, and
    the original manifest carried Windows backslash paths pointing outside this
    bundle. This produces a runtime copy with absolute, platform-correct paths.
    """
    src = resolve(cfg["retrain"]["input_manifest"])
    man = json.loads(src.read_text(encoding="utf-8"))
    for item in man.get("added_cycle_paths", []):
        item["path"] = str(resolve(item["path"]))
    man["replay_root"] = str(resolve(cfg["retrain"]["replay_dir"]))
    out = ensure_dir(resolve(cfg["retrain"]["output_root"]).parent) / "_manifest_runtime.json"
    out.write_text(json.dumps(man, indent=2), encoding="utf-8")
    return out


def build_train_config(cfg, manifest_path: Path) -> Path:
    r = cfg["retrain"]
    train_cfg = {
        "seed": int(cfg["seed"]),
        "num_workers": 0,
        "device": "cuda",
        "meta": {"cycle_name": "cycle2"},
        "model": {
            "num_classes": int(cfg["model"]["num_classes"]),
            "base_checkpoint": str(resolve(cfg["retrain"]["base_checkpoint"])),
        },
        "data": {
            "train_dir": str(resolve(r["train_dir"])),
            "replay_dir": str(resolve(r["replay_dir"])),
            "val_dir": str(resolve(r["val_dir"])),
            "test_dir": str(resolve(r["test_dir"])),
            "image_size": int(r["image_size"]),
            "resize_size": int(r["resize_size"]),
        },
        "train": {
            "batch_size": int(r["batch_size"]),
            "weight_decay": float(r["weight_decay"]),
            "max_epochs": int(r["max_epochs"]),
            "min_epochs": int(r["min_epochs"]),
            "patience": int(r["patience"]),
            "lr_g3": float(r["lr_g3"]),
            "lr_head": float(r["lr_head"]),
        },
        "replay": {
            "old_per_batch": int(r["old_per_batch"]),
            "new_per_batch": int(r["new_per_batch"]),
            "add_per_class_after_cycle": int(r["add_per_class_after_cycle"]),
            "input_manifest": str(manifest_path),
        },
        "output": {"root": str(resolve(r["output_root"]))},
    }
    out = resolve(r["output_root"]).parent / "_train_config_runtime.yaml"
    ensure_dir(out.parent)
    with open(out, "w", encoding="utf-8") as f:
        yaml.safe_dump(train_cfg, f, sort_keys=False)
    return out


def require_cuda() -> str:
    """Steps 1/2/4 refuse to run without CUDA. Step 3 must too.

    train_catA_replay.py is a verbatim copy of the main-repo script and silently
    falls back to CPU:

        device = torch.device("cuda" if ... and torch.cuda.is_available() else "cpu")

    On CPU it still completes, writes peak_vram_mb: null, and reports a wall-clock
    several times too large. That wall-clock is exactly what backs the paper's
    "fits the target device's envelope" claim. A silent CPU run would therefore
    produce a wrong number that looks entirely valid. Guard it here rather than
    editing the verbatim copy.
    """
    import torch
    if not torch.cuda.is_available():
        raise SystemExit(
            "CUDA not available -- refusing to run the retrain cycle.\n"
            f"  torch {torch.__version__}\n"
            "A CPU fallback would still produce a wall-clock number, and that number\n"
            "backs the paper's device-envelope claim. It must not be measured on CPU.\n"
            "Likely cause: the PyPI 'manylinux_2_28_aarch64' torch wheel (no CUDA)\n"
            "was installed instead of the Jetson 'linux_aarch64' wheel. See README.")
    return torch.cuda.get_device_name(0)


def main():
    cfg = load_cfg()
    check_hardware_filled(cfg)
    if not cfg["retrain"].get("enabled", True):
        print("retrain.enabled is false -- skipping.")
        return

    gpu = require_cuda()
    print(f"CUDA check: OK ({gpu})")

    # The cycle must START from the incumbent (cycle 1), not from the DEPLOYED
    # model (cycle 2, used by steps 1/2/5). Pointing both at one entry retrains
    # cycle-2 on data it has already seen: it does not crash, it converges early
    # from an implausibly low epoch-1 loss, and it reports an inflated accuracy
    # measured on memorised material. Nothing downstream can detect that, so it
    # is checked here.
    base = resolve(cfg["retrain"]["base_checkpoint"])
    deployed = resolve(cfg["model"]["checkpoint"])
    if not base.exists():
        raise SystemExit(f"retrain.base_checkpoint not found: {base}")
    if base.resolve() == deployed.resolve():
        raise SystemExit(
            "retrain.base_checkpoint is the DEPLOYED checkpoint "
            f"({deployed.name}).\nThe cycle must start from the previous cycle's "
            "model, not its own output -- otherwise it trains and tests on data "
            "the model has already learned, and the accuracy is meaningless.")
    print(f"retrain base   : {base.name}")
    print(f"deployed model : {deployed.name}  (steps 1/2/5, not used here)")

    manifest = materialize_manifest(cfg)
    train_config = build_train_config(cfg, manifest)
    train_script = Path(__file__).resolve().parent / "train_catA_replay.py"
    if not train_script.exists():
        raise SystemExit(f"missing {train_script}")

    print(f"running retrain cycle -> {resolve(cfg['retrain']['output_root'])}")
    print("(this is the real cycle-2 config; nothing in the main repo is touched)\n")

    t0 = time.perf_counter()
    proc = subprocess.run([sys.executable, str(train_script), "--config", str(train_config)],
                          cwd=str(BUNDLE_ROOT))
    wall = time.perf_counter() - t0

    if proc.returncode != 0:
        write_metrics("03_retrain_cycle_FAILED.json",
                      {"driver_wall_seconds": round(wall, 2),
                       "returncode": proc.returncode}, cfg)
        raise SystemExit(f"training failed with code {proc.returncode}")

    # the training script writes its own metrics.json (per-epoch time, peak VRAM,
    # process RAM, wall time) -- surface it alongside the driver timing
    inner = None
    mdir = resolve(cfg["retrain"]["output_root"]) / "metrics"
    for cand in sorted(mdir.glob("*.json")) if mdir.exists() else []:
        try:
            d = json.loads(cand.read_text(encoding="utf-8"))
            if "runtime" in d:
                inner = {"file": cand.name, **d}
                break
        except Exception:
            continue

    # The driver checked CUDA in ITS process; confirm the training subprocess
    # actually used the GPU. peak_vram_mb is None only on a CPU run.
    ran_on_gpu = bool(inner and inner.get("runtime", {}).get("peak_vram_mb") is not None)
    if not ran_on_gpu:
        print("\n" + "!" * 66)
        print("  peak_vram_mb is null -- the training subprocess ran on CPU.")
        print("  The wall-clock below is NOT a Jetson GPU measurement and must not")
        print("  be reported as one. Fix the environment and re-run.")
        print("!" * 66 + "\n")

    write_metrics("03_retrain_cycle.json", {
        "driver_wall_seconds": round(wall, 2),
        "output_root": str(resolve(cfg["retrain"]["output_root"]).relative_to(BUNDLE_ROOT)),
        "device_used": gpu if ran_on_gpu else "CPU -- RESULT INVALID",
        "gpu_confirmed": ran_on_gpu,
        "dataloader_workers": 0,
        "measurement_caveat": (
            "num_workers=0, so JPEG decode and augmentation run single-threaded on "
            "one CPU core while the GPU waits. This wall-clock is a data-loading "
            "bound, not a training-compute bound; peak memory is unaffected."),
        "training_metrics": inner if inner else "not found -- check outputs/retrain_cycle/metrics/",
    }, cfg)
    print(f"\nSTEP 3 done -- {wall:.1f} s wall clock.")
    if not ran_on_gpu:
        raise SystemExit("retrain cycle ran on CPU -- result invalid")


if __name__ == "__main__":
    main()
