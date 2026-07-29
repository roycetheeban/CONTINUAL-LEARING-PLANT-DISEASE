"""Re-time the REAL Case 2 + Replay cycle-2 training config on this laptop's GPU.

Same data, same hyperparameters, same seed as the actual cycle-2 run already used
elsewhere in the paper -- this script only redirects the output directory so the
real `catA_replay_cycle2/checkpoints/model.pth` (which E2 depends on) is never
touched. The training script itself already logs everything E3 needs: per-epoch
time (train_log.csv), peak VRAM (torch.cuda.max_memory_allocated), end-of-run
process RAM (psutil), and total/train wall time (metrics.json) -- this driver just
invokes it with a patched config and copies the result into this experiment's
outputs folder."""
import json
import shutil
import subprocess
import sys

import yaml

from common_ondevice import CONFIG_PATH, ensure_dirs, load_cfg, OUTPUT_DIR, REPO_ROOT


def main():
    cfg = load_cfg(CONFIG_PATH)
    rt_cfg = cfg["retrain_timing"]

    source_config_path = REPO_ROOT / rt_cfg["source_config"]
    output_root = REPO_ROOT / rt_cfg["output_root"]
    ensure_dirs(output_root.parent)

    with open(source_config_path, "r", encoding="utf-8") as f:
        train_cfg = yaml.safe_load(f)

    # only the output location changes -- data, hyperparameters, seed all identical
    # to the real cycle-2 run, so this is a genuine re-timing, not a different setup.
    train_cfg["output"] = {"root": str(output_root).replace("\\", "/")}

    patched_config_path = OUTPUT_DIR / "catA_replay_cycle2_timing.yaml"
    with open(patched_config_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(train_cfg, f, sort_keys=False)

    train_script = (REPO_ROOT / "experiments" / "part1_case2_imagenet_finetune" /
                     "Continuation" / "catA_same_classes" / "experience_replay" /
                     "code" / "train_catA_replay.py")

    print(f"Re-timing cycle-2 replay training -> {output_root}")
    print(f"(real cycle-2 checkpoint at "
          f"{REPO_ROOT / 'experiments/part1_case2_imagenet_finetune/Continuation/catA_same_classes/experience_replay/outputs/catA_replay_cycle2/checkpoints/model.pth'} "
          f"is NOT touched by this run)")

    result = subprocess.run(
        [sys.executable, str(train_script), "--config", str(patched_config_path)],
        cwd=str(REPO_ROOT), check=True,
    )

    src_metrics = output_root / "metrics" / "metrics.json"
    with open(src_metrics, "r", encoding="utf-8") as f:
        train_metrics = json.load(f)

    ensure_dirs(OUTPUT_DIR)
    out_path = OUTPUT_DIR / "retrain_cycle_metrics.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "note": "re-timing of the real cycle-2 replay config on this laptop's GPU; "
                     "same data/hyperparameters/seed as the real cycle-2 run, output "
                     "redirected so the real artifact is untouched",
            "runtime": train_metrics["runtime"],
            "test_accuracy": train_metrics["test"]["accuracy"],
            "test_macro_f1": train_metrics["test"]["macro_f1"],
        }, f, indent=2)
    shutil.copy(output_root / "logs" / "train_log.csv", OUTPUT_DIR / "retrain_epoch_log.csv")
    print(f"wrote {out_path}")
    print(f"runtime: {train_metrics['runtime']}")


if __name__ == "__main__":
    main()
