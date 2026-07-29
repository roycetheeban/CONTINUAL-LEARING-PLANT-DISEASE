"""E4 orchestrator -- runs the full matrix: seeds x methods x cycles.

For each seed:  base -> {replay, ewc, naive} x {cycle 1, cycle 2}
Cycle 2 of a method resumes from cycle 1 of the SAME method, so each method forms
a genuine continual sequence rather than two independent fine-tunes.

Multi-seed is not optional here: with a 107-image test set a single run's accuracy
carries a 95% CI of about +/-5.7 pp, so any single number is nearly meaningless.
Table XIV reports mean +/- std across seeds.

Usage:
    python run_all.py                 # every seed in the config
    python run_all.py --seeds 42      # one seed
    python run_all.py --skip-base     # reuse existing base checkpoints
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

from common_turmeric import CONFIG_PATH, REPO_ROOT, load_cfg

CODE = Path(__file__).resolve().parent


def run(script: str, extra: list[str]) -> float:
    cmd = [sys.executable, str(CODE / script)] + extra
    print(f"\n{'='*72}\n$ {' '.join(cmd[1:])}\n{'='*72}")
    t0 = time.perf_counter()
    res = subprocess.run(cmd, cwd=str(CODE))
    dt = time.perf_counter() - t0
    if res.returncode != 0:
        raise SystemExit(f"FAILED ({res.returncode}): {script} {extra}")
    print(f"-- ok in {dt:.1f}s")
    return dt


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(CONFIG_PATH))
    ap.add_argument("--seeds", type=int, nargs="*", default=None)
    ap.add_argument("--methods", nargs="*", default=None)
    ap.add_argument("--skip-base", action="store_true")
    args = ap.parse_args()

    cfg = load_cfg(args.config)
    seeds = args.seeds if args.seeds else cfg["seeds"]
    methods = args.methods if args.methods else cfg["methods"]
    out_root = REPO_ROOT / cfg["output"]["root"]

    print(f"seeds={seeds} methods={methods}")
    total = time.perf_counter()

    for seed in seeds:
        base_ckpt = out_root / f"base_seed{seed}" / "checkpoints" / "base_model.pth"
        if args.skip_base and base_ckpt.exists():
            print(f"\n[seed {seed}] base exists, skipping")
        else:
            run("train_base_turmeric.py", ["--seed", str(seed), "--config", args.config])

        for method in methods:
            for cycle in (1, 2):
                run("train_cl_turmeric.py",
                    ["--method", method, "--cycle", str(cycle),
                     "--seed", str(seed), "--config", args.config])

    print(f"\n{'='*72}")
    print(f"ALL RUNS COMPLETE in {(time.perf_counter()-total)/60:.1f} min")
    print("next: python analyze_results.py")


if __name__ == "__main__":
    main()
