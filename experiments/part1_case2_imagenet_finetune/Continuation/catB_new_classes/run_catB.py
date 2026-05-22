#!/usr/bin/env python3
import subprocess
import sys
from pathlib import Path


def run_method(method_name: str, cycle: int) -> bool:
    method_configs = {
        "ewc": f"ewc_head_expand/configs/catB_ewc_cycle{cycle}.yaml",
        "replay": f"experience_replay_head_expand/configs/catB_replay_cycle{cycle}.yaml",
        "hybrid": f"hybrid_ewc_replay_head_expand/configs/catB_hybrid_cycle{cycle}.yaml",
        "isolation": f"parameter_isolation_new_branch/configs/catB_isolation_cycle{cycle}.yaml",
        "naive_full_retrain": f"naive_full_retrain_head_expand/configs/catB_naive_full_retrain_cycle{cycle}.yaml",
    }
    method_scripts = {
        "ewc": "ewc_head_expand/code/train_catB_ewc.py",
        "replay": "experience_replay_head_expand/code/train_catB_replay.py",
        "hybrid": "hybrid_ewc_replay_head_expand/code/train_catB_hybrid.py",
        "isolation": "parameter_isolation_new_branch/code/train_catB_isolation.py",
        "naive_full_retrain": "naive_full_retrain_head_expand/code/train_catB_naive_full_retrain.py",
    }

    if method_name not in method_configs:
        print("Unknown method:", method_name)
        print("Available: ewc, replay, hybrid, isolation, naive_full_retrain")
        return False

    root = Path(__file__).parent
    config_path = root / method_configs[method_name]
    script_path = root / method_scripts[method_name]

    if not config_path.exists():
        print("Config not found:", config_path)
        return False
    if not script_path.exists():
        print("Script not found:", script_path)
        return False

    cmd = [sys.executable, str(script_path), "--config", str(config_path)]
    try:
        subprocess.run(cmd, check=True, capture_output=False)
        print(f"{method_name} cycle{cycle} completed")
        return True
    except subprocess.CalledProcessError as e:
        print(f"{method_name} cycle{cycle} failed: {e.returncode}")
        return False


def main():
    if len(sys.argv) < 2:
        print("Usage: python run_catB.py <method> [cycle]")
        print("Methods: ewc, replay, hybrid, isolation, naive_full_retrain")
        print("Cycles: 1, 2")
        return

    method = sys.argv[1].lower()
    cycle = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    run_method(method, cycle)


if __name__ == "__main__":
    main()
