"""Run all four Jetson measurement steps in order.

Steps 1-2 are the critical pair (TensorRT engine + conversion gate + latency) and
take a couple of minutes. Step 3 (retrain cycle) is the long one. Step 4 needs
ultralytics installed.

  python3 code/run_all.py              # all steps
  python3 code/run_all.py --only 1 2   # just the critical pair
  python3 code/run_all.py --skip 3     # everything except the long retrain
"""
import argparse
import subprocess
import sys
from pathlib import Path

CODE = Path(__file__).resolve().parent
STEPS = [
    ("1", "01_export_build_trt.py", "ONNX export + TensorRT FP16 engine"),
    ("2", "02_gate_and_latency.py", "Conversion gate + classification latency"),
    ("3", "03_retrain_cycle.py", "On-device retraining cycle (LONG)"),
    ("4", "04_bench_detect_seg.py", "Detection / segmentation latency"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=None, help="step numbers to run")
    ap.add_argument("--skip", nargs="*", default=[], help="step numbers to skip")
    args = ap.parse_args()

    selected = [s for s in STEPS
                if (args.only is None or s[0] in args.only) and s[0] not in args.skip]
    if not selected:
        raise SystemExit("nothing selected")

    print("=" * 66)
    print("Jetson Orin Nano on-device measurement")
    for num, script, desc in selected:
        print(f"  [{num}] {desc}")
    print("=" * 66)

    failed = []
    for num, script, desc in selected:
        print(f"\n{'-' * 66}\n[{num}] {desc}\n{'-' * 66}")
        rc = subprocess.run([sys.executable, str(CODE / script)]).returncode
        if rc != 0:
            print(f"[{num}] FAILED (exit {rc})")
            failed.append(num)

    print("\n" + "=" * 66)
    if failed:
        print(f"FAILED steps: {', '.join(failed)}")
        print("Partial results are still in outputs/ -- check the *_FAILED.json files.")
        sys.exit(1)
    print("All steps completed. Results in outputs/*.json")
    print("Send those JSON files back and they go straight into Table XIII.")
    print("=" * 66)


if __name__ == "__main__":
    main()
