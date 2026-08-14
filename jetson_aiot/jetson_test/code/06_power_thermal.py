"""STEP 6 -- power draw and thermal behaviour under each workload.

Why this exists
---------------
The bundle originally declared power and thermals out of scope: "Server/headless
access; thermal and power draw not measured." That is incorrect on this unit -- the
INA3221 rails and all thermal zones are readable from sysfs without root or physical
access. For a paper whose central claim is that a Jetson runs the full loop on
device, the immediate reviewer question is what that costs in watts, and leaving it
unanswered is a choice rather than a limitation.

Reports, per workload:
  * mean / peak board power (VDD_IN) and the CPU+GPU and SoC rails
  * energy in joules, and millijoules per unit of work
  * temperature start / peak / rise, to show whether anything throttles

Idle is measured first and subtracted, so both figures are available: board TOTAL
answers "what does the device draw", marginal-above-idle answers "what does this
workload cost". Report whichever the claim needs, but do not silently mix them.
"""
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common_jetson import (build_eval_transform, check_hardware_filled,  # noqa: E402
                           load_cfg, load_test_tensors, resolve, set_seed,
                           write_metrics)
from power_monitor import PowerSampler, available, subtract_idle  # noqa: E402


def _load_trt_runner(engine_path: Path):
    import importlib.util
    p = Path(__file__).resolve().parent / "02_gate_and_latency.py"
    spec = importlib.util.spec_from_file_location("_gate", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.TRTRunner(engine_path)


def measure(label, fn, items, item_name, interval, idle=None):
    """Run fn() under sampling; return the summary dict."""
    with PowerSampler(interval) as ps:
        fn()
    s = ps.summary(work_items=items, item_name=item_name)
    if idle:
        s["above_idle_w"] = subtract_idle(s, idle)
    vin = s["rails_w"].get("VDD_IN", {})
    tj = s["thermal_c"].get("tj-thermal", {})
    extra = ""
    if "per_item" in s:
        extra = f" | {s['per_item']['energy_mj_per_item']:.2f} mJ/{item_name}"
    print(f"  {label:26s} {vin.get('mean', 0):5.2f} W mean  {vin.get('peak', 0):5.2f} W peak"
          f"  tj {tj.get('peak', 0):.1f}C (+{tj.get('rise', 0):.1f}){extra}")
    return s


def main():
    cfg = load_cfg()
    check_hardware_filled(cfg)
    set_seed(int(cfg["seed"]))
    if not available():
        raise SystemExit("No INA3221 rails found -- power sampling unavailable here.")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA not available -- this must run on the Jetson.")
    device = torch.device("cuda")

    pcfg = cfg.get("power", {})
    if not pcfg.get("enabled", True):
        print("power.enabled is false -- skipping.")
        return
    interval = float(pcfg.get("sample_interval_s", 0.1))
    idle_s = float(pcfg.get("idle_seconds", 30))
    reps = int(pcfg.get("reps", 200))

    engine_path = resolve(cfg["trt"]["engine_path"])
    if not engine_path.exists():
        raise SystemExit(f"No engine at {engine_path}. Run 01_export_build_trt.py first.")

    print(f"sampling every {interval*1000:.0f} ms | idle baseline {idle_s:.0f} s")
    print("NOTE: keep the device otherwise idle for the duration of this step.\n")

    results = {}

    # ---- idle baseline -----------------------------------------------------
    idle = measure("idle baseline", lambda: time.sleep(idle_s), 0, "n/a", interval)
    results["idle"] = idle

    # ---- classification through the deployed TensorRT engine ---------------
    x, y = load_test_tensors(cfg, build_eval_transform(cfg), device)
    clf = _load_trt_runner(engine_path)
    k = int(cfg["data"]["capture_set_size"])
    one, cap = x[:1], x[:k]
    for _ in range(20):
        clf.infer(one)

    results["trt_classification_single"] = measure(
        "TRT classify (1 img)", lambda: [clf.infer(one) for _ in range(reps)],
        reps, "image", interval, idle)
    results["trt_classification_capture_set"] = measure(
        f"TRT classify ({k} imgs)", lambda: [clf.infer(cap) for _ in range(reps)],
        reps * k, "image", interval, idle)

    # ---- detection / segmentation ------------------------------------------
    dcfg = cfg["detect_seg"]
    try:
        from ultralytics import YOLO
    except ImportError:
        YOLO = None
        print("  ultralytics missing -- skipping detection/segmentation power")

    if YOLO is not None:
        root = resolve(cfg["data"]["test_dir"])
        imgs = []
        for cname in cfg["model"]["class_names"]:
            for p in sorted((root / cname).glob("*")):
                if p.suffix.lower() in (".jpg", ".jpeg", ".png"):
                    imgs.append(str(p))
                    break
        n_yolo = int(pcfg.get("yolo_reps", 100))
        imgsz, half = int(dcfg["imgsz"]), bool(dcfg.get("half", True))
        for key, wkey, label in (("detection", "detect_weights", "YOLOv8n detect"),
                                 ("segmentation", "seg_weights", "YOLOv8n segment")):
            w = resolve(dcfg[wkey])
            if not w.exists():
                continue
            mdl = YOLO(str(w))
            for i in range(10):
                mdl.predict(imgs[i % len(imgs)], imgsz=imgsz, half=half,
                            device=0, verbose=False)
            results[f"yolo_{key}"] = measure(
                label,
                lambda m=mdl: [m.predict(imgs[i % len(imgs)], imgsz=imgsz, half=half,
                                         device=0, verbose=False)
                               for i in range(n_yolo)],
                n_yolo, "frame", interval, idle)

    # ---- the full cascade loop ---------------------------------------------
    # Step 5 samples power around its OWN timed section and writes it into its
    # metrics file. Wrapping the subprocess here instead would charge process
    # startup, two YOLO loads, engine deserialisation and warmup to the per-loop
    # energy -- about half the window, i.e. roughly double the true figure.
    loop_script = Path(__file__).resolve().parent / "05_end_to_end_loop.py"
    if loop_script.exists() and pcfg.get("include_loop", True):
        import json
        import subprocess
        print("  running cascade (step 5 self-samples its timed section) ...")
        subprocess.run([sys.executable, str(loop_script)], capture_output=True)
        loop_json = resolve("outputs") / "05_end_to_end_loop.json"
        if loop_json.exists():
            blk = json.loads(loop_json.read_text()).get("power")
            if blk:
                blk["above_idle_w"] = subtract_idle(blk, idle)
                results["end_to_end_loop"] = blk
                vin = blk["rails_w"].get("VDD_IN", {})
                tj = blk["thermal_c"].get("tj-thermal", {})
                pi = blk.get("per_item", {})
                print(f"  {'full cascade loop':26s} {vin.get('mean', 0):5.2f} W mean  "
                      f"{vin.get('peak', 0):5.2f} W peak  tj {tj.get('peak', 0):.1f}C"
                      f" (+{tj.get('rise', 0):.1f}) | "
                      f"{pi.get('energy_mj_per_item', 0):.2f} mJ/capture_set")

    write_metrics("06_power_thermal.json", {
        "what_this_measures": (
            "Board power and thermal behaviour per workload, read from the INA3221 "
            "rails and thermal zones via sysfs. Supersedes the bundle's original "
            "claim that these were unmeasurable under headless access."),
        "rails": {"VDD_IN": "total board power -- report this one",
                  "VDD_CPU_GPU_CV": "CPU + GPU + CV accelerators",
                  "VDD_SOC": "SoC / memory / fabric"},
        "power_mode": cfg["hardware"].get("power_mode"),
        "sample_interval_s": interval,
        "idle_baseline_seconds": idle_s,
        "methodology": (
            "Sampling runs on a background thread and adds no latency to the timed "
            "path. Energy is integrated trapezoidally over real sample timestamps. "
            "'above_idle_w' is the marginal cost of the workload; rails_w is the "
            "board total. Both are reported -- do not mix them in one claim."),
        "caveat": (
            "Rail readings are the on-board INA3221 sensors, not an external power "
            "meter, and exclude losses in the supply. They are the standard figures "
            "reported for Jetson platforms and are what tegrastats surfaces."),
        "workloads": results,
    }, cfg)
    print("\nSTEP 6 done.")


if __name__ == "__main__":
    main()
