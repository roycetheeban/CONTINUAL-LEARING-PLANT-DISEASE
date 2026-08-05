"""STEP 4 -- detection and segmentation LATENCY on the target device.

Table XIII currently cites 12.2 ms detection and 5.0 ms segmentation from the
prior conference paper [ref21]. Those are the only rows in the table not measured
for this paper, and it is not established that they were measured on a Jetson
rather than on a training GPU. Measuring them here removes that ambiguity.

SCOPE -- read this before reporting anything from this script:
  * LATENCY ONLY. This bundle ships no detection/segmentation ground truth, so
    mAP@0.5, pixel accuracy and mIoU are NOT re-measured and must stay cited to
    [ref21].
  * Inputs are real leaf images from the test set, letterboxed to 640x640 by
    ultralytics. Content does not affect latency materially; resolution does.
"""
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common_jetson import (check_hardware_filled, load_cfg, resolve,  # noqa: E402
                           set_seed, write_metrics)


def collect_images(cfg, limit=32):
    root = resolve(cfg["data"]["test_dir"])
    imgs = []
    for cname in cfg["model"]["class_names"]:
        for p in sorted((root / cname).glob("*")):
            if p.suffix.lower() in (".jpg", ".jpeg", ".png"):
                imgs.append(str(p))
                break
    first = root / cfg["model"]["class_names"][0]
    for p in sorted(first.glob("*")):
        if len(imgs) >= limit:
            break
        if p.suffix.lower() in (".jpg", ".jpeg", ".png") and str(p) not in imgs:
            imgs.append(str(p))
    return imgs


def bench(model, imgs, dcfg, label):
    imgsz = int(dcfg["imgsz"])
    half = bool(dcfg.get("half", True))
    warm, iters = int(dcfg["warmup_iters"]), int(dcfg["timed_iters"])

    for i in range(warm):
        model.predict(imgs[i % len(imgs)], imgsz=imgsz, half=half,
                      device=0, verbose=False)

    ts = []
    for i in range(iters):
        src = imgs[i % len(imgs)]
        t0 = time.perf_counter()
        model.predict(src, imgsz=imgsz, half=half, device=0, verbose=False)
        ts.append((time.perf_counter() - t0) * 1000.0)

    a = np.array(ts)
    r = {"mean_ms": round(float(a.mean()), 3), "std_ms": round(float(a.std()), 3),
         "p50_ms": round(float(np.percentile(a, 50)), 3),
         "p95_ms": round(float(np.percentile(a, 95)), 3),
         "fps_from_mean": round(1000.0 / float(a.mean()), 2)}
    print(f"  {label:14s} {r['mean_ms']:7.3f} ms   ({r['fps_from_mean']} FPS)")
    return r


def main():
    cfg = load_cfg()
    check_hardware_filled(cfg)
    set_seed(int(cfg["seed"]))

    dcfg = cfg["detect_seg"]
    if not dcfg.get("enabled", True):
        print("detect_seg.enabled is false -- skipping.")
        return

    try:
        from ultralytics import YOLO
    except ImportError:
        raise SystemExit(
            "ultralytics not installed.\n"
            "  pip install ultralytics --no-deps\n"
            "(--no-deps avoids pulling an x86 torch wheel over JetPack's build)")

    imgs = collect_images(cfg)
    if not imgs:
        raise SystemExit("no test images found")
    print(f"benchmarking on {len(imgs)} images, imgsz={dcfg['imgsz']}, half={dcfg.get('half', True)}")

    results = {}
    for key, wkey, label in [("detection", "detect_weights", "detection"),
                             ("segmentation", "seg_weights", "segmentation")]:
        w = resolve(dcfg[wkey])
        if not w.exists():
            print(f"  {label}: weights missing at {w} -- skipped")
            continue
        results[key] = bench(YOLO(str(w)), imgs, dcfg, label)
        results[key]["weights"] = w.name

    write_metrics("04_detect_seg_latency.json", {
        "scope": "LATENCY ONLY -- mAP/mIoU not re-measured; keep citing ref21 for accuracy.",
        "imgsz": int(dcfg["imgsz"]),
        "half": bool(dcfg.get("half", True)),
        "n_source_images": len(imgs),
        "results": results,
    }, cfg)
    print("\nSTEP 4 done.")


if __name__ == "__main__":
    main()
