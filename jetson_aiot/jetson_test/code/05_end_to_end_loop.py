"""STEP 5 -- the COMPLETE perception-to-adaptation loop, run as a TRUE CASCADE.

Why this exists
---------------
Steps 1-4 measure components in isolation. The abstract claims something stronger:

    "a Jetson Orin Nano executes the complete perception-to-adaptation loop on device"

Separate component timings do not demonstrate that. This script runs the real
LEAFSENSE data flow, each stage consuming the previous stage's OUTPUT:

    capture -> detect single leaves -> crop each leaf -> segment each crop
            -> remove background -> classify each masked crop -> confidence routing

That distinction is not cosmetic. Segmentation runs once PER CROP, not once per
frame, and classification runs once PER DETECTION, not once per frame. An earlier
version of this script ran all stages on the 8 full frames independently and
reported 384.3 ms; because a capture set of 8 frames yields ~16 detections, that
figure understated both the segmentation and classification stages. Do not
reintroduce the un-chained version -- it measures a pipeline that does not exist.

SCOPE -- read before reporting
------------------------------
EXCLUDED, deliberately:
  * GAN occlusion recovery -- optional in the design and not shipped in this
    bundle. Excluded rather than approximated; a stand-in would put a fabricated
    number inside a latency budget.
  * Environmental risk fusion -- implementation not part of this bundle. It is
    scalar arithmetic over a few sensor readings and is expected to be negligible
    beside a 640x640 YOLO pass, but that is an argument, not a measurement.

Classification runs through the REAL TensorRT FP16 engine from step 1 -- the
artifact actually deployed. Detection and segmentation run through PyTorch FP16 via
ultralytics, matching step 4; this bundle ships no TensorRT engines for them.
"""
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common_jetson import (build_eval_transform, check_hardware_filled,  # noqa: E402
                           load_cfg, resolve, set_seed, write_metrics)


def _load_trt_runner(engine_path: Path):
    """Reuse step 2's TRTRunner (module name starts with a digit -> import by path)."""
    import importlib.util
    p = Path(__file__).resolve().parent / "02_gate_and_latency.py"
    spec = importlib.util.spec_from_file_location("_gate", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.TRTRunner(engine_path)


def collect_capture_set(cfg, k: int):
    """k image paths drawn round-robin across classes -- one greenhouse capture batch."""
    root = resolve(cfg["data"]["test_dir"])
    per_class = []
    for cname in cfg["model"]["class_names"]:
        per_class.append([p for p in sorted((root / cname).glob("*"))
                          if p.suffix.lower() in (".jpg", ".jpeg", ".png")])
    out, i = [], 0
    while len(out) < k and any(per_class):
        bucket = per_class[i % len(per_class)]
        if bucket:
            out.append(str(bucket.pop(0)))
        i += 1
    return out


def _pct(a):
    a = np.asarray(a, dtype=float)
    if a.size == 0:
        return {"mean_ms": 0.0, "std_ms": 0.0, "p50_ms": 0.0, "p95_ms": 0.0}
    return {"mean_ms": round(float(a.mean()), 3), "std_ms": round(float(a.std()), 3),
            "p50_ms": round(float(np.percentile(a, 50)), 3),
            "p95_ms": round(float(np.percentile(a, 95)), 3)}


def main():
    cfg = load_cfg()
    check_hardware_filled(cfg)
    set_seed(int(cfg["seed"]))
    if not torch.cuda.is_available():
        raise SystemExit("CUDA not available -- this must run on the Jetson.")
    device = torch.device("cuda")

    lcfg = cfg.get("loop", {})
    if not lcfg.get("enabled", True):
        print("loop.enabled is false -- skipping.")
        return
    k = int(cfg["data"]["capture_set_size"])
    warm = int(lcfg.get("warmup_iters", 3))
    iters = int(lcfg.get("timed_iters", 30))
    conf_thr = float(lcfg.get("confidence_threshold", 0.90))
    min_crop = int(lcfg.get("min_crop_px", 16))
    imgsz = int(cfg["detect_seg"]["imgsz"])
    half = bool(cfg["detect_seg"].get("half", True))
    trt_max_batch = int(lcfg.get("trt_max_batch", 8))   # step 1 profile max

    engine_path = resolve(cfg["trt"]["engine_path"])
    if not engine_path.exists():
        raise SystemExit(f"No engine at {engine_path}. Run 01_export_build_trt.py first.")

    try:
        from ultralytics import YOLO
    except ImportError:
        raise SystemExit("ultralytics not installed -- pip install ultralytics --no-deps")

    from PIL import Image

    paths = collect_capture_set(cfg, k)
    if len(paths) < k:
        raise SystemExit(f"only found {len(paths)} images, need {k}")
    print(f"capture set: {k} full frames | {iters} timed loops (warmup {warm})")
    print("cascade: detect -> crop -> segment(per crop) -> mask -> classify(per crop)")

    det = YOLO(str(resolve(cfg["detect_seg"]["detect_weights"])))
    seg = YOLO(str(resolve(cfg["detect_seg"]["seg_weights"])))
    clf = _load_trt_runner(engine_path)
    tf = build_eval_transform(cfg)

    def one_loop():
        t = {}

        # --- 1. capture: read the batch off disk as a camera write would land it
        t0 = time.perf_counter()
        frames = [Image.open(p).convert("RGB") for p in paths]
        t["capture"] = (time.perf_counter() - t0) * 1000.0

        # --- 2. detection: locate single leaves in each frame
        t0 = time.perf_counter()
        det_res = det.predict(paths, imgsz=imgsz, half=half, device=0, verbose=False)
        t["detection"] = (time.perf_counter() - t0) * 1000.0

        # --- 3. crop: one image per detected leaf (consumes detector output)
        t0 = time.perf_counter()
        crops = []
        for frame, r in zip(frames, det_res):
            W, H = frame.size
            for box in r.boxes.xyxy.detach().cpu().numpy():
                x1, y1, x2, y2 = [int(round(v)) for v in box[:4]]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(W, x2), min(H, y2)
                if (x2 - x1) >= min_crop and (y2 - y1) >= min_crop:
                    crops.append(frame.crop((x1, y1, x2, y2)))
        t["crop"] = (time.perf_counter() - t0) * 1000.0
        n_crops = len(crops)

        # --- 4. segmentation: PER CROP, not per frame
        t0 = time.perf_counter()
        seg_res = (seg.predict(crops, imgsz=imgsz, half=half, device=0, verbose=False)
                   if crops else [])
        t["segmentation"] = (time.perf_counter() - t0) * 1000.0

        # --- 5. background removal: apply the mask, zero everything else
        t0 = time.perf_counter()
        masked, n_masked = [], 0
        for crop, r in zip(crops, seg_res):
            arr = torch.from_numpy(np.asarray(crop, dtype=np.uint8)).permute(2, 0, 1)
            if r.masks is not None and len(r.masks) > 0:
                # ultralytics returns masks on the inference device; arr is CPU
                m = r.masks.data.detach().float().amax(0)[None, None]     # (1,1,h,w)
                m = torch.nn.functional.interpolate(
                    m, size=(arr.shape[1], arr.shape[2]), mode="nearest")[0].cpu()
                arr = (arr.float() * (m > 0.5).float()).to(torch.uint8)
                n_masked += 1
            masked.append(Image.fromarray(arr.permute(1, 2, 0).numpy()))
        t["background_removal"] = (time.perf_counter() - t0) * 1000.0

        # --- 6. classification: PER CROP, through the deployed TensorRT engine
        t0 = time.perf_counter()
        confs, preds = [], []
        if masked:
            x = torch.stack([tf(im) for im in masked]).to(device)
            for i in range(0, x.shape[0], trt_max_batch):   # engine profile max
                p = torch.softmax(clf.infer(x[i:i + trt_max_batch]), dim=1)
                c, q = p.max(dim=1)
                confs.append(c)
                preds.append(q)
            confs = torch.cat(confs)
            preds = torch.cat(preds)
            torch.cuda.synchronize()
        t["classification"] = (time.perf_counter() - t0) * 1000.0

        # --- 7. confidence routing: low-confidence crops feed the retrain buffer
        t0 = time.perf_counter()
        if len(confs) if isinstance(confs, torch.Tensor) else 0:
            n_routed = int((confs < conf_thr).sum().item())
            mean_conf = float(confs.mean().item())
        else:
            n_routed, mean_conf = 0, float("nan")
        t["confidence_routing"] = (time.perf_counter() - t0) * 1000.0

        t["_total"] = sum(v for kk, v in t.items() if not kk.startswith("_"))
        return t, n_crops, n_masked, n_routed, mean_conf

    for _ in range(warm):
        one_loop()

    stages = ["capture", "detection", "crop", "segmentation", "background_removal",
              "classification", "confidence_routing"]
    acc = {s: [] for s in stages}
    totals, crop_counts, confs_seen = [], [], []
    last = None

    # Power is sampled around ONLY the timed loops. Wrapping the whole process
    # would charge Python startup, two YOLO model loads, TRT deserialisation and
    # warmup to the per-loop energy -- roughly half the window, and therefore
    # roughly double the true figure.
    sampler = None
    try:
        from power_monitor import PowerSampler, available
        if available() and cfg.get("power", {}).get("enabled", True):
            sampler = PowerSampler(float(cfg["power"].get("sample_interval_s", 0.1)))
    except Exception:
        sampler = None

    if sampler is not None:
        sampler.__enter__()
    for _ in range(iters):
        t, n_crops, n_masked, n_routed, mean_conf = one_loop()
        for s in stages:
            acc[s].append(t[s])
        totals.append(t["_total"])
        crop_counts.append(n_crops)
        if not np.isnan(mean_conf):
            confs_seen.append(mean_conf)
        last = (n_crops, n_masked, n_routed, mean_conf)
    power_block = None
    if sampler is not None:
        sampler.__exit__(None, None, None)
        power_block = sampler.summary(work_items=iters, item_name="capture_set")
        power_block["attribution"] = (
            "Sampled around the timed loops ONLY -- excludes process startup, model "
            "loading, engine deserialisation and warmup.")

    total_stats = _pct(totals)
    mean_total = total_stats["mean_ms"]
    mean_crops = float(np.mean(crop_counts))
    per_stage = {s: {**_pct(acc[s]),
                     "share_pct": round(float(np.mean(acc[s])) / mean_total * 100, 2)}
                 for s in stages}

    print(f"\n  {'stage':22s} {'mean ms':>9s}  {'share':>7s}")
    for s in stages:
        print(f"  {s:22s} {per_stage[s]['mean_ms']:9.3f}  {per_stage[s]['share_pct']:6.2f}%")
    print(f"  {'-' * 22} {'-' * 9}")
    print(f"  {'TOTAL LOOP':22s} {mean_total:9.3f} ms  "
          f"({1000.0 / mean_total:.2f} capture-sets/s, "
          f"{k * 1000.0 / mean_total:.1f} frames/s)")

    n_crops, n_masked, n_routed, mean_conf = last
    print(f"\n  crops/loop {mean_crops:.1f} (from {k} frames) | masks applied {n_masked}"
          f" | routed {n_routed} | mean conf {mean_conf:.4f}")
    print(f"  per-frame {mean_total / k:.3f} ms | per-crop {mean_total / max(mean_crops, 1):.3f} ms")

    write_metrics("05_end_to_end_loop.json", {
        "claim_supported": (
            "Complete perception-to-adaptation loop executed on device as a TRUE "
            "CASCADE: capture -> detect leaves -> crop per detection -> segment each "
            "crop -> remove background -> classify each masked crop -> confidence "
            "routing. Each stage consumes the previous stage's output."),
        "scope_exclusions": [
            "GAN occlusion recovery -- optional in the design, not shipped in this "
            "bundle; excluded rather than approximated.",
            "Environmental risk fusion -- implementation not part of this bundle.",
        ],
        "DOMAIN_CAVEAT": {
            "issue": (
                "The detector is trained on class 'turmeric_leaf' and the segmenter on "
                "'middle_leaf', but this bundle ships only TOMATO (PlantVillage) "
                "imagery. The cascade therefore runs turmeric-domain detection and "
                "segmentation over tomato frames before a tomato classifier. That "
                "cross-domain chain does not exist in the deployed system, where the "
                "turmeric and tomato pipelines are separate."),
            "latency_is_valid": (
                "YES. YOLO and MobileNetV3 cost is set by input resolution and crop "
                "count, not image content, so the per-stage and total timings are a "
                "sound measurement of the device's compute cost for this pipeline "
                "shape."),
            "semantics_are_NOT_valid": (
                "NO. mean_classifier_confidence, routed_for_retraining and the crop "
                "count are products of out-of-domain detection. They must NOT be "
                "reported as system behaviour, routing rates, or evidence about "
                "confidence calibration."),
            "to_fix_properly": (
                "Re-run with turmeric imagery matching the detector/segmenter domain, "
                "and a classifier trained on that same crop distribution."),
        },
        "backends": {
            "detection": "PyTorch FP16 (ultralytics)",
            "segmentation": "PyTorch FP16 (ultralytics), run PER CROP",
            "classification": "TensorRT FP16 engine (the deployed artifact), PER CROP",
        },
        "capture_set_frames": k,
        "mean_crops_per_loop": round(mean_crops, 2),
        "timed_loops": iters,
        "warmup_loops": warm,
        "imgsz_detect_seg": imgsz,
        "confidence_threshold": conf_thr,
        "trt_max_batch": trt_max_batch,
        "total_loop_ms": total_stats,
        "capture_sets_per_second": round(1000.0 / mean_total, 3),
        "frames_per_second": round(k * 1000.0 / mean_total, 2),
        "ms_per_frame": round(mean_total / k, 3),
        "ms_per_crop": round(mean_total / max(mean_crops, 1), 3),
        "per_stage_ms": per_stage,
        "last_loop_observations": {
            "crops": n_crops, "masks_applied": n_masked,
            "routed_for_retraining": n_routed,
            "mean_classifier_confidence": (None if np.isnan(mean_conf)
                                           else round(mean_conf, 4)),
        },
        "power": power_block,
        "note_vs_earlier_run": (
            "Supersedes an un-chained version that ran every stage on the 8 full "
            "frames independently and reported 384.335 ms. That undercounted "
            "segmentation and classification, both of which scale with detection "
            "count, not frame count."),
    }, cfg)
    print("\nSTEP 5 done.")


if __name__ == "__main__":
    main()
