"""STEP 7 -- redeploy the RETRAINED model: convert, gate, decide.

Why this exists
---------------
Step 3 retrains a model on device. Step 2 gates the INCUMBENT. Nothing converted or
gated the model step 3 actually produced -- so the bundle could show that the loop
retrains on device, but not that it REDEPLOYS on device. Redeployment is the
"adaptation" half of "perception-to-adaptation".

This closes that. It runs the full promotion path on the retrained checkpoint:

    retrained .pth -> ONNX -> TensorRT FP16 engine -> conversion gate
                                                   -> deployment gate -> decision

Two distinct gates, and the paper should not conflate them:

  CONVERSION GATE  retrained-FP32 vs retrained-TRT-FP16
      "is the compiled artifact equivalent to the model I validated?"
      This is the epistemic gate -- it justifies measuring the deployed artifact.

  DEPLOYMENT GATE  retrained-TRT-FP16 vs incumbent-TRT-FP16
      "does the new model actually beat the one in production?"
      This is the promotion decision. Both are measured on the SAME fixed test set,
      through the SAME runtime, which is the only comparison that means anything.

Both engines are evaluated in this process on identical preloaded tensors, so the
comparison carries no cross-run variance.
"""
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common_jetson import (BUNDLE_ROOT, build_eval_transform,  # noqa: E402
                           check_hardware_filled, ensure_dir, load_cfg,
                           load_test_tensors, resolve, set_seed, write_metrics)


def _load_trt_runner(engine_path: Path):
    import importlib.util
    p = Path(__file__).resolve().parent / "02_gate_and_latency.py"
    spec = importlib.util.spec_from_file_location("_gate", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.TRTRunner(engine_path)


def build_model_from(ckpt: Path, cfg, device, half=False):
    """Same architecture as common_jetson.build_model, arbitrary checkpoint."""
    import torch.nn as nn
    from torchvision import models
    model = models.mobilenet_v3_small(weights=None)
    model.classifier[3] = nn.Linear(1024, int(cfg["model"]["num_classes"]))
    model = model.to(device)
    state = torch.load(ckpt, map_location=device, weights_only=True)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model.half() if half else model


def export_and_build(model, cfg, device, onnx_path: Path, engine_path: Path):
    """Mirrors step 1 exactly, so build times are comparable."""
    import tensorrt as trt
    size = int(cfg["model"]["image_size"])
    ensure_dir(onnx_path.parent)

    t0 = time.perf_counter()
    torch.onnx.export(
        model, torch.randn(1, 3, size, size, device=device), str(onnx_path),
        input_names=["input"], output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=int(cfg["export"]["opset"]))
    export_s = time.perf_counter() - t0

    tcfg = cfg["trt"]
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)
    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            errs = [str(parser.get_error(i)) for i in range(parser.num_errors)]
            raise RuntimeError("ONNX parse failed: " + "; ".join(errs))

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE,
                                 int(tcfg.get("workspace_mb", 1024)) * 1024 * 1024)
    if tcfg.get("fp16", True):
        if not builder.platform_has_fast_fp16:
            raise RuntimeError("Platform reports no fast FP16 support.")
        config.set_flag(trt.BuilderFlag.FP16)
    profile = builder.create_optimization_profile()
    profile.set_shape("input", (1, 3, size, size), (1, 3, size, size), (8, 3, size, size))
    config.add_optimization_profile(profile)

    t0 = time.perf_counter()
    serialized = builder.build_serialized_network(network, config)
    build_s = time.perf_counter() - t0
    if serialized is None:
        raise RuntimeError("build_serialized_network returned None (build failed).")
    engine_path.write_bytes(serialized)
    return export_s, build_s


def acc_torch(model, x, y, batch=64, half=False):
    correct = 0
    with torch.no_grad():
        for i in range(0, x.shape[0], batch):
            xb = x[i:i + batch]
            correct += (model(xb.half() if half else xb).argmax(1)
                        == y[i:i + batch]).sum().item()
    return correct / x.shape[0]


def preds_trt(runner, x, batch=8):
    return torch.cat([runner.infer(x[i:i + batch]).argmax(1)
                      for i in range(0, x.shape[0], batch)])


def mcnemar(a_correct, b_correct):
    """Exact McNemar on paired per-image correctness. b=a-right/b-wrong, c=reverse."""
    b = int((a_correct & ~b_correct).sum())
    c = int((~a_correct & b_correct).sum())
    p = None
    if b + c > 0:
        try:
            from scipy.stats import binomtest
            p = float(binomtest(c, b + c, 0.5).pvalue)
        except Exception:
            p = None
    return {"b_first_only_correct": b, "c_second_only_correct": c,
            "discordant": b + c, "exact_mcnemar_p": None if p is None else round(p, 6),
            "significant_at_0.05": None if p is None else bool(p < 0.05)}


def main():
    cfg = load_cfg()
    check_hardware_filled(cfg)
    set_seed(int(cfg["seed"]))
    if not torch.cuda.is_available():
        raise SystemExit("CUDA not available -- this must run on the Jetson.")
    device = torch.device("cuda")

    retrained = resolve(cfg["retrain"]["output_root"]) / "checkpoints" / "model.pth"
    if not retrained.exists():
        raise SystemExit(f"No retrained checkpoint at {retrained}. Run 03 first.")
    # The incumbent is the model the cycle STARTED from (cycle 1) -- that is what
    # was in production when the cycle ran, and therefore what the promotion
    # decision must be measured against. Using the cycle-2 engine from step 1 here
    # would compare the retrained model against a near-copy of itself.
    incumbent_ckpt = resolve(cfg["retrain"]["base_checkpoint"])
    if not incumbent_ckpt.exists():
        raise SystemExit(f"No incumbent checkpoint at {incumbent_ckpt}.")

    print("loading test set ...")
    x, y = load_test_tensors(cfg, build_eval_transform(cfg), device)
    n = x.shape[0]
    print(f"  {n} images")

    out_dir = ensure_dir(resolve("outputs"))

    # Build the incumbent's engine BEFORE the redeploy clock starts. In production
    # that engine already exists, so its build time is not a cost of this redeploy.
    print("building incumbent engine (not counted in redeploy cost) ...")
    inc_onnx = out_dir / "model_incumbent.onnx"
    inc_eng = out_dir / "model_incumbent_fp16.engine"
    export_and_build(build_model_from(incumbent_ckpt, cfg, device),
                     cfg, device, inc_onnx, inc_eng)

    print("converting retrained checkpoint ...")
    onnx_p = out_dir / "model_retrained.onnx"
    eng_p = out_dir / "model_retrained_fp16.engine"
    m32 = build_model_from(retrained, cfg, device)
    t_redeploy0 = time.perf_counter()
    export_s, build_s = export_and_build(m32, cfg, device, onnx_p, eng_p)
    print(f"  ONNX {export_s:.2f} s / {onnx_p.stat().st_size/1e6:.2f} MB | "
          f"engine {build_s:.2f} s / {eng_p.stat().st_size/1e6:.2f} MB")

    print("evaluating ...")
    new_trt = _load_trt_runner(eng_p)
    old_trt = _load_trt_runner(inc_eng)

    a_new32 = acc_torch(m32, x, y)
    p_new = preds_trt(new_trt, x, int(cfg["data"]["capture_set_size"]))
    p_old = preds_trt(old_trt, x, int(cfg["data"]["capture_set_size"]))
    gate_eval_s = 0.0
    t0 = time.perf_counter()
    _ = preds_trt(new_trt, x, int(cfg["data"]["capture_set_size"]))
    gate_eval_s = time.perf_counter() - t0
    redeploy_total_s = time.perf_counter() - t_redeploy0

    c_new = (p_new == y)
    c_old = (p_old == y)
    a_new = float(c_new.float().mean())
    a_old = float(c_old.float().mean())

    # conversion gate: retrained FP32 vs retrained TRT FP16
    with torch.no_grad():
        p_new32 = torch.cat([m32(x[i:i + 64]).argmax(1) for i in range(0, n, 64)])
    conv = mcnemar((p_new32 == y), c_new)
    conv_delta = (a_new - a_new32) * 100

    # deployment gate: retrained TRT vs incumbent TRT
    dep = mcnemar(c_old, c_new)
    dep_delta = (a_new - a_old) * 100
    promote = a_new > a_old

    print(f"\n  CONVERSION GATE  retrained fp32 {a_new32*100:.4f}%  ->  "
          f"retrained TRT {a_new*100:.4f}%   delta {conv_delta:+.4f} pp")
    print(f"    discordant {conv['discordant']}  p={conv['exact_mcnemar_p']}")
    print(f"\n  DEPLOYMENT GATE  incumbent TRT {a_old*100:.4f}%  ->  "
          f"retrained TRT {a_new*100:.4f}%   delta {dep_delta:+.4f} pp")
    print(f"    discordant {dep['discordant']}  p={dep['exact_mcnemar_p']}"
          f"  significant={dep['significant_at_0.05']}")
    print(f"\n  DECISION: {'PROMOTE' if promote else 'REJECT'} the retrained model")
    print(f"  redeploy cost: export {export_s:.2f} s + build {build_s:.2f} s + "
          f"gate eval {gate_eval_s:.2f} s = {redeploy_total_s:.2f} s total")

    write_metrics("07_redeploy_gate.json", {
        "what_this_measures": (
            "Full on-device redeployment of the model produced by step 3: convert to "
            "TensorRT, run the conversion gate, run the deployment gate, decide. "
            "Closes the adaptation half of the perception-to-adaptation loop."),
        "test_set_size": n,
        "retrained_checkpoint": str(retrained.relative_to(BUNDLE_ROOT)),
        "incumbent_checkpoint": str(incumbent_ckpt.relative_to(BUNDLE_ROOT)),
        "incumbent_note": (
            "The incumbent is the model the cycle STARTED from -- what was in "
            "production when the cycle ran. Its engine is built before the "
            "redeploy clock starts, since in production it already exists."),
        "redeploy_cost_seconds": {
            "onnx_export": round(export_s, 3),
            "tensorrt_build": round(build_s, 3),
            "gate_evaluation": round(gate_eval_s, 3),
            "total": round(redeploy_total_s, 3),
        },
        "artifact_sizes_mb": {
            "onnx": round(onnx_p.stat().st_size / 1e6, 3),
            "engine": round(eng_p.stat().st_size / 1e6, 3),
        },
        "conversion_gate": {
            "question": "Is the compiled artifact equivalent to the validated model?",
            "retrained_fp32_pct": round(a_new32 * 100, 4),
            "retrained_trt_fp16_pct": round(a_new * 100, 4),
            "delta_pp": round(conv_delta, 4),
            "mcnemar": conv,
            "identical_predictions": conv["discordant"] == 0,
            "statistically_equivalent": (conv["exact_mcnemar_p"] is None
                                         or conv["exact_mcnemar_p"] >= 0.05),
        },
        "deployment_gate": {
            "question": "Does the new model beat the incumbent on the fixed test set?",
            "incumbent_trt_fp16_pct": round(a_old * 100, 4),
            "retrained_trt_fp16_pct": round(a_new * 100, 4),
            "delta_pp": round(dep_delta, 4),
            "delta_images": int(round((a_new - a_old) * n)),
            "mcnemar": dep,
            "decision": "PROMOTE" if promote else "REJECT",
            "decision_rule": "promote iff retrained TRT accuracy > incumbent TRT accuracy",
            "caveat": ("The decision rule is a strict accuracy comparison, as the "
                       "design specifies. Statistical significance is reported "
                       "separately and is NOT part of the rule -- do not describe the "
                       "promotion as statistically significant unless p < 0.05."),
        },
    }, cfg)
    print("\nSTEP 7 done.")


if __name__ == "__main__":
    main()
