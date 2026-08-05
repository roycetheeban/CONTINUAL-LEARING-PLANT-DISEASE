"""STEP 1 -- ONNX export + real TensorRT FP16 engine build.

This is the step the laptop could not do: TensorRT would not install on
Windows/CUDA 12.1, so Table XIII currently reports "n/a" for the engine build and
substitutes an ONNX Runtime graph for the compiled artifact. JetPack ships
TensorRT, so this should now simply work.

On failure it records the reason and exits NON-ZERO rather than silently falling
back -- a fallback number reported as a TensorRT number would be worse than no
number at all.
"""
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common_jetson import (BUNDLE_ROOT, build_model, check_hardware_filled,  # noqa: E402
                           ensure_dir, load_cfg, resolve, set_seed, write_metrics)


def export_onnx(model, cfg, device):
    onnx_path = resolve(cfg["export"]["onnx_path"])
    ensure_dir(onnx_path.parent)
    size = int(cfg["model"]["image_size"])
    dummy = torch.randn(1, 3, size, size, device=device)

    start = time.perf_counter()
    torch.onnx.export(
        model, dummy, str(onnx_path),
        input_names=["input"], output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=int(cfg["export"]["opset"]),
    )
    return onnx_path, time.perf_counter() - start


def build_engine(onnx_path: Path, cfg):
    import tensorrt as trt

    tcfg = cfg["trt"]
    engine_path = resolve(tcfg["engine_path"])
    ensure_dir(engine_path.parent)

    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)

    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            errs = [str(parser.get_error(i)) for i in range(parser.num_errors)]
            raise RuntimeError("ONNX parse failed: " + "; ".join(errs))

    config = builder.create_builder_config()
    ws = int(tcfg.get("workspace_mb", 1024)) * 1024 * 1024
    try:
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, ws)
    except AttributeError:                       # TensorRT < 8.4
        config.max_workspace_size = ws

    used_fp16 = False
    if tcfg.get("fp16", True):
        if not builder.platform_has_fast_fp16:
            raise RuntimeError("Platform reports no fast FP16 support -- unexpected on Orin.")
        config.set_flag(trt.BuilderFlag.FP16)
        used_fp16 = True

    # fixed batch 1 for the latency-critical single-image path
    profile = builder.create_optimization_profile()
    size = int(cfg["model"]["image_size"])
    profile.set_shape("input", (1, 3, size, size), (1, 3, size, size), (8, 3, size, size))
    config.add_optimization_profile(profile)

    start = time.perf_counter()
    serialized = builder.build_serialized_network(network, config)
    build_s = time.perf_counter() - start
    if serialized is None:
        raise RuntimeError("build_serialized_network returned None (build failed).")

    engine_path.write_bytes(serialized)
    return engine_path, build_s, used_fp16


def main():
    cfg = load_cfg()
    check_hardware_filled(cfg)
    set_seed(int(cfg["seed"]))

    if not torch.cuda.is_available():
        raise SystemExit("CUDA not available -- this must run on the Jetson.")
    device = torch.device("cuda")

    print("[1/2] exporting ONNX ...")
    model = build_model(cfg, device)
    onnx_path, export_s = export_onnx(model, cfg, device)
    onnx_mb = onnx_path.stat().st_size / 1e6
    print(f"      {export_s:.2f} s, {onnx_mb:.2f} MB")

    print("[2/2] building TensorRT engine ...")
    try:
        engine_path, build_s, used_fp16 = build_engine(onnx_path, cfg)
    except Exception as e:
        write_metrics("01_export_build_trt_FAILED.json", {
            "onnx_export_seconds": round(export_s, 3),
            "onnx_size_mb": round(onnx_mb, 3),
            "tensorrt_build": "FAILED",
            "failure": f"{type(e).__name__}: {e}",
        }, cfg)
        raise SystemExit(
            f"\nTensorRT build FAILED: {type(e).__name__}: {e}\n"
            "Not falling back -- a fallback number must never be reported as TensorRT.\n"
            "Check that JetPack's tensorrt python module is importable.")

    engine_mb = engine_path.stat().st_size / 1e6
    print(f"      {build_s:.2f} s, {engine_mb:.2f} MB, fp16={used_fp16}")

    write_metrics("01_export_build_trt.json", {
        "onnx_export_seconds": round(export_s, 3),
        "onnx_size_mb": round(onnx_mb, 3),
        "tensorrt_build_seconds": round(build_s, 3),
        "engine_size_mb": round(engine_mb, 3),
        "fp16_enabled": used_fp16,
        "engine_path": str(engine_path.relative_to(BUNDLE_ROOT)),
    }, cfg)
    print("\nSTEP 1 done.")


if __name__ == "__main__":
    main()
