"""Export the deployed classifier to ONNX and build an FP16 "compiled" artifact.

Primary path: TensorRT FP16 engine (what the paper's deployment design specifies).
Fallback: PyTorch native FP16 (`model.half()`), used only if TensorRT is unavailable
or fails to build on this machine. Whichever path actually ran is recorded in
metrics.json and MUST be reflected accurately wherever the paper describes it --
never report a TensorRT number that was actually produced by the fallback."""
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
from torchvision import models

from common_ondevice import CONFIG_PATH, ensure_dirs, load_cfg, set_seed, OUTPUT_DIR, REPO_ROOT


def build_model(cfg: dict, device: torch.device) -> nn.Module:
    m = cfg["model"]
    model = models.mobilenet_v3_small(weights=None)
    model.classifier[3] = nn.Linear(1024, int(m["num_classes"]))
    model = model.to(device)
    state = torch.load(REPO_ROOT / m["base_checkpoint"], map_location=device, weights_only=True)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


def export_onnx(model, cfg, device) -> tuple[Path, float]:
    onnx_path = REPO_ROOT / cfg["export"]["onnx_path"]
    ensure_dirs(onnx_path.parent)
    size = int(cfg["model"]["image_size"])
    dummy = torch.randn(1, 3, size, size, device=device)

    start = time.perf_counter()
    torch.onnx.export(
        model, dummy, str(onnx_path),
        input_names=["input"], output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=int(cfg["export"]["opset"]),
    )
    elapsed = time.perf_counter() - start
    return onnx_path, elapsed


def try_tensorrt(onnx_path: Path, cfg: dict):
    """Returns (engine_path, build_seconds) or raises."""
    import tensorrt as trt

    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)

    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            errs = [str(parser.get_error(i)) for i in range(parser.num_errors)]
            raise RuntimeError("ONNX parse failed: " + "; ".join(errs))

    config = builder.create_builder_config()
    if not builder.platform_has_fast_fp16:
        raise RuntimeError("platform reports no fast FP16 support")
    config.set_flag(trt.BuilderFlag.FP16)

    profile = builder.create_optimization_profile()
    size = int(cfg["model"]["image_size"])
    profile.set_shape("input", (1, 3, size, size), (8, 3, size, size), (32, 3, size, size))
    config.add_optimization_profile(profile)

    start = time.perf_counter()
    serialized = builder.build_serialized_network(network, config)
    build_s = time.perf_counter() - start
    if serialized is None:
        raise RuntimeError("TensorRT returned no serialized engine")

    engine_path = OUTPUT_DIR / "model_fp16.engine"
    with open(engine_path, "wb") as f:
        f.write(serialized)
    return engine_path, build_s


def main():
    cfg = load_cfg(CONFIG_PATH)
    set_seed(cfg["seed"])
    ensure_dirs(OUTPUT_DIR)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    model = build_model(cfg, device)
    onnx_path, onnx_s = export_onnx(model, cfg, device)
    print(f"ONNX export: {onnx_s:.3f}s -> {onnx_path}")

    result = {
        "onnx_export_sec": round(onnx_s, 4),
        "onnx_path": str(onnx_path),
        "onnx_size_mb": round(onnx_path.stat().st_size / (1024 ** 2), 3),
    }

    if cfg["compile"].get("attempt_tensorrt", True):
        try:
            engine_path, build_s = try_tensorrt(onnx_path, cfg)
            result.update({
                "compiled_path_used": "tensorrt_fp16",
                "engine_build_sec": round(build_s, 4),
                "engine_path": str(engine_path),
                "engine_size_mb": round(engine_path.stat().st_size / (1024 ** 2), 3),
            })
            print(f"TensorRT FP16 engine built in {build_s:.3f}s")
        except Exception as e:  # noqa: BLE001 - any failure means we fall back, but must be logged
            result.update({
                "compiled_path_used": "pytorch_fp16_fallback",
                "tensorrt_failure_reason": f"{type(e).__name__}: {e}",
                "tensorrt_install_note": (
                    "TensorRT could not be installed in this environment: the pip "
                    "packages (tensorrt / tensorrt-cu12) pull nvidia-cuda-runtime-cu13, "
                    "whose wheel fails to build against this machine's CUDA 12.1 torch "
                    "build on Windows. Not a code defect -- an environment constraint. "
                    "ONNX Runtime GPU (CUDAExecutionProvider) is benchmarked instead as "
                    "the compiled-graph comparison point."
                ),
                "engine_build_sec": None,
            })
            print(f"TensorRT unavailable/failed -> falling back to PyTorch FP16. Reason: {e}")
    else:
        result.update({"compiled_path_used": "pytorch_fp16_fallback",
                        "tensorrt_failure_reason": "disabled in config"})

    with open(OUTPUT_DIR / "export_compile_metrics.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
