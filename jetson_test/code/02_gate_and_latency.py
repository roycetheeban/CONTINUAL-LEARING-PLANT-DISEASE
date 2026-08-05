"""STEP 2 -- the conversion gate, measured through the REAL TensorRT engine.

This is the measurement the paper's deployment-gate argument actually requires.
The argument is: "a system promoting a converted artifact into production cannot
know it is equivalent without measuring the artifact it actually deploys." On the
laptop that artifact was an ONNX Runtime graph (delta 0.00), which made the gate
look decorative. Here it is the genuine TensorRT FP16 engine.

A NON-ZERO delta would strengthen the paper -- it would make the gate empirically
justified rather than only epistemically justified. Either result is reportable;
do not tune anything to chase one.

Also measures classification latency: PyTorch FP32 eager vs the TRT FP16 engine,
per image and per 8-image capture set.
"""
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common_jetson import (build_eval_transform, build_model,  # noqa: E402
                           check_hardware_filled, load_cfg, load_test_tensors,
                           resolve, set_seed, write_metrics)


# --------------------------------------------------------------- TRT runner
class TRTRunner:
    """Minimal TensorRT execution wrapper (explicit batch, single input/output)."""

    def __init__(self, engine_path: Path):
        import tensorrt as trt
        self.trt = trt
        logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(logger)
        self.engine = runtime.deserialize_cuda_engine(engine_path.read_bytes())
        if self.engine is None:
            raise RuntimeError(f"Failed to deserialize engine at {engine_path}")
        self.ctx = self.engine.create_execution_context()
        self.in_name, self.out_name = self._io_names()

    def _io_names(self):
        e = self.engine
        if hasattr(e, "num_io_tensors"):                       # TensorRT >= 8.5
            names = [e.get_tensor_name(i) for i in range(e.num_io_tensors)]
            ins = [n for n in names if e.get_tensor_mode(n) == self.trt.TensorIOMode.INPUT]
            outs = [n for n in names if e.get_tensor_mode(n) == self.trt.TensorIOMode.OUTPUT]
            return ins[0], outs[0]
        names = [e.get_binding_name(i) for i in range(e.num_bindings)]
        ins = [n for n in names if e.binding_is_input(n)]
        outs = [n for n in names if not e.binding_is_input(n)]
        return ins[0], outs[0]

    def infer(self, x: torch.Tensor) -> torch.Tensor:
        """x: float32 CUDA tensor (N,3,H,W). Returns logits as float32 CUDA tensor."""
        x = x.half().contiguous()
        n = x.shape[0]
        if hasattr(self.ctx, "set_input_shape"):
            self.ctx.set_input_shape(self.in_name, tuple(x.shape))
        out = torch.empty((n, 5), dtype=torch.float16, device="cuda")
        if hasattr(self.ctx, "set_tensor_address"):
            self.ctx.set_tensor_address(self.in_name, x.data_ptr())
            self.ctx.set_tensor_address(self.out_name, out.data_ptr())
            self.ctx.execute_async_v3(torch.cuda.current_stream().cuda_stream)
        else:
            self.ctx.execute_v2([int(x.data_ptr()), int(out.data_ptr())])
        torch.cuda.synchronize()
        return out.float()


# ---------------------------------------------------------------- helpers
def accuracy_torch(model, x, y, batch=64, half=False):
    correct = 0
    with torch.no_grad():
        for i in range(0, x.shape[0], batch):
            xb = x[i:i + batch]
            if half:
                xb = xb.half()
            correct += (model(xb).argmax(1) == y[i:i + batch]).sum().item()
    return correct / x.shape[0]


def accuracy_trt(runner, x, y, batch=8):
    correct = 0
    for i in range(0, x.shape[0], batch):
        logits = runner.infer(x[i:i + batch])
        correct += (logits.argmax(1) == y[i:i + batch]).sum().item()
    return correct / x.shape[0]


def time_call(fn, warmup, iters):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1000.0)
    a = np.array(ts)
    return {"mean_ms": round(float(a.mean()), 3),
            "std_ms": round(float(a.std()), 3),
            "p50_ms": round(float(np.percentile(a, 50)), 3),
            "p95_ms": round(float(np.percentile(a, 95)), 3)}


def main():
    cfg = load_cfg()
    check_hardware_filled(cfg)
    set_seed(int(cfg["seed"]))
    if not torch.cuda.is_available():
        raise SystemExit("CUDA not available -- this must run on the Jetson.")
    device = torch.device("cuda")

    engine_path = resolve(cfg["trt"]["engine_path"])
    if not engine_path.exists():
        raise SystemExit(f"No engine at {engine_path}. Run 01_export_build_trt.py first.")

    print("loading test set ...")
    tf = build_eval_transform(cfg)
    x, y = load_test_tensors(cfg, tf, device)
    n = x.shape[0]
    print(f"  {n} images")

    print("building models ...")
    fp32 = build_model(cfg, device, half=False)
    fp16 = build_model(cfg, device, half=True)
    runner = TRTRunner(engine_path)

    print("accuracy ...")
    acc_fp32 = accuracy_torch(fp32, x, y, half=False)
    acc_fp16 = accuracy_torch(fp16, x, y, half=True)
    acc_trt = accuracy_trt(runner, x, y, batch=int(cfg["data"]["capture_set_size"]))
    print(f"  torch fp32 {acc_fp32*100:.2f}% | torch fp16 {acc_fp16*100:.2f}% | TRT fp16 {acc_trt*100:.2f}%")

    print("latency ...")
    lcfg = cfg["latency"]
    w, it = int(lcfg["warmup_iters"]), int(lcfg["timed_iters"])
    k = int(cfg["data"]["capture_set_size"])
    one, cap = x[:1], x[:k]

    with torch.no_grad():
        lat = {
            "torch_fp32_single": time_call(lambda: fp32(one), w, it),
            "torch_fp32_capture_set": time_call(lambda: fp32(cap), w, it),
            "trt_fp16_single": time_call(lambda: runner.infer(one), w, it),
            "trt_fp16_capture_set": time_call(lambda: runner.infer(cap), w, it),
        }
    for kk, v in lat.items():
        print(f"  {kk:26s} {v['mean_ms']:7.3f} ms")

    speedup = lat["torch_fp32_single"]["mean_ms"] / lat["trt_fp16_single"]["mean_ms"]

    write_metrics("02_gate_and_latency.json", {
        "test_set_size": n,
        "accuracy": {
            "torch_fp32": round(acc_fp32 * 100, 4),
            "torch_fp16": round(acc_fp16 * 100, 4),
            "tensorrt_fp16": round(acc_trt * 100, 4),
            "delta_trt_minus_fp32_pp": round((acc_trt - acc_fp32) * 100, 4),
            "delta_images": int(round((acc_trt - acc_fp32) * n)),
        },
        "latency_ms": lat,
        "trt_speedup_vs_fp32_single": round(speedup, 3),
        "capture_set_size": k,
    }, cfg)
    print("\nSTEP 2 done.")


if __name__ == "__main__":
    main()
