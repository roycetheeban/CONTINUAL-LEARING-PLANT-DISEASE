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
def _trt_to_torch_dtype(trt) -> dict:
    """TensorRT DataType -> torch dtype. Optional members guarded by version."""
    m = {
        trt.DataType.FLOAT: torch.float32,
        trt.DataType.HALF: torch.float16,
        trt.DataType.INT8: torch.int8,
        trt.DataType.INT32: torch.int32,
        trt.DataType.BOOL: torch.bool,
    }
    for trt_name, torch_name in (("INT64", "int64"), ("BF16", "bfloat16"),
                                 ("UINT8", "uint8")):
        if hasattr(trt.DataType, trt_name):
            m[getattr(trt.DataType, trt_name)] = getattr(torch, torch_name)
    return m


class TRTRunner:
    """Minimal TensorRT execution wrapper (explicit batch, single input/output).

    I/O dtypes are read FROM THE ENGINE, never assumed.

    BuilderFlag.FP16 permits FP16 *kernels* internally; it does NOT change the
    network's I/O dtypes, which come from the ONNX graph and stay FLOAT (fp32).
    Verified on this unit: building the step-1 graph with the FP16 flag yields
    input/output tensors of DataType.FLOAT.

    An earlier version hardcoded `x.half()` and an fp16 output buffer. TensorRT
    then reinterpreted fp16 bytes as fp32 (and vice versa), producing garbage
    logits and a TRT accuracy near chance -- i.e. a spectacular ~-78 pp
    "conversion gate delta" that is pure plumbing error. Since a non-zero delta
    is exactly the result this experiment hopes for, that failure mode is far
    more dangerous than a crash. Do not reintroduce a hardcoded dtype here.
    """

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

        dmap = _trt_to_torch_dtype(trt)
        self.in_trt_dtype = self._tensor_dtype(self.in_name)
        self.out_trt_dtype = self._tensor_dtype(self.out_name)
        for nm, dt in ((self.in_name, self.in_trt_dtype), (self.out_name, self.out_trt_dtype)):
            if dt not in dmap:
                raise RuntimeError(f"Unmapped TensorRT dtype {dt} on tensor '{nm}'")
        self.in_dtype = dmap[self.in_trt_dtype]
        self.out_dtype = dmap[self.out_trt_dtype]

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

    def _tensor_dtype(self, name):
        e = self.engine
        if hasattr(e, "get_tensor_dtype"):                     # TensorRT >= 8.5
            return e.get_tensor_dtype(name)
        return e.get_binding_dtype(e.get_binding_index(name))  # legacy

    def io_spec(self) -> dict:
        """Recorded in the metrics file so the gate delta is auditable."""
        return {"input_name": self.in_name, "input_trt_dtype": str(self.in_trt_dtype),
                "output_name": self.out_name, "output_trt_dtype": str(self.out_trt_dtype),
                "note": "dtypes read from the engine, not assumed"}

    def infer(self, x: torch.Tensor) -> torch.Tensor:
        """x: float32 CUDA tensor (N,3,H,W). Returns logits as float32 CUDA tensor."""
        x = x.to(self.in_dtype).contiguous()
        if hasattr(self.ctx, "set_input_shape"):
            self.ctx.set_input_shape(self.in_name, tuple(x.shape))
            out_shape = tuple(self.ctx.get_tensor_shape(self.out_name))
        else:
            out_shape = (x.shape[0], int(self.engine.get_binding_shape(
                self.engine.get_binding_index(self.out_name))[-1]))
        out = torch.empty(out_shape, dtype=self.out_dtype, device="cuda")
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

    print(f"  engine I/O: {runner.in_name} {runner.in_trt_dtype} -> "
          f"{runner.out_name} {runner.out_trt_dtype}")

    print("accuracy ...")
    acc_fp32 = accuracy_torch(fp32, x, y, half=False)
    acc_fp16 = accuracy_torch(fp16, x, y, half=True)
    acc_trt = accuracy_trt(runner, x, y, batch=int(cfg["data"]["capture_set_size"]))
    print(f"  torch fp32 {acc_fp32*100:.2f}% | torch fp16 {acc_fp16*100:.2f}% | TRT fp16 {acc_trt*100:.2f}%")

    # ---- plumbing sanity check --------------------------------------------
    # A conversion gate delta is a scientific finding; a dtype/binding error is
    # a bug. They are indistinguishable from the delta alone, and the bug
    # happens to produce the large non-zero delta this experiment hopes to see.
    # Flag implausible deltas so one can never be reported as the other.
    # This does NOT tune anything -- it only labels the result.
    delta_pp = (acc_trt - acc_fp32) * 100
    chance_pp = 100.0 / int(cfg["model"]["num_classes"])
    implausible = abs(delta_pp) > 5.0
    near_chance = acc_trt * 100 < chance_pp + 10.0
    if implausible or near_chance:
        print("\n" + "!" * 66)
        print("  SUSPECT RESULT -- treat as a bug until proven otherwise.")
        print(f"  TRT accuracy {acc_trt*100:.2f}% vs FP32 {acc_fp32*100:.2f}%  "
              f"(delta {delta_pp:+.2f} pp)")
        if near_chance:
            print(f"  TRT accuracy is near chance ({chance_pp:.1f}% for "
                  f"{cfg['model']['num_classes']} classes) -- this is the signature")
            print("  of an I/O dtype or binding mismatch, NOT a conversion-gate finding.")
        print("  FP16 layer fusion does not move accuracy by more than ~1-2 pp on a")
        print("  model this small. Verify the engine I/O dtypes printed above before")
        print("  reporting this number anywhere.")
        print("!" * 66 + "\n")

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
        "engine_io": runner.io_spec(),
        "sanity": {
            "plausible": not (implausible or near_chance),
            "abs_delta_pp": round(abs(delta_pp), 4),
            "chance_accuracy_pp": round(chance_pp, 2),
            "verdict": ("SUSPECT -- verify engine I/O dtypes before reporting"
                        if (implausible or near_chance) else
                        "delta within the range FP16 fusion can plausibly explain"),
        },
        "latency_ms": lat,
        "trt_speedup_vs_fp32_single": round(speedup, 3),
        "capture_set_size": k,
    }, cfg)
    print("\nSTEP 2 done.")


if __name__ == "__main__":
    main()
