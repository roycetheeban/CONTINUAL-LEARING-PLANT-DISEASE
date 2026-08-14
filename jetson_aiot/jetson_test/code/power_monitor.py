"""Power and thermal sampling for the Jetson, straight off sysfs.

The bundle originally declared these unmeasurable ("server/headless access only").
They are not: the INA3221 rails and every thermal zone are readable from sysfs with
no physical access and no root. For an edge-deployment paper, power is close to a
mandatory metric -- a claim that a device runs a workload invites the question of
what it costs to run it.

Rails on Orin Nano (5 V board):
    VDD_IN            total board power -- the number to report
    VDD_CPU_GPU_CV    CPU + GPU + CV accelerators
    VDD_SOC           SoC / memory / fabric

Sampling runs on a background thread, so it adds no latency to the timed path.
Energy is integrated trapezoidally over the actual sample timestamps rather than
assuming a fixed interval, because thread scheduling is not exact.
"""
import threading
import time
from pathlib import Path

I2C_GLOB = "/sys/bus/i2c/drivers/ina3221*/*/hwmon/hwmon*"
THERMAL_GLOB = "/sys/devices/virtual/thermal/thermal_zone*"


def _read(p: Path, cast=int, default=None):
    try:
        return cast(p.read_text().strip())
    except Exception:
        return default


def discover_rails() -> list[dict]:
    """[{label, volt_path, curr_path}] for each labelled INA3221 channel."""
    rails = []
    for hwmon in sorted(Path("/").glob(I2C_GLOB.lstrip("/"))):
        for i in (1, 2, 3):
            lbl = hwmon / f"in{i}_label"
            v, c = hwmon / f"in{i}_input", hwmon / f"curr{i}_input"
            if lbl.exists() and v.exists() and c.exists():
                name = _read(lbl, str)
                if name:
                    rails.append({"label": name, "volt": v, "curr": c})
    return rails


def discover_zones() -> list[dict]:
    """[{type, path}] for thermal zones reporting a plausible temperature."""
    zones = []
    for z in sorted(Path("/").glob(THERMAL_GLOB.lstrip("/"))):
        t, temp = z / "type", z / "temp"
        if t.exists() and temp.exists():
            val = _read(temp)
            if val is not None and val > 0:          # cv*-thermal read 0 when idle
                zones.append({"type": _read(t, str), "path": temp})
    return zones


def available() -> bool:
    return bool(discover_rails())


class PowerSampler:
    """Background sampler. Use as a context manager, then read .summary()."""

    def __init__(self, interval_s: float = 0.1):
        self.interval = float(interval_s)
        self.rails = discover_rails()
        self.zones = discover_zones()
        self._stop = threading.Event()
        self._thread = None
        self.t: list[float] = []
        self.watts: dict[str, list[float]] = {r["label"]: [] for r in self.rails}
        self.temps: dict[str, list[float]] = {z["type"]: [] for z in self.zones}

    def _sample_once(self):
        self.t.append(time.perf_counter())
        for r in self.rails:
            mv = _read(r["volt"], int, 0) or 0
            ma = _read(r["curr"], int, 0) or 0
            self.watts[r["label"]].append(mv * ma / 1e6)      # mV * mA -> W
        for z in self.zones:
            mc = _read(z["path"], int, 0) or 0
            self.temps[z["type"]].append(mc / 1000.0)         # milli-C -> C

    def _run(self):
        while not self._stop.is_set():
            self._sample_once()
            self._stop.wait(self.interval)

    def __enter__(self):
        if not self.rails:
            raise RuntimeError("no INA3221 rails found -- power sampling unavailable")
        self._sample_once()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)
        self._sample_once()
        return False

    def _energy_j(self, label: str) -> float:
        """Trapezoidal integration over real timestamps (thread timing is not exact)."""
        w, t = self.watts[label], self.t
        n = min(len(w), len(t))
        return sum((w[i] + w[i + 1]) / 2.0 * (t[i + 1] - t[i]) for i in range(n - 1))

    def summary(self, work_items: int = 0, item_name: str = "item") -> dict:
        if not self.t:
            return {"error": "no samples"}
        dur = self.t[-1] - self.t[0]
        out = {"duration_s": round(dur, 3), "samples": len(self.t),
               "sample_interval_s": self.interval, "rails_w": {}, "energy_j": {},
               "thermal_c": {}}
        for label, w in self.watts.items():
            if not w:
                continue
            out["rails_w"][label] = {"mean": round(sum(w) / len(w), 4),
                                     "peak": round(max(w), 4),
                                     "min": round(min(w), 4)}
            out["energy_j"][label] = round(self._energy_j(label), 4)
        for zt, c in self.temps.items():
            if c:
                out["thermal_c"][zt] = {"start": round(c[0], 2), "peak": round(max(c), 2),
                                        "end": round(c[-1], 2),
                                        "rise": round(max(c) - c[0], 2)}
        if work_items > 0 and "VDD_IN" in out["energy_j"]:
            out["per_item"] = {
                "item": item_name, "count": work_items,
                "energy_mj_per_item": round(out["energy_j"]["VDD_IN"] / work_items * 1000, 4),
                "ms_per_item": round(dur / work_items * 1000, 4)}
        return out


def subtract_idle(load: dict, idle: dict) -> dict:
    """Marginal cost above idle. Board total is reported separately -- both matter:
    total answers 'what does the device draw', marginal answers 'what does the
    workload cost'."""
    out = {}
    for label, v in load.get("rails_w", {}).items():
        base = idle.get("rails_w", {}).get(label, {}).get("mean")
        if base is not None:
            out[label] = {"mean_above_idle_w": round(v["mean"] - base, 4),
                          "peak_above_idle_w": round(v["peak"] - base, 4)}
    return out
