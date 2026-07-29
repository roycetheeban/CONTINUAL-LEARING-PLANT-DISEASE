"""Simulate realistic greenhouse sensor traces (turmeric context).
Hourly resolution: diurnal cycle + recurring seasonal (wet/dry) cycle + AR(1)
drift + sensor noise + dropout. The seasonal cycle recurs across the whole
timeline so both the train and test blocks span the full range of conditions
(a fair future-holdout, not an extrapolation to an unseen regime).
SIMULATED data -- disclosed in the paper."""
import numpy as np
import pandas as pd

from common_env import CONFIG_PATH, load_cfg, set_seed


def _series(n, base, diurnal_amp, seasonal_amp, noise_sd, ar1, lo, hi,
            seasonal_period_h, rng, phase=0.0):
    """One variable over n hours: diurnal + recurring seasonal + AR(1) noise."""
    hours = np.arange(n)
    diurnal = diurnal_amp * np.sin(2 * np.pi * (hours % 24) / 24.0 + phase)
    seasonal = seasonal_amp * np.sin(2 * np.pi * hours / seasonal_period_h)
    e = rng.normal(0, noise_sd, n)
    ar = np.zeros(n)
    for t in range(1, n):
        ar[t] = ar1 * ar[t - 1] + e[t]
    return np.clip(base + diurnal + seasonal + ar, lo, hi)


def _p(d):
    return dict(base=d["base"], diurnal_amp=d["diurnal_amp"], seasonal_amp=d["seasonal_amp"],
                noise_sd=d["noise_sd"], ar1=d["ar1"], lo=d["lo"], hi=d["hi"])


def simulate(cfg: dict) -> pd.DataFrame:
    set_seed(cfg["seed"])
    rng = np.random.default_rng(cfg["seed"])
    days = int(cfg["sim"]["days"])
    n = days * 24
    sp_h = int(cfg["sim"]["seasonal_period_days"]) * 24
    env = cfg["environment"]

    temp = _series(n, **_p(env["temp"]), seasonal_period_h=sp_h, rng=rng, phase=0.0)
    rh = _series(n, **_p(env["humidity"]), seasonal_period_h=sp_h, rng=rng, phase=np.pi)
    soil = _series(n, **_p(env["soil"]), seasonal_period_h=sp_h, rng=rng, phase=np.pi)
    co2 = _series(n, **_p(env["co2"]), seasonal_period_h=sp_h, rng=rng, phase=np.pi)

    idx = np.arange(n)
    df = pd.DataFrame({
        "t_hour": idx, "day": idx // 24 + 1, "hour": idx % 24,
        "temp_C": np.round(temp, 2), "rh_pct": np.round(rh, 2),
        "soil_pct": np.round(soil, 2), "co2_ppm": np.round(co2, 1),
    })
    drop = float(env["dropout_frac"])
    for col in ["temp_C", "rh_pct", "soil_pct", "co2_ppm"]:
        df.loc[rng.random(n) < drop, col] = np.nan
    return df


if __name__ == "__main__":
    cfg = load_cfg(CONFIG_PATH)
    df = simulate(cfg)
    print(df[["temp_C", "rh_pct", "soil_pct", "co2_ppm"]].describe().round(2).to_string())
    print("\nrows:", len(df), "| days:", int(df['day'].max()))
