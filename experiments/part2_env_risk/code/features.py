"""Aggregate the hourly stream into one feature row per 24h capture window.

Features are deliberately RAW-ish aggregates the device can compute directly --
NOT the oracle internals (no w(T), no >=90% wet-hour count, no accumulated
severity). This preserves a genuine learning gap so the task is not circular.
"""
import numpy as np
import pandas as pd

FEATURE_COLS = [
    "temp_mean", "temp_min", "temp_max", "temp_range", "hours_temp_in_band",
    "rh_mean", "rh_max", "rh_min", "rh_std", "hours_rh_above_85", "rh_trend",
    "soil_mean", "co2_mean",
]


def _slope(y):
    y = np.asarray(y, dtype=float)
    x = np.arange(len(y))
    ok = ~np.isnan(y)
    if ok.sum() < 2:
        return 0.0
    return float(np.polyfit(x[ok], y[ok], 1)[0])


def make_windows(df_hourly: pd.DataFrame, labels_by_day: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    fc = cfg["features"]
    win = int(fc["window_hours"])
    band_lo, band_hi = fc["temp_band_lo"], fc["temp_band_hi"]
    rh_hi = fc["rh_high_threshold"]
    risk_by_day = dict(zip(labels_by_day["day"], labels_by_day["risk"]))

    d = df_hourly.set_index("t_hour").sort_index()
    n = int(df_hourly["t_hour"].max()) + 1
    days = int(df_hourly["day"].max())

    rows = []
    for day in range(1, days + 1):
        for ch in fc["capture_hours"]:
            cap = (day - 1) * 24 + ch          # absolute capture hour
            start = cap - win + 1
            if start < 0:
                continue                        # not enough trailing history
            w = d.loc[start:cap]
            if w.empty:
                continue
            temp, rh = w["temp_C"].values, w["rh_pct"].values
            soil, co2 = w["soil_pct"].values, w["co2_ppm"].values
            feat = {
                "day": day, "capture_hour": ch,
                "temp_mean": np.nanmean(temp), "temp_min": np.nanmin(temp),
                "temp_max": np.nanmax(temp), "temp_range": np.nanmax(temp) - np.nanmin(temp),
                "hours_temp_in_band": int(np.nansum((temp >= band_lo) & (temp <= band_hi))),
                "rh_mean": np.nanmean(rh), "rh_max": np.nanmax(rh), "rh_min": np.nanmin(rh),
                "rh_std": np.nanstd(rh), "hours_rh_above_85": int(np.nansum(rh >= rh_hi)),
                "rh_trend": _slope(rh),
                "soil_mean": np.nanmean(soil), "co2_mean": np.nanmean(co2),
                "risk": risk_by_day.get(day, "Low"),
            }
            rows.append(feat)
    out = pd.DataFrame(rows)
    out[FEATURE_COLS] = out[FEATURE_COLS].round(3)
    return out


def add_label_noise(df: pd.DataFrame, frac: float, seed: int) -> pd.DataFrame:
    """Flip a fraction of labels to a random *other* class (diagnostic uncertainty)."""
    rng = np.random.default_rng(seed + 7)
    classes = np.array(["Low", "Medium", "High"])
    df = df.copy()
    flip = rng.random(len(df)) < frac
    for i in np.where(flip)[0]:
        others = classes[classes != df.iloc[i]["risk"]]
        df.iat[i, df.columns.get_loc("risk")] = rng.choice(others)
    return df


if __name__ == "__main__":
    from common_env import CONFIG_PATH, load_cfg
    from simulate_sensors import simulate
    from infection_risk_labels import bin_risk, calibrate_thresholds, label_days
    cfg = load_cfg(CONFIG_PATH)
    df = simulate(cfg)
    lab = label_days(df, cfg)
    im, sp = cfg["infection_model"], cfg["split"]
    low, med = calibrate_thresholds(lab[lab["day"] <= sp["train_end_day"]]["accum_sev"],
                                    im["calib_low_pct"], im["calib_med_pct"])
    lab = bin_risk(lab, low, med)
    win = make_windows(df, lab, cfg)
    win = add_label_noise(win, cfg["labels"]["noise_frac"], cfg["seed"])
    print("windows:", len(win))
    print(win[["day", "capture_hour", "rh_mean", "hours_rh_above_85", "temp_mean", "risk"]].head(8).to_string(index=False))
    print("\nrisk dist:", win["risk"].value_counts(normalize=True).round(3).to_dict())
