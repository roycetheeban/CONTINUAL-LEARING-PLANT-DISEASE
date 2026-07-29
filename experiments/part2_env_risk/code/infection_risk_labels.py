"""Turmeric-parameterised infection-risk oracle (label generator).

Implements the Magarey, Sutton & Thayer (2005) generic foliar-fungal infection
model [Phytopathology 95:92-100, doi 10.1094/PHYTO-95-0092]:

    w(T) = ((Tmax - T)/(Tmax - Topt)) *
           ((T - Tmin)/(Topt - Tmin)) ** ((Topt - Tmin)/(Tmax - Topt))     (0..1)

parameterised for turmeric leaf blotch (Taphrina maculans) / leaf spot
(Colletotrichum) per Gohel et al. (2022) [Indian Phytopathol. 75:487-491].

Daily severity: from leaf-wetness hours (RH >= threshold) and the temperature
response during the wet period; accumulated over a trailing window and binned to
Low / Medium / High. This is the LABEL ORACLE only -- the models never see w(T)
or the wet-hour count directly (see features.py), so the task is not circular.
"""
import numpy as np
import pandas as pd


def temp_response(T, T_min, T_opt, T_max):
    """Magarey (2005) beta temperature-response function, clipped to [0, 1]."""
    T = np.asarray(T, dtype=float)
    w = np.zeros_like(T)
    m = (T > T_min) & (T < T_max)
    expo = (T_opt - T_min) / (T_max - T_opt)
    base = ((T_max - T[m]) / (T_max - T_opt)) * \
           (((T[m] - T_min) / (T_opt - T_min)) ** expo)
    w[m] = np.clip(base, 0.0, 1.0)
    return w


def daily_severity(df_hourly: pd.DataFrame, im: dict) -> pd.DataFrame:
    """Per-day disease-severity value in ~[0,4] from wet hours x temp response."""
    d = df_hourly.copy()
    # impute short gaps for the oracle (dropout affects the models, not the truth)
    d[["temp_C", "rh_pct"]] = d[["temp_C", "rh_pct"]].interpolate(limit=6).ffill().bfill()
    d["leaf_wet"] = (d["rh_pct"] >= im["rh_wet_threshold"]).astype(int)
    d["wT"] = temp_response(d["temp_C"].values, im["T_min"], im["T_opt"], im["T_max"])

    rows = []
    for day, g in d.groupby("day"):
        wet_hours = int(g["leaf_wet"].sum())
        if wet_hours > 0:
            mean_w = float(g.loc[g["leaf_wet"] == 1, "wT"].mean())
        else:
            mean_w = 0.0
        # required wetness duration grows away from the temperature optimum
        required = im["wetness_min_hours"] / max(mean_w, 1e-3)
        if wet_hours >= required and mean_w > 0:
            wet_eff = min(wet_hours, im["wetness_max_hours"]) / im["wetness_max_hours"]
            dsv = 4.0 * mean_w * wet_eff          # ~[0,4], TOMCAST/FAST-style DSV
        else:
            dsv = 0.0
        rows.append({"day": int(day), "wet_hours": wet_hours,
                     "mean_wT": round(mean_w, 3), "dsv": round(dsv, 3)})
    return pd.DataFrame(rows)


def label_days(df_hourly: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Oracle up to accumulation: hourly sensors -> per-day accumulated severity
    (no risk binning yet -- the caller calibrates thresholds on the train period)."""
    im = cfg["infection_model"]
    dsv = daily_severity(df_hourly, im)
    w = int(im["accum_window_days"])
    dsv = dsv.copy()
    dsv["accum_sev"] = dsv["dsv"].rolling(window=w, min_periods=1).sum().round(3)
    return dsv


def calibrate_thresholds(accum_train, low_pct, med_pct):
    """Low|Med and Med|High cut-points from the TRAIN-period accumulated severity."""
    return (round(float(np.percentile(accum_train, low_pct)), 3),
            round(float(np.percentile(accum_train, med_pct)), 3))


def bin_risk(dsv_df: pd.DataFrame, low, med) -> pd.DataFrame:
    a = dsv_df["accum_sev"].values
    dsv_df = dsv_df.copy()
    dsv_df["risk"] = np.where(a < low, "Low", np.where(a < med, "Medium", "High"))
    return dsv_df


if __name__ == "__main__":
    from common_env import CONFIG_PATH, load_cfg
    from simulate_sensors import simulate
    cfg = load_cfg(CONFIG_PATH)
    df = simulate(cfg)
    lab = label_days(df, cfg)
    im = cfg["infection_model"]
    tr = lab[lab["day"] <= cfg["split"]["train_end_day"]]["accum_sev"]
    low, med = calibrate_thresholds(tr, im["calib_low_pct"], im["calib_med_pct"])
    lab = bin_risk(lab, low, med)
    print("thresholds Low|Med=%.2f  Med|High=%.2f" % (low, med))
    print("accum_sev range:", round(lab['accum_sev'].min(), 2), "..", round(lab['accum_sev'].max(), 2))
    print("overall risk dist:\n", lab["risk"].value_counts(normalize=True).round(3).to_string())
