"""Construct the E2 fusion-evaluation episodes.

Each episode = a 5-day window. Per day, real test-pool images are sampled and fed
through the (already-cached) classifier predictions to produce a per-class leaf
count -- the target-class day-count SHAPE (rising/stable/falling) is engineered per
config, but the count itself is the model's real prediction tally, including its
real ~1.5% error rate. The environmental risk state is a REAL E1 test-period window
(day > val_end_day), except for the sensor_failure_rising category where risk is
deliberately marked unavailable.

Ground truth per episode category is assigned from an INDEPENDENTLY reasoned true
cause (see README), not read off the fusion rule table, so vision-only / sensor-only
/ fusion can genuinely disagree and be scored against the same target."""
import json
import random
from pathlib import Path

import pandas as pd

from common_fusion import CONFIG_PATH, ensure_dirs, load_cfg, set_seed, OUTPUT_DIR, REPO_ROOT


def load_pool_predictions(cfg) -> pd.DataFrame:
    df = pd.read_csv(OUTPUT_DIR / "test_pool_predictions.csv")
    return df


def load_risk_windows(cfg) -> pd.DataFrame:
    df = pd.read_csv(REPO_ROOT / cfg["env_risk"]["windows_csv"])
    start_day = cfg["env_risk"]["test_period_start_day"]
    return df[df["day"] >= start_day].reset_index(drop=True)


def risk_bin_to_rows(risk_windows: pd.DataFrame, risk_bin: str) -> pd.DataFrame:
    if risk_bin == "Low_Medium":
        return risk_windows[risk_windows["risk"].isin(["Low", "Medium"])]
    if risk_bin == "NA":
        return None
    return risk_windows[risk_windows["risk"] == risk_bin]


def sample_day_counts(rng: random.Random, cfg: dict, trend: str) -> list[int]:
    base = cfg["episodes"]["day_counts"][trend]
    # a single per-episode jitter applied uniformly to every day so replicate
    # episodes vary without perturbing the intended first-vs-last day trend shape
    # (independent per-day jitter on small integers made "stable" episodes flip
    # into spurious rising/falling trends -- caught during verification)
    jitter = rng.choice([-1, 0, 0, 1])
    return [max(0, c + jitter) for c in base]


def build_day_leaf_counts(rng: random.Random, pool: pd.DataFrame, cfg: dict,
                            target_class: str, class_names: list[str],
                            target_day_counts: list[int]) -> list[dict]:
    """For each day, sample target-class images to hit the intended target count and
    background images from other classes, then tally by PREDICTED class."""
    bg_lo, bg_hi = cfg["episodes"]["background_classes_per_day"]
    target_pool = pool[pool["gt_class"] == target_class]
    other_classes = [c for c in class_names if c != target_class]
    other_pool = pool[pool["gt_class"].isin(other_classes)]

    days = []
    for target_n in target_day_counts:
        sampled = target_pool.sample(n=target_n, replace=True, random_state=rng.randint(0, 10**9))
        n_bg = rng.randint(bg_lo, bg_hi)
        bg_sampled = other_pool.sample(n=n_bg, replace=True, random_state=rng.randint(0, 10**9))
        day_images = pd.concat([sampled, bg_sampled])
        counts = {c: 0 for c in class_names}
        for pred in day_images["pred_class"]:
            counts[pred] += 1
        days.append(counts)
    return days


def classify_trend(day_counts: list[int], rising_pct: float, falling_pct: float) -> str:
    first, last = day_counts[0], day_counts[-1]
    if first == 0:
        pct = 100.0 if last > 0 else 0.0
    else:
        pct = 100.0 * (last - first) / first
    if pct >= rising_pct:
        return "rising"
    if pct <= falling_pct:
        return "falling"
    return "stable"


def main():
    cfg = load_cfg(CONFIG_PATH)
    set_seed(cfg["seed"])
    rng = random.Random(cfg["seed"])
    ensure_dirs(OUTPUT_DIR)

    class_names = cfg["model"]["class_names"]
    target_class = cfg["data"]["target_class"]
    pool = load_pool_predictions(cfg)
    risk_windows = load_risk_windows(cfg)

    n_rep = cfg["episodes"]["n_replicates_per_category"]
    rising_pct = cfg["episodes"]["rising_threshold_pct"]
    falling_pct = cfg["episodes"]["falling_threshold_pct"]

    episodes = []
    episode_id = 0
    for cat_name, cat_cfg in cfg["categories"].items():
        trend_shape = cat_cfg["trend"]
        risk_bin = cat_cfg["risk_bin"]
        ground_truth = cat_cfg["ground_truth"]
        candidate_risk_rows = risk_bin_to_rows(risk_windows, risk_bin)

        for rep in range(n_rep):
            episode_id += 1
            target_day_counts = sample_day_counts(rng, cfg, trend_shape)
            day_counts_full = build_day_leaf_counts(
                rng, pool, cfg, target_class, class_names, target_day_counts)
            target_series = [d[target_class] for d in day_counts_full]
            realized_trend = classify_trend(target_series, rising_pct, falling_pct)

            if candidate_risk_rows is None:
                risk_state = None
                risk_row = None
            else:
                row = candidate_risk_rows.sample(n=1, random_state=rng.randint(0, 10**9)).iloc[0]
                risk_state = row["risk"]
                risk_row = {"day": int(row["day"]), "capture_hour": int(row["capture_hour"]),
                             "rh_mean": float(row["rh_mean"]), "soil_mean": float(row["soil_mean"])}

            delta_pct = (100.0 * (target_series[-1] - target_series[0]) / target_series[0]
                         if target_series[0] > 0 else (100.0 if target_series[-1] > 0 else 0.0))

            episodes.append({
                "episode_id": episode_id,
                "category": cat_name,
                "intended_trend": trend_shape,
                "realized_trend": realized_trend,
                "risk_bin_intended": risk_bin,
                "risk_state": risk_state,
                "risk_source_window": json.dumps(risk_row) if risk_row else None,
                "target_class": target_class,
                "day_counts_target": json.dumps(target_series),
                "day_counts_full": json.dumps(day_counts_full),
                "delta_pct": round(delta_pct, 1),
                "ground_truth": ground_truth,
            })

    df = pd.DataFrame(episodes)
    mismatches = (df["intended_trend"] != df["realized_trend"]).sum()
    print(f"episodes built: {len(df)}; trend-shape mismatches (intended vs realized): {mismatches}")

    out_path = OUTPUT_DIR / "episodes.csv"
    df.to_csv(out_path, index=False)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
