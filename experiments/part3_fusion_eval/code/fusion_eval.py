"""Score all 42 constructed episodes under three decision sources -- vision-only,
sensor-only, label-level fusion (Phi) -- against each episode's independently
reasoned ground truth. Produces Table IX (model_comparison-style CSV) plus
false-alarm rate, a sensor-failure degradation check, and early-warning lead time
for the stable_high category (per 06_EXPERIMENT_PROTOCOL.md PART 2 step 4)."""
import json

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

from common_fusion import CONFIG_PATH, ensure_dirs, load_cfg, OUTPUT_DIR


ALERT_LIKE = {"Alert"}  # states a farmer would act on immediately


def vision_only_predict(cfg, realized_trend: str) -> str:
    return cfg["vision_only_rule"][realized_trend]


def sensor_only_predict(cfg, risk_state) -> str:
    key = "NA" if pd.isna(risk_state) else risk_state
    return cfg["sensor_only_rule"][key]


def fusion_predict(cfg, realized_trend: str, risk_state) -> str:
    if pd.isna(risk_state):
        # graceful degradation: sensor unavailable -> fall back to vision-only
        return vision_only_predict(cfg, realized_trend)
    for row in cfg["fusion_rule"]["table"]:
        if row["risk"] == "NA":
            continue
        trend_ok = (row["trend"] == realized_trend) or (row["trend"] == "any")
        if row["risk"] == "Low_Medium":
            risk_ok = risk_state in ("Low", "Medium")
        elif row["risk"] == "any":
            risk_ok = True
        else:
            risk_ok = (row["risk"] == risk_state)
        if trend_ok and risk_ok:
            return row["output"]
    raise ValueError(f"no fusion rule matched trend={realized_trend} risk={risk_state}")


def score_source(df: pd.DataFrame, pred_col: str, labels: list[str]) -> dict:
    y_true = df["ground_truth"]
    y_pred = df[pred_col]
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", labels=labels, zero_division=0)
    # false alarm = predicted Alert/Watch but ground truth is Normal/Observe (a farmer
    # acted, or was told to watch closely, for nothing)
    escalated_pred = y_pred.isin(["Alert", "Watch"])
    benign_truth = y_true.isin(["Normal", "Observe"])
    false_alarms = int((escalated_pred & benign_truth).sum())
    false_alarm_rate = false_alarms / len(df)
    return {
        "accuracy": round(100 * acc, 2),
        "macro_f1": round(100 * macro_f1, 2),
        "false_alarms_n": false_alarms,
        "false_alarm_rate_pct": round(100 * false_alarm_rate, 2),
    }


def main():
    cfg = load_cfg(CONFIG_PATH)
    ensure_dirs(OUTPUT_DIR)

    df = pd.read_csv(OUTPUT_DIR / "episodes.csv")

    df["vision_only_pred"] = df["realized_trend"].apply(lambda t: vision_only_predict(cfg, t))
    df["sensor_only_pred"] = df["risk_state"].apply(lambda r: sensor_only_predict(cfg, r))
    df["fusion_pred"] = df.apply(
        lambda row: fusion_predict(cfg, row["realized_trend"], row["risk_state"]), axis=1)
    df["vision_only_correct"] = (df["vision_only_pred"] == df["ground_truth"]).astype(int)
    df["sensor_only_correct"] = (df["sensor_only_pred"] == df["ground_truth"]).astype(int)
    df["fusion_correct"] = (df["fusion_pred"] == df["ground_truth"]).astype(int)

    labels = sorted(set(df["ground_truth"]) | set(df["vision_only_pred"])
                     | set(df["sensor_only_pred"]) | set(df["fusion_pred"]))

    comparison = {
        "Vision only (counts)": score_source(df, "vision_only_pred", labels),
        "Sensor only (risk state)": score_source(df, "sensor_only_pred", labels),
        "Label-level fusion": score_source(df, "fusion_pred", labels),
    }

    # sensor-failure degradation check: fusion should exactly match vision-only on
    # the sensor_failure_rising category (graceful degradation, no corrupted state)
    fail_df = df[df["category"] == "sensor_failure_rising"]
    degradation_ok = bool((fail_df["fusion_pred"] == fail_df["vision_only_pred"]).all())

    # early-warning lead time: stable_high episodes are the case where fusion (and
    # sensor-only) issue "Watch" while symptoms are not yet visible (vision-only
    # says Normal). Lead time = how many days earlier the warning fires relative to
    # when vision-only would first flag it, IF the scenario had continued to the
    # rising_high trajectory (i.e., the average day-count-to-threshold gap).
    # We report this as a worked-example number using the rising_high day-shape,
    # not a second live simulation -- documented in the README.
    rising_high_days = cfg["episodes"]["day_counts"]["rising"]
    rising_pct = cfg["episodes"]["rising_threshold_pct"]
    first = rising_high_days[0]
    lead_time_days = None
    for i, c in enumerate(rising_high_days):
        pct = 100.0 * (c - first) / first if first > 0 else (100.0 if c > 0 else 0.0)
        if pct >= rising_pct:
            lead_time_days = i  # day index (0-based) vision-only would first fire
            break
    # fusion/sensor-only fire on day 0 for a stable_high (already at High risk);
    # vision-only only fires once the count trajectory crosses threshold.
    early_warning_lead_days = lead_time_days if lead_time_days is not None else 0

    comparison_df = pd.DataFrame(comparison).T
    comparison_df.index.name = "decision_source"
    comparison_df.to_csv(OUTPUT_DIR / "table_ix_comparison.csv")

    df.to_csv(OUTPUT_DIR / "episodes_scored.csv", index=False)

    metrics = {
        "n_episodes": int(len(df)),
        "n_categories": int(df["category"].nunique()),
        "table_ix": comparison,
        "sensor_failure_degrades_to_vision_only": degradation_ok,
        "early_warning_lead_time_days": early_warning_lead_days,
        "category_accuracy": {
            cat: {
                "vision_only": round(100 * float(g["vision_only_correct"].mean()), 1),
                "sensor_only": round(100 * float(g["sensor_only_correct"].mean()), 1),
                "fusion": round(100 * float(g["fusion_correct"].mean()), 1),
            }
            for cat, g in df.groupby("category")
        },
    }
    with open(OUTPUT_DIR / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(comparison_df.to_string())
    print(f"\nsensor-failure graceful degradation OK: {degradation_ok}")
    print(f"early-warning lead time (worked example): {early_warning_lead_days} day(s)")
    print(f"wrote table_ix_comparison.csv, episodes_scored.csv, metrics.json")


if __name__ == "__main__":
    main()
