"""E1 orchestrator: simulate -> label (turmeric infection oracle) -> features ->
time-block split -> train 5 models -> evaluate + robustness -> save artifacts.

Deploys the Decision Tree; the comparison quantifies the price of interpretability.
All environmental data is SIMULATED (disclosed in the paper).
"""
import argparse
import json
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from common_env import CONFIG_PATH, OUTPUT_DIR, ensure_dirs, get_process_ram_mb, load_cfg, set_seed
from features import FEATURE_COLS, add_label_noise, make_windows
from infection_risk_labels import bin_risk, calibrate_thresholds, label_days
from simulate_sensors import simulate

CLASSES = ["Low", "Medium", "High"]


def build_models(cfg):
    dt = cfg["decision_tree"]; rf = cfg["random_forest"]; gb = cfg["gradient_boosting"]
    seed = cfg["seed"]
    return {
        "Majority baseline": (DummyClassifier(strategy="most_frequent"), False),
        "Logistic Regression": (make_pipeline(StandardScaler(),
            LogisticRegression(max_iter=1000, class_weight="balanced")), False),
        "Decision Tree": (DecisionTreeClassifier(
            max_depth=dt["max_depth"], min_samples_leaf=dt["min_samples_leaf"],
            class_weight="balanced", random_state=seed), True),
        "Random Forest": (RandomForestClassifier(
            n_estimators=rf["n_estimators"], max_depth=rf["max_depth"],
            min_samples_leaf=rf["min_samples_leaf"], class_weight="balanced",
            random_state=seed, n_jobs=-1), False),
        "Gradient Boosting": (GradientBoostingClassifier(
            n_estimators=gb["n_estimators"], max_depth=gb["max_depth"],
            learning_rate=gb["learning_rate"], random_state=seed), False),
    }


def measure_latency_us(model, X, repeats=50):
    """Mean per-sample inference latency in microseconds."""
    t0 = time.perf_counter()
    for _ in range(repeats):
        model.predict(X)                        # keep DataFrame -> preserves feature names
    dt = (time.perf_counter() - t0) / repeats
    return round(dt / len(X) * 1e6, 2)


def model_size_kb(model):
    return round(len(pickle.dumps(model)) / 1024.0, 2)


def perturb(X, kind, cfg, seed):
    rng = np.random.default_rng(seed + 99)
    Xp = X.astype(float).copy()                 # float so noise/impute don't clash with int cols
    if kind == "noise":
        k = cfg["robustness"]["noise_multiplier"] - 1.0
        for c in X.columns:
            Xp[c] = X[c] + rng.normal(0, k * X[c].std(), len(X))
    elif kind == "dropout":
        frac = cfg["robustness"]["dropout_frac"]
        for c in X.columns:
            m = rng.random(len(X)) < frac
            Xp.loc[m, c] = X[c].mean()          # impute dropped cells with column mean
    return Xp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(CONFIG_PATH))
    args = ap.parse_args()
    cfg = load_cfg(args.config)
    set_seed(cfg["seed"])
    out = ensure_dirs(OUTPUT_DIR)

    # ---- data pipeline ----
    sensors = simulate(cfg)
    labels = label_days(sensors, cfg)                       # per-day accumulated severity
    im, sp = cfg["infection_model"], cfg["split"]
    train_acc = labels[labels["day"] <= sp["train_end_day"]]["accum_sev"]
    low, med = calibrate_thresholds(train_acc, im["calib_low_pct"], im["calib_med_pct"])
    labels = bin_risk(labels, low, med)                     # Low/Med/High from train-calibrated cuts
    windows = make_windows(sensors, labels, cfg)
    windows = add_label_noise(windows, cfg["labels"]["noise_frac"], cfg["seed"])
    sensors.to_csv(out / "sensors_raw.csv", index=False)
    windows.to_csv(out / "windows_labelled.csv", index=False)

    # ---- time-block split (never random) ----
    tr = windows[windows["day"] <= sp["train_end_day"]]
    te = windows[windows["day"] > sp["val_end_day"]]
    Xtr, ytr = tr[FEATURE_COLS], tr["risk"]
    Xte, yte = te[FEATURE_COLS], te["risk"]
    print(f"train={len(tr)} test={len(te)} | risk cuts Low|Med={low:.2f} Med|High={med:.2f}")
    print("train dist:", tr['risk'].value_counts(normalize=True).round(2).to_dict())
    print("test  dist:", te['risk'].value_counts(normalize=True).round(2).to_dict())

    # ---- train + evaluate all models ----
    rows, dt_model = [], None
    for name, (model, interpretable) in build_models(cfg).items():
        model.fit(Xtr, ytr)
        pred = model.predict(Xte)
        rows.append({
            "model": name,
            "accuracy": round(accuracy_score(yte, pred) * 100, 2),
            "macro_f1": round(f1_score(yte, pred, average="macro", labels=CLASSES) * 100, 2),
            "latency_us": measure_latency_us(model, Xte),
            "size_kb": model_size_kb(model),
            "interpretable": "yes" if interpretable else "no",
        })
        if name == "Decision Tree":
            dt_model = model
    comp = pd.DataFrame(rows)
    comp.to_csv(out / "model_comparison.csv", index=False)

    # ---- Decision Tree detail (deployed model) ----
    dt_pred = dt_model.predict(Xte)
    rep = classification_report(yte, dt_pred, labels=CLASSES, output_dict=True, zero_division=0)
    per_class = pd.DataFrame({c: {"precision": round(rep[c]["precision"], 3),
                                  "recall": round(rep[c]["recall"], 3),
                                  "f1": round(rep[c]["f1-score"], 3),
                                  "support": int(rep[c]["support"])} for c in CLASSES}).T
    per_class.index.name = "risk_class"
    per_class.to_csv(out / "dt_per_class.csv")
    cm = confusion_matrix(yte, dt_pred, labels=CLASSES)
    pd.DataFrame(cm, index=CLASSES, columns=CLASSES).to_csv(out / "confusion_dt.csv")
    imp = pd.DataFrame({"feature": FEATURE_COLS,
                        "importance": np.round(dt_model.feature_importances_, 4)}
                       ).sort_values("importance", ascending=False)
    imp.to_csv(out / "feature_importances.csv", index=False)

    # ---- robustness (DT) ----
    rob = [{"condition": "clean", "accuracy": round(accuracy_score(yte, dt_pred) * 100, 2)}]
    for kind in ("noise", "dropout"):
        Xp = perturb(Xte, kind, cfg, cfg["seed"])
        rob.append({"condition": {"noise": "sensor noise x2", "dropout": "10% dropout"}[kind],
                    "accuracy": round(accuracy_score(yte, dt_model.predict(Xp)) * 100, 2)})
    pd.DataFrame(rob).to_csv(out / "robustness.csv", index=False)

    # ---- deployed artifact + roll-up ----
    joblib.dump(dt_model, out / "decision_tree.pkl")
    dt_depth = dt_model.get_depth(); dt_leaves = dt_model.get_n_leaves()
    metrics = {
        "n_windows": len(windows), "n_train": len(tr), "n_test": len(te),
        "risk_distribution_overall": windows["risk"].value_counts(normalize=True).round(3).to_dict(),
        "deployed": "Decision Tree",
        "dt_depth": int(dt_depth), "dt_leaves": int(dt_leaves),
        "dt_test_accuracy": comp.loc[comp.model == "Decision Tree", "accuracy"].iat[0],
        "dt_test_macro_f1": comp.loc[comp.model == "Decision Tree", "macro_f1"].iat[0],
        "best_model": comp.sort_values("macro_f1", ascending=False).iloc[0]["model"],
        "top_features": imp.head(4).set_index("feature")["importance"].to_dict(),
        "robustness": {r["condition"]: r["accuracy"] for r in rob},
        "process_ram_mb": get_process_ram_mb(),
        "config": args.config,
        "note": "Environmental data is SIMULATED. Labels from Magarey et al. (2005) "
                "generic infection model parameterised for turmeric (Gohel et al. 2022).",
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    # ---- console summary ----
    print("\n=== MODEL COMPARISON (test) ===")
    print(comp.to_string(index=False))
    print(f"\nDT depth={dt_depth} leaves={dt_leaves}")
    print("\nDT per-class:\n", per_class.to_string())
    print("\nTop features:\n", imp.head(6).to_string(index=False))
    print("\nRobustness:\n", pd.DataFrame(rob).to_string(index=False))
    print("\nArtifacts ->", out)


if __name__ == "__main__":
    main()
