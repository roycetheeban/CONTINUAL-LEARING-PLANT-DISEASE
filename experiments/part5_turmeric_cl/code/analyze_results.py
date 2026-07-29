"""E4 analysis -- aggregate the run matrix into Table XIV, with honest statistics.

Two things this does that a plain accuracy table does not:

1. mean +/- std across seeds. A single run on a 107-image test set carries a 95%
   CI of about +/-5.7 pp, so one number cannot support a claim.

2. McNemar's paired test for method-vs-method comparison. Both models are scored
   on the SAME test images, so the images they agree on carry no information --
   only the disagreements do. Comparing two overlapping confidence intervals is
   the wrong test here and would declare "no difference" almost regardless of the
   data. McNemar uses the discordant pairs directly and is far more sensitive.

Writes: table_xiv.csv, pairwise_mcnemar.csv, metrics_summary.json
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

from common_turmeric import CONFIG_PATH, OUTPUT_DIR, REPO_ROOT, load_cfg, write_json


def mean_std(xs: list[float]) -> tuple[float, float]:
    if not xs:
        return float("nan"), float("nan")
    m = sum(xs) / len(xs)
    if len(xs) == 1:
        return m, 0.0
    var = sum((x - m) ** 2 for x in xs) / (len(xs) - 1)
    return m, math.sqrt(var)


def binom_two_sided(k: int, n: int) -> float:
    """Exact two-sided binomial p-value at p=0.5 (McNemar exact)."""
    if n == 0:
        return 1.0
    k = min(k, n - k)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n)
    return min(1.0, 2 * tail)


def load_preds(path: Path) -> dict[str, int]:
    """image path -> correct (1/0)"""
    out = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            out[row["path"]] = int(row["correct"])
    return out


def mcnemar(a: dict[str, int], b: dict[str, int]) -> dict:
    """a, b: path -> correct. Returns discordant counts and exact p."""
    keys = sorted(set(a) & set(b))
    b_only = sum(1 for k in keys if a[k] == 1 and b[k] == 0)   # a right, b wrong
    c_only = sum(1 for k in keys if a[k] == 0 and b[k] == 1)   # b right, a wrong
    n_disc = b_only + c_only
    return {
        "n_compared": len(keys),
        "a_right_b_wrong": b_only,
        "b_right_a_wrong": c_only,
        "n_discordant": n_disc,
        "p_value": binom_two_sided(min(b_only, c_only), n_disc),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(CONFIG_PATH))
    args = ap.parse_args()

    cfg = load_cfg(args.config)
    out_root = REPO_ROOT / cfg["output"]["root"]
    seeds, methods = cfg["seeds"], cfg["methods"]

    # ---- collect ----------------------------------------------------------
    acc = defaultdict(list)   # (stage) -> [acc per seed]
    f1 = defaultdict(list)
    preds: dict[tuple, dict] = {}
    missing = []

    for seed in seeds:
        p = out_root / f"base_seed{seed}" / "metrics" / "metrics.json"
        if p.exists():
            m = json.loads(p.read_text())
            acc["base"].append(m["test"]["accuracy"])
            f1["base"].append(m["test"]["macro_f1"])
            pp = p.parent / "test_predictions.csv"
            if pp.exists():
                preds[("base", seed)] = load_preds(pp)
        else:
            missing.append(str(p))

        for method in methods:
            for cycle in (1, 2):
                d = out_root / f"{method}_cycle{cycle}_seed{seed}"
                p = d / "metrics" / "metrics.json"
                if not p.exists():
                    missing.append(str(p))
                    continue
                m = json.loads(p.read_text())
                key = f"{method}_cycle{cycle}"
                acc[key].append(m["test"]["accuracy"])
                f1[key].append(m["test"]["macro_f1"])
                pp = d / "metrics" / "test_predictions.csv"
                if pp.exists():
                    preds[(key, seed)] = load_preds(pp)

    if missing:
        print(f"WARNING: {len(missing)} run(s) missing; results are partial")
        for m in missing[:6]:
            print("   ", m)

    # ---- Table XIV --------------------------------------------------------
    rows = []
    order = ["base"] + [f"{m}_cycle{c}" for m in methods for c in (1, 2)]
    print(f"\n{'stage':<20}{'n':>3}{'acc mean':>10}{'std':>8}{'F1 mean':>10}{'std':>8}")
    print("-" * 59)
    for k in order:
        if not acc[k]:
            continue
        am, asd = mean_std(acc[k])
        fm, fsd = mean_std(f1[k])
        rows.append({
            "stage": k, "n_seeds": len(acc[k]),
            "accuracy_mean": round(100 * am, 2), "accuracy_std": round(100 * asd, 2),
            "macro_f1_mean": round(100 * fm, 2), "macro_f1_std": round(100 * fsd, 2),
            "accuracy_per_seed": "|".join(f"{100*x:.2f}" for x in acc[k]),
        })
        print(f"{k:<20}{len(acc[k]):>3}{100*am:>9.2f}%{100*asd:>8.2f}"
              f"{100*fm:>9.2f}%{100*fsd:>8.2f}")

    with open(OUTPUT_DIR / "table_xiv.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # ---- pairwise McNemar, at the final cycle -----------------------------
    print(f"\n{'='*72}\nMcNemar (paired) -- final cycle, per seed\n{'='*72}")
    mc_rows = []
    finals = [f"{m}_cycle2" for m in methods]
    for i, A in enumerate(finals):
        for B in finals[i + 1:]:
            for seed in seeds:
                if (A, seed) not in preds or (B, seed) not in preds:
                    continue
                r = mcnemar(preds[(A, seed)], preds[(B, seed)])
                sig = "SIGNIFICANT" if r["p_value"] < 0.05 else "not significant"
                mc_rows.append({"seed": seed, "model_a": A, "model_b": B, **r,
                                 "verdict": sig})
                print(f"seed {seed}  {A:<16} vs {B:<16} "
                      f"disc={r['n_discordant']:>3} "
                      f"({r['a_right_b_wrong']}/{r['b_right_a_wrong']})  "
                      f"p={r['p_value']:.4f}  {sig}")

    if mc_rows:
        with open(OUTPUT_DIR / "pairwise_mcnemar.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(mc_rows[0].keys()))
            w.writeheader()
            w.writerows(mc_rows)

    # ---- retention: did the CL cycles hold onto base performance? ----------
    print(f"\n{'='*72}\nRetention vs base (final cycle)\n{'='*72}")
    retention = {}
    base_m, _ = mean_std(acc["base"])
    for m in methods:
        k = f"{m}_cycle2"
        if not acc[k]:
            continue
        fm, fsd = mean_std(acc[k])
        retention[m] = {
            "final_acc_mean_pct": round(100 * fm, 2),
            "delta_vs_base_pp": round(100 * (fm - base_m), 2),
        }
        print(f"  {m:<10} {100*fm:>6.2f}%  (base {100*base_m:.2f}%, "
              f"delta {100*(fm-base_m):+.2f} pp)")

    n_test = len(next(iter(preds.values()))) if preds else 0
    ci = 1.96 * math.sqrt(0.9 * 0.1 / n_test) * 100 if n_test else float("nan")

    write_json(OUTPUT_DIR / "metrics_summary.json", {
        "seeds": seeds, "methods": methods,
        "n_test_images": n_test,
        "single_run_95ci_halfwidth_pp_at_p90": round(ci, 2),
        "table_xiv": rows,
        "retention_vs_base": retention,
        "mcnemar": mc_rows,
        "missing_runs": missing,
        "interpretation_note": (
            "With n_test=%d a single run's 95%% CI is about +/-%.1f pp, so "
            "differences smaller than that are NOT resolvable from one run. "
            "Method comparisons use the paired McNemar test on identical test "
            "images; where p >= 0.05 the methods are not distinguishable on this "
            "dataset and must not be reported as different. E4 is a feasibility "
            "check that the recipe transfers, not a precision benchmark."
            % (n_test, ci)
        ),
    })

    print(f"\nn_test={n_test}; single-run 95% CI ~ +/-{ci:.1f} pp")
    print(f"wrote table_xiv.csv, pairwise_mcnemar.csv, metrics_summary.json -> {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
