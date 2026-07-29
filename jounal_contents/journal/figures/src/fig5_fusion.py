"""Fig 5 -- Label-level fusion example chain.
Panels (a)/(b) use REAL data: (a) is the real trained Case2+Replay classifier's
per-class predictions on real held-out tomato test images, sampled into a
constructed 5-day episode (experiments/part3_fusion_eval/outputs/episodes_scored.csv,
episode_id=1, category=rising_high); (b) is the REAL E1 hourly sensor trace for the
matching days (experiments/part2_env_risk/outputs/sensors_raw.csv). Panel (c) states
the real Phi inputs/output for this episode. The episode's day-by-day image
sequencing is constructed (no chronological tomato photos exist), and the vision
(tomato) / risk (turmeric-parameterised) branches are paired for illustration only
-- both disclosed in the caption, matching the honesty standard used for Table VIII/IX.
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.gridspec import GridSpec

from ls_style import PAL, COL1, save
from ls_diagram import box, arrow


def find_repo_root(start: Path) -> Path:
    p = start.resolve()
    for parent in [p] + list(p.parents):
        if (parent / "experiments").is_dir() and (parent / "jounal_contents").is_dir():
            return parent
    raise RuntimeError("could not locate repo root")


REPO_ROOT = find_repo_root(Path(__file__))

EPISODES_CSV = REPO_ROOT / "experiments/part3_fusion_eval/outputs/episodes_scored.csv"
SENSORS_CSV = REPO_ROOT / "experiments/part2_env_risk/outputs/sensors_raw.csv"

CLASS_SHORT = {
    "Tomato___Bacterial_spot": "bact._spot",
    "Tomato___Early_blight": "early_blight",
    "Tomato___Late_blight": "late_blight",
    "Tomato___Leaf_Mold": "leaf_mold",
    "Tomato___healthy": "healthy",
}
FOCUS_CLASS = "Tomato___Early_blight"
FOCUS_COLOR = PAL["vermillion"]
OTHER_COLOR = PAL["grey"]
MARKERS = ["o", "s", "^", "D", "x"]


def load_episode():
    df = pd.read_csv(EPISODES_CSV)
    row = df[(df["episode_id"] == 1) & (df["category"] == "rising_high")].iloc[0]
    day_counts_full = json.loads(row["day_counts_full"])
    risk_window = json.loads(row["risk_source_window"])
    return row, day_counts_full, risk_window


def main():
    row, day_counts_full, risk_window = load_episode()
    classes = list(day_counts_full[0].keys())
    days = list(range(1, len(day_counts_full) + 1))
    anchor_day = risk_window["day"]
    sim_days = [anchor_day - (len(days) - 1) + (d - 1) for d in days]  # days ending on anchor_day

    sensors = pd.read_csv(SENSORS_CSV)
    trace = sensors[(sensors["day"] >= sim_days[0]) & (sensors["day"] <= sim_days[-1])].copy()
    trace["t"] = trace["day"] + trace["hour"] / 24.0

    fig = plt.figure(figsize=(COL1, 5.7))
    gs = GridSpec(3, 1, height_ratios=[1.0, 0.85, 1.0], hspace=0.5)

    # ---- panel (a): per-class leaf counts across days (REAL classifier output) --
    ax_a = fig.add_subplot(gs[0])
    for i, c in enumerate(classes):
        y = [d[c] for d in day_counts_full]
        if c == FOCUS_CLASS:
            ax_a.plot(days, y, "-", marker=MARKERS[i], color=FOCUS_COLOR, ms=4.5,
                      lw=1.6, zorder=5, label=CLASS_SHORT[c])
        else:
            ax_a.plot(days, y, "-", marker=MARKERS[i], color=OTHER_COLOR, ms=3,
                      lw=0.9, alpha=0.75, zorder=3, label=CLASS_SHORT[c])
    ax_a.set_xlabel("Day")
    ax_a.set_ylabel("Leaf count")
    ax_a.set_xticks(days)
    ax_a.legend(fontsize=4.6, ncol=3, frameon=False, loc="upper left",
                columnspacing=0.8, handlelength=1.3)
    ax_a.yaxis.grid(True, ls=":", lw=0.4, color=PAL["faint"])
    ax_a.set_axisbelow(True)
    for s in ("top", "right"):
        ax_a.spines[s].set_visible(False)
    ax_a.text(0.01, 1.14, "(a)", transform=ax_a.transAxes, fontsize=7, fontweight="bold")

    # ---- panel (b): aligned REAL environmental trace ----------------------------
    ax_b = fig.add_subplot(gs[1])
    ax_b.plot(trace["t"], trace["rh_pct"], color=PAL["blue"], lw=1.1, label="Humidity (%)")
    ax_b.axhspan(85, 100, color=PAL["blue"], alpha=0.10, lw=0)
    ax_b.axhline(85, color=PAL["blue"], lw=0.6, ls=":")
    ax_b2 = ax_b.twinx()
    ax_b2.plot(trace["t"], trace["soil_pct"], color=PAL["green"], lw=1.0, ls="--",
               label="Soil moisture (%)")
    ax_b.set_xlabel("Day")
    ax_b.set_ylabel("Humidity (%)", color=PAL["blue"])
    ax_b2.set_ylabel("Soil (%)", color=PAL["green"])
    ax_b.tick_params(axis="y", labelcolor=PAL["blue"])
    ax_b2.tick_params(axis="y", labelcolor=PAL["green"])
    lines1, labels1 = ax_b.get_legend_handles_labels()
    lines2, labels2 = ax_b2.get_legend_handles_labels()
    ax_b.legend(lines1 + lines2, labels1 + labels2, fontsize=4.6, frameon=False,
                loc="lower right")
    for s in ("top",):
        ax_b.spines[s].set_visible(False)
        ax_b2.spines[s].set_visible(False)
    ax_b.text(0.01, 1.18, "(b)", transform=ax_b.transAxes, fontsize=7, fontweight="bold")

    # ---- panel (c): fusion box (real Delta-count + real risk -> real output) ----
    ax_c = fig.add_subplot(gs[2])
    ax_c.set_xlim(0, 100)
    ax_c.set_ylim(0, 50)
    ax_c.axis("off")
    ax_c.text(0.01, 1.03, "(c)", transform=ax_c.transAxes, fontsize=7, fontweight="bold")

    target_series = json.loads(row["day_counts_target"])
    delta_count = target_series[-1] - target_series[0]
    risk_state = row["risk_state"]
    fused = row["fusion_pred"]

    ax_c.text(50, 49, f"Inputs from the Day 1–{len(days)} window in (a)/(b)",
              ha="center", va="center", fontsize=5.2, color="#666", style="italic")

    # two input branches (top row) -- text kept short so it sits inside the boxes
    box(ax_c, 26, 40, 40, 12,
        f"Vision branch\n+{delta_count} early_blight leaves\nin {len(days)} days  (rising)",
        fc="white", ec="#333", fs=5.3, bold=True)
    box(ax_c, 74, 40, 40, 12,
        f"Sensor branch\nRisk = {risk_state}\n(humidity > 85%)",
        fc="white", ec="#333", fs=5.3, bold=True)

    # explicit Phi combine node -- WHY the two inputs give this output, not just that
    box(ax_c, 50, 23, 60, 9,
        "Φ:  rising  +  High risk   →   Alert",
        fc="#eeeeee", ec="#333", fs=5.3, bold=True)

    # arrows routed through the gaps so the heads are clearly visible
    arrow(ax_c, (26, 33.6), (40, 27.4), color="#333", lw=1.3)
    arrow(ax_c, (74, 33.6), (60, 27.4), color="#333", lw=1.3)
    arrow(ax_c, (50, 18.7), (50, 13.6), color="#333", lw=1.3)

    box(ax_c, 50, 8, 92, 9,
        f'Fused output: "{fused.upper()}"\ninspect and consider fungicide',
        fc=PAL["green"], tc="white", ec="#1b6b50", fs=5.2, bold=True)

    save(fig, "fig5_fusion")


if __name__ == "__main__":
    main()
