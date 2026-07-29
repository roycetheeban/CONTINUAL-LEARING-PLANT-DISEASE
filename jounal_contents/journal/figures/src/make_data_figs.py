"""Figs 6, 7, 8 -- data-driven, from the REAL CL-chapter numbers (Tables VI, VII, IX)."""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.ticker import NullLocator
from ls_style import PAL, COL1, save


# ---------------------------------------------------------------- Fig 6: CatA
def fig6():
    cases = ["Case 1\nScratch", "Case 2\nImageNet", "Case 3\nPlantVillage"]
    methods = ["EWC", "Replay", "Isolation", "Naive"]
    avg = {  # average CL accuracy, Table VI
        "EWC":       [97.67, 98.21, 97.54],
        "Replay":    [97.32, 98.25, 98.15],
        "Isolation": [96.44, 97.23, 96.57],
        "Naive":     [97.58, 98.42, 97.36],
    }
    col = {"EWC": PAL["blue"], "Replay": PAL["green"],
           "Isolation": PAL["orange"], "Naive": PAL["grey"]}

    fig, ax = plt.subplots(figsize=(COL1, 2.55))
    x = np.arange(3); w = 0.2
    for i, m in enumerate(methods):
        lbl = m + (" (ref)" if m == "Naive" else "")
        bars = ax.bar(x + (i - 1.5) * w, avg[m], w, label=lbl, color=col[m],
                      edgecolor="#222", linewidth=0.5,
                      hatch="////" if m == "Naive" else None)
        ax.bar_label(bars, fmt="%.1f", fontsize=4.7, padding=1.5, rotation=90)

    # mark best (Case 2 + Replay = 98.25, Replay is index 1 -> offset -0.5w at x=1)
    ax.annotate("best", xy=(1 - 0.5 * w, 98.25), xytext=(1 - 0.5 * w, 99.15),
                ha="center", fontsize=6, color=PAL["green"],
                arrowprops=dict(arrowstyle="-", color=PAL["green"], lw=0.7))

    ax.set_ylim(95.5, 99.5)
    ax.set_xticks(x); ax.set_xticklabels(cases)
    ax.set_ylabel("Average CL accuracy (%)")
    ax.yaxis.grid(True, ls=":", lw=0.4, color=PAL["faint"]); ax.set_axisbelow(True)
    ax.legend(ncol=4, loc="lower center", frameon=False, fontsize=5.8,
              columnspacing=1.0, handlelength=1.2, bbox_to_anchor=(0.5, -0.32))
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    save(fig, "fig6_cata")


# ------------------------------------------------------------- Fig 7: lambda
def fig7():
    lam = np.array([1000, 3000, 5000, 8000, 10000, 15000])
    c1 = [97.71, 97.80, 97.63, 97.89, 97.98, 97.89]
    c2 = [98.15, 98.24, 98.24, 98.07, 98.24, 98.15]

    fig, ax = plt.subplots(figsize=(COL1, 2.55))
    ax.set_xscale("log")
    ax.axvspan(880, 1450, color=PAL["grey"], alpha=0.10, lw=0)
    ax.axvspan(11500, 17000, color=PAL["grey"], alpha=0.10, lw=0)
    ax.plot(lam, c1, "-o", color=PAL["blue"], ms=4, label="Cycle 1")
    ax.plot(lam, c2, "--s", color=PAL["orange"], ms=4, label="Cycle 2")
    ax.plot(10000, 97.98, "*", color=PAL["blue"], ms=13, mec="#222", mew=0.4, zorder=5)
    ax.plot(3000, 98.24, "*", color=PAL["orange"], ms=13, mec="#222", mew=0.4, zorder=5)
    ax.annotate("C1 opt.", (10000, 97.98), xytext=(10000, 97.66), ha="center",
                fontsize=5.5, color=PAL["blue"])
    ax.annotate("C2 opt.", (3000, 98.24), xytext=(3000, 98.33), ha="center",
                fontsize=5.5, color=PAL["orange"])
    ax.text(1080, 97.52, "under-\nreg.", fontsize=5.3, color="#666", ha="center")
    ax.text(13500, 97.52, "over-\nreg.", fontsize=5.3, color="#666", ha="center")

    ax.set_xlabel(r"EWC penalty  $\lambda$  (log scale)")
    ax.set_ylabel("Test accuracy (%)")
    ax.set_ylim(97.45, 98.45)
    ax.set_xticks(lam)
    ax.set_xticklabels([f"{l//1000}k" for l in lam])
    ax.xaxis.set_minor_locator(NullLocator())
    ax.yaxis.grid(True, ls=":", lw=0.4, color=PAL["faint"]); ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.02))
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    save(fig, "fig7_lambda")


# -------------------------------------------------------------- Fig 8: CatB
def fig8():
    # (case, method, old_retention, avg_new)  -- Table IX
    D = [
        (1, "EWC", 87.34, 82.79), (1, "Replay", 84.78, 92.85),
        (1, "Naive", 87.07, 93.91), (1, "Hybrid", 92.44, 84.43),
        (2, "EWC", 95.95, 94.20), (2, "Replay", 97.19, 98.07),
        (2, "Isolation", 100, 50.87), (2, "Naive", 87.51, 95.74),
        (2, "Hybrid", 98.24, 95.26),
        (3, "EWC", 95.78, 90.62), (3, "Replay", 96.92, 96.42),
        (3, "Isolation", 100, 15.38), (3, "Naive", 94.72, 96.91),
        (3, "Hybrid", 97.54, 92.84),
    ]
    mk = {"EWC": "o", "Replay": "s", "Isolation": "^", "Naive": "X", "Hybrid": "D"}
    cc = {1: PAL["sky"], 2: PAL["blue"], 3: PAL["green"]}

    fig, ax = plt.subplots(figsize=(COL1, 3.05))
    ax.add_patch(Rectangle((95.5, 95.5), 7, 6, color=PAL["green"], alpha=0.09, zorder=0))
    ax.text(99.0, 99.6, "desirable", fontsize=5.6, color=PAL["green"], ha="center",
            style="italic")

    for case, method, xo, yn in D:
        ax.scatter(xo, yn, marker=mk[method], s=34, facecolor=cc[case],
                   edgecolor="#222", linewidth=0.5, zorder=3)

    # highlight winner
    ax.scatter(97.19, 98.07, s=180, facecolor="none", edgecolor=PAL["vermillion"],
               linewidth=1.3, zorder=4)
    ax.annotate("Case 2 + Replay\n(best on both)", (97.19, 98.07),
                xytext=(88.5, 99.0), fontsize=5.8, color=PAL["vermillion"], ha="center",
                arrowprops=dict(arrowstyle="->", color=PAL["vermillion"], lw=0.8))
    # isolation failure region (two triangles at x=100: Case 2 y=50.9, Case 3 y=15.4)
    ax.annotate("Isolation (Case 2 & 3):\nzero forgetting,\nno acquisition", (99.7, 50.87),
                xytext=(88.5, 46), fontsize=5.6, color="#555", ha="center",
                arrowprops=dict(arrowstyle="->", color="#888", lw=0.7))
    ax.annotate("", (99.7, 15.38), xytext=(90.5, 42),
                arrowprops=dict(arrowstyle="->", color="#bbb", lw=0.6))

    ax.set_xlabel("Old-class retention (%)")
    ax.set_ylabel("New-class accuracy (%)")
    ax.set_xlim(82, 103); ax.set_ylim(8, 103)
    ax.grid(True, ls=":", lw=0.4, color=PAL["faint"]); ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    meth_h = [Line2D([], [], marker=mk[m], color="#444", ls="", ms=5, label=m)
              for m in ["EWC", "Replay", "Isolation", "Naive", "Hybrid"]]
    case_h = [Line2D([], [], marker="o", color=cc[c], ls="", ms=5, label=f"Case {c}")
              for c in (1, 2, 3)]
    leg1 = ax.legend(handles=meth_h, title="Method", loc="lower left",
                     fontsize=5.4, title_fontsize=5.8, frameon=True, framealpha=0.92,
                     handletextpad=0.3, borderpad=0.3)
    leg1.get_frame().set_edgecolor("#ccc")
    ax.add_artist(leg1)
    leg2 = ax.legend(handles=case_h, title="Init.", loc="center left",
                     fontsize=5.4, title_fontsize=5.8, frameon=True, framealpha=0.92,
                     handletextpad=0.3, borderpad=0.3, bbox_to_anchor=(0.0, 0.60))
    leg2.get_frame().set_edgecolor("#ccc")
    save(fig, "fig8_catb_scatter")


if __name__ == "__main__":
    fig6(); fig7(); fig8()
