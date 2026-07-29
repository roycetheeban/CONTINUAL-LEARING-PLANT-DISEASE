"""Shared LEAFSENSE figure style: IEEE-compliant, colour-blind-safe, greyscale-legible."""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Colour-blind-safe palette (Okabe-Ito based) -- same across all 9 figures
PAL = {
    "blue":      "#0072B2",
    "orange":    "#E69F00",
    "green":     "#009E73",
    "vermillion":"#D55E00",
    "pink":      "#CC79A7",
    "sky":       "#56B4E9",
    "grey":      "#999999",
    "ink":       "#222222",
    "faint":     "#CCCCCC",
}

# IEEE column widths (inches)
COL1 = 3.46   # single column  (88 mm)
COL2 = 7.16   # double column  (181 mm)

matplotlib.rcParams.update({
    "pdf.fonttype": 42,      # embed TrueType (editable, IEEE-safe)
    "ps.fonttype": 42,
    "svg.fonttype": "none",
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
    "font.size": 8,
    "axes.linewidth": 0.8,
    "axes.edgecolor": "#333333",
    "axes.labelsize": 8,
    "axes.titlesize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 6.5,
    "lines.linewidth": 1.3,
    "figure.dpi": 150,
})

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.abspath(os.path.join(HERE, ".."))          # journal/figures/
PREV = os.path.join(OUT, "_preview")                      # journal/figures/_preview/
os.makedirs(PREV, exist_ok=True)

def save(fig, name):
    """Write both the paper PDF and a PNG preview."""
    fig.savefig(os.path.join(OUT, name + ".pdf"), bbox_inches="tight", pad_inches=0.02)
    fig.savefig(os.path.join(PREV, name + ".png"), bbox_inches="tight",
                pad_inches=0.02, dpi=200)
    plt.close(fig)
    print("wrote", name + ".pdf")
