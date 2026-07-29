"""Box/arrow helpers for the LEAFSENSE concept diagrams (Figs 1-4).
Uses a square data unit so boxes keep their proportions: set the canvas with
canvas(ax, w_in, h_in) and then draw in a 0..100 (x) by 0..(100*h/w) (y) space."""
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from ls_style import PAL


def canvas(w_in, h_in):
    fig, ax = plt.subplots(figsize=(w_in, h_in))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100 * h_in / w_in)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    return fig, ax


def box(ax, cx, cy, w, h, text="", fc="white", ec=PAL["ink"], tc="#111",
        fs=6.5, lw=1.1, ls="-", pad=0.02, bold=False, hatch=None, round=True,
        align="center", z=2):
    style = f"round,pad=0,rounding_size={min(w, h) * 0.14:.2f}" if round else "square,pad=0"
    p = FancyBboxPatch((cx - w / 2, cy - h / 2), w, h, boxstyle=style,
                       linewidth=lw, edgecolor=ec, facecolor=fc, linestyle=ls,
                       hatch=hatch, zorder=z, mutation_aspect=1)
    ax.add_patch(p)
    if text:
        ax.text(cx, cy, text, ha="center", va="center", fontsize=fs, color=tc,
                fontweight="bold" if bold else "normal", zorder=z + 1, linespacing=1.15)
    return (cx, cy)


def label(ax, x, y, text, fs=5.6, color="#555", ha="center", va="center",
          italic=False, bold=False, z=5):
    ax.text(x, y, text, ha=ha, va=va, fontsize=fs, color=color,
            fontstyle="italic" if italic else "normal",
            fontweight="bold" if bold else "normal", zorder=z, linespacing=1.15)


def arrow(ax, p0, p1, color=PAL["ink"], lw=1.1, ls="-", style="-|>", rad=0.0,
          shrink=2.0, z=1):
    a = FancyArrowPatch(p0, p1, arrowstyle=style, mutation_scale=9,
                        linewidth=lw, color=color, linestyle=ls,
                        connectionstyle=f"arc3,rad={rad}",
                        shrinkA=shrink, shrinkB=shrink, zorder=z)
    ax.add_patch(a)
