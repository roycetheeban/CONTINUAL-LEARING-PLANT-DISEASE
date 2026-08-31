"""Fig 3 -- on-device monthly continual-learning cycle (adapt -> verify -> gate)."""
from matplotlib.patches import Polygon
from ls_style import PAL, COL1, save
from ls_diagram import canvas, box, arrow, label

W, H = COL1, 4.90
fig, ax = canvas(W, H)
YT = 100 * H / W

X = 52  # main spine


def diamond(ax, cx, cy, w, h, text, fc, ec):
    pts = [(cx, cy + h / 2), (cx + w / 2, cy), (cx, cy - h / 2), (cx - w / 2, cy)]
    ax.add_patch(Polygon(pts, closed=True, facecolor=fc, edgecolor=ec, linewidth=1.3, zorder=2))
    ax.text(cx, cy, text, ha="center", va="center", fontsize=5.8, zorder=3,
            color="#111", linespacing=1.15)


# nodes (top -> bottom)
box(ax, X, YT - 6, 52, 8, "Monthly trigger  /  dashboard button",
    fc="#eef3f8", ec=PAL["blue"], fs=5.8, bold=True)
box(ax, X, YT - 20, 60, 10, "Load retrain buffer +\nreturned relabelled samples", fs=5.8)
box(ax, X, YT - 36, 60, 12,
    "EWC / Replay fine-tune\n(G1–G2 frozen · checkpoint each epoch)", fs=5.8)
box(ax, X, YT - 52, 60, 10, "Export ONNX  →  build\nTensorRT FP16 engine", fs=5.8)
box(ax, X, YT - 70, 64, 13,
    "Evaluate candidate ENGINE\non FIXED held-out test set",
    fc="#fff5e6", ec=PAL["orange"], lw=1.6, bold=True, fs=5.8)
# Offset LEFT of the spine so the arrow into the diamond does not run through it.
label(ax, X - 4, YT - 79.6, "same TRT FP16 runtime\nthat will serve",
      fs=4.7, color=PAL["vermillion"], italic=True, ha="right")

dy = YT - 92
diamond(ax, X, dy, 34, 18, "better than\nincumbent?", "#f3f3f3", "#555")

# YES -> deploy
box(ax, X, dy - 20, 60, 11,
    "Atomic swap + bump version\n(keep old engine for rollback)",
    fc="#e8f5f0", ec=PAL["green"], fs=5.6, bold=True)
box(ax, X, dy - 34, 42, 7, "Deployed for next month",
    fc="#eef3f8", ec=PAL["blue"], fs=5.6)

# NO -> keep incumbent (left channel) and loop back up
box(ax, 15, dy, 22, 18, "Keep\nincumbent;\ncarry data\nforward",
    fc="#fbeee7", ec=PAL["vermillion"], fs=5.4)

# ---- arrows ----
arrow(ax, (X, YT - 10), (X, YT - 15))
arrow(ax, (X, YT - 25), (X, YT - 30))
arrow(ax, (X, YT - 42), (X, YT - 47))
arrow(ax, (X, YT - 57), (X, YT - 63.5))
arrow(ax, (X, YT - 76.5), (X, dy + 9))          # into diamond
arrow(ax, (X, dy - 9), (X, dy - 14.5))          # yes out
arrow(ax, (X, dy - 25.5), (X, dy - 30.5))
label(ax, X + 2.5, dy - 11.5, "yes", fs=5.2, color=PAL["green"], ha="left")

# No branch: diamond left -> keep-incumbent box -> up the left channel -> box 2.
# The "no" label sits ABOVE the midpoint of the horizontal segment (x 35 -> 26),
# not beside the diamond vertex, so it annotates the arrow it belongs to.
arrow(ax, (X - 17, dy), (26, dy), color=PAL["vermillion"])
label(ax, 30.5, dy + 2.6, "no", fs=5.2, color=PAL["vermillion"], ha="center")
arrow(ax, (15, dy + 9), (15, YT - 20), color=PAL["vermillion"], rad=0.0)
arrow(ax, (15, YT - 20), (X - 30, YT - 20), color=PAL["vermillion"])
# Offset LEFT of the vertical channel (x=15): the boxes start at x=22, so there is
# no room on the right. Sits in the outer margin, clear of both line and boxes.
label(ax, 12.5, (dy + 9 + YT - 20) / 2, "retry\nnext\ncycle", fs=4.7,
      color=PAL["vermillion"], ha="right")

save(fig, "fig3_cl_cycle")
