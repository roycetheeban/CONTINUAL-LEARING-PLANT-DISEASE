"""Fig 1 -- LEAFSENSE system architecture (offline-first edge + optional cloud)."""
from matplotlib.patches import FancyBboxPatch
from ls_style import PAL, COL2, save
from ls_diagram import canvas, box, arrow, label

W, H = COL2, 4.35
fig, ax = canvas(W, H)
YT = 100 * H / W  # ~60.8


def container(cx, cy, w, h, title, ec, fc, ls="-", tcol=None):
    ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h,
                 boxstyle="round,pad=0,rounding_size=1.2", linewidth=1.4,
                 edgecolor=ec, facecolor=fc, linestyle=ls, zorder=0))
    ax.text(cx - w / 2 + 1.5, cy + h / 2 - 2.2, title, ha="left", va="center",
            fontsize=6.2, fontweight="bold", color=tcol or ec, zorder=1)


# ============ CLOUD tier (optional, greyed, dashed) ============
container(50, YT - 6.5, 92, 11, "CLOUD  —  optional (subscription / support)",
          "#999", "#f4f4f4", ls="--", tcol="#777")
box(ax, 22, YT - 8.0, 24, 5.5, "Relabelling service", fc="white", ec="#aaa", fs=5.0)
box(ax, 50, YT - 8.0, 24, 5.5, "Model registry / OTA", fc="white", ec="#aaa", fs=5.0)
box(ax, 78, YT - 8.0, 26, 5.5, "RAG chatbot / insights", fc="white", ec="#aaa", fs=5.0)

# intermittent sync arrow between tiers
arrow(ax, (50, YT - 12.5), (50, YT - 18.5), color="#999", ls="--", lw=1.2, style="<|-|>")
label(ax, 62, YT - 15.5, "intermittent sync\n(relabel queue · updates)", fs=4.6,
      color="#999", ha="left", italic=True)

# ============ EDGE tier (dominant, solid) ============
EY = YT - 38
container(50, EY, 96, 34, "EDGE  —  NVIDIA Jetson Orin Nano  (offline-first)",
          PAL["ink"], "white")

# --- inputs (left) ---
box(ax, 12, EY + 8, 17, 7, "Rotating camera\n360° / 45°  (8 pos)", fc="#eef3f8",
    ec=PAL["blue"], fs=4.7)
box(ax, 12, EY - 4, 17, 7, "Sensors: CO₂, temp,\nhumidity, soil", fc="#fff5e6",
    ec=PAL["orange"], fs=4.7)

# --- image path (top lane) ---
img = [
    (30, "YOLOv8n\ndetect"),
    (43, "crop\n≥124²"),
    (56, "YOLOv8n\n-seg"),
    (69, "MobileNetV3\nclassify"),
]
for x, t in img:
    box(ax, x, EY + 8, 11, 7, t, fc="#eef3f8", ec=PAL["blue"], fs=4.6)
# optional GAN chip between seg and classify (dashed, small, above the lane)
box(ax, 62.5, EY + 13.4, 10, 4.0, "GAN (opt.)", fc="#f2f2f2", ec="#999", ls="--", fs=4.4)
arrow(ax, (57.5, EY + 11.2), (60, EY + 12.2), color="#bbb", ls="--", lw=0.8)
arrow(ax, (65, EY + 12.2), (67.5, EY + 11.2), color="#bbb", ls="--", lw=0.8)
for a, b in [(30, 43), (43, 56), (56, 69)]:
    arrow(ax, (a + 5.5, EY + 8), (b - 5.5, EY + 8), color="#555", lw=0.9)
arrow(ax, (20.5, EY + 8), (24.5, EY + 8), color="#555", lw=0.9)   # camera -> detect

# --- sensor path (bottom lane) ---
box(ax, 34, EY - 4, 16, 7, "Decision-tree\nenv. risk", fc="#fff5e6",
    ec=PAL["orange"], fs=4.7)
arrow(ax, (20.5, EY - 4), (26, EY - 4), color="#555", lw=0.9)      # sensors -> tree

# --- fusion (centre-right) ---
box(ax, 83, EY + 2, 14, 9, "Label-level\nfusion", fc="#eef7f2", ec=PAL["green"],
    bold=True, fs=5.2)
arrow(ax, (74.5, EY + 7), (76, EY + 4), color="#555", lw=0.9)      # classify -> fusion
arrow(ax, (42, EY - 4), (76.5, EY + 0.5), color="#555", lw=0.9, rad=-0.1)  # tree -> fusion

# --- dashboard / insights ---
box(ax, 89, EY - 9, 16, 6, "Streamlit dashboard\n+ rule insights", fc="white",
    ec=PAL["ink"], fs=4.7)
arrow(ax, (83, EY - 2.5), (88, EY - 6), color="#555", lw=0.9)

# --- CL loop (bottom band inside edge) ---
cly = EY - 12.5
box(ax, 33, cly, 60, 6.5,
    "Confidence routing  →  retrain buffer / relabel queue  →  "
    "monthly EWC · metric gate · engine swap",
    fc="#f7f0f7", ec=PAL["pink"], fs=4.7)
# single feedback arrow from CL loop back up into the classifier (self-improve)
arrow(ax, (60, cly + 3.3), (69, EY + 4.3), color=PAL["pink"], lw=1.2, rad=0.28)
label(ax, 54, cly + 7.0, "self-improve", fs=4.6, color=PAL["pink"], italic=True)

save(fig, "fig1_architecture")
