"""Fig 1 -- LEAFSENSE system architecture (offline-first edge tier).

Layout contract (keep these invariants if you edit):
  * Three horizontal bands inside one EDGE container, top to bottom:
        PERCEPTION (y=PY)  ->  FUSION/OUTPUT (y=FY)  ->  ADAPTATION (y=AY)
  * Flow is left-to-right within a band, top-to-bottom between bands.
  * NO arrow crosses another arrow. The self-improve feedback runs along a
    dedicated channel BELOW the adaptation band and re-enters the pipeline
    vertically, so it never cuts through the fusion arrows.
  * The container title owns the strip above TITLE_Y; nothing is drawn into it.
"""
from matplotlib.patches import FancyBboxPatch
from ls_style import PAL, COL2, save
from ls_diagram import canvas, box, arrow, label

W, H = COL2, 3.55
fig, ax = canvas(W, H)
YT = 100 * H / W                      # ~46.8 usable y-units

# ---------------------------------------------------------------- geometry
CX, CW = 50, 96                       # container centre / width
CTOP, CBOT = YT - 1.5, 3.0            # container top / bottom edges
TITLE_Y = CTOP - 3.0                  # title baseline; keep content below this

PY = CTOP - 10.0                      # perception band
FY = PY - 13.5                        # fusion / output band
AY = FY - 12.5                        # adaptation band
FEED_Y = AY - 6.2                     # feedback return channel (below)

BH = 7.4                              # standard box height
BLUE_FC, ORANGE_FC = "#eef3f8", "#fff5e6"


def container(cx, top, bot, w, title, ec, fc="white"):
    h = top - bot
    ax.add_patch(FancyBboxPatch((cx - w / 2, bot), w, h,
                 boxstyle="round,pad=0,rounding_size=1.4", linewidth=1.4,
                 edgecolor=ec, facecolor=fc, zorder=0))
    ax.text(cx - w / 2 + 2.0, TITLE_Y, title, ha="left", va="center",
            fontsize=6.6, fontweight="bold", color=ec, zorder=1)


def band_label(y, text):
    """Left-margin tier label, rotated, outside the flow."""
    ax.text(3.6, y, text, ha="center", va="center", fontsize=5.0,
            color=PAL["grey"], fontweight="bold", rotation=90, zorder=1)


container(CX, CTOP, CBOT, CW,
          "EDGE  —  NVIDIA Jetson Orin Nano Super  (offline-first)",
          PAL["ink"])

# ======================================================== PERCEPTION band
band_label(PY, "PERCEPTION")

CAMW, SW, GW = 15.0, 10.0, 8.4        # camera / stage / GAN widths
cam = box(ax, 14.0, PY, CAMW, BH, "Rotating camera\n360° / 45°  (8 pos)",
          fc=BLUE_FC, ec=PAL["blue"], fs=4.8)

stages = [(31.0, "YOLOv8n\ndetect"), (45.0, "crop\n≥124²"),
          (59.0, "YOLOv8n\n-seg"), (86.0, "MobileNetV3\nclassify")]
for x, t in stages:
    box(ax, x, PY, SW, BH, t, fc=BLUE_FC, ec=PAL["blue"], fs=4.7)

# optional GAN: INLINE in the lane between seg and classify, dashed.
GX = 72.5
box(ax, GX, PY, GW, BH, "GAN\n(opt.)", fc="#f4f4f4", ec="#8c8c8c",
    ls=(0, (3, 2)), fs=4.4, tc="#5a5a5a")

arrow(ax, (14.0 + CAMW / 2, PY), (31.0 - SW / 2, PY), color="#555", lw=0.9)
for a, b in [(31.0, 45.0), (45.0, 59.0)]:
    arrow(ax, (a + SW / 2, PY), (b - SW / 2, PY), color="#555", lw=0.9)
arrow(ax, (59.0 + SW / 2, PY), (GX - GW / 2, PY),
      color="#8c8c8c", lw=0.8, ls=(0, (3, 2)))
arrow(ax, (GX + GW / 2, PY), (86.0 - SW / 2, PY),
      color="#8c8c8c", lw=0.8, ls=(0, (3, 2)))

# =================================================== FUSION / OUTPUT band
band_label(FY, "FUSION")

SENW, RISKW, FUSEW, DASHW = 15.0, 14.0, 13.0, 14.5
sens = box(ax, 14.0, FY, SENW, BH, "Sensors: CO₂, temp,\nhumidity, soil",
           fc=ORANGE_FC, ec=PAL["orange"], fs=4.8)
risk = box(ax, 37.0, FY, RISKW, BH, "Decision-tree\nenv. risk",
           fc=ORANGE_FC, ec=PAL["orange"], fs=4.8)
fuse = box(ax, 59.5, FY, FUSEW, BH + 1.2, "Label-level\nfusion",
           fc="#eef7f2", ec=PAL["green"], bold=True, fs=5.2)
dash = box(ax, 84.0, FY, DASHW, BH, "Streamlit dashboard\n+ rule insights",
           fc="white", ec=PAL["ink"], fs=4.7)

arrow(ax, (14.0 + SENW / 2, FY), (37.0 - RISKW / 2, FY), color="#555", lw=0.9)
arrow(ax, (37.0 + RISKW / 2, FY), (59.5 - FUSEW / 2, FY), color="#555", lw=0.9)
arrow(ax, (59.5 + FUSEW / 2, FY), (84.0 - DASHW / 2, FY), color="#555", lw=0.9)

# classify -> fusion: down then left then down. Orthogonal, no crossing.
# The vertical leg drops at CJX, INSIDE the classify box footprint but left of
# the pink low-confidence channel, so the two never intersect.
CJX = 83.5
JY = (PY - BH / 2 + FY + (BH + 1.2) / 2) / 2
arrow(ax, (CJX, PY - BH / 2 - 0.15), (CJX, JY), color="#555", lw=0.9, style="-",
      shrink=0)
arrow(ax, (CJX, JY), (59.5, JY), color="#555", lw=0.9, style="-")
arrow(ax, (59.5, JY), (59.5, FY + (BH + 1.2) / 2), color="#555", lw=0.9)
label(ax, 70.5, JY + 1.8, "per-class leaf counts", fs=4.3, color="#666")

# ======================================================== ADAPTATION band
band_label(AY, "ADAPTATION")

ABX, ABW, ABH = 48.0, 72.0, 6.8
box(ax, ABX, AY, ABW, ABH,
    "Confidence routing  →  retrain buffer / relabel queue  →  "
    "monthly EWC · metric gate · engine swap",
    fc="#f7f0f7", ec=PAL["pink"], fs=4.8)

# low-confidence crops leave the classifier's right edge and drop down the far
# right channel into adaptation -- outside every other route, no crossings.
RCH = 93.0                                     # right-hand channel
arrow(ax, (86.0 + SW / 2, PY), (RCH, PY), color=PAL["pink"], lw=1.0, style="-")
arrow(ax, (RCH, PY), (RCH, AY), color=PAL["pink"], lw=1.0, style="-")
arrow(ax, (RCH, AY), (ABX + ABW / 2, AY), color=PAL["pink"], lw=1.0)
label(ax, 91.0, FY - 8.5, "low-confidence crops", fs=4.3, color=PAL["pink"],
      italic=True, ha="center")
ax.texts[-1].set_rotation(90)

# self-improve: dedicated channel BELOW the band, re-enters classify vertically
# at RTN -- left of the grey jog (CJX) and of the pink outbound channel (RCH).
LCH, RTN = 8.5, 89.0
arrow(ax, (ABX - ABW / 2, AY), (LCH, AY), color=PAL["pink"], lw=1.2, style="-")
arrow(ax, (LCH, AY), (LCH, FEED_Y), color=PAL["pink"], lw=1.2, style="-")
arrow(ax, (LCH, FEED_Y), (RTN, FEED_Y), color=PAL["pink"], lw=1.2, style="-")
arrow(ax, (RTN, FEED_Y), (RTN, PY - BH / 2), color=PAL["pink"], lw=1.2)
label(ax, 47.0, FEED_Y - 2.0, "self-improve: updated engine swapped in",
      fs=4.5, color=PAL["pink"], italic=True)

save(fig, "fig1_architecture")
