"""Fig 4 -- MobileNetV3-Small functional layer groups G1-G4 + split-LR head."""
from ls_style import PAL, COL1, save
from ls_diagram import canvas, box, arrow, label

W, H = COL1, 3.05
fig, ax = canvas(W, H)
YT = 100 * H / W  # top of canvas (~88)

# ---- title strip ------------------------------------------------------------
label(ax, 50, YT - 4, "MobileNetV3-Small backbone", fs=7, color="#111", bold=True)

# ---- backbone: 4 group bands ------------------------------------------------
# Convention (matches Figs 1-3): PALE TINT fill + saturated palette EDGE. Filling
# with the raw palette colour makes this figure louder than the rest of the paper
# and puts dark text on saturated ground, which fails contrast. Tint + edge keeps
# the Okabe-Ito hue coding while staying legible in greyscale.
groups = [
    ("G1 Early", "feat[0–3]", "edges,\ncolour", "FROZEN\n(always)",
     "#e7f2fa", PAL["sky"]),
    ("G2 Mid",   "feat[4–8]", "venation,\ntexture", "frozen\n(CatA)",
     "#e6f5f0", PAL["green"]),
    ("G3 Late",  "feat[9–12]", "lesion,\nclass margin", "LOW LR\nEWC target",
     "#fdf0dc", PAL["orange"]),
    ("G4 Head",  "classifier", "class\nprobs", "split-LR\n(CatB)",
     "#fbeadf", PAL["vermillion"]),
]
x0, x1 = 6, 94
n = len(groups)
gw = (x1 - x0) / n
yb = YT - 34          # band vertical centre
bh = 15
centres = []
for i, (name, blk, feat, status, fc, ec) in enumerate(groups):
    cx = x0 + gw * (i + 0.5)
    centres.append(cx)
    # G3 is the primary forgetting site: emphasised by a heavier rule, not by a
    # different colour, so the emphasis survives greyscale printing.
    lw = 2.0 if name.startswith("G3") else 1.2
    box(ax, cx, yb, gw * 0.86, bh, "", fc=fc, ec=ec, lw=lw, round=True, z=2)
    label(ax, cx, yb + 3.0, name, fs=6.2, color="#111", bold=True, z=4)
    label(ax, cx, yb - 0.2, blk, fs=5.2, color="#111", z=4)
    label(ax, cx, yb - 3.6, status, fs=4.9, color="#111", z=4)
    # feature-type caption above each band
    label(ax, cx, yb + bh / 2 + 4.2, feat, fs=4.9, color="#666", z=4)

# flow arrows through the backbone
for i in range(n - 1):
    arrow(ax, (centres[i] + gw * 0.43, yb), (centres[i + 1] - gw * 0.43, yb),
          color="#555", lw=1.0)
# input / output stubs
arrow(ax, (x0 - 4.5, yb), (x0 - 0.3, yb), color="#555", lw=1.0)
label(ax, x0 - 5.5, yb, "224²\n×3", fs=4.8, color="#666", ha="right")
arrow(ax, (x1 + 0.3, yb), (x1 + 4.5, yb), color="#555", lw=1.0)

# G3 = primary forgetting site callout
label(ax, centres[2], yb - bh / 2 - 3.3, "primary forgetting site",
      fs=5.2, color=PAL["vermillion"], bold=True)

# ---- split-LR head inset ----------------------------------------------------
iy = yb - 30
box(ax, 50, iy, 84, 20, "", fc="#f7f7f7", ec="#bbb", lw=1.0, round=True, z=1)
label(ax, 50, iy + 7.5, "Classifier head expansion  5 → 7  (+2,050 params, 0.13%)",
      fs=5.6, color="#111", bold=True, z=3)

# old neurons (grey, low LR) + new (vermillion, high LR)
ny = iy - 1.5
oldx = [18, 25, 32, 39, 46]
for x in oldx:
    box(ax, x, ny, 4.4, 4.4, "", fc="#dedede", ec="#777", lw=0.8, round=True, z=3)
label(ax, 32, ny - 5.2, "old classes T1–T5   (LR ↓ protected)",
      fs=4.9, color="#333", z=4)
newx = [66, 73]
for x in newx:
    box(ax, x, ny, 4.4, 4.4, "", fc="#fbeadf", ec=PAL["vermillion"], lw=1.1,
        round=True, z=3)
label(ax, 69.5, ny - 5.2, "new T6–T7   (LR ↑ fast)", fs=4.9, color="#333", z=4)
# divider bracket
arrow(ax, (52, ny), (61, ny), color="#999", lw=0.9, style="-|>")

# arrow from G4 down to inset
arrow(ax, (centres[3], yb - bh / 2 - 6.0), (72, iy + 10.2), color="#888",
      lw=1.0, rad=-0.2)

save(fig, "fig4_layer_groups")
