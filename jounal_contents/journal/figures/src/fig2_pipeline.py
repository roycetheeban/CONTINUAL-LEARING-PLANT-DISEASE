"""Fig 2 -- three-stage inference pipeline with confidence routing."""
from ls_style import PAL, COL1, save
from ls_diagram import canvas, box, arrow, label

W, H = COL1, 4.85
fig, ax = canvas(W, H)
YT = 100 * H / W

X = 30           # main spine
MW = 46          # main box width

rows = [
    (136, 8,  "Input image (camera / upload)", "white", "#333", None),
    (124, 8,  "Letterbox → 640×640", "white", "#333", None),
    (111, 9,  "YOLOv8n detection", "#eef3f8", PAL["blue"], "mAP@0.5\n0.632"),
    (98, 9,   "Crop leaves + ≥124² filter", "white", "#333", None),
    (85, 10,  "YOLOv8n-seg\n(mask · bg → black)", "#eef3f8", PAL["blue"], "99.5% px\n0.915 mIoU"),
    (72, 8,   "Resize 200² → pad 224²", "white", "#333", None),
]
for y, h, txt, fc, ec, met in rows:
    box(ax, X, y, MW, h, txt, fc=fc, ec=ec, fs=5.7, bold=(fc != "white"))
    if met:
        label(ax, X - MW / 2 - 3, y, met, fs=4.7, color="#888", ha="right", italic=True)

# optional GAN (dashed, greyed)
box(ax, X, 59, MW, 9, "GAN occlusion recovery", fc="#f2f2f2", ec="#999",
    ls="--", fs=5.5)
label(ax, X - MW / 2 - 3, 59, "off by\ndefault", fs=4.7, color="#999", ha="right", italic=True)

# classifier + output
box(ax, X, 44, MW, 10, "MobileNetV3-Small\nclassifier", fc="#eef3f8",
    ec=PAL["blue"], bold=True, fs=5.7)
label(ax, X - MW / 2 - 3, 44, "89.81%\n5-class", fs=4.7, color="#888", ha="right", italic=True)
box(ax, X, 28, MW, 10, "Per-class leaf counts\n{healthy, blotch, spot, …}",
    fc="#eef7f2", ec=PAL["green"], bold=True, fs=5.5)

# main spine arrows (fixed gaps between box edges)
spine = [(132, 128), (120, 115.5), (106.5, 102.5), (93.5, 90),
         (80, 76), (68, 63.5), (54.5, 49), (39, 33)]
for a, b in spine:
    arrow(ax, (X, a), (X, b))

# ---- confidence routing branch (right) ----
RX = 78
box(ax, RX, 50, 40, 12, "conf ≥ τ_high →\nRetrain buffer\n(auto-labelled)",
    fc="#e8f5f0", ec=PAL["green"], fs=5.2, bold=True)
box(ax, RX, 32, 40, 12, "conf < τ_low →\nRelabel queue\n(human/LLM review)",
    fc="#fbeee7", ec=PAL["vermillion"], fs=5.2, bold=True)
# split point
sx, sy = X + MW / 2, 44
label(ax, 62, 58, "route by\nconfidence", fs=4.8, color="#555", italic=True)
arrow(ax, (sx, sy), (RX - 20, 50), color=PAL["green"], rad=-0.15)
arrow(ax, (sx, sy - 2), (RX - 20, 32), color=PAL["vermillion"], rad=-0.2)

save(fig, "fig2_pipeline")
