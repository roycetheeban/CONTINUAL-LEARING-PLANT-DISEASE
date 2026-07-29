# Figure Production Guide — 9 Figures

Labels are final and match `04_FULL_PAPER_LATEX.md`. For each: what tool, what exactly to draw, and the IEEE Access constraints.

---

## ⭐ BUILD STATUS (updated 2026-07-29)

**All 8 figures are generated and finalised** — Fig. 9 was **removed from the paper** (see below), so the set is now 8, not 9. They are as vector PDFs in `journal/figures/`, produced **programmatically with Python + Matplotlib** (not draw.io — we generated the diagrams as code so they are reproducible and consistent with the data plots). Each was visually inspected and polished.

| Fig | File | Status | How it was made | Data source |
|---|---|---|---|---|
| 1 | `fig1_architecture.pdf` | ✅ **FINAL** | matplotlib (`src/fig1_architecture.py`) | conceptual |
| 2 | `fig2_pipeline.pdf` | ✅ **FINAL** | matplotlib (`src/fig2_pipeline.py`) | conceptual |
| 3 | `fig3_cl_cycle.pdf` | ✅ **FINAL** | matplotlib (`src/fig3_cl_cycle.py`) | conceptual |
| 4 | `fig4_layer_groups.pdf` | ✅ **FINAL** | matplotlib (`src/fig4_layer_groups.py`) | conceptual |
| 5 | `fig5_fusion.pdf` | ✅ **FINAL** | matplotlib (`src/fig5_fusion.py`) | **E1 + E2 — real.** Panel (a): real Case2+Replay classifier predictions on real held-out tomato images (constructed episode sequencing). Panel (b): real E1 hourly sensor trace. Panel (c): real Φ inputs/output, with an explicit Φ rule box. Revised 2026-07-29: absolute leaf count ("+7 leaves") replaces the misleading "+700%"; arrows/box widths fixed. |
| 6 | `fig6_cata.pdf` | ✅ **FINAL** | matplotlib (`src/make_data_figs.py`) | **Table VI — real** |
| 7 | `fig7_lambda.pdf` | ✅ **FINAL** | matplotlib (`src/make_data_figs.py`) | **Table VII — real** |
| 8 | `fig8_catb_scatter.pdf` | ✅ **FINAL** | matplotlib (`src/make_data_figs.py`) | **Table X — real** |
| ~~9~~ | ~~`fig9_deployment.pdf`~~ | ❌ **REMOVED 2026-07-29** | — | No unit is installed, so the deployment photograph cannot be produced, and a dashboard screenshot alone does not carry the claim. The figure was never referenced from the body text, so nothing else depended on it. Deployment-conditional wording elsewhere was softened at the same time. **If a unit is later installed, restore this figure AND revisit those sentences.** |

**Reproduce / edit any finalised figure:** scripts live in `journal/figures/src/` (`ls_style.py` = shared palette + rcParams; `ls_diagram.py` = box/arrow helpers). Run e.g. `python make_data_figs.py` or `python fig1_architecture.py`. PNG previews are written to `journal/figures/_preview/`.

### What still needs refinement / your input

| Item | Action needed |
|---|---|
| ~~Fig 9~~ | **Removed** — no longer an outstanding item. |
| Fig 6/7/8 numbers | Already real — no change unless the underlying CL tables change. |
| Optional Fig 6 add-on | A second panel with training-time (log axis) was left out for clarity; add if you want the EWC-efficiency point shown visually. |
| ~~Optional Fig 5/9 palette pass~~ | Moot — Fig 9 removed. Fig 5 uses the shared `ls_style.py` palette, consistent with Figs 1–8. |

> **Note:** the per-figure draw.io specs below (Figs 1–5) are kept for reference and manual editing only. All 8 figures are already built as code (Fig 5 via `src/fig5_fusion.py`) — the draw.io route is optional if you'd rather hand-edit one. **The Fig. 9 spec further below is retained only as a build recipe should hardware become available; the figure is not currently in the paper.**

---

## Tool summary

| Figures       | Tool                                                                | Why                                                                        |
| ------------- | ------------------------------------------------------------------- | -------------------------------------------------------------------------- |
| 1, 2, 3, 4, 5 | **draw.io** (diagrams.net, free, desktop or app.diagrams.net) | Block diagrams & flowcharts. Exports true vector PDF.                      |
| 6, 7, 8       | **Python + Matplotlib**                                       | Data plots — must come from your real metrics JSON, never redrawn by hand |
| 9             | **Camera + screenshot**, composed in draw.io or PowerPoint    | Photo panel + dashboard screenshot                                         |

**Alternatives if you prefer:** Figma or Excalidraw for 1–5 (Excalidraw looks hand-drawn — avoid for a journal); Inkscape if you want full vector control; PowerPoint works but export quality is worse — only use it for the (now removed) Fig. 9 recipe.

## IEEE Access rules that apply to every figure

- **Vector PDF preferred** (`File → Export as → PDF`, tick *Crop to diagram*). If raster, **600 dpi minimum**, PNG not JPG.
- **Width:** single column ≈ **88 mm**, double column (`figure*`) ≈ **181 mm**. Fig. 1 is the only `figure*`.
- **Fonts:** sans-serif (Arial/Helvetica), **minimum 8 pt after scaling down to column width**. This is the #1 rejection cause — draw at final size, don't shrink a big diagram.
- **Must survive greyscale printing.** Never encode meaning in colour alone — always pair colour with a shape, pattern, or label.
- **Colour-blind safe palette** (use these hex values throughout, all 8 figures):
  `#0072B2` blue · `#E69F00` orange · `#009E73` green · `#D55E00` vermillion · `#CC79A7` pink · `#56B4E9` sky · `#999999` grey
- No figure titles inside the image — the LaTeX `\caption{}` is the title.
- Save editable sources in `journal/figures/src/`, exports in `journal/figures/`.

---

# Fig. 1 — LEAFSENSE System Architecture

**Tool: draw.io** · **Double column (`figure*`), 181 mm wide** · file: `fig1_architecture.pdf`

The paper's offline-first claim must be *visible* here: the edge tier should dominate the frame and the cloud tier should look clearly optional.

**Layout — two horizontal bands:**

*Upper band (~25% height) — "CLOUD (optional, subscription)":* draw in light grey fill `#F0F0F0` with a **dashed border**. Three small boxes: `Relabelling Service`, `Model Registry / OTA Updates`, `RAG Chatbot + Daily Insights`. Keep these visually small and plain.

*Connector between bands:* a single **dashed** bidirectional arrow labelled `intermittent sync — relabel queue, model updates`. Dashed = not required for operation. This one detail carries the whole offline-first argument.

*Lower band (~75% height) — "EDGE — NVIDIA Jetson Orin Nano":* solid dark border, white fill, visually dominant. Inside, left-to-right:

1. **Inputs column:** `Rotating Camera 360°/45° → 8 positions` and below it `Sensors: CO₂ · Temp · Humidity · Soil Moisture`. Small camera and thermometer icons help (draw.io shape library → search "camera", "sensor").
2. **Image path** (horizontal chain, use `#0072B2` blue borders): `YOLOv8n Detect → Crop + Filter ≥124×124 → YOLOv8n-Seg → [GAN — optional]` — draw the GAN box **dashed with grey fill** and a small tag `off by default` → `MobileNetV3-Small Classify`.
3. **Sensor path** (parallel, below, `#E69F00` orange borders): `Timestamp Match → Decision Tree Risk`.
4. **Convergence:** both paths arrow into one box `Label-Level Fusion` (`#009E73` green, make it slightly larger — it's a contribution).
5. **Outputs:** `Streamlit Dashboard (local)` and `Rule-Based Insights`.
6. **Adaptation loop** — draw as a distinct sub-block, ideally in a rounded container labelled `Continual Learning Loop`: from the classifier, `Confidence Routing` splits into `Retrain Buffer (auto-labelled)` and `Relabel Queue (reviewed)`; both feed `Monthly EWC/Replay Retrain → Metric Gate → Engine Swap`; a **feedback arrow curves back to the classifier box**. That curved arrow is what makes the system look self-improving rather than linear — don't omit it.

**Tips:** `Arrange → Layout → Horizontal Flow` to align the chain, then nudge manually. Use `Edit → Find/Replace style` to keep colours consistent. Turn on grid snapping.

---

# Fig. 2 — Three-Stage Inference Pipeline with Confidence Routing

**Tool: draw.io** · **Single column, 88 mm** · file: `fig2_pipeline.pdf`

Vertical top-to-bottom flow (fits a narrow column better than horizontal):

```
Input image (any resolution)
   ↓
Letterbox → 640×640 (aspect preserved, black pad)
   ↓
YOLOv8n Detection                    ← side annotation: mAP@0.5 = 0.632
   ↓
Crop leaves │ Quality filter ≥124×124   ← annotate: "smaller crops discarded"
   ↓
YOLOv8n-Seg  →  mask, background → black   ← 99.5% pixel acc · 0.915 mIoU
   ↓
Resize 200×200 → pad 224×224
   ↓
┌ ─ ─ Lightweight GAN (optional) ─ ─┐    ← DASHED box, grey, tag "disabled by default"
   ↓
MobileNetV3-Small Classifier          ← 89.81% (5-class turmeric)
   ↓
Per-class leaf counts {healthy: 41, blotch: 3, leafspot: 5, dry: 2, aphid: 0}
```

**Right-hand branch off the classifier** (this is the part that matters — it's the data engine):

- `conf ≥ τ_high` → `Retrain Buffer (auto-labelled, per class)` — green `#009E73`
- `conf < τ_low` → `Relabel Queue (never trained unreviewed)` — orange `#E69F00`

Put those two boxes to the right of the main spine with a small diamond or split point. Use italic small text for the metric annotations so they read as callouts, not flow steps.

---

# Fig. 3 — On-Device Monthly CL Cycle with Metric Gate

**Tool: draw.io** (use *Flowchart* shape library) · **Single column, 88 mm** · file: `fig3_cl_cycle.pdf`

Standard flowchart symbols: rounded rectangle = start/end, rectangle = process, diamond = decision.

```
( Monthly trigger  OR  dashboard "Retrain" button )
   ↓
[ Load retrain buffer + returned relabelled samples ]
   ↓
[ EWC / Replay fine-tune — PyTorch, on-device
  layer-wise LR schedule, checkpoint every epoch ]     ← side note: "power-cut safe"
   ↓
[ Export classifier.onnx ]                              ← "~seconds"
   ↓
[ trtexec → classifier_candidate.engine (FP16) ]        ← "~1–2 min"
   ↓
[ Evaluate candidate ENGINE on FIXED held-out test set ]
   ↑ callout box, highlighted: "evaluated through the same
     TensorRT FP16 runtime that will serve it"
   ↓
< Metrics better than incumbent? >
   ├── YES → [ Atomic file swap + bump manifest version ]
   │            → ( Deployed. Previous engine kept for rollback )
   └── NO  → [ Keep incumbent; carry data to next cycle ] ──┐
                                                            │
   └──────────── loop arrow back to top ────────────────────┘
```

Shade the evaluation box and its callout in a highlight colour — it's the novel part of the loop and reviewers should see it immediately. Colour the YES path `#009E73` green and the NO path `#D55E00` vermillion, but **also label both arrows in text** so it survives greyscale.

---

# Fig. 4 — MobileNetV3-Small Layer Groups G1–G4 & Split-LR Head

**Tool: draw.io** · **Single column, 88 mm** · file: `fig4_layer_groups.pdf`

**Main panel:** horizontal stack of blocks representing the network, colour-banded into four groups. Draw ~13 small uniform blocks left to right, then group them with brackets underneath:

| Group    | Blocks          | Colour                 | Label under bracket                                              |
| -------- | --------------- | ---------------------- | ---------------------------------------------------------------- |
| G1 Early | features[0–3]  | `#56B4E9` sky        | `Edges, colour gradients` / **FROZEN always**            |
| G2 Mid   | features[4–8]  | `#009E73` green      | `Venation, texture` / **Frozen (CatA)**                  |
| G3 Late  | features[9–12] | `#E69F00` orange     | `Lesion shape, class margins` / **LOW LR — EWC target** |
| G4 Head  | classifier      | `#D55E00` vermillion | `Class probabilities` / **Split-LR**                     |

Add a **padlock icon** on G1/G2 and a **flame or gradient icon** on G3 — this makes the frozen/plastic distinction readable in greyscale, not just by colour.

Label G3 prominently: `PRIMARY FORGETTING SITE`. That's the paper's core methodological claim.

**Inset panel (right side or below):** the head expanding 5 → 7. Draw 5 old neurons in one colour with a label `LR = 1e-4 (protected)` and 2 new neurons in a contrasting colour with `LR = 1e-3 (fast acquisition)`. Add the annotation `+2,050 params (+0.13%)`. Use a bracket or dashed outline to show the 2 new neurons were appended.

---

# Fig. 5 — Label-Level Fusion Example Chain

**Tool: hybrid — Matplotlib for the two data panels, draw.io to compose with the fusion box** · **Single column, 88 mm** · file: `fig5_fusion.pdf`

Easiest route: generate panels (a) and (b) in Matplotlib as one stacked PDF, then import into draw.io and add panel (c) below it. Or do the whole thing in Matplotlib with a text box drawn via `ax.annotate` — also fine and more reproducible.

**Panel (a) — per-class counts over time:** 5 lines (one per class), x-axis Day 1–7, y-axis leaf count. Make `leaf_spot` visibly rise 3 → 5 → 9 in `#D55E00`; keep the other four flat and muted grey. Use distinct markers (○ □ △ ◇ ×) so lines are distinguishable in greyscale.

**Panel (b) — aligned environmental trace:** same x-axis (critical — align them exactly). Humidity line sustained above 85% with a shaded band above the 85% threshold; soil moisture as a second line or a small bar strip.

**Panel (c) — the fusion box:** two inputs merging into one output.

- Left input: `Δ leaf_spot = +6 over 3 days ↑`
- Right input: `Risk state = HIGH (humidity > 85% sustained)`
- → Output box (`#009E73`): **"Elevated fungal risk — inspect and consider fungicide"**

Add small `(a)`, `(b)`, `(c)` labels in the top-left of each panel. (The original spec said to colour-match the dashboard in Fig. 9; **Fig. 9 has since been removed**, so Fig. 5 simply uses the shared `ls_style.py` palette.)

---

# Fig. 6 — CatA Accuracy Across Cases and Methods

**Tool: Python + Matplotlib** · **Single column, 88 mm** · file: `fig6_cata.pdf`
**Data: Table III (already real).**

Grouped bar chart. X-axis: 3 groups (Case 1 Scratch / Case 2 ImageNet / Case 3 PlantVillage). 4 bars per group: EWC, Replay, Isolation, Naive.

**Critical:** set `ax.set_ylim(95, 99)` — at full 0–100 scale every bar looks identical and the figure says nothing. Add a note in the caption that the axis is truncated (reviewers expect this disclosure).

- Bar colours: EWC `#0072B2`, Replay `#009E73`, Isolation `#E69F00`, Naive `#999999`.
- **Hatch the Naive bars** (`hatch='//'`) and label them `reference only` — it is not a valid CL method and must not read as a competitor.
- Value labels on top of each bar (`ax.bar_label(fmt='%.2f')`).
- Star ★ annotation on Case 2 + Replay (98.25%).
- Optional second panel below sharing the x-axis: training time in seconds on a **log scale**. This makes the EWC efficiency argument visual — 171 s vs 620 s is the practical headline, and a log axis shows it cleanly.

```python
plt.rcParams.update({'font.family':'sans-serif','font.size':8,'figure.figsize':(3.46,2.6)})
# 3.46 in = 88 mm. Save with: plt.savefig('fig6_cata.pdf', bbox_inches='tight')
```

---

# Fig. 7 — λ Sensitivity Curve

**Tool: Python + Matplotlib** · **Single column, 88 mm** · file: `fig7_lambda.pdf`
**Data: Table IV (already real).**

Two lines over λ ∈ {1000, 3000, 5000, 8000, 10000, 15000}, **x-axis log scale** (`ax.set_xscale('log')`).

- Cycle 1 accuracy — solid line, circle markers, `#0072B2`
- Cycle 2 accuracy — dashed line, square markers, `#E69F00`
- Mark each optimum with a large ★: C1 at λ=10,000 (97.98%), C2 at λ=3,000 (98.24%).
- Annotate the two failure regimes with light shaded background spans: left region `under-regularized — weights drift`, right region `over-regularized — plasticity lost`. Use `ax.axvspan(..., alpha=0.1)`.
- Y-limits roughly 97.4–98.4 so the variation is legible.

The story the figure must tell in one glance: **the optimum moves left between cycles.** Consider a small curved arrow from the C1 star to the C2 star labelled `optimum shifts down as model converges`.

---

# Fig. 8 — CatB Stability–Plasticity Scatter ★ KEY FIGURE

**Tool: Python + Matplotlib** · **Single column, 88 mm** · file: `fig8_catb_scatter.pdf`
**Data: Table VI (already real).**

This is the most important figure in the paper — it makes three findings visible simultaneously. Invest the most time here.

- **X-axis:** old-class retention (%), range ~80–101
- **Y-axis:** new-class accuracy (%), range ~0–100
- **One marker per case/method.** Encode **method by marker shape** (EWC ○, Replay ■, Isolation ▲, Naive ×, Hybrid ◆) and **case by colour** (Case 1 `#56B4E9`, Case 2 `#0072B2`, Case 3 `#009E73`). Shape-encoding the method is what makes it greyscale-safe.
- **Shade the top-right corner** lightly green with the label `desirable: high plasticity + high stability`.
- **Circle Case 2 + Replay** (97.19, 98.07) and annotate it `Best — 98.07 / 97.19`.
- **Annotate the isolation cluster** at the far right, bottom: they sit at ~100% retention but 15–51% new accuracy. Label that region `zero forgetting, no acquisition`. The visual absurdity of those points is the entire failure-analysis argument.
- Draw a **dashed Pareto frontier** through the non-dominated points.
- Optionally a faint diagonal line where new = old, to show balance.

---

# ~~Fig. 9~~ — Greenhouse Deployment & Dashboard  ❌ REMOVED FROM THE PAPER (2026-07-29)

> Retained below only as a **build recipe** in case a unit is installed later. This figure is **not** in the current draft — there is no deployment to photograph, and the paper's deployment wording was softened to conditional to match. Restoring it means restoring those sentences too.

**Tool: camera + screenshot, composed in draw.io or PowerPoint** · **Single column, 88 mm** · file: `fig9_deployment.pdf`

**Panel (a) — deployment photograph.** Shoot the installed unit in the greenhouse. Requirements: good even lighting (avoid harsh backlight through greenhouse glazing); the Jetson enclosure, the rotating camera mount, and at least one sensor probe all visible in frame; include something for scale. Add small leader labels with arrows: `Jetson Orin Nano`, `Rotating camera mount (45° steps)`, `Environmental sensors`. Shoot at highest resolution; crop, don't zoom.

**Panel (b) — dashboard screenshot.** Capture the Streamlit dashboard showing: the 5-line per-class daily count trend graph, the current fused risk state, and the `Retrain` / `Capture Now` controls. Crop tightly to the useful region — a full browser window with address bar and whitespace wastes the column. If the on-screen text will be unreadable at 88 mm, screenshot a **zoomed browser view** (Ctrl + `+` to ~150% before capturing) rather than shrinking a wide screenshot.

Compose side by side or stacked, label `(a)` and `(b)`. Export at 600 dpi minimum. Photographs must be PNG or TIFF — **not JPG**, IEEE flags compression artefacts.

**Privacy:** if any person appears in the photo, get consent or crop them out.

---

## Suggested build order

1. **Figs 6, 7, 8 first** — data already exists, pure scripting, and Fig. 8 will tell you whether the CatB story lands.
2. **Figs 2, 4** — smallest diagrams, good for settling your draw.io colour/font style.
3. **Figs 1, 3** — largest diagrams, reuse the style from step 2.
4. **Fig. 5** — needs the decision-tree work (E1) finished so the risk state in panel (c) is real.
5. ~~**Fig. 9**~~ — removed; no longer part of the build order.

Write one Matplotlib style block and `%run` it at the top of all three plot scripts so Figs 6–8 are visually identical siblings.
