# 08 — Pending Work: Ordered Path to Submission

**Purpose:** everything still outstanding for the journal paper, in execution order, with the full flow for each.
**Companion:** [`07_CHANGES_MADE.md`](07_CHANGES_MADE.md) — what has already been done and why it deviated from plan.
**Scope:** journal paper only. Production/Jetson app work is tracked separately in `app_dev/jetson_flow.md`.

---

## Priority overview

| #  | Item                                         | Effort     | Blocks submission?                | Depends on              |
| -- | -------------------------------------------- | ---------- | --------------------------------- | ----------------------- |
| 1  | Compile & verify layout                      | 30 min     | **Yes**                     | —                      |
| 2  | Fill §VI-C implementation details           | 1 h        | **Yes**                     | —                      |
| 3  | Refs [21] [22] [27] venues/DOIs              | 1 h        | **Yes**                     | your publication status |
| 4  | Verify all 35 refs on IEEE Xplore            | 2 h        | **Yes**                     | —                      |
| 5  | Fix stale table numbers in planning docs     | 20 min     | No                                | —                      |
| 6  | Fig. 9 deployment + dashboard                | 2–4 h     | **Yes** (figure referenced) | hardware access         |
| 7  | i                                            | 1–2 days  | **Yes** (table referenced)  | —                      |
| 8  | Author bios + photos                         | 1 h        | **Yes**                     | co-authors              |
| 9  | Final formatting / page-count pass           | 2 h        | **Yes**                     | all above               |
| 10 | *(Optional)* Real Jetson E3 re-measurement | 1 day      | No                                | Jetson hardware         |
| 11 | *(Optional)* Real sensor logs for E1       | 4–8 weeks | No                                | deployed unit           |

**Critical path:** items 1 → 2 → 3 → 4 → 6 → 7 → 8 → 9.
Items 5, 10, 11 can run in parallel or be skipped.

---

# 1. Compile and verify layout ⚠️ DO THIS FIRST

**Why first:** table-collision fixes from `07_CHANGES_MADE.md` §17 were made blind — there is no LaTeX toolchain on the dev machine. Everything downstream assumes the paper renders correctly.

**Flow:**

1. Upload `jounal_contents/journal/leafsense.tex` + the `figures/` folder to Overleaf.
2. Compile with pdfLaTeX. Expect zero errors; warnings about float placement are acceptable.
3. Check specifically:
   - Tables XI, XII, XIII no longer overlap (the reported bug)
   - Table XIII footnotes (`†`/`‡`) render below the table, inside the float
   - Fig. 5 panel (c) is legible at single-column width (88 mm)
   - No table runs past the column margin
4. If Table XIII still crowds → switch it to `\begin{table*}` (spans both columns).
5. Confirm page count is near the ~15-page target.

**Done when:** clean compile, no overlaps, page count acceptable.

---

# 2. Fill §VI-C implementation details

**Why it matters:** IEEE Access reviewers routinely ask for this, and its absence is an easy desk-reject-adjacent complaint. Currently a placeholder comment in `leafsense.tex`.

**What to supply:**

| Field                                 | Where to find it                                                              |
| ------------------------------------- | ----------------------------------------------------------------------------- |
| GPU model used for benchmark training | your training machine spec                                                    |
| PyTorch / CUDA / torchvision versions | `environment.yml` → torch 2.5.1+cu121, torchvision 0.20.1+cu121, CUDA 12.1 |
| Optimizer                             | training scripts →**Adam**                                             |
| LR schedule                           | `ReduceLROnPlateau`, patience 3, factor 0.5, min_lr 1e-6                    |
| Batch size                            | cycle configs →**32**                                                  |
| Epochs per cycle                      | `max_epochs: 12`, `min_epochs: 5`, patience 5                             |
| Layer-wise LRs                        | cycle 2 →`lr_g3: 5e-5`, `lr_head: 2e-4`                                  |
| Random seed / seed count              | **42**, single seed — ⚠️ state this honestly                         |
| Image size                            | 224×224, resize 256                                                          |

**Flow:** read the values from `configs/part1_case2_imagenet/continuation/cat_a/replay/cycle2/catA_replay_cycle2.yaml` and `environment.yml` → write the paragraph → replace the placeholder comment → sync `.md`.

⚠️ **Single-seed disclosure:** all CL results are one seed. Either state it as a limitation, or run 3 seeds and report mean±std. Stating it is acceptable; hiding it is not.

---

# 3. Fill references [21] [22] [27]

Three bibliography entries still contain `[VENUE]`, `[YEAR]`, `[PP--PP]` placeholders.

| Ref       | What it is                                                                   | Needed                                                                                       |
| --------- | ---------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------- |
| `ref21` | *A Lightweight Modular Pipeline for Edge-Optimized Leaf Disease Detection* | Venue, year, pages, DOI —**load-bearing**: Table XIII cites it for detect/seg latency |
| `ref22` | GAN-based leaf inpainting paper                                              | Venue, year, pages, DOI                                                                      |
| `ref27` | Turmeric dataset                                                             | Author names — Mendeley DOIs`10.17632/jtttfbx342.1` and `10.17632/g46dvrcvwn.1`         |

**Flow:**

1. Check IEEE Xplore indexing status for your two papers.
2. If not yet indexed → use the accepted-but-unpublished IEEE form: `to be published` / `early access`. Do **not** invent page numbers.
3. Open both Mendeley DOIs → copy exact author lists for `ref27`.
4. Update `leafsense.tex` bibliography → sync `.md`.

⚠️ `ref21` is the citation backing two numbers in Table XIII. If it cannot be cited properly, those rows need re-measuring or removing.

---

# 4. Verify all 35 references on IEEE Xplore

**Flow:** for each `\bibitem`, confirm author list, title, venue, volume/issue, pages, year, DOI. Fix mismatches. Mark verified.

**Known items:**

- [24], [26], [28], [34], [35] → already Crossref-verified ✅
- [31] FedAvg, [33] FixMatch → marked `%% VERIFY` in the `.tex`, **page ranges unconfirmed**
- [26] has a published correction (`10.1186/s13007-024-01140-3`) → check whether it affects anything relied upon

---

# 5. Fix stale table numbers in planning docs

Planning docs still use the *planned* table numbers, which no longer match what compiles (see `07_CHANGES_MADE.md` §19).

| Doc says        | Actually renders as  |
| --------------- | -------------------- |
| Table VIII (E1) | **Table IV**   |
| Table IX (E2)   | **Table V**    |
| Table X (E3/E5) | **Table XIII** |
| Table XI (E4)   | **Table XIV**  |

**Flow:** update `01_SCOPE_AND_COVERAGE.md`, `06_EXPERIMENT_PROTOCOL.md`, `05_FIGURE_GUIDE.md`, and the placeholder index in `04_FULL_PAPER_LATEX.md`.

**Cosmetic only** — the `.tex` uses `\ref{}` so the compiled paper is already correct. Do it to stop future confusion, not because the paper is wrong.

---

# 6. Fig. 9 — deployment photo + dashboard screenshot

Last unbuilt figure. Two independent panels.

### Panel (a) — deployment photograph *(needs hardware)*

- Shoot the installed unit in the greenhouse
- Must show: Jetson enclosure, rotating camera mount, at least one sensor probe
- Even lighting — avoid harsh backlight through glazing
- Include something for scale
- Add leader labels: `Jetson Orin Nano`, `Rotating camera mount (45° steps)`, `Environmental sensors`
- **PNG or TIFF, not JPG** — IEEE flags compression artefacts
- Crop, don't zoom. Get consent or crop out any person.

### Panel (b) — dashboard screenshot *(software only — can be done now)*

You said the Streamlit dashboard is already built locally. This panel does **not** need the greenhouse.

- Capture: 5-line per-class daily count trend, current fused risk state, `Retrain` / `Capture Now` controls
- Zoom browser to ~150% **before** capturing so text stays legible at 88 mm
- Crop tightly — no address bar, no whitespace

### Compose

Stack or place side by side, label `(a)` / `(b)`, export ≥600 dpi as `figures/fig9_deployment.pdf`.

**Note:** panel (b) is unblocked *now*. If the greenhouse install is delayed, consider whether a dashboard-only Fig. 9 with an amended caption is acceptable — that is a judgement call about how much the paper's deployment claim rests on the photograph.

---

# 7. E4 — Turmeric CL validation → Table XIV

Currently deferred. The **data and a trained classifier both already exist** (see `07_CHANGES_MADE.md` §18, §20), so this is closer to done than the status suggests.

### What exists

- **Data:** `data/turmeric_5_classes_splitted/` — 5 classes, ~152–184 train / 19–23 val / 19–24 test per class, 224×224 RGB, pre-segmented
- **Trained model:** `other models/model/MobilenetV3_Phase3_EWC_Incremental_Results/best_phase3_model.pth` — 98.59% val, 89.81% retention, EWC λ=10,000

### ⚠️ Two traps before touching it

1. **Class order is NOT alphabetical** — `Healthy=0, Leaf_Spot=1, Blotch=2, Dry=3, Aphids=4`. Read `class_to_idx` from the checkpoint; never assume ImageFolder order. Getting this wrong mislabels every prediction *silently*.
2. **timm, not torchvision** — keys are `backbone.conv_stem.*`; 2.19 M params; custom head `Linear(1024→512)→BN→ReLU→Dropout→Linear(512→5)`. Requires `timm` (now installed) plus the `EnhancedMobileNetV3Phase3` class. The tomato loading code will **not** work.

### Flow

1. Reconstruct the architecture and confirm `load_state_dict` is clean (no missing/unexpected keys).
2. Reproduce the reported 89.81% retention on the turmeric test set — if it doesn't reproduce, the architecture reconstruction is wrong, stop and fix.
3. Run **Case 2 + Replay** and **Case 2 + EWC** for 2 cycles, mirroring the tomato CatA protocol.
4. Apply the turmeric-specific replay buffer rule: `min(40, 50% of initial_train)` — the tomato rule ("15%, min 100") is unsatisfiable at this volume. **State the change and reason in the paper.**
5. Report mean ± std over 3–5 seeds.
6. Fill Table XIV + write the 2 discussion paragraphs.

### ⚠️ Framing constraint

A ~20-image test set per class means **1 image ≈ 5 percentage points**. Differences under ~5 pp between methods are **not resolvable**. Present E4 as a **feasibility check that the recipe transfers**, not a precision benchmark. Overclaiming on 20 test images is exactly what a reviewer will catch.

---

# 8. Author biographies and photos

Standard IEEE requirement. Collect from each co-author: photo, degrees, current affiliation, research interests, IEEE membership grade.

---

# 9. Final formatting and page-count pass

**Flow:**

1. Recompile clean.
2. Page count near ~15.
3. All 9 figures present, vector PDF (or ≥600 dpi raster).
4. All 14 tables render inside their columns; no overflow, no collisions.
5. Every `\ref{}` and `\cite{}` resolves — no `??` in the PDF.
6. Abstract and index terms match final content.
7. Confirm all three §VI disclosures survived editing (simulated sensors, constructed episodes, proxy hardware).
8. If switching to `ieeeaccess.cls`: change `\documentclass`, restore IEEE Access front matter, **delete the figure-fallback block**.

---

# 10. *(Optional)* Real Jetson E3 re-measurement

Would upgrade Table XIII from proxy to genuine on-device numbers and let the caption say "Jetson Orin Nano" without qualification.

**Flow:** copy `experiments/part4_ondevice_proxy/` to the Jetson → record `nvpmodel -q` power mode, JetPack/TensorRT versions, ambient temp → re-run all three scripts → build a **real TensorRT FP16 engine** (JetPack ships TensorRT, so the Windows blocker disappears) → sample `tegrastats` every second for peak RAM and thermal throttling → re-measure detection/segmentation on-device so those rows become measured rather than cited → update Table XIII and remove the proxy footnotes.

**Not blocking** — current numbers are real and honestly disclosed.

---

# 11. *(Optional)* Real sensor logs for E1

The largest standing caveat: everything in E1 is simulated.

**Flow:** install the unit → log sensors for 4–8 weeks → replace `sensors_raw.csv` → re-run `train_env_risk.py` → update Table IV → **re-run E2 as well** (it consumes E1's risk output) → remove or soften the §VI simulation disclosure.

**Timeline makes this unlikely before submission.** Recommend submitting with the disclosure and treating real-data validation as follow-up work.

---

# Pre-submission checklist

```
[ ]  1. Compiles clean; tables XI/XII/XIII do not overlap
[ ]  2. §VI-C implementation details filled (incl. single-seed disclosure)
[ ]  3. Refs [21] [22] [27] have real venues/DOIs
[ ]  4. All 35 refs verified on IEEE Xplore
[ ]  5. Planning-doc table numbers corrected
[ ]  6. Fig. 9 produced (both panels)
[ ]  7. Table XIV filled from E4; framed as feasibility check
[ ]  8. Author bios + photos added
[ ]  9. Page count ~15; no ?? in PDF; all 3 disclosures intact
```

**Currently done:** Tables IV, V, XIII carry real numbers · Figs 1–8 final · 35/35 citations resolve · `.tex`/`.md` synced.

---

# Known standing limitations

State these plainly rather than hoping they go unnoticed:

| # | Limitation                                        | Mitigation in place                                                       |
| - | ------------------------------------------------- | ------------------------------------------------------------------------- |
| 1 | E1 sensor data is simulated                       | §VI disclosure; oracle from published model, not self-defined thresholds |
| 2 | E2 episodes are constructed                       | §VI disclosure; vision + risk branches are both real                     |
| 3 | E3 measured on laptop, not Jetson                 | Table XIII footnote + §VI disclosure                                     |
| 4 | ONNX Runtime substituted for TensorRT             | Labelled everywhere; JetPack ships TensorRT for real deployment           |
| 5 | Single seed throughout                            | Must be stated in §VI-C                                                  |
| 6 | Fig. 5 pairs tomato vision with turmeric risk     | Disclosed in caption                                                      |
| 7 | E4 test set too small for fine-grained claims     | Frame as feasibility check                                                |
| 8 | Fusion fails on the adversarial confound category | Reported as a measured limitation, not hidden                             |
