# 08 — Pending Work: Ordered Path to Submission

**Purpose:** everything still outstanding for the journal paper, in execution order, with the full flow for each.
**Updated:** 2026-07-29 — items 3, 4, 5, 6, 7 of the previous list are now **closed**; see `07_CHANGES_MADE.md` PART A2.
**Companion:** [`07_CHANGES_MADE.md`](07_CHANGES_MADE.md) — what has already been done and why it deviated from plan.
**Scope:** journal paper only. Production/Jetson app work is tracked separately in `app_dev/jetson_flow.md`.

---

## Where the paper stands

| | Status |
|---|---|
| Result tables | ✅ **All 4 filled with real numbers** — IV (E1), V (E2), XIII (E3/E5), XIV (E4) |
| Figures | ✅ **8 final** — Fig. 9 removed, see `07_CHANGES_MADE.md` §25 |
| References | ✅ **35/35 complete**; 20 of 21 DOIs resolve; IEEE-format compliant |
| Placeholders in `.tex` | ✅ **Zero**, except §VI-C implementation details |
| `.tex` ↔ `.md` sync | ✅ byte-identical |

**Four blocking items remain**, all needing input only you can supply (item 2 is partly draftable from the repo).

---

## Priority overview

| # | Item | Effort | Blocks submission? | Depends on |
|---|---|---|---|---|
| 1 | Compile & verify layout | 30 min | **Yes** | — |
| 2 | Fill §VI-C implementation details | 1 h | **Yes** | GPU model from you |
| 3 | Author bios + photos | 1 h | **Yes** | co-authors |
| 4 | Final formatting / page-count pass | 2 h | **Yes** | items 1–3 |
| 5 | *(Optional)* Real Jetson E3 re-measurement | 1 day | No | Jetson hardware |
| 6 | *(Optional)* Real sensor logs for E1 | 4–8 weeks | No | deployed unit |
| 7 | *(Optional)* More turmeric data → upgrade E4 | — | No | data collection |
| 8 | *(Optional)* Audit prior papers' reference lists | 1 h | No | — |

**Critical path:** 1 → 2 → 3 → 4. Items 5–8 can run in parallel or be skipped.

---

# 1. Compile and verify layout ⚠️ DO THIS FIRST

**Why first:** the Table XIII collision fix (`07_CHANGES_MADE.md` §28) was made **without a LaTeX toolchain** — it is the only change in the whole project that could not be verified locally. Everything downstream assumes the paper renders correctly.

**Flow:**

1. Upload `jounal_contents/journal/leafsense.tex` + the `figures/` folder to Overleaf.
2. Compile with pdfLaTeX. Expect zero errors; float-placement warnings are acceptable.
3. Check specifically:
   - Tables XI, XII, XIII no longer overlap (the reported bug)
   - Table XIII's `†`/`‡` footnotes render below the table, inside the float
   - **Table XIV** (new) renders cleanly and inside its column
   - Fig. 5 panel (c) is legible at single-column width (88 mm)
   - No table runs past the column margin
   - No `??` anywhere — all `\ref`/`\cite` resolve
   - **No leftover Fig. 9 reference** — `\ref{fig:deploy}` should appear zero times
4. If Table XIII still crowds → switch it to `\begin{table*}` (spans both columns).
5. Confirm page count is near the ~15-page target.

**Done when:** clean compile, no overlaps, no `??`, page count acceptable.

---

# 2. Fill §VI-C implementation details

**Why it matters:** IEEE Access reviewers routinely ask for this. It is the **only remaining content gap** in the paper (`leafsense.tex`, placeholder comment in §VI-C).

**Already known — fill straight from the repo:**

| Field | Value | Source |
|---|---|---|
| PyTorch / CUDA / torchvision | torch 2.5.1+cu121, torchvision 0.20.1+cu121, CUDA 12.1 | `environment.yml` |
| Optimizer | **Adam** | training scripts |
| LR schedule | `ReduceLROnPlateau`, patience 3, factor 0.5, min_lr 1e-6 | training scripts |
| Batch size | **32** | cycle configs |
| Epochs per cycle | `max_epochs: 12`, `min_epochs: 5`, patience 5 | `catA_replay_cycle2.yaml` |
| Layer-wise LRs | cycle 2 → `lr_g3: 5e-5`, `lr_head: 2e-4` | cycle configs |
| Image size | 224×224 (resize 256) | configs |
| Random seed | **42** | configs |

**Still needed from you:** the **exact GPU model** used for the CL benchmark training runs. (The E3 proxy work used an RTX 4050 laptop GPU, but the original benchmark was run elsewhere — don't assume they are the same machine.)

**Flow:** read the remaining values → write the paragraph → replace the placeholder comment in `leafsense.tex` → sync `04_FULL_PAPER_LATEX.md`.

⚠️ **Single-seed disclosure.** All CL benchmark results (Tables VII–XII) are **one seed**. Either state this plainly as a limitation, or re-run 3 seeds and report mean ± std. Stating it is acceptable; omitting it is not — and note that **E4 now reports 3 seeds**, so the paper reads inconsistently if the benchmark's single-seed status goes unmentioned.

---

# 3. Author biographies and photos

Standard IEEE requirement. Collect from each co-author: photo, degrees, current affiliation, research interests, IEEE membership grade.

---

# 4. Final formatting and page-count pass

**Flow:**

1. Recompile clean.
2. Page count near ~15.
3. All **8** figures present, vector PDF (or ≥600 dpi raster).
4. All 14 tables render inside their columns; no overflow, no collisions.
5. Every `\ref{}` and `\cite{}` resolves — no `??` in the PDF.
6. Abstract and index terms match final content.
7. **Confirm all four §VI disclosures survived editing:**
   - simulated sensor traces (E1)
   - constructed episodes (E2)
   - proxy hardware + ONNX Runtime substitution (E3)
   - feasibility framing + non-significance (E4)
8. If switching to `ieeeaccess.cls`: change `\documentclass`, restore IEEE Access front matter, **delete the figure-fallback block**.

---

# 5. *(Optional)* Real Jetson E3 re-measurement

Would upgrade Table XIII from proxy to genuine on-device numbers and let the caption say "Jetson Orin Nano" without qualification.

**Flow:** copy `experiments/part4_ondevice_proxy/` to the Jetson → record `nvpmodel -q` power mode, JetPack/TensorRT versions, ambient temp → re-run all three scripts → build a **real TensorRT FP16 engine** (JetPack ships TensorRT, so the Windows blocker disappears) → sample `tegrastats` every second for peak RAM and thermal throttling → re-measure detection/segmentation on-device so those rows become measured rather than cited → update Table XIII and remove the proxy footnotes.

**Not blocking** — current numbers are real and honestly disclosed.

---

# 6. *(Optional)* Real sensor logs for E1

The largest standing caveat: everything in E1 is simulated.

**Flow:** install the unit → log sensors for 4–8 weeks → replace `sensors_raw.csv` → re-run `train_env_risk.py` → update Table IV → **re-run E2 as well** (it consumes E1's risk output) → soften the §VI simulation disclosure.

If a unit is actually installed, **also reconsider Fig. 9** (the removal in `07_CHANGES_MADE.md` §25 was conditional on there being no hardware) and revisit the deployment-conditional wording throughout §VI.

**Timeline makes this unlikely before submission.** Recommend submitting with the disclosure and treating real-data validation as follow-up work.

---

# 7. *(Optional)* More turmeric data → upgrade E4

E4's binding constraint is the **107-image test set** (95% CI ≈ ±5.7 pp), which is why no pairwise McNemar test reaches p<0.05 and the result is framed as a feasibility check rather than a ranking.

At roughly **400 images per class** the test set reaches ~60/class, the CI tightens to ~±3 pp, and method ranking becomes possible — upgrading Table XIV from a feasibility check to a precision benchmark. Nothing in `experiments/part5_turmeric_cl/` would need rewriting; re-run `split_turmeric_5cls.py` then `run_all.py`.

---

# 8. *(Optional)* Audit the prior papers' reference lists

Reference `[5]` in this paper was **fabricated** and was inherited from the prior conference paper (`07_CHANGES_MADE.md` §26). Spot-reading the ICDTSA GAN paper's bibliography shows several more apparently non-existent venues (*"IEEE Transactions on Agriculture"*, *"IEEE Transactions on Remote Sensing"*, *"Agricultural AI Journal"* — none of these journals exist).

Both papers are already published, so this does **not** block this submission — but it is better to know before someone else checks. The same Crossref script used here can be pointed at either bibliography.

---

# Pre-submission checklist

```
[ ]  1. Compiles clean; tables XI/XII/XIII do not overlap; no ?? in PDF
[ ]  2. §VI-C filled (incl. GPU model + single-seed disclosure)
[ ]  3. Author bios + photos added
[ ]  4. Page count ~15; 8 figures; all 4 §VI disclosures intact
```

**Already done:** Tables IV, V, XIII, **XIV** filled with real numbers · Figs 1–8 final · **Fig. 9 removed** and deployment claims softened · **35/35 references complete and verified** (fabricated [5] replaced) · `.tex`/`.md` byte-identical · planning-doc table numbers corrected.

---

# Known standing limitations

State these plainly rather than hoping they go unnoticed. All except #6 are already disclosed in the paper.

| # | Limitation | Mitigation in place |
|---|---|---|
| 1 | E1 sensor data is simulated | §VI disclosure; oracle from a published model, not self-defined thresholds |
| 2 | E2 episodes are constructed | §VI disclosure; vision and risk branches are both real |
| 3 | E3 measured on laptop, not Jetson | Table XIII footnote + §VI disclosure |
| 4 | ONNX Runtime substituted for TensorRT | Labelled everywhere; JetPack ships TensorRT for real deployment |
| 5 | E4 differences not statistically resolvable (n=107) | Framed as a feasibility check; McNemar p-values reported |
| 6 | **CL benchmark (Tables VII–XII) is single-seed** | ⚠️ **Not yet stated** — see item 2 |
| 7 | Fig. 5 pairs tomato vision with turmeric risk | Disclosed in the caption |
| 8 | Fusion fails on the adversarial confound category | Reported as a measured limitation, not hidden |
| 9 | No installed unit; no in-situ deployment evidence | Deployment wording is conditional throughout; Fig. 9 removed rather than faked |
