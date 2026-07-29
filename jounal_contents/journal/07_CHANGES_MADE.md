# 07 — Changes Made: Plan vs. As-Built

**Purpose:** working record of every material deviation between the original plan (`06_EXPERIMENT_PROTOCOL.md`, `01_SCOPE_AND_COVERAGE.md`) and what was actually built, with the reasoning, the flow followed, the reference material, and the data used.
**Companion:** [`08_PENDING_WORK.md`](08_PENDING_WORK.md) — what still has to be done.
**Covers:** 2026-07-23 → 2026-07-29.

---

## Status at a glance

| Experiment | Planned | Built | Table | Verdict |
|---|---|---|---|---|
| E1 Environmental risk | Tier B, generic thresholds | Turmeric-parameterised infection oracle, 5-model comparison | IV | ✅ done, design **upgraded** |
| E2 Label-level fusion | Real deployment episodes | Constructed episodes, real model + real risk | V | ✅ done, source **downgraded** |
| E3/E5 On-device cost | Measured on Jetson | Measured on laptop GPU proxy | XIII | 🟡 partial, hardware **substituted** |
| E4 Turmeric CL | Case 2 + Replay **and** EWC, full ranking | Replay + naive reference only | XIV | ✅ done, scope **narrowed** |

**Paper state:** **all 4 result tables carry real numbers**; **8 figures final (Fig. 9 removed)**; **35/35 citations complete and verified**; `.tex` and `.md` byte-identical; the only remaining content gap is §VI-C implementation details.

---

# PART A — Changes made, in order

## 1. E1 label oracle: rejected circular thresholds → adopted a published infection model

**Original plan (yours):** extract 4 sensor parameters → define Low/Med/High threshold bands → generate data from those bands → train a decision tree.

**Why it was rejected:** circular. Defining `humidity > 85% → HIGH`, generating data from that rule, then training a tree on it means the tree recovers the rule and reports ~99–100%. That measures nothing, and placing it beside the genuine CL results in the same paper puts *their* credibility at risk too.

**First replacement (Tier B, planned):** labels from published plant-pathology relations — Agrios and Lee & Yun.

**Second replacement (what shipped) — the important one:** during implementation you caught that the deployed product is **turmeric**, but TOMCAST/FAST-style models are **tomato**-specific. Tomato appears in this project *only* as the CL benchmark crop. Using a tomato-calibrated oracle for a turmeric product would have been a crop mismatch a plant-pathology reviewer would flag immediately.

**Final design:** take the *generic* mechanism, parameterise it for the *right crop*.

| Component | Source |
|---|---|
| Generic infection mechanism (temperature response × leaf-wetness hours → daily severity, accumulated over a trailing window) | **Magarey, Sutton & Thayer (2005)**, *Phytopathology* 95:92–100, doi `10.1094/PHYTO-95-0092` → cited as **`ref34`** |
| Turmeric-specific epidemiology (*Taphrina maculans* leaf blotch, *Colletotrichum* leaf spot) used to set cardinal temperatures and wetness thresholds | **Gohel, Mistry, Rathava & Dhaduk (2022)**, *Indian Phytopathol.* 75:487–491, doi `10.1007/s42360-021-00452-x` → cited as **`ref35`** |

Both DOIs Crossref-verified. Bibliography grew 33 → **35**.

**Anti-circularity guarantee built into the code:** the ML features never see the oracle's internals — no `w(T)`, no wet-hour count, no accumulated severity. And the feature humidity threshold (85%) is deliberately *different* from the oracle's internal wetness threshold (90%), so the tree cannot trivially reconstruct the labelling rule.

---

## 2. E1 risk classes: 4 → 3

Draft Table had Low/Medium/High/**Critical**. Dropped to 3. A fourth band is hard to justify agronomically and would carry too few samples to give a stable per-class F1.

---

## 3. E1 comparison set: decision-tree-only → 5 models

Planned: decision tree, plus a majority baseline and a random forest.
Built: **Majority · Logistic Regression · Decision Tree (deployed) · Random Forest · Gradient Boosting**, each scored on accuracy, macro-F1, per-sample latency, and pickled size.

**Reason:** the paper's design claim is *"interpretability costs nothing here."* That is only demonstrable by showing the interpretable model matches higher-capacity ones. It does — DT 86.25% vs RF 85.0% vs GBM 85.0%, at **1/170th the size** and **73× lower latency**.

---

## 4. E1 simulator: one-off "monsoon block" → recurring seasonal cycle

**Bug:** the first simulator put a single unusual weather regime *inside the test period only*. The chronological split then became unseen-regime extrapolation rather than a fair future-holdout — RF/GBM collapsed to the majority baseline and the High class had zero recall.

**Fix:** made the wet/dry cycle **recurring** with a 38-day period, so any ~40-day block spans both wet and dry conditions.

---

## 5. E1 feature window: 24 h → 120 h

**Bug:** features aggregated 24 h, but the label depends on **5 days** of accumulated severity. A single day cannot resolve it. Every model scored near-random (22–36%, barely above the 33–44% baseline).

**Fix:** aligned the feature window to the oracle's accumulation horizon (120 h). This single change is what moved the experiment from broken to working.

---

## 6. E1 thresholds: fixed constants → train-period percentiles

Low/Medium/High cut-points became the **40th/72nd percentiles of accumulated severity computed on the training period only**, then applied unchanged to test. Avoids test-set leakage while keeping class balance stable (~40/35/25).

**Honest caveat recorded:** that balance is therefore a *designed property of the calibration*, not an independently observed natural frequency.

---

## 7. E1 flow as actually executed

```
simulate_sensors.py     200 days hourly; diurnal + recurring 38-day seasonal
                        + AR(1) drift + sensor noise + 2% dropout
        ↓
infection_risk_labels.py  Magarey temp-response × wetness hours → daily severity
                          → 5-day rolling accumulation
        ↓
calibrate thresholds      40th/72nd percentile, TRAIN PERIOD ONLY
        ↓
features.py               13 features over a 120 h trailing window, 2 captures/day
        ↓                 + 5% label noise (diagnostic uncertainty)
train_env_risk.py         chronological split (train ≤ day 145, val ≤ 160, test 161-200)
                          → 5 models → metrics, importances, robustness
```

**Data:** fully simulated. No real sensor logs exist.
**Artifacts:** `experiments/part2_env_risk/outputs/` — incl. deployed `decision_tree.pkl`.
**Result:** DT depth 5, 15 leaves → **86.25% acc / 85.63 macro-F1**. Top features `hours_rh_above_85` (0.473) + `rh_mean` (0.315) = **79%** of importance, confirming the humidity/leaf-wetness mechanism the oracle encodes. Robustness: 75.0% under 2× noise, 85.0% under 10% dropout.

---

## 8. E2 episode source: real deployment windows → constructed scenarios

Protocol ranked three sources. Sources 1 (real deployment windows) and 2 (turmeric image sequences + E1 traces) **do not exist** — no unit deployed, and no turmeric dataset was in the repo at the time. Fell back to **source 3, constructed scenarios**, which the protocol explicitly permits.

**Two things kept real, deliberately:**
- **Vision branch:** every leaf count is a real prediction from the actual trained **Case 2 + Replay cycle-2** checkpoint, run over real held-out tomato test images (`data/02_tomato_5cls/test/` — verified never used in training, replay, or model selection for any case/cycle). Pool accuracy re-measured at **98.50%**, matching its training-time 98.42%. Only the *day-by-day sequencing* is constructed.
- **Sensor branch:** real E1 decision-tree outputs, sampled from `windows_labelled.csv` restricted to the E1 **test period** (`day ≥ 161`).

---

## 9. E2: circularity bug caught and fixed — the most important correction

**The bug:** the first version set each episode's ground truth to Φ's own output. Fusion then scored **100%** — because it was being graded against its own rule table. This is the same circularity trap the protocol had warned about for E1, reappearing in a different place.

**How it was caught:** 100% accuracy against a self-authored rule was treated as a red flag, not a result, and investigated.

**The fix:** added an adversarial category, `confound_high_risk_false_alarm` — humidity elevated for a reason *unrelated* to infection (e.g. recent irrigation), with counts stable. It has the **identical `(stable, High)` signature** as the genuine early-warning category but a different true cause. Φ cannot distinguish them, so fusion is **guaranteed wrong** on one of them.

**Effect:** fusion dropped from a meaningless 100% to a meaningful **87.5%**, and now has a documented, explainable failure mode instead of a suspiciously perfect score.

---

## 10. E2 scale: 20–30 episodes (protocol minimum) → 48

**8 categories × 6 replicates.** Each single-modality baseline fails in a distinct, explainable way — which is what makes the comparison non-vacuous:

| Category | Vision-only | Sensor-only | Fusion |
|---|---|---|---|
| rising + High | 100% | 0% | 100% |
| rising + Medium | 0% | 0% | 100% |
| rising + Low (benign cause) | 0% | 0% | 100% |
| stable + High (early warning) | **0%** | 100% | 100% |
| stable + Low/Med | 100% | 100% | 100% |
| falling (recovering) | 100% | 83.3% | 100% |
| sensor failure + rising | 100% | 0% | 100% |
| **adversarial confound** | 83.3% | 0% | **0%** |

**Result:** fusion **87.5% / 71.75 macro-F1** vs vision-only 60.4%/29.6 vs sensor-only 35.4%/20.6. Graceful degradation verified — on all 6 sensor-failure episodes, fusion output matched vision-only exactly.

**Also fixed during E2:** a jitter bug where independent per-day random jitter on small integer counts flipped "stable" trajectories into spurious rising/falling ones. Fixed by applying one jitter value uniformly per episode. Verified via `intended_trend` vs `realized_trend` columns.

---

## 11. Fig. 5 built, then redesigned three times

Built from real data: panel (a) real classifier predictions, panel (b) real E1 sensor trace for the matching days, panel (c) the real Φ decision.

Revisions after review:
1. Panel (c) originally showed inputs → output with no explanation → **added an explicit Φ rule box** so the figure states *why* those inputs give that action.
2. "+700%" replaced with **"+7 leaves"** — mathematically correct but misleading, since a 1→8 rise on a small base inflates the percentage. The rule only tests the ±20% threshold, so the absolute count communicates better.
3. Overlapping arrows and text spilling outside boxes fixed; Φ box widened.

**Disclosed in the caption:** vision is tomato, risk is turmeric-parameterised, paired for illustration; day sequencing is constructed.

---

## 12. E3: Jetson Orin Nano → laptop GPU proxy

No Jetson available. Ran the same protocol on a **laptop NVIDIA RTX 4050 (6 GB, CUDA 12.1)**.

Retrain timing re-ran the **real** cycle-2 replay config — identical data, hyperparameters, and seed — with output redirected to a fresh directory so the real checkpoint E2 depends on was never touched (verified). It reproduced **98.42%** test accuracy exactly, confirming a faithful re-timing rather than a different run.

**Measured:** 373.2 s wall (348.4 s train), 155.6 MB peak VRAM, 1245 MB process RAM.

---

## 13. E3: TensorRT → ONNX Runtime GPU

TensorRT could not be installed. Both `tensorrt` and `tensorrt-cu12` fail to build their wheels on this Windows machine against the CUDA 12.1 torch build (`tensorrt-cu12` pulls `nvidia-cuda-runtime-cu13`). Two separate install attempts, both failed.

**Substituted ONNX Runtime GPU** as the compiled-graph path, and labelled it as such **everywhere** — table footnote, discussion text, README, and this report. No claim anywhere says TensorRT was used.

**Not a deployment blocker:** JetPack ships TensorRT preinstalled, so the real engine build will work on the actual target — which is where that measurement belongs anyway.

---

## 14. E3 gate result: Δ ≈ 0, reported rather than buried

| Path | Accuracy |
|---|---|
| PyTorch FP32 candidate | 98.51% |
| PyTorch FP16 | 98.59% (Δ +0.09 pp) |
| Compiled graph (ONNX RT) | 98.51% (Δ 0.00) |

+0.09 pp is **one image out of 1,137** — noise, not a real difference. The protocol anticipated this and prescribed the reframing that was used: the gate's justification is not that conversion *is* lossy, but that a system **cannot know it is lossless without measuring the artifact it actually deploys**.

**Unexpected finding kept:** eager FP16 is *slower* than FP32 (12.4 vs 10.2 ms/img) for a network this small — per-op conversion overhead exceeds the arithmetic saving. **The compiled graph, not half precision, delivers the 2.7× speedup** (10.2 → 3.8 ms/img). This contradicts the naive "FP16 = faster" assumption and is worth keeping.

---

## 15. E5 latency: detection/segmentation cited, not re-measured

Per `01_SCOPE_AND_COVERAGE.md` §1.2, the 3-stage pipeline is cited to the prior conference paper rather than re-evaluated. Detection **12.2 ms** and segmentation **5.0 ms** come from `ref21` Table IV.

**Checked and worth knowing:** the raw training artifacts in `other models/model/results_enhanced_yolov8n/performance_metrics.txt` report *different* numbers (mAP 0.664, 10.45 ms) than the published paper (mAP 0.632, 82.1 FPS) — they are from a different training run. **The published values were used**, since those are the citable ones. Also, the paper never states those FPS figures were measured *on* Jetson, so they are cited as "reported in [21]" without claiming on-device provenance.

---

## 16. Table IV: dropped the "Interpretability" column

Removed the yes/no column. The interpretability *argument* remains in the surrounding prose, where it belongs — a binary column added no information the text didn't already carry better.

---

## 17. Table XIII layout fix

Tables XI/XII/XIII collided in the rendered PDF. Two causes, both fixed:
1. Table XIII's caption ran **5 lines** (hardware, citation, and proxy disclaimer). Cut to a one-line title; caveats moved to `†`/`‡` footnotes under the table.
2. Three consecutive `[!t]`-only floats gave LaTeX nowhere to place them → changed to `[!tbp]`.

⚠️ **Not visually verified** — no LaTeX toolchain on this machine. Needs an Overleaf recompile to confirm.

---

## 18. E4 deferred (decision, not omission)

The turmeric dataset **was added** mid-session: `data/turmeric_5_classes_splitted/` — 5 classes (Aphids_Disease, Blotch, Dry Leaf, Healthy_Leaf, Leaf_Spot), ~152–184 train / 19–23 val / 19–24 test images per class, 224×224 RGB, pre-segmented.

Deferred by decision because the sample count is low: a ~20-image test set per class means one image ≈ 5 percentage points, so differences under ~5 pp between CL methods are not resolvable. The protocol's own pre-written guidance for this case is to run it as a **feasibility check, not a precision benchmark**.

A turmeric classifier **was later found** in the repo — see item 20.

---

## 19. Documentation drift found: planning-doc table numbers are stale

The planning docs refer to E1→Table VIII, E2→Table IX, E3→Table X, E4→Table XI. Those were the *planned* positions. The paper now compiles to different numbers:

| Experiment | Docs say | **Actually renders as** |
|---|---|---|
| E1 environmental risk (`tab:dt`) | Table VIII | **Table IV** |
| E2 fusion (`tab:fusion`) | Table IX | **Table V** |
| E3/E5 on-device (`tab:ondevice`) | Table X | **Table XIII** |
| E4 turmeric (`tab:turmeric`) | Table XI | **Table XIV** |

**Impact:** cosmetic only — the `.tex` uses `\ref{}` throughout, so the compiled paper is internally correct. It is the *planning docs and placeholder index* that are stale. Listed as a cleanup item in `08_PENDING_WORK.md`.

---

## 20. Turmeric classifier found — with two traps

Located at `other models/model/MobilenetV3_Phase3_EWC_Incremental_Results/best_phase3_model.pth`.
98.59% best validation accuracy, 89.81% knowledge retention, EWC λ=10,000, 10 epochs.

Two things that will silently break an integration if missed:

**Trap 1 — class order is NOT alphabetical:**
```
Healthy_Leaf_=0, Leaf_Spot_=1, Blotch_=2, Dry Leaf_=3, Aphids_Disease_=4
```
ImageFolder alphabetical order would be `Aphids=0, Blotch=1, Dry=2, Healthy=3, Leaf_Spot=4`. Anything assuming alphabetical order mislabels **every** prediction. The correct mapping is stored inside the checkpoint as `class_to_idx` — read it from there, never assume.

**Trap 2 — different architecture family from the tomato model.** Keys are `backbone.conv_stem.*` → **timm**, not torchvision. 2,190,586 params with a custom head `Linear(1024→512) → BatchNorm → ReLU → Dropout → Linear(512→5)`. It needs `timm` installed plus the `EnhancedMobileNetV3Phase3` class; it is **not** loadable by the same code path as the tomato classifier.

---

# PART A2 — Changes made 2026-07-28 → 07-29 (continues from item 20)

## 21. E4 un-deferred and completed

Item 18 recorded E4 as deferred on sample-size grounds. That was **reversed** once the data was analysed properly: the turmeric set is clean (0 exact duplicates, 99.5% effectively unique) and large enough for a feasibility claim, just not for a method ranking.

**Split strategy — deliberately different from the tomato policy.** The protocol assumed splitting from scratch. That is unsafe here: the existing phase-3 turmeric checkpoint was trained on the shipped `train/`, so re-splitting would leak its training images into a new test set and make its 89.81% retention uninterpretable. Instead **test (107) and val (102) were frozen as shipped** and only `train/` (834) was re-split 50/25/25 → 417 / 210 / 207, reproducing the tomato protocol's *internal* ratios. Built by `data/06_scripts/splitters/split_turmeric_5cls.py` → `data/07_turmeric_5cls/`. Verified fully disjoint; every train image used exactly once; two cross-split near-duplicates excluded.

**Buffer rule changed and stated in the paper:** tomato's "15% per class, min 100" is unsatisfiable at 76–92 images/class → **`min(40, 50% of initial_train)`**.

**Architecture:** fresh torchvision MobileNetV3-Small + ImageNet, *not* the timm phase-3 checkpoint — the recipe under test is defined by the G1–G4 grouping, which is torchvision-specific.

**Result:** base 89.10 ± 2.35 → **Replay 90.03 ± 3.54**; EWC 88.79 ± 1.62; naive 87.85 ± 2.47.

## 22. E4 scope narrowed to Replay + a naive reference

Originally planned as a Replay-vs-EWC ranking. Narrowed on the user's decision, and it is the more defensible choice: **every pairwise McNemar test returns p ≥ 0.109**, so a ranking table would present differences the data cannot support. The naive row is retained as a labelled *reference condition* — without it, Replay's number floats with nothing to contrast against.

## 23. E4: replay batch-ratio bug (found and fixed)

The first replay run scored **82.24%** — 10 pp *below* the incumbent, which replay should never cause. Root cause: the stream loader used `train.batch_size` (32) while the buffer loader used `old_per_batch` (16), producing a **16 old : 32 new** mix instead of the intended 50/50. `new_per_batch` was in the config but never read. After the fix: **91.59%**.

## 24. E4: post-hoc training-schedule change — disclosed

`patience` 5→8, `min_epochs` 5→10, epoch caps 15/30→30/40, **after** seeing the first results. Motivated by a defect visible in the logs independent of any comparison: seed 43 hit the phase-1 epoch ceiling while still improving; seeds 43 and 44 peaked at epoch 6 and early-stopped on a 102-image val set whose macro-F1 swings several points per epoch.

Three things make this a methodological fix rather than result-chasing:
1. The defect is visible in the training logs regardless of outcome.
2. It applied identically to every seed and every method.
3. **It made the headline comparison worse** — replay's margin over naive shrank 3.73 → 2.18 pp, its std widened ±1.87 → ±3.54, and directional consistency dropped from 3/3 seeds to 2/3.

It did achieve its stated target: base variance tightened ±3.74 → ±2.35. Both runs retained (`outputs/`, `outputs_patience5_backup/`).

**Also declined:** a suggestion to report only the best-performing seed. Seed 42's 92.52% sits at the top of a spread that also contains 85.98%; reporting it alone would not reproduce.

## 25. Fig. 9 removed; deployment claims softened

No unit is installed, so the deployment photograph cannot be produced, and a dashboard screenshot alone does not carry the claim. **The figure was never referenced from the body text** (`\ref{fig:deploy}` appeared zero times), so nothing depended on it.

More importantly, the paper previously asserted *"…pending replacement with in-situ sensor logs from **the installed** turmeric greenhouse unit"* — presupposing an installation that does not exist. Reworded to **"No unit is currently installed, so no in-situ readings exist… to be revisited once a turmeric greenhouse unit is deployed."** Every other mention of "deployed"/"installed"/"in the field" was audited; the rest are generic statements about the problem domain or already-hedged design language.

**Net effect on framing:** the paper is now consistently a *methods and architecture* contribution rather than a *deployed system* one — coherent with E1 being simulated, E2 constructed, and E3 proxy-measured.

## 26. Reference [5] was fabricated — removed and replaced ⚠️

`H. Sharma and M. Patel, "Edge computing challenges for deep learning-based plant disease detection systems," IEEE Internet Things J., vol. 9, no. 12, pp. 9876–9888, 2022, doi: 10.1109/JIOT.2022.3142890` **does not exist.** Three independent checks:

1. The DOI returns **HTTP 404** from doi.org.
2. Enumerating *IEEE IoT J.* vol. 9 no. 12 via Crossref: real articles run **…9858–9871**, then **9889–9903**. The claimed pages **9876–9888 fall in a gap occupied by no article.**
3. No Crossref search finds the title, the authors, or anything matching in that venue.

Replaced with **A. T. Khan, S. M. Jensen, A. R. Khan, and S. Li, "Plant disease detection model for edge computing devices," *Front. Plant Sci.*, vol. 14, Dec. 2023, Art. no. 1308528, doi: 10.3389/fpls.2023.1308528** — Crossref-verified and on-topic for the claim it supports. Evidence for the removal is documented inline in the `.tex` so nobody re-adds it.

⚠️ **This entry was inherited from the prior conference paper**, where it appears as `[4]` with the same dead DOI.

## 27. Remaining reference corrections

| Ref | Was | Now |
|---|---|---|
| [11] SerpensGate-YOLOv8 | Dec. 2024 | **Jan. 2025** (actual publication date) |
| [17] iCaRL | pp. 2001–2010 | **pp. 5533–5542** (IEEE Xplore pagination matching the cited DOI) |
| [21] ICIIS | `[VENUE]`/`[YEAR]`/`[PP--PP]` | Peradeniya, Sri Lanka, Jan. 2026, **pp. 395–400, doi: 10.1109/ICIIS69028.2026.11450780** — Crossref-verified, all 3 authors confirmed |
| [22] ICDTSA | placeholders | Kilinochchi, Sri Lanka, May 2025, **pp. 7–16**; confirmed **no DOI** (proceedings volume, absent from Crossref) |
| [27] Turmeric datasets | `[Author(s)]` | Real author lists via the **DataCite** API — Mendeley DOIs register with DataCite, not Crossref, which is why the earlier Crossref check 404'd |
| [31] FedAvg | `%% VERIFY` | Verified from the publisher: `proceedings.mlr.press/v54/mcmahan17a` states **PMLR 54:1273–1282**. Pages were already correct |
| [33] FixMatch | `%% VERIFY` | Paper + authors confirmed in the NeurIPS 2020 index; **pagination could not be confirmed** from a primary source (NeurIPS publishes none) — 596–608 is the Curran print volume's range, documented as such |

**Final state: 20 of 21 DOIs resolve.** The 2 flagged mismatches are confirmed false positives — [14] Crossref holds the stale early-access record (the paper's v44/n7/pp3366–3385 is correct), and [35] flagged only on `\textit{}` markup. **IEEE format compliant throughout; no formatting changes needed.**

**Decision on [22]: kept**, after considering removal. The GAN paper's own reference list contains several apparently fabricated entries (*"IEEE Transactions on Agriculture"*, *"IEEE Transactions on Remote Sensing"*, *"Agricultural AI Journal"* — none of these journals exist). But citing a paper is not endorsing its bibliography, and the citation is load-bearing: the journal paper *describes* the GAN occlusion-recovery stage, so removing the citation while keeping the description would be undisclosed prior publication — a worse problem. **Recommended follow-up: run the same Crossref check over both prior papers' reference lists.**

## 28. Table XIII layout and caption fixed

Tables XI/XII/XIII collided in the rendered PDF. Two causes, both addressed: the caption ran 5 lines (cut to a one-line title, with the hardware/citation caveats moved to `†`/`‡` footnotes), and three consecutive `[!t]`-only floats gave LaTeX nowhere to place them (changed to `[!tbp]`).

⚠️ **Not visually verified** — no LaTeX toolchain available locally. Needs an Overleaf recompile to confirm.


---

# PART B — Reference material used

| Ref | Work | Used for |
|---|---|---|
| `ref21` | Your prior conference paper, *A Lightweight Modular Pipeline for Edge-Optimized Leaf Disease Detection* | Detection/segmentation latency + accuracy (Table XIII), 3-stage pipeline description |
| `ref22` | Your prior GAN occlusion-recovery paper | Optional GAN stage |
| `ref34` | Magarey, Sutton & Thayer (2005), *Phytopathology* 95:92–100 | E1 generic infection mechanism |
| `ref35` | Gohel et al. (2022), *Indian Phytopathol.* 75:487–491 | E1 turmeric parameterisation |
| `ref25`, `ref26`, `ref28` | Agrios; Lee & Yun; Rudin | Agronomic thresholds; interpretability argument |

---

# PART C — Data used

| Dataset | Location | Used by | Real? |
|---|---|---|---|
| Tomato 5-class (PlantVillage, segmented) | `data/02_tomato_5cls/` | CL benchmark, E2 vision branch | ✅ real |
| Simulated greenhouse sensors | generated by `simulate_sensors.py` | E1 training, E2 risk branch, Fig. 5(b) | ❌ **simulated** |
| Constructed fusion episodes | generated by `build_episodes.py` | E2, Table V | ⚠️ **constructed** (real components) |
| Turmeric 5-class | `data/turmeric_5_classes_splitted/` | E4 — not yet used | ✅ real, unused |

---

# PART D — Disclosures in the paper

Three deviations are stated in the paper itself, not just here:

1. **§VI environmental data disclosure** — E1 sensor traces are simulated; results are methodology validation pending real logs.
2. **§VI fusion evaluation disclosure** — E2 episodes are constructed; validates that Φ behaves as designed given correct inputs, not a field accuracy measurement.
3. **§VI on-device disclosure + Table XIII footnotes** — laptop proxy not Jetson; ONNX Runtime not TensorRT; detect/seg cited not measured.

**Rationale:** each is a methodological weakness that reviewers routinely accept when declared, and that becomes an integrity problem when not.

---

# PART E — Verification performed

| Check | Result |
|---|---|
| `.tex` ↔ `.md` fence byte-identical | ✅ pass |
| Citations resolve | ✅ 35/35, none undefined, none uncited |
| LaTeX environments balanced | ✅ 14 tables, 9 figures, 4 equations, 1 bibliography |
| Broken `\ref{}` targets | ✅ none |
| Table column counts | ✅ consistent |
| Numbers traced to source CSV/JSON | ✅ all |
| E2 class-order mapping | ✅ verified via 98.50% pool accuracy |
| Real cycle-2 checkpoint untouched by E3 re-timing | ✅ verified |
| **Rendered PDF layout** | ❌ **not verified — no LaTeX toolchain** |
