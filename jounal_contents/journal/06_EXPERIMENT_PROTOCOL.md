# Remaining Experiments — Full Protocol
**Scope:** E1 decision tree · E2 label-level fusion · E3 on-device cycle · E4 turmeric CL validation.
**Omitted per instruction:** cassava batch-size study (E6) — dropped, not referenced anywhere in the draft.

---

# PART 0 — REVIEW OF YOUR DECISION-TREE PLAN

## 0.1 Your plan, restated

> Take the environmental docs → extract the 4 measurement parameters → define low / medium / high risk threshold bands → generate dummy data from those bands → train the decision tree on it.

## 0.2 The problem: this is circular

If you define the rule `humidity > 85% AND soil moisture high → HIGH risk`, then generate data labelled by that rule, then train a decision tree on it — **the tree will recover your rule and report ~99–100% accuracy.** That number measures nothing. It says "a decision tree can fit a decision rule", which was never in question.

This matters concretely for the paper. Table VIII would present that accuracy alongside the genuinely-earned CL accuracies in Tables III–VI, and a reviewer who notices the labels were synthesized from the same thresholds the model recovers will treat it as inflating the contribution — which puts the credibility of the *real* results at risk too. The CL benchmark is strong enough that it should not be exposed to that.

Two further technical issues in the plan as stated:

**CO₂ is not a disease driver.** Temperature, humidity and soil moisture have direct causal links to fungal infection. CO₂ does not — elevated CO₂ in a closed greenhouse indicates **poor ventilation**, which lengthens leaf wetness duration, which raises fungal risk. It is a *proxy variable*, and the paper must say so. A plant-pathology reviewer will otherwise ask why CO₂ is in the feature set, and "it was one of our sensors" is not an answer.

**Random train/test splitting on time-series data leaks.** Consecutive sensor readings are strongly autocorrelated — the reading one minute later is nearly identical. A random split puts near-duplicate rows on both sides and inflates test accuracy dramatically. You must split by **time block** (e.g. whole days), never randomly.

## 0.3 Three ways forward

**Tier C — your plan, honestly reframed.** Keep the approach but stop calling it learning. Present it as *rule distillation*: published agronomic thresholds compiled into a decision tree so they execute cheaply and auditably on the edge. Report **fidelity to the rule set** and inference cost, not "accuracy". Fully honest, but contributes little scientifically.

**Tier B — RECOMMENDED, unblocks the paper now.** Same synthetic-data approach, but with two changes that make it defensible:
1. **Labels come from published plant-pathology relations, not from thresholds you invented.** For the turmeric fungal pathogens the driver is leaf-wetness duration combined with temperature; cite Agrios [25] and the environmental-risk modelling of Lee & Yun [26] as the basis. The tree then learns an *external, citable* relationship.
2. **The simulator generates realistic traces, not uniform noise** — diurnal temperature and humidity cycles, autocorrelation, sensor noise, occasional dropouts, and seasonal drift. Then the honest, reportable results are *generalization to environmental regimes not seen in training* and *robustness to sensor noise and dropout*. Those are real questions with non-trivial answers.

**Tier A — best, if your data allows.** Make the label an **observed outcome** instead of a rule: using the per-class leaf counts your own pipeline already produces daily, define the target as *"does the count of any disease class rise by more than X% over the next N days?"* Now the tree performs genuine forecasting from real sensors against real outcomes, and Table VIII becomes a legitimate result. Requires roughly 4–8 weeks of paired sensor + capture data from the deployed unit.

**Recommendation:** build **Tier B now** so the paper is not blocked, and start accumulating Tier A data in parallel from the moment the unit is installed. If Tier A data matures before submission, swap it in — the protocol below is written so that only the labelling step changes.

**Whichever tier you use, the paper must state plainly that the environmental data is simulated.** Non-disclosure here is the one thing that would turn a methodological weakness into an integrity problem. One sentence in Section VI is enough, and it costs you nothing — reviewers accept simulated sensor data in a systems paper when it is declared.

## 0.4 One change to the draft

Your plan says **low / medium / high** (3 classes); the draft's Table VIII currently has 4 (Low/Medium/High/Critical). **Go with 3.** A fourth "Critical" band is hard to justify agronomically and would carry very few samples, producing an unstable per-class F1. I will note this in the checklist — update Table VIII to 3 rows.

---

# PART 1 — E1: DECISION-TREE ENVIRONMENTAL RISK MODEL
*Fills Table VIII (P10). Written for Tier B; the Tier A variant differs only at Step 3.*

> ## ✅ AS-RUN RECORD (2026-07-24) — supersedes the Tier B plan below
>
> The protocol below (Tier B, tomato-agnostic published thresholds) was **superseded during implementation** by a stronger design, once we caught that the deployed product is **turmeric**, not tomato — a TOMCAST-style tomato model would have been a crop mismatch. What was actually built:
>
> - **Label oracle:** the *generic* Magarey, Sutton & Thayer (2005) foliar-fungal infection model [`ref34`, Phytopathology 95:92–100, doi 10.1094/PHYTO-95-0092] — the same leaf-wetness × temperature mechanism behind TOMCAST/FAST, but crop-agnostic — **parameterised with turmeric leaf-blotch (*Taphrina maculans*) / leaf-spot (*Colletotrichum*) epidemiology** from Gohel et al. (2022) [`ref35`, Indian Phytopathol. 75:487–491, doi 10.1007/s42360-021-00452-x]. Not self-defined thresholds → avoids the circularity problem this doc originally flagged.
> - **Models compared (not DT-only):** Majority baseline, Logistic Regression, **Decision Tree (deployed)**, Random Forest, Gradient Boosting.
> - **Input:** simulated hourly traces — diurnal + a **recurring** (not one-off) seasonal wet/dry cycle + AR(1) drift + sensor noise + 2% dropout. The seasonal period is short enough (38 days) that the chronological test block still spans both wet and dry conditions — a fair future-holdout rather than an unseen-regime extrapolation.
> - **Features:** trailing **5-day** aggregates (matching the oracle's accumulation window) — NOT a 24h window. An earlier 24h-window attempt produced near-random accuracy because the label depends on multi-day accumulated severity that a single day cannot resolve.
> - **Risk thresholds:** Low/Medium/High cut-points are **percentiles of accumulated severity calibrated on the training period only** (40th/72nd), not fixed constants — avoids test-set leakage and keeps class balance stable (~40/35/25).
> - **Result:** Decision Tree (depth 5, 15 leaves) = **86.25% accuracy, 85.6 macro-F1** — matches/beats Random Forest (85.0%) and Gradient Boosting (85.0%) at 1/170th the size and 73× lower latency. Top features: `hours_rh_above_85` + `rh_mean` = 79% of importance (confirms the humidity/leaf-wetness mechanism, exactly as predicted). Robustness: 75.0% under 2× sensor noise, 85.0% under 10% dropout.
> - **Code:** `experiments/part2_env_risk/code/` (`simulate_sensors.py`, `infection_risk_labels.py`, `features.py`, `train_env_risk.py`), config `configs/part2_env_risk.yaml`, artifacts + deployed `decision_tree.pkl` in `experiments/part2_env_risk/outputs/`.
> - **Still open:** real deployment-site sensor logs to replace the simulated traces when the unit is installed. (Fig. 5 worked example is now done — see PART 2's AS-RUN record below.)
>
> The step-by-step plan below is kept as background on the original (superseded) Tier B design and the reasoning that led to the change.

## Step 1 — Feature definition

| # | Variable | Unit | Sensor | Role |
|---|---|---|---|---|
| 1 | Air temperature | °C | DHT22 | Direct — governs pathogen development rate |
| 2 | Relative humidity | % | DHT22 | Direct — primary fungal infection driver |
| 3 | Soil moisture | % VWC | Capacitive probe | Direct — waterlogging; raises canopy humidity |
| 4 | CO₂ concentration | ppm | MH-Z19 / equivalent | **Proxy for ventilation** — stagnant air extends leaf wetness |

Derive engineered features from the raw window — a tree over instantaneous readings alone cannot express "sustained" conditions, which is what actually drives infection:

- `humidity_mean_24h`, `humidity_max_24h`
- **`hours_humidity_above_85_24h`** ← expected to be the dominant feature
- `temp_mean_24h`, `temp_min_24h`, `temp_range_24h`
- `hours_temp_in_22_30_band_24h` (pathogen-favourable range)
- `soil_moisture_mean_24h`
- `co2_mean_24h` (ventilation proxy)
- `humidity_trend_24h` (slope, rising vs falling)

One feature row = one 24-hour window, aligned to a capture set. This is the correct granularity: it matches the daily cadence of the vision branch, so fusion in E2 lines up without resampling.

## Step 2 — Synthetic trace generation

Generate **≥ 180 simulated days** (≈ 6 months, so multiple seasonal regimes appear). For each variable:

- **Diurnal cycle:** sinusoid — temperature peaks mid-afternoon, humidity anti-correlated (peaks pre-dawn).
- **Seasonal drift:** slow sinusoid over the 180 days, plus a distinct "monsoon" block of ~30 consecutive days with sustained high humidity, to create a genuinely different regime.
- **Autocorrelation:** AR(1) noise, not i.i.d. — real sensors drift smoothly.
- **Sensor noise:** Gaussian, per-sensor σ matched to datasheet accuracy (DHT22: ±0.5 °C, ±2–5% RH).
- **Dropouts:** randomly null ~2% of readings to test robustness.

Save the generated raw trace as CSV, at the same 1-reading-per-second rate as the dev CSV described in your Jetson doc, then aggregate into the 24-hour feature windows.

## Step 3 — Labelling (**the step that differs by tier**)

**Tier B:** label each 24-hour window by an infection-pressure score built from *published* relations — the accumulation of hours in the pathogen-favourable temperature band while humidity is above the wetness threshold, modulated by the ventilation proxy. Bin the resulting score into Low / Medium / High. Document the exact function in the paper and cite its basis. **Add label noise (~5%)** to prevent the tree from achieving a degenerate 100% and to simulate real-world diagnostic uncertainty.

**Tier A:** label each window by the *observed* outcome from the vision branch — whether any disease-class leaf count rose by more than X% over the following N days. No score function needed; the label is empirical.

Check class balance and target roughly 40 / 35 / 25 (Low / Medium / High). Do not force exact balance — high-risk days genuinely are rarer, and the model should see that prior.

## Step 4 — Splitting (**do not skip**)

Split by **contiguous time blocks**, never randomly:
- Train: days 1–126 (70%)
- Validation: days 127–153 (15%)
- Test: days 154–180 (15%)

Ensure the monsoon regime falls at least partly in the **test** block — that gives you the generalization claim: the tree holds up on an environmental regime it did not train on. State the split policy and the reason in the paper; it signals methodological care.

## Step 5 — Training

`sklearn.tree.DecisionTreeClassifier`. Grid search over `max_depth ∈ {3,4,5,6,8}`, `min_samples_leaf ∈ {5,10,20}`, `criterion ∈ {gini, entropy}`, with `class_weight='balanced'`. Select on **validation macro-F1**, not accuracy.

**Constrain depth to ≤ 5.** The entire justification for using a tree is auditability; a depth-12 tree is not auditable and you lose the argument that motivated the choice. If a shallow tree costs a little accuracy, that trade *is* the finding — report it.

## Step 6 — Reporting (Table VIII, 3 classes)

- Per-class precision / recall / F1 / support, plus overall accuracy and macro-F1
- Confusion matrix (Low↔Medium confusion is acceptable and expected; Low↔High is not)
- Tree depth and leaf count
- **Ranked feature importances** — `hours_humidity_above_85_24h` should dominate; if it does not, something is wrong with the label function
- Inference latency (expect microseconds — makes the "negligible alongside three neural stages" claim concrete)
- **Baseline comparison rows:** majority-class baseline, plus a random forest. If the RF beats the tree by only ~1–2 points, that gap is the quantified price of interpretability and directly justifies the design choice.
- **Robustness:** re-evaluate with sensor noise doubled, and with 10% dropout. Degradation curves here are worth more than the headline accuracy.

## Step 7 — Also produce Fig. 5 panel (c) input
Export one worked example — a real window from the test set where the tree outputs HIGH — to use as the concrete chain in Fig. 5.

---

# PART 2 — E2: LABEL-LEVEL FUSION EVALUATION
*Fills Table IX (P11).*

> ## ✅ AS-RUN RECORD (2026-07-24) — supersedes the "source 1/2" plan below for now
>
> The protocol below lists real deployment windows and turmeric-image sequences as
> the preferred episode sources (in that priority order). Neither exists yet — no
> unit is deployed, and no turmeric image dataset exists in the repo (E4 not
> started) — so this run uses **source 3, constructed scenarios**, exactly as the
> protocol allows, with two additions beyond the minimum spec:
>
> - **Vision branch is real, not invented:** every leaf count comes from running the
>   actual trained **Case 2 + Replay (cycle 2)** checkpoint (98.42% test acc) over
>   real held-out tomato test images (`data/02_tomato_5cls/test/`, confirmed never
>   used in training/replay/model-selection for any Case/cycle). Only the
>   *day-by-day sequencing* of which images go into which day is constructed.
> - **Sensor branch is real E1 output**, sampled from `windows_labelled.csv`
>   restricted to the E1 **test period** (`day ≥ 161`), not invented risk values.
> - **Ground truth is independently reasoned per scenario category**, not read off
>   Φ's own output — an early draft of this evaluation made exactly that mistake
>   (ground truth = Φ's output column), which made fusion trivially score 100% and
>   was caught during verification. Fixed by adding an explicit **adversarial
>   category** (`confound_high_risk_false_alarm`) with the same `(trend, risk)`
>   signature as the early-warning category but a different true cause — Φ cannot
>   tell them apart, so fusion is *guaranteed* wrong on one of them.
> - **8 categories × 6 replicates = 48 episodes** (protocol's minimum is 20–30),
>   covering: confirmed rising risk at High/Medium/Low, an early-warning case
>   (stable count, High risk — no symptoms yet), a recovering case, a sensor-failure
>   case, and the adversarial confound above.
> - **Result:** Label-level fusion = **87.5% accuracy, 71.75 macro-F1**, vs. 60.4%/
>   29.6 (vision-only) and 35.4%/20.6 (sensor-only). False-alarm rate: fusion and
>   sensor-only tie at 12.5%, both below vision-only's 14.6%. Fusion degrades
>   gracefully to vision-only's output on every sensor-failure episode (confirmed
>   exact match). Fusion is **not perfect** — it fails on the adversarial confound
>   category, inheriting sensor-only's mistake, since a 2-input discrete rule has no
>   signal to distinguish a coincidental high-humidity reading from a causal one.
>   This is reported as an honest limitation in the paper, not hidden.
> - **Early-warning lead time:** reported as a worked-example calculation (~1 day),
>   not measured from a continuously-extended simulation — flagged as a limitation
>   in `experiments/part3_fusion_eval/README.md`.
> - **Code:** `experiments/part3_fusion_eval/code/` (`vision_infer.py`,
>   `build_episodes.py`, `fusion_eval.py`), config `configs/part3_fusion_eval.yaml`,
>   artifacts in `experiments/part3_fusion_eval/outputs/`. Full methodology,
>   assumptions, and open items: `experiments/part3_fusion_eval/README.md`.
> - **Still open:** real turmeric images and real deployment episodes (sources 1/2
>   below) would let this be re-run as a genuine field measurement rather than a
>   methodology validation.
>
> The step-by-step plan below is kept as background on the full protocol, including
> the preferred (not-yet-available) episode sources.

## Step 1 — Define the fusion rule Φ explicitly

The draft states `s_t = Φ(Δc_t, r_t)`. Write out Φ as a rule table before evaluating — the paper needs it stated, and it must be fixed before you measure, not tuned afterwards on the test episodes.

Suggested form (tune the thresholds on a validation split, then freeze):

| Δ disease count | Risk state | Fused output |
|---|---|---|
| Rising (> +20% over 3 days) | High | **Alert** — probable active infection |
| Rising | Medium | **Watch** — monitor closely |
| Rising | Low | **Observe** — possible non-environmental cause |
| Stable | High | **Watch** — conditions favourable, no symptoms yet |
| Stable | Low/Med | **Normal** |
| Falling | any | **Normal / recovering** |

Note that `Stable + High → Watch` is the case that justifies fusion at all: it is the *early warning* the vision branch alone cannot produce, because there are no symptoms yet.

## Step 2 — Build the evaluation episode set

An "episode" = one 3–7 day window with a ground-truth outcome label. Aim for **30–50 episodes** minimum. Sources, in order of preference:
1. Real deployment windows labelled by an agronomist or by the eventual observed outcome
2. Turmeric dataset image sequences paired with the simulated environmental traces from E1
3. Constructed scenarios covering all six rows of the rule table — including the deliberately adversarial ones

**Include negative and adversarial episodes** or the table is worthless: rising counts under benign conditions (should NOT alert), high risk with no symptoms (early warning), and a sensor-dropout episode (should degrade to vision-only, not produce a wrong fused state).

## Step 3 — Compare three decision sources

| Source | Input |
|---|---|
| Vision only | Δ counts, ignore environment |
| Sensor only | Decision-tree risk state, ignore images |
| **Label-level fusion** | Both, via Φ |

## Step 4 — Report (Table IX)

- Accuracy and macro-F1 on the final state
- **False-alarm rate** — the metric a farmer actually cares about; a system that cries wolf gets switched off
- **Early-warning lead time** (days between first Watch/Alert and confirmed outbreak), if your episodes support it. This is the single most persuasive number in the whole fusion section if you can produce it.
- A **sensor-failure row**: fusion behaviour when `r_t` is unavailable, demonstrating graceful degradation to vision-only

**Be prepared for fusion not to win on raw accuracy.** If vision-only matches it, say so, and argue the case on lead time and false-alarm rate instead. That is a more honest and more interesting result than a marginal accuracy bump, and reviewers respond well to it.

---

# PART 3 — E3: ON-DEVICE CYCLE COST AND LATENCY
*Fills Table X (P12). Must be measured **on the Jetson Orin Nano**, not on the training GPU.*

> ## 🟡 AS-RUN RECORD (2026-07-26) — PARTIAL, ON PROXY HARDWARE
>
> **The Jetson measurement this protocol specifies was NOT performed** — no Jetson
> hardware available. What was actually run, and how it deviates:
>
> | Protocol step | What actually happened |
> |---|---|
> | Step 1 — measurement conditions | Recorded, but for a **laptop NVIDIA RTX 4050 (6 GB, CUDA 12.1)**, not a Jetson. No `nvpmodel` power mode / thermal data (Jetson-specific). |
> | Step 2 — retraining cycle | ✅ Done on proxy HW. Re-ran the **real** Case 2 + Replay cycle-2 config (identical data/hyperparams/seed, output redirected so the real checkpoint E2 depends on was untouched — verified). **373.2 s** wall (348.4 s train), **155.6 MB** peak VRAM, 1245 MB process RAM. Reproduced 98.42% test acc exactly → confirms faithful re-timing. |
> | Step 3 — FP16 gate ★ | ✅ Done, **but with a substitution**: TensorRT could not be installed (pip `tensorrt`/`tensorrt-cu12` pull `nvidia-cuda-runtime-cu13`, whose wheel fails against this machine's CUDA 12.1 torch build on Windows). **ONNX Runtime GPU** used as the compiled-graph path instead. Result: FP32 98.51% / PyTorch-FP16 98.59% (Δ +0.09 pp) / compiled 98.51% (Δ 0.00). |
> | Step 4 — inference latency | 🟡 Partial. Classification measured (eager 10.2 ms/img, compiled 3.8 ms/img, median-of-5×200 after warm-up — laptop GPU clocks were too noisy for single-shot timing). Detection/segmentation **cited from prior paper [21]** rather than re-measured, per `01_SCOPE_AND_COVERAGE.md` §1.2. Power draw not measured. |
>
> **On the Δ ≈ 0 result:** this protocol explicitly anticipated it ("If it is
> essentially zero, report that honestly and reframe the argument as *the gate must
> verify this rather than assume it*"). That is exactly what the paper now argues —
> +0.09 pp is one image out of 1,137, i.e. noise, and the gate's justification is
> that a system cannot know conversion is lossless without measuring the artifact it
> deploys.
>
> **Unexpected finding worth keeping:** PyTorch eager FP16 is *slower* than FP32
> (12.4 vs 10.2 ms/img) for a network this small — per-op conversion overhead
> exceeds the arithmetic saving. The compiled graph, not half precision, is what
> delivers the 2.7× speedup.
>
> Code: `experiments/part4_ondevice_proxy/`. Full honest inventory of gaps:
> `experiments/part4_ondevice_proxy/README.md` §7.
>
> The step-by-step protocol below is kept as the specification for the real
> on-device run, which remains outstanding.

## Step 1 — Fix the measurement conditions
Record and report: JetPack / TensorRT / PyTorch versions, **power mode** (`nvpmodel -q`, e.g. 15 W vs 25 W), fan setting, and ambient temperature. Different power modes change these numbers by a factor of two — an unreported power mode makes the results irreproducible.

## Step 2 — Retraining cycle
Run one complete monthly cycle on device with the winning recipe. Measure:
- Wall-clock time for the fine-tune (`time`), with per-epoch breakdown
- Peak RAM (`tegrastats` sampled every second throughout — log to file, take the max)
- ONNX export time
- `trtexec` FP16 engine build time
- Fixed-test-set evaluation time
- **Thermal:** confirm no throttling occurred (`tegrastats` reports temperatures); if the SoC throttles, say so — it is a legitimate deployment finding

## Step 3 — The FP16 gate measurement ★
This is the measurement that justifies the paper's gating design, so run it deliberately:

1. Evaluate the retrained **PyTorch** candidate on the fixed test set → accuracy A_pt
2. Compile to a TensorRT FP16 engine
3. Evaluate the **compiled engine** on the same fixed test set → accuracy A_trt
4. Report both and **Δ = A_trt − A_pt**

**Report Δ whatever it is.** If it is non-zero, that is direct evidence for gating on the engine. If it is essentially zero, report that honestly and reframe the argument as *"the gate must verify this rather than assume it"* — which remains a valid design principle. Do not quietly drop the measurement if it comes out small; a reviewer asking "did you check?" and getting "yes, and it was 0.1%" is a much better position than having no answer.

If you can, repeat over 3 retraining runs and report mean ± std — a single Δ could be run-to-run noise.

## Step 4 — Inference latency
For one complete 8-image capture set, measure per stage (warm models, discard the first run):
detection per image · segmentation per leaf · classification per batch of crops · decision tree · fusion · **end-to-end per set**

Report leaves detected per image so the per-leaf figures are interpretable. Then compute device utilization: with 2 sets/day at the measured end-to-end time, the device is busy roughly *(2 × t_set) / 86400* of the day. That percentage is the argument that retraining contends with nothing — make it explicit.

Optionally add idle and peak power draw (W) from `tegrastats`; useful if solar-powered deployment is ever in scope.

---

# PART 4 — E4: TURMERIC CL DEPLOYMENT VALIDATION
*Fills Table XI (P13). Full protocol from data selection through layer-level training.*

## Step 1 — Dataset assembly

Two Mendeley sources (from your slides):
- `10.17632/jtttfbx342.1` — Image Dataset for Turmeric Plant Leaf Disease Detection
- `10.17632/g46dvrcvwn.1` — Turmeric Plant Disease Dataset

Five classes: **healthy · leaf blotch · leaf spot · dry leaf · aphid disease.**

**Pool both sources** and deduplicate — you need every image you can get (see Step 2). Check for near-duplicates across the two datasets with perceptual hashing before pooling; overlap between public datasets is common and would leak across splits.

Add any greenhouse-collected turmeric images you have. Record the per-class counts you actually end up with — this drives everything downstream.

## Step 2 — ⚠️ The volume problem, and how to handle it

At ~200 images/class, applying the tomato split policy directly gives:

| Split | Fraction | Images/class |
|---|---|---|
| Test (fixed, held out forever) | 15% | 30 |
| Validation | 10% | 20 |
| Initial training | 37.5% | 75 |
| Cycle 1 stream | 18.75% | ~37 |
| Cycle 2 stream | 18.75% | ~37 |

Two consequences you must handle explicitly:

**(a) The replay buffer floor breaks.** The tomato protocol uses "15% per class, minimum 100 images". Here 100 > the 75 available initial-training images — the rule is unsatisfiable. **Redefine for turmeric as `min(40, 50% of initial_train)`**, state the change and the reason in the paper. Silently applying a different buffer rule to the two datasets is exactly the kind of inconsistency reviewers find.

**(b) A 30-image test set gives coarse resolution.** One image = 3.3 percentage points, so differences under ~3 points between methods are not resolvable. Options, best first:
1. **Pool both datasets** to push toward 400/class (your original target) — then 60 test images/class, and the numbers become meaningful
2. Report **mean ± std over 3–5 seeds** and state that differences within noise are not claimed as differences
3. Reduce to a **single CL cycle** — larger streams, but you lose the cross-cycle trend

Do (1) and (2) together if at all possible. If you cannot get past ~200/class, **say so in the limitations** and present E4 as a feasibility check rather than a precision benchmark. That framing is defensible; overclaiming on 30 test images is not.

## Step 3 — Preprocessing (must match deployment exactly)

Feed the classifier what it will see in the field — not raw dataset images:
1. Run source images through **YOLOv8n detection** → leaf crops
2. Apply the **≥124×124 quality filter**
3. Run **YOLOv8n-seg** → mask, background filled black
4. Resize to **200×200**, zero-pad to **224×224**
5. Normalize to [0,1]

If a source image is already a single isolated leaf, skip to step 3. **Do not skip segmentation** — the tomato benchmark used segmented images, and an unsegmented turmeric set would make the two incomparable.

**Split by leaf identity if the metadata allows it** (as in the tomato protocol). If the datasets do not carry leaf IDs, cluster near-duplicates by perceptual hash and keep each cluster wholly within one split. State which you did.

Augmentation on training splits only — horizontal/vertical flip, ±15° rotation, mild brightness/contrast jitter. **No augmentation on val or test.** Given the small volume, augmentation matters more here than it did for tomato; consider augmenting the stream data ~2–3× to give each cycle enough gradient steps.

## Step 4 — Model initialization (Case 2 recipe)

The benchmark selected **Case 2 (ImageNet)**. Reproduce it for turmeric:

- Backbone: MobileNetV3-Small, ImageNet weights (torchvision or `timm`)
- Replace head with 5-class output
- Input 224×224×3

*(Optional, valuable if time permits: also run Case 3 — pretrain the backbone on the non-turmeric PlantVillage classes, then fine-tune. The tomato benchmark found Case 3 gives the best old-class retention, and if that reproduces on turmeric it materially strengthens the transferability claim. Frame as a secondary result.)*

## Step 5 — Base training, layer by layer

Follow Table II of the draft exactly.

**Phase i — classifier warm-up (10–15 epochs)**
| G1 features[0–3] | G2 features[4–8] | G3 features[9–12] | G4 head |
|---|---|---|---|
| Frozen | Frozen | Frozen | **1e-3** |

Train only the new 5-class head. Purpose: adapt the randomly-initialized head before any gradient reaches the backbone. Skipping this lets large early head gradients corrupt the pretrained features — the single most common transfer-learning mistake.

**Phase ii — unfreeze G3**
| G1 | G2 | G3 | G4 |
|---|---|---|---|
| Frozen | Frozen | **1e-4** | **1e-3** |

G3 is where disease-discriminative features live, so this is where turmeric-specific adaptation happens. Train to convergence with early stopping on validation macro-F1 (patience 10, **minimum 5 epochs before early stopping activates** — the small classes produce noisy validation curves and will otherwise trigger premature stops).

Optimizer: AdamW, weight decay 1e-4. Batch size 32 (drop to 16 if VRAM-limited). Loss: cross-entropy with class weighting if the classes are imbalanced.

**Save this checkpoint as the Part-1 baseline** and evaluate it on the fixed test set. Every CL number is reported relative to it.

## Step 6 — Continual learning cycles (CatA protocol)

Two cycles, three methods (Replay, EWC, Naive as reference).

**Cycle 1**
| G1 | G2 | G3 | G4 |
|---|---|---|---|
| Frozen | Frozen | **1e-4** | **5e-4** |

**Cycle 2**
| G1 | G2 | G3 | G4 |
|---|---|---|---|
| Frozen | Frozen | **5e-5** | **2e-4** |

Per method:
- **Replay:** buffer = `min(40, 50% of initial_train)` per class, sampled from initial-training data only — **never** from val or test. Mix 50/50 with stream data in every batch.
- **EWC:** compute the Fisher matrix over the same buffer, applying the penalty **only to trainable parameters (G3, G4)**. λ = **10,000 for Cycle 1, 3,000 for Cycle 2** (the tuned schedule). *If time permits, re-run the λ grid on turmeric — the benchmark showed λ sensitivity depends on backbone domain specificity, so the tomato-optimal values may not transfer, and confirming or refuting that is itself a reportable result.*
- **Naive:** no protection, reference only.

Evaluate on the **fixed** test set after every cycle. Also track validation accuracy (old-class only) as the real-time forgetting signal.

## Step 7 — Report (Table XI)

Same columns as Table III so the comparison is direct: C1 Acc, C1 F1, C2 Acc, C2 F1, Avg, training time. Add rows for the Part-1 baseline and Naive.

**The three questions the table must answer:**
1. Does the tomato-derived recipe transfer to turmeric at all?
2. **Is the method ordering preserved** (Replay ≥ EWC > Naive on retention)? This is the real transferability claim — the ordering matters more than the absolute values.
3. How much does the ~3× smaller per-class volume cost, and how did the buffer-floor change affect replay?

Write the two discussion paragraphs the draft has placeholders for. Address the volume question head-on rather than waiting to be asked — it is the first thing a reviewer will probe.

---

# PART 5 — CONCLUSION: WHAT THE PAPER SHOULD CLAIM

You asked for a conclusion idea. Here is what the evidence supports — and, as importantly, what it does not.

## 5.1 The defensible headline

> **Continual learning for plant disease is not a single problem with a single best method, and the right choice is determined by the incremental scenario, the initialization, and the device's memory budget — not by a leaderboard.**

This is stronger than "replay wins" because it is what your 25 runs actually demonstrate, and it is genuinely useful to a practitioner. Three findings support it, and each is a complete argument on its own:

1. **Method ranking inverts across scenarios.** Isolation is within ~1 point of best in CatA and collapses to 6.38% in CatB. Any paper reporting one ranking is misleading its readers.
2. **Initialization sets the ceiling, and interacts with the method.** Case 2 is best overall; Case 3 gives the best old-class *retention*; Case 1 is worst everywhere. λ sensitivity itself scales with backbone domain specificity (+1.14% for Case 3 vs +0.09% for Case 1). Pretraining and CL-method choice are not independent decisions.
3. **Protection mechanisms can interfere.** Hybrid EWC+Replay is *worse* than replay alone while costing 37–40% more. More protection is not better protection — the mechanisms constrain and drive the same parameters in opposition.

## 5.2 Lead with the negative results

Your two negative findings are the most publishable material in the paper, and it is worth being deliberate about that:

- **Isolation's inverted failure** — Case 3 (6.38%) doing *worse* than Case 2 (45.45%) despite the stronger backbone is counter-intuitive and mechanistically explicable: the better the frozen backbone, the wider the gap to a randomly-initialized branch. That is a transferable insight about architecture-based CL generally, not a quirk of plant disease.
- **Hybrid underperforming its components** — a clean, well-motivated negative result with a mechanistic explanation.

Positive results ("replay reached 98%") are expected and easily matched. Negative results with mechanisms are what gets cited.

## 5.3 The systems claim

> **The complete adapt-and-verify loop fits on a single edge device with no cloud dependency** — 5.82 MB models, a retraining cycle inside the device's memory envelope, and a deployment gate that evaluates the compiled engine rather than the training artifact.

The gating-on-the-compiled-engine detail is small but distinctive: it is the kind of thing that only surfaces when a system is actually deployed rather than simulated, and it signals to reviewers that this is real work.

## 5.4 What NOT to claim

- ❌ Do not claim field validation of the environmental model if E1 uses simulated data. Say "simulated" plainly.
- ❌ Do not claim the fusion layer improves diagnostic accuracy unless E2 shows it. Claim interpretability, graceful degradation, and — if measurable — early-warning lead time.
- ❌ Do not claim long-term deployment stability from 2 cycles. Two cycles reveal forgetting dynamics; they do not characterize a year of operation.
- ❌ Do not present naive fine-tuning as a competitor it beats. It uses the full data history and is explicitly not a valid CL method under the storage constraint.

## 5.5 Closing paragraph — the arc to aim for

Something like: *this work moves plant disease monitoring from a model that is trained once and degrades, to a system that observes, doubts itself, relabels, retrains, and verifies before it trusts a new version of itself — entirely on a device in a greenhouse with no internet.* Then the future work (LLM teacher closing the relabel loop, agentic orchestration) reads as the natural next step in that arc rather than a list of leftovers.

---

# CHECKLIST

**Decide first**
- [ ] Choose Tier A / B / C for the decision tree (recommend **B now**, A later if data matures)
- [ ] Confirm 3 risk classes → **update draft Table VIII from 4 rows to 3**
- [ ] Confirm how many turmeric images/class you actually have after pooling both Mendeley sources

**E1 — decision tree**
- [ ] Generate ≥180 days of realistic traces (diurnal + seasonal + AR(1) + noise + dropouts)
- [ ] Label from published relations; add ~5% label noise
- [ ] Split by time block, monsoon regime in the test block
- [ ] Depth ≤ 5; select on validation macro-F1
- [ ] Report + baselines (majority, random forest) + noise/dropout robustness
- [ ] Export one worked example for Fig. 5(c)

**E2 — fusion**
- [ ] Write and freeze the rule table Φ before evaluating
- [ ] Build 30–50 episodes including adversarial and sensor-failure cases
- [ ] Report accuracy, macro-F1, false-alarm rate, lead time, degradation row

**E3 — on-device**
- [ ] Record JetPack/TensorRT versions and power mode
- [ ] Full cycle timing + peak RAM via tegrastats; check for thermal throttling
- [ ] **Measure Δ = A_trt − A_pt and report it whatever it is** (ideally over 3 runs)
- [ ] Per-stage and end-to-end latency; compute device utilization %

**E4 — turmeric**
- [ ] Pool + deduplicate both Mendeley datasets
- [ ] Redefine replay buffer floor to `min(40, 50% of initial_train)` and state it
- [ ] Preprocess through the real pipeline (detect → filter → segment → 224×224)
- [ ] Phase i warm-up, then Phase ii G3 unfreeze; save Part-1 baseline
- [ ] Two CL cycles × {Replay, EWC, Naive}; 3–5 seeds, report mean ± std
- [ ] Answer the three questions in Step 7 explicitly in the discussion

**Paper-wide**
- [ ] Add the "environmental data is simulated" disclosure to Section VI
- [ ] Add CO₂-as-ventilation-proxy justification to Section V-A
- [ ] Add the time-block splitting rationale to Section VI
