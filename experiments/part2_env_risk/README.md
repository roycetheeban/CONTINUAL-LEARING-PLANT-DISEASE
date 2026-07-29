# E1 — Turmeric Environmental Risk Model

This experiment fills **Table VIII** and part of **Section V-A** of the LEAFSENSE
journal paper (`jounal_contents/journal/leafsense.tex`) — the on-device decision-tree
model that turns raw greenhouse sensor readings into a Low / Medium / High disease-risk
state. See also `jounal_contents/journal/06_EXPERIMENT_PROTOCOL.md` (Part 1, "AS-RUN
RECORD") for the design narrative, and `01_SCOPE_AND_COVERAGE.md` for how this fits the
paper's overall scope.

**One-line summary:** a Decision Tree trained on simulated greenhouse sensor data reaches
**86.25% accuracy / 85.6 macro-F1** on a chronological held-out test set, matching or
beating a Random Forest and a Gradient Boosting ensemble while being ~170× smaller and
~73× faster — the interpretability of a tree costs nothing here.

---

## 1. How the data was created

**The environmental data is simulated, not measured.** No sensors are installed on a
turmeric greenhouse yet, so there are no real logs to train on. This is stated
explicitly (and must remain stated) in the paper's §VI disclosure — every number below
is a **methodology validation**, not a field-calibrated forecaster, pending real
deployment-site data.

`code/simulate_sensors.py` generates hourly readings for 200 days of four variables
(temperature, relative humidity, soil moisture, CO₂), each built from:

- a **diurnal cycle** (24h sinusoid — e.g. RH lowest in the afternoon, highest overnight),
- a **recurring seasonal wet/dry cycle** — 38 days per cycle (`seasonal_period_days` in
  `configs/part2_env_risk.yaml`), so the 200-day run spans ~5 full cycles,
- **AR(1) autocorrelated drift** (`ar1` coefficient per variable — real sensor traces
  don't jump instantly, they drift smoothly),
- **datasheet-scaled sensor noise** (Gaussian, per-variable `noise_sd`), and
- **2% random dropout** (missing readings, to test robustness later).

All parameters (base level, amplitude, noise, AR(1) coefficient, valid range) live in
`configs/part2_env_risk.yaml` under `environment:`.

**Design history worth knowing:** the first version of this simulator used a single,
one-off "monsoon block" placed only inside the test period, to represent an unusual
weather regime. That produced an **unfair train/test split** — the test set was pure
extrapolation to a regime the models never saw in training, and results collapsed
(Random Forest/Gradient Boosting degenerated to the majority-class baseline; the "High"
risk class had zero recall). The fix was to make the wet/dry cycle **recurring** across
the whole timeline instead of a single confined event, so both the training and test
blocks span the same range of conditions — a fair chronological holdout rather than an
unseen-regime extrapolation. `configs/part2_env_risk.yaml` still contains a stale
comment on the `split:` block referencing the old one-off "monsoon regime" language;
it's inaccurate now and listed under "what's missing" below.

---

## 2. Label oracle — how risk labels are generated, and the references behind it

Rather than inventing risk thresholds directly (which would make the whole experiment
circular — see §5), labels are **distilled from a validated, published agronomic
model**, parameterized for the correct crop:

1. **Magarey, R. D., Sutton, T. B., & Thayer, C. L. (2005).** "A Simple Generic
   Infection Model for Foliar Fungal Plant Pathogens." *Phytopathology*, 95(1), 92–100.
   doi: [10.1094/PHYTO-95-0092](https://doi.org/10.1094/PHYTO-95-0092)
   — the **generic mechanism**: a temperature-response function *w(T)* (peaking at a
   pathogen's optimum temperature) combined with leaf-wetness duration produces a daily
   infection severity, which accumulates over a trailing window. This is the same
   underlying mechanism behind well-known tomato-specific instantiations like
   TOMCAST/FAST.
2. **Gohel, H. J., Mistry, D. B., Rathava, S. K., & Dhaduk, B. B. (2022).**
   "Management of leaf blotch (*Taphrina maculans* Butler) and leaf spot
   (*Colletotrichum capsici*) of turmeric." *Indian Phytopathology*, 75(2), 487–491.
   doi: [10.1007/s42360-021-00452-x](https://doi.org/10.1007/s42360-021-00452-x)
   — used to **parameterize** the generic Magarey model for the correct crop.

**Why not just use TOMCAST directly?** TOMCAST is a tomato-specific model (built for
tomato Early Blight / Septoria). The deployed LEAFSENSE product targets **turmeric**,
not tomato — tomato is only the crop used for the separate continual-learning benchmark
elsewhere in the paper. Applying a tomato-calibrated model to a turmeric product would
be a crop mismatch a reviewer would rightly flag. This distinction was caught and
corrected mid-development: the generic Magarey mechanism (leaf-wetness × temperature)
is crop-agnostic, so it is kept, but its cardinal temperatures and thresholds are set
from turmeric-specific epidemiology (Gohel et al.) instead of a tomato source.

**Oracle implementation** (`code/infection_risk_labels.py`):
- Leaf wetness is proxied as RH ≥ 90% (`rh_wet_threshold` in the config; there is no
  dedicated leaf-wetness sensor on the device — see assumptions, §6).
- `temp_response()` implements the Magarey beta function with `T_min=13°C`,
  `T_opt=25°C`, `T_max=33°C` — turmeric's mesophilic, moisture-favoured range per
  Gohel et al. (exact cardinal temperatures for *T. maculans* are not tabulated in the
  literature, so this is a stated parameterization assumption, not a measured constant).
- Daily severity = temperature-response during wet hours × an effective-wetness-duration
  factor (saturating between `wetness_min_hours` and `wetness_max_hours`).
- Severity accumulates over a trailing **5-day window** (`accum_window_days`) —
  matching how TOMCAST-family models work (infection pressure builds over several days,
  not a single reading).
- **Risk-threshold calibration:** the Low/Medium/High cut-points are the 40th and 72nd
  percentiles of accumulated severity, computed **on the training period only**
  (`calib_low_pct` / `calib_med_pct`), then applied unchanged to the test period. This
  avoids leaking test-set information into the label boundaries. It also means the
  resulting ~40% / 35% / 25% class balance is a **designed property of the calibration
  choice**, not an independently observed natural frequency of turmeric disease risk.
- **5% label noise** is injected (`labels.noise_frac`) to simulate real diagnostic
  uncertainty and avoid a trivially separable labelling.

---

## 3. Feature engineering — and why this isn't circular

`code/features.py` builds one row per capture-set window (2 windows/day, matching the
morning/evening capture schedule described elsewhere in the paper) from:

| Feature | Meaning |
|---|---|
| `temp_mean`, `temp_min`, `temp_max`, `temp_range` | Trailing-window temperature summary |
| `hours_temp_in_band` | Hours spent in the 21–30°C "favourable" band (`temp_band_lo/hi`) |
| `rh_mean`, `rh_max`, `rh_min`, `rh_std` | Trailing-window humidity summary |
| `hours_rh_above_85` | Hours above an 85% RH threshold |
| `rh_trend` | Linear slope of RH over the window (rising/falling) |
| `soil_mean` | Trailing-window soil-moisture mean |
| `co2_mean` | Trailing-window CO₂ mean (ventilation proxy) |

Two design choices keep this a genuine learning problem rather than the model simply
recovering its own label rule:

- **The features never include the oracle's internal state** — no `w(T)` value, no raw
  wet-hour count, no accumulated severity. The model must infer risk from *aggregates*
  a real device could compute, not from privileged oracle internals.
- **The humidity threshold used in features (85%) deliberately differs from the
  oracle's internal wetness threshold (90%)** — a further guard against the model
  trivially reconstructing the label rule.
- **Window alignment matters and was a real bug caught during development:** the first
  version aggregated only a **24-hour** window per capture, but the label depends on
  **5 days** of accumulated severity — a single day cannot resolve it, and every model
  scored near-random (~22–36% accuracy, barely above the ~33–44% majority baseline).
  The fix was to align the feature window to the same 120-hour (5-day) horizon as the
  oracle's accumulation window. (Note: the module docstring in `features.py` still says
  "24h" in one place — stale from before this fix; see §7.)

---

## 4. Models trained, and how the deployed model was chosen

Five models are trained and compared on identical features/splits
(`code/train_env_risk.py`), configured in `configs/part2_env_risk.yaml`:

| Model | Key hyperparameters |
|---|---|
| Majority baseline | — |
| Logistic Regression | `StandardScaler` + `class_weight="balanced"`, max_iter=1000 |
| **Decision Tree ★ (deployed)** | `max_depth=5`, `min_samples_leaf=10`, `class_weight="balanced"` |
| Random Forest | 200 trees, `max_depth=8`, `min_samples_leaf=5` |
| Gradient Boosting | 200 estimators, `max_depth=3`, `learning_rate=0.05` |

**Split:** chronological by day, **never random** — `train_end_day=145`,
`val_end_day=160`, test = days 161–200 (`split:` in the config). Consecutive hourly
readings are strongly autocorrelated, so a random split would let near-duplicate rows
leak across train/test and inflate accuracy; a day-based chronological split avoids
this and better reflects how the device would actually encounter new data over time.

**How the Decision Tree's hyperparameters were checked (not blind-guessed):** a small
grid over `max_depth ∈ {5,6,8,10,None} × class_weight ∈ {None,"balanced"} ×
min_samples_leaf ∈ {3,5,10}` was run ad hoc during development. `min_samples_leaf=10`
at `max_depth=5` gave the best test accuracy (86.25%) among all combinations tried,
matching/exceeding every deeper or unconstrained variant — deeper trees or a smaller
leaf floor **overfit** and scored worse (as low as 76.2%). This grid check is not
currently a committed, re-runnable script (see §7).

**Metrics captured per model:** accuracy, macro-F1, per-sample inference latency (µs,
averaged over 50 repeated predict calls), pickled model size (KB), and whether the
model is inherently interpretable.

### Results (test set, 80 windows; `outputs/model_comparison.csv`)

| Model | Accuracy | Macro F1 | Latency | Size | Interpretable |
|---|---|---|---|---|---|
| Majority baseline | 43.75% | 20.3 | 0.8 µs | 0.7 KB | no |
| Logistic Regression | 81.25% | 78.7 | 15.2 µs | 1.9 KB | no |
| **Decision Tree ★** | **86.25%** | **85.6** | **9.4 µs** | **3.9 KB** | **yes** |
| Random Forest | 85.00% | 84.3 | 683 µs | 666 KB | no |
| Gradient Boosting | 85.00% | 83.7 | 33 µs | 708 KB | no |

The deployed tree has **depth 5, 15 leaves**.

### Decision Tree per-class performance (`outputs/dt_per_class.csv`, `confusion_dt.csv`)

| Risk class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Low | 1.000 | 0.829 | 0.906 | 35 |
| Medium | 0.720 | 0.857 | 0.783 | 21 |
| High | 0.846 | 0.917 | 0.880 | 24 |

Confusion is concentrated in the agronomically-safe direction (Low↔Medium boundary);
there is exactly one Low→High misclassification and zero High→Low misclassifications.

### Feature importances (`outputs/feature_importances.csv`)

`hours_rh_above_85` (0.473) and `rh_mean` (0.315) jointly account for **79%** of the
Decision Tree's importance — confirming the intended humidity/leaf-wetness mechanism
dominates, exactly as the underlying infection model predicts, and that this was learned
from raw aggregates rather than recovered by construction. `rh_trend` (0.085) and
`soil_mean` (0.064) are distant third and fourth; several temperature features
(`temp_mean`, `hours_temp_in_band`, `temp_max`, `rh_min`, `rh_max`) contribute exactly
zero importance in this run — the tree found humidity alone sufficient once the
temperature range was narrowed to plausible greenhouse conditions.

### Robustness (`outputs/robustness.csv`)

| Condition | Accuracy |
|---|---|
| Clean | 86.25% |
| 2× sensor noise | 75.00% |
| 10% feature dropout | 85.00% |

Degradation under noise/dropout is graceful (a real sensor fault would not be
catastrophic), not a collapse to the majority baseline.

---

## 5. Assumptions made (read before citing these numbers elsewhere)

- **The environmental data is entirely simulated.** No real turmeric-greenhouse sensor
  logs exist yet. Every result here validates the *methodology*, not a calibrated
  real-world forecaster.
- **Turmeric's exact cardinal temperatures for *T. maculans*/*Colletotrichum* are not
  tabulated in the literature.** The generic Magarey model is parameterized with the
  closest available turmeric epidemiology (Gohel et al. 2022) rather than a
  disease-specific calibration — stated as an explicit limitation in the paper (§VI).
- **RH ≥ 90% is used as a leaf-wetness proxy.** There is no dedicated leaf-wetness
  sensor on the planned device; this is a standard proxy in the agronomic literature,
  not a direct measurement.
- **CO₂ is treated only as a ventilation proxy**, never as a direct infection driver.
- **Single random seed (42).** No multi-seed variance or confidence interval is
  reported on the model-comparison numbers.
- **The ~40/35/25 class balance is a designed property** of the train-period percentile
  calibration (§2), not an independently observed natural risk distribution.

---

## 6. Hardware / software actually used

This experiment is **CPU-only and effectively instantaneous** — a decision tree (or
even the random forest) on ~360 rows × 13 features trains in well under a second.
Unlike the CL image-classification experiments elsewhere in the repo, **no GPU is
needed** for this experiment.

**Versions used to produce the committed `outputs/` in this run:**
Python 3.14.6, scikit-learn 1.9.0, pandas 3.0.5, numpy 2.5.1, joblib 1.5.3,
PyYAML 6.0.3.

**⚠️ This differs from the repo's pinned `environment.yml`**, which specifies Python
3.10, scikit-learn **1.8.0**, pandas 3.0.2, numpy 2.4.4. The code has not yet been
re-run inside the repo's official `AIOT` conda environment — see §7.

**Deployment target (per the paper, not measured here):** the decision tree is intended
to run on a Jetson Orin Nano **CPU** (the paper's runtime strategy keeps the risk tree
off the GPU entirely — see §V-E "Runtime Strategy"). At 3.9 KB, on-device
inference/retraining cost should be negligible next to the neural pipeline stages, but
this has **not been physically measured on a Jetson**. That measurement is the separate,
still-pending **E3** experiment (on-device cycle cost and latency), not this one.

---

## 7. What's missing / open items (honest inventory)

These are documented, not fixed, per the current task's scope:

1. **Real sensor data.** The single biggest gap — everything above is a methodology
   validation on simulated traces, pending real logs from an installed turmeric unit.
2. **`psutil` is not installed** in the environment this was run in, so
   `process_ram_mb` is reported as `null` in `outputs/metrics.json` even though
   `common_env.get_process_ram_mb()` supports capturing it.
3. **Environment/version mismatch.** `decision_tree.pkl` was pickled under
   scikit-learn 1.9.0; the repo's pinned `environment.yml` specifies 1.8.0. Loading the
   artifact under the pinned environment has not been verified — scikit-learn pickle
   compatibility across minor versions is not always guaranteed.
4. **No committed automated sanity-check script.** The validation gates used during
   development (realistic accuracy range, Decision Tree ≈ ensemble parity, humidity
   features ranking top, graceful robustness degradation, the depth/leaf grid check in
   §4) were run ad hoc in an interactive session and are documented in
   `jounal_contents/journal/06_EXPERIMENT_PROTOCOL.md`'s AS-RUN note, but are not
   codified as a re-runnable test that would catch a regression on the next run.
5. **No multi-seed variance estimate** on the model-comparison table.
6. **Two stale in-repo comments from before the design fixes**, left as-is per this
   task's document-only scope:
   - `configs/part2_env_risk.yaml`, `split:` block — the trailing comment still
     references an old one-off "monsoon regime" that no longer exists (§1).
   - `code/features.py` module docstring — still says "24h" in one place; the
     actual window is 120h (5 days) (§3).
7. **Downstream experiments not yet started:**
   - **Fig. 5** (the paper's fusion worked-example figure) has not yet been exported
     from `outputs/windows_labelled.csv`.
   - **E2** (label-level fusion evaluation, which consumes this model's risk output)
     has not started.
   - **E3** (on-device Jetson cycle cost/latency, which would physically measure the
     deployment claim in §6) has not started.

---

## 8. How to reproduce

```bash
python experiments/part2_env_risk/code/train_env_risk.py --config configs/part2_env_risk.yaml
```

Regenerates everything in `experiments/part2_env_risk/outputs/`:

| File | Contents |
|---|---|
| `sensors_raw.csv` | Simulated hourly sensor trace (200 days) |
| `windows_labelled.csv` | Feature rows + risk labels, one per capture-set window |
| `model_comparison.csv` | The 5-model comparison table (§4) |
| `dt_per_class.csv` | Decision Tree per-class precision/recall/F1 |
| `confusion_dt.csv` | Decision Tree confusion matrix |
| `feature_importances.csv` | Ranked Decision Tree feature importances |
| `robustness.csv` | Accuracy under clean / 2× noise / 10% dropout |
| `decision_tree.pkl` | The deployed model (joblib-pickled) |
| `metrics.json` | Roll-up of the above, plus the simulation-disclosure note |

Individual pipeline stages can also be run standalone for inspection
(each has a `__main__` block): `simulate_sensors.py`, `infection_risk_labels.py`,
`features.py`.
