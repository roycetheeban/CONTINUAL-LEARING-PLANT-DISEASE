# E2 — Label-Level Fusion Evaluation

This experiment fills **Table IX** (`\label{tab:fusion}`) and produces **Fig. 5** of the
LEAFSENSE journal paper (`jounal_contents/journal/leafsense.tex`) — comparing three
decision sources (vision-only leaf counts, sensor-only risk state, and label-level
fusion Φ) on a set of constructed evaluation episodes. See also
`jounal_contents/journal/06_EXPERIMENT_PROTOCOL.md` (PART 2) for the design narrative
this follows, and `experiments/part2_env_risk/README.md` for the E1 model this
consumes.

**One-line summary:** on 48 constructed episodes, **label-level fusion reaches 87.5%
accuracy / 71.75 macro-F1**, against 60.4%/29.6 for vision-only and 35.4%/20.6 for
sensor-only, while matching sensor-only's false-alarm rate (12.5%) rather than
vision-only's (14.6%). Fusion is **not perfect** — it shares sensor-only's exact
blind spot on one deliberately adversarial category (see §2) — which is the honest,
non-trivial result this experiment was designed to be able to produce.

---

## 1. How the vision-branch data was created

**The classifier inference is real; the day-by-day episode sequencing is
constructed.** There is no chronological (multi-day, same-plant) tomato photo series
anywhere in the repo — PlantVillage-style datasets are flat, unordered single-shot
images with no severity/date metadata (confirmed by checking filenames, EXIF, and
for any manifest — none exists). So "day-to-day change" cannot come from real
chronology; it has to come from choosing how many real images of which class are fed
to the real classifier on each synthetic day.

`code/vision_infer.py` runs the actual trained continual-learning checkpoint —
**Case 2 (ImageNet) + Replay, cycle 2** (the paper's best CatA recipe, 98.42% test
accuracy / macro-F1 0.9827) — once over the **entire held-out tomato test pool**
(`data/02_tomato_5cls/test/`, 1,137 images across 5 classes), caching every image's
real predicted class to `outputs/test_pool_predictions.csv`. Pool-wide accuracy in
this run was **98.50%** (1,120/1,137), consistent with the checkpoint's original
training-time evaluation — this is a built-in sanity gate (`vision_infer.py` exits
with an error if pool accuracy drops below 95%, which would indicate a class-order or
preprocessing mismatch).

`code/build_episodes.py` then builds episodes by **sampling** from this cache: for
each synthetic day, it draws real `Tomato___Early_blight` images to hit an intended
target-class count and real images from the other 4 classes as background, then
tallies counts by the classifier's **predicted** class (not the folder ground-truth
label) — so the leaf-count vector a fusion episode sees is genuinely what the
deployed model would report, imperfections and all, not a hand-typed number.
Sampling is with replacement across episodes/days; this is inference-only (no
gradient step), so there is no train/test leakage concern beyond the original
disjoint split (`test/` is confirmed never used in training, replay, or
model-selection for any Case/cycle — verified against the splitter script and both
replay configs).

**Why tomato images for a turmeric product?** No turmeric image dataset exists in
this repo (confirmed — E4, the turmeric CL validation, has not started). Tomato is
the only real image pool available, consistent with the paper's existing framing
(the CL benchmark itself is tomato; only the environmental oracle in E1 is
turmeric-parameterized). Fig. 5 therefore pairs a real tomato vision-branch output
with a real turmeric-parameterized risk output — disclosed in the figure caption, the
same honesty standard already applied to E1's simulated sensor data.

---

## 2. Episode design, the fusion rule Φ, and why the comparison isn't circular

Per the protocol (`06_EXPERIMENT_PROTOCOL.md` PART 2, step 1), Φ is fixed **before**
evaluation, as a lookup on `(trend, risk)`:

| Trend | Risk | Fused output |
|---|---|---|
| Rising (Δ ≥ +20% over the window) | High | **Alert** |
| Rising | Medium | **Watch** |
| Rising | Low | **Observe** |
| Stable | High | **Watch** |
| Stable | Low/Medium | **Normal** |
| Falling | any | **Normal** |
| any | sensor unavailable | falls back to **vision-only's** output |

`vision-only` predicts from Δcount alone (`rising → Alert`, else `Normal`) — it never
sees the risk state. `sensor-only` predicts from risk alone (`High → Watch`,
`Medium → Observe`, `Low → Normal`, `unavailable → Unknown`) — it never sees the
counts. Both are genuinely blind to information the other has, by construction.

**48 episodes** (5-day windows, matching E1's `accum_window_days=5`) were built, 6
replicates across **8 categories**. Each category has a **ground-truth outcome
derived from an independently reasoned "true cause"** — not read off Φ's output
column — so the three sources can genuinely disagree:

| Category | Trend | Risk | True cause | Ground truth |
|---|---|---|---|---|
| `rising_high` | rising | High | genuine progression, favourable environment | Alert |
| `rising_medium` | rising | Medium | genuine progression, moderate environment | Watch |
| `rising_low_benign` | rising | Low | **non-infectious** count rise (e.g. insect/mechanical damage) despite unfavourable environment | Observe |
| `stable_high` | stable | High | favourable environment, symptoms not yet visible (early warning) | Watch |
| `stable_low_medium` | stable | Low/Med | nothing happening (true negative) | Normal |
| `falling_recovering` | falling | Low/Med | recovering | Normal |
| `sensor_failure_rising` | rising | *unavailable* | genuine progression, sensor down | Alert |
| `confound_high_risk_false_alarm` | stable | High | humidity elevated for an **unrelated** reason (e.g. recent irrigation), no real infection | Normal |

**An earlier version of this evaluation made a methodological mistake worth
recording**: the first pass literally copied Φ's own output column as each
category's ground truth, which made fusion trivially score 100% — the exact
circularity trap this design is supposed to avoid (fusion would "win" a test built
to match its own rule, telling a reviewer nothing). This was caught during
verification (100% accuracy against a self-authored rule table is a red flag, not a
result) and fixed by adding the **`confound_high_risk_false_alarm`** category: it has
the *identical* `(trend=stable, risk=High)` signature as `stable_high`, but a
different true cause and therefore a different ground truth (Normal, not Watch). Φ
cannot tell the two apart — it necessarily predicts Watch for both — so fusion is
*guaranteed* to be wrong on one of them. This is deliberate: it demonstrates the
irreducible error of a simple 2-input discrete rule against a coincidental risk
factor, rather than only evaluating Φ on cases it was designed to get right.

A day-count jitter bug was also caught and fixed during development: applying
independent random jitter to each day of a "stable" trajectory (small integers, e.g.
`[2,2,3,2,2]`) sometimes flipped the realized first-vs-last-day trend into a spurious
rise or fall. Fixed by applying one jitter value uniformly across all days of an
episode, which varies replicate episodes without perturbing the intended shape.
(`outputs/episodes.csv` logs both `intended_trend` and `realized_trend` per episode
so this class of bug is directly checkable — after the fix, 47/48 episodes realize
their intended shape.)

---

## 3. The sensor/risk side — real E1 windows, not invented values

`code/build_episodes.py` samples risk states from **real** rows of
`experiments/part2_env_risk/outputs/windows_labelled.csv`, restricted to the E1
**test period** (`day ≥ 161`, i.e. `val_end_day + 1` from `configs/part2_env_risk.yaml`)
— so every risk state an episode sees is an actual Decision Tree output on held-out
E1 data, not a value chosen to make Φ look good. All three risk classes (Low/Medium/
High) have real windows available in that period (35/21/24 windows respectively).

---

## 4. Results

### Table IX (`outputs/table_ix_comparison.csv`)

| Decision source | Accuracy | Macro F1 | False alarms |
|---|---|---|---|
| Vision only (counts) | 60.42% | 29.56 | 7 (14.58%) |
| Sensor only (risk state) | 35.42% | 20.57 | 6 (12.50%) |
| **Label-level fusion** | **87.50%** | **71.75** | 6 (12.50%) |

**False alarm** = the source escalates to Alert/Watch while the true outcome is
Normal/Observe.

### Per-category accuracy (`outputs/episodes_scored.csv`, `metrics.json`)

| Category | Vision-only | Sensor-only | Fusion |
|---|---|---|---|
| `rising_high` | 100% | 0% | 100% |
| `rising_medium` | 0% | 0% | 100% |
| `rising_low_benign` | 0% | 0% | 100% |
| `stable_high` | 0% | 100% | 100% |
| `stable_low_medium` | 100% | 100% | 100% |
| `falling_recovering` | 100% | 83.3% | 100% |
| `sensor_failure_rising` | 100% | 0% | 100% |
| `confound_high_risk_false_alarm` | 83.3% | 0% | **0%** |

Each single-modality source fails in a real, explainable way: vision-only over-alerts
whenever a non-infectious cause raises the count (`rising_low_benign`) and completely
misses the early-warning case where risk is high but no symptoms exist yet
(`stable_high`, 0%). Sensor-only cannot distinguish confirmed disease from mere
favourable conditions (0% on `rising_high`/`rising_medium`/`rising_low_benign`,
since it always answers from risk state alone) and produces nothing at all when the
sensor fails (`sensor_failure_rising`, 0%, scored as `Unknown` against `Alert`).
Fusion recovers from all of these **except** the adversarial confound category,
where it inherits sensor-only's exact mistake because Φ genuinely cannot see any
signal that distinguishes it from `stable_high` — an honest limitation, not hidden.

**Sensor-failure graceful degradation:** confirmed — on all 6
`sensor_failure_rising` episodes, fusion's output exactly matches vision-only's
output (`metrics.json: sensor_failure_degrades_to_vision_only = true`), i.e. a dead
sensor degrades to the vision-only decision rather than producing a corrupted joint
state, as claimed in §V ("Label-Level Fusion") of the paper.

**Early-warning lead time:** reported as a **worked example**, not a second live
simulation — using the `rising` day-count shape (`[2,3,5,7,9]` in
`configs/part3_fusion_eval.yaml`), vision-only's Δ-based rule would not cross its
+20% threshold until day 2 of the window, whereas fusion (and sensor-only) already
flag `Watch` on day 1 of a `stable_high` scenario, since risk alone is already High.
That gives a **1-day** lead-time figure (`metrics.json:
early_warning_lead_time_days`). This is a coarse, config-derived estimate, not
measured from a continuously-running simulation across the day boundary — flagged
under "what's missing" below.

---

## 5. Assumptions made

- **Episode day-sequencing is constructed**, not chronological — no real multi-day
  tomato image series exists. The *classifier inference* on each image is real; the
  choice of which images to feed on which day is engineered to realize each
  category's trend shape.
- **Tomato images stand in for turmeric** in this worked example (no turmeric image
  dataset exists yet in the repo). The environmental/risk half is genuinely
  turmeric-parameterized (from E1); the vision half is genuinely tomato. This mixed
  pairing is for illustration only and is disclosed in the paper.
- **Ground truth per category is a human-reasoned judgement** about each
  constructed scenario's true cause (e.g. "this count rise is from insect damage,
  not disease"), not derived from any measurement — this is inherent to using
  constructed rather than field-observed episodes, and is why the categories, not
  just the raw episode count, matter for interpreting the table.
- **A single disease class (`Tomato___Early_blight`) is used as the focus class**
  across all episodes for consistency between Table IX and Fig. 5. Simultaneous
  multi-disease fusion is out of scope here.
- **Single random seed (42).** No multi-seed variance estimate on Table IX.
- **The +20%/-20% rising/falling thresholds** (`rising_threshold_pct`,
  `falling_threshold_pct` in the config) are fixed a priori from the protocol
  document's suggested rule, not tuned against these episodes.

---

## 6. Hardware / software actually used

Unlike E1 (which ran in an ad hoc environment that drifted from the repo's pinned
versions), this experiment was run inside the repo's **official `AIOT` conda
environment**, matching `environment.yml` exactly:

- Python 3.10, **torch 2.5.1+cu121, torchvision 0.20.1+cu121** (CUDA available,
  `torch.cuda.is_available() == True` on the machine this ran on), pandas 3.0.2,
  numpy 2.4.4, scikit-learn 1.8.0, Pillow 12.2.0.

Vision inference (1,137 images through MobileNetV3-Small) ran on **GPU** and
completed in well under a minute; episode construction and fusion scoring
(pandas/sklearn only, no deep learning) are CPU-only and near-instantaneous. No
Jetson measurement was taken — the deployment-hardware cost of the fusion layer
itself is negligible (it is a lookup table), and the vision-branch cost is already
covered by the paper's CL/deployment sections.

---

## 7. What's missing / open items (honest inventory)

1. **Real multi-day image sequences.** The biggest gap — episodes are constructed
   from single-shot images sampled to fit an intended trend, not photographed on
   consecutive real days from the same plants.
2. **No turmeric images.** The vision half of every episode is tomato; only the risk
   half is turmeric-parameterized. A real E2 on turmeric needs E4's dataset first.
3. **Early-warning lead time is a worked-example calculation**, not measured by
   continuing a `stable_high` episode forward in simulated time until symptoms
   actually appear and counting elapsed days. A more rigorous version would extend
   selected episodes with confirmatory "outbreak" days and measure the gap directly.
4. **Ground truth is human-reasoned per category**, not independently measured —
   inherent to constructed episodes and disclosed throughout this README, but worth
   restating: this is a methodology demonstration of what fusion *can* do given
   correctly-labelled scenarios, not a field accuracy measurement.
5. **Single random seed**, no variance estimate on Table IX.
6. **No committed automated regression test** for the fusion-rule scoring logic
   (`fusion_eval.py`) — correctness was checked by manual inspection of
   `episodes_scored.csv` and the category-accuracy breakdown in this README, not by
   a re-runnable assertion suite.
7. This experiment consumes E1's risk output; if E1 is ever re-run with real
   sensor data, this experiment should be re-run afterward to stay consistent.

---

## 8. How to reproduce

```bash
python experiments/part3_fusion_eval/code/vision_infer.py
python experiments/part3_fusion_eval/code/build_episodes.py
python experiments/part3_fusion_eval/code/fusion_eval.py
```

(Run with the repo's `AIOT` conda environment active, or an equivalent environment
matching `environment.yml`; requires E1's outputs — `experiments/part2_env_risk/outputs/
windows_labelled.csv` and `sensors_raw.csv` — to already exist.)

Regenerates everything in `experiments/part3_fusion_eval/outputs/`:

| File | Contents |
|---|---|
| `test_pool_predictions.csv` | Real classifier prediction cache, all 1,137 test-pool images |
| `episodes.csv` | The 48 constructed episodes (pre-scoring) |
| `episodes_scored.csv` | Episodes with all three decision-source predictions and correctness flags |
| `table_ix_comparison.csv` | The 3-row Table IX comparison |
| `metrics.json` | Roll-up: Table IX, per-category accuracy, degradation check, lead-time worked example |

Fig. 5 (`jounal_contents/journal/figures/fig5_fusion.pdf`) is built from
`episodes_scored.csv` (episode_id=1, category `rising_high`) and E1's
`sensors_raw.csv` by `jounal_contents/journal/figures/src/fig5_fusion.py`.
