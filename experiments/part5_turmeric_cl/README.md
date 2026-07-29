# E4 — Turmeric Continual-Learning Validation

Fills **Table XIV** of the LEAFSENSE paper. Tests whether the CL recipe selected on
tomato (Case 2 ImageNet init + Replay, layer groups G1–G4 with split learning rates)
transfers to the deployed target crop.

**Headline:** it transfers. Replay reaches **90.03% ± 3.54** on turmeric after two
CL cycles, above the 89.10% ± 2.35 base and above naive fine-tuning (87.85% ± 2.47).
**No method-vs-method difference is statistically resolvable on this dataset** — every
pairwise McNemar test returns p ≥ 0.109. E4 is a **feasibility result**, not a
precision benchmark, and must be reported as such.

---

## 1. Data

Source: `data/turmeric_5_classes_splitted/` (1,045 images, 5 classes, 224×224 RGB,
pre-segmented). Re-split by `data/06_scripts/splitters/split_turmeric_5cls.py` into
`data/07_turmeric_5cls/`.

| Split | Images | Per class |
|---|---|---|
| initial_train | 417 | 76–92 |
| cl_cycle1_stream | 210 | 38–46 |
| cl_cycle2_stream | 207 | 38–46 |
| replay_buffer | 198 | 38–40 |
| val | 102 | 19–23 |
| test | 107 | 19–24 |

### Why test/ and val/ were carried over unchanged
The shipped data already had a train/val/test split, and the existing checkpoint
`other models/model/MobilenetV3_Phase3_EWC_Incremental_Results/best_phase3_model.pth`
was trained against exactly that split. Re-splitting from scratch would place images
that model trained on into a new test set, making its reported 89.81% retention
uninterpretable. So only `train/` (834 images) was re-split, 50/25/25 — which
reproduces the tomato protocol's internal ratios (37.5 : 18.75 : 18.75 of the whole).

### Data quality checks performed
- **Exact duplicates:** 0
- **Near-duplicates** (16×16 dHash, Hamming ≤ 10): 5 pairs, of which **2 crossed
  splits**. Both were excluded from test/val (their partners remain in train), so
  test = 107 and val = 102 rather than 108/103.
- **Effective unique images:** 99.5% (1040/1045)

> An earlier check using an 8×8 aHash reported 187 "near-duplicates". That was a
> false alarm — at that resolution any two similarly-posed leaf silhouettes on black
> backgrounds collide. The 16×16 result above is the correct one.

### Replay buffer rule — changed, and must be stated in the paper
Tomato used *"15% per class, minimum 100"*. That is **unsatisfiable** here:
initial_train holds only 76–92 images per class. Redefined for turmeric as
**`min(40, 50% of initial_train)`** → 38–40 per class. Silently applying different
buffer rules across datasets is exactly the inconsistency reviewers look for.

---

## 2. Method

Deliberately mirrors the tomato CatA pipeline so the comparison means something.

**Architecture: torchvision MobileNetV3-Small, ImageNet weights (Case 2).**
*Not* the existing timm phase3 checkpoint — that model has a different block
structure and a custom 2-layer head (2.19M params), and the recipe under test is
defined by layer groups `features[0:4] / [4:9] / [9:13] / classifier`, which are
torchvision-specific. Building on it would answer a different question. It remains an
independent reference point.

**Base training** (Table II of the paper):
- Phase i — backbone frozen, head only (lr 1e-3). Adapts the randomly-initialised
  5-class head before any gradient reaches the pretrained features.
- Phase ii — unfreeze G3 (`features[9:]`) at 1e-4, head at 1e-3. G1–G2 stay frozen.

**CL cycles** — G1–G2 frozen throughout; per-cycle learning rates from the paper's
schedule (cycle 1: G3 1e-4 / head 5e-4; cycle 2: G3 5e-5 / head 2e-4).

| Method | Behaviour |
|---|---|
| `replay` | Every batch is 16 buffer + 16 stream samples (50/50) |
| `ewc` | Stream only, plus λ·Fisher penalty anchoring to previous weights (λ=10,000 cycle 1; 3,000 cycle 2) |
| `naive` | Stream only, no protection. **Reference condition — not a valid CL method** |

Cycle 2 of a method resumes from cycle 1 **of the same method**, so each forms a
genuine continual sequence. 3 seeds (42/43/44) × 3 methods × 2 cycles + 3 base = 21 runs.

---

## 3. Results

### Table XIV (test set, n=107, mean ± std over 3 seeds)

| Stage | Accuracy | Macro-F1 |
|---|---|---|
| Base (Case 2, ImageNet) | 89.10 ± 2.35 | 89.18 ± 2.16 |
| Replay — cycle 1 | 92.52 ± 3.37 | 92.29 ± 3.76 |
| **Replay — cycle 2** | **90.03 ± 3.54** | **89.93 ± 3.73** |
| Naive — cycle 2 *(reference only)* | 87.85 ± 2.47 | 87.72 ± 2.68 |
| EWC — cycle 2 | 88.79 ± 1.62 | 88.58 ± 1.97 |

Retention vs base after two cycles: **Replay +0.93 pp**, EWC −0.31 pp, naive −1.25 pp.
Replay is the only method that does not end below its starting point.

### Statistical reality — read this before quoting any number

With **107 test images**, one image = 0.93 pp and a single run's 95% CI is
**±5.7 pp**. Every pairwise McNemar test on the final cycle returns **p ≥ 0.109**:

| Comparison | Seeds (p) |
|---|---|
| Replay vs EWC | 1.000 / 0.453 / 0.727 |
| Replay vs naive | 1.000 / 0.109 / 1.000 |
| EWC vs naive | 1.000 / 0.453 / 1.000 |

Per-seed accuracies show why:

```
replay_cycle2   85.98 | 92.52 | 91.59      6.5 pp spread
naive_cycle2    85.98 | 86.92 | 90.65
```

Seed 42's replay run went 93.46% (cycle 1) → 85.98% (cycle 2): a 7.5 pp swing
*within a single method*. Run-to-run variance (±2–4 pp) swamps the method
differences (~1–2 pp).

**Claimable:** the recipe transfers — Replay trains successfully on turmeric,
ranks highest, and is the only method that does not degrade the base model.
**Not claimable:** that Replay is significantly better than EWC or naive here.

McNemar is used rather than comparing confidence intervals because both models are
scored on the *same* images — only the disagreements carry information, and
overlapping-CI comparison would declare "no difference" almost regardless of data.

---

## 4. Two corrections made during development

### 4.1 Replay batch-ratio bug (fixed)
The stream loader initially used `train.batch_size` (32) while the buffer loader used
`old_per_batch` (16), producing a **16 old : 32 new** mix — a 1:2 ratio instead of the
intended 50/50. `new_per_batch` was in the config but never read.

Caught because replay scored **82.24%**, a 10 pp drop below the incumbent, which
replay should never cause. After the fix: **91.59%**. The tomato script uses separate
batch sizes for exactly this reason.

### 4.2 Training schedule corrected (post-hoc — disclosed)
The first matrix run used `patience=5`, `phase1.max_epochs=15`, `phase2.max_epochs=30`.
Inspecting the logs showed a genuine defect:

- seed 43's phase 1 **hit the 15-epoch ceiling while still improving**
- seeds 43 and 44 both peaked at **epoch 6** then early-stopped, because a 102-image
  val set makes macro-F1 swing several points epoch-to-epoch (0.52 → 0.91 → 0.67 →
  0.81 within one run), tripping `patience=5` mid-climb

Changed to `patience=8`, `min_epochs=10`, `phase1.max_epochs=30`, `phase2.max_epochs=40`
and re-ran all 21 runs.

**This config change was made after seeing the first results, so it is disclosed
here.** Three things make it a methodological fix rather than result-chasing:
1. It was motivated by a defect visible in the training logs (epoch ceiling, stop
   mid-improvement) independent of any method comparison.
2. It applied identically to every seed and every method.
3. **It made the headline comparison worse, not better** — replay's margin over
   naive shrank from 3.73 to 2.18 pp, its std widened from ±1.87 to ±3.54, and
   directional consistency dropped from 3/3 seeds to 2/3.

It did achieve what it targeted: base-model variance tightened from ±3.74 to ±2.35.

Both runs are retained — `outputs/` (final, patience 8) and
`outputs_patience5_backup/` (first run) — so the comparison is verifiable.

| Stage | patience 5 | patience 8 (reported) |
|---|---|---|
| base | 88.79 ± 3.74 | 89.10 ± 2.35 |
| replay cycle 2 | 90.65 ± 1.87 | 90.03 ± 3.54 |
| ewc cycle 2 | 86.60 ± 2.16 | 88.79 ± 1.62 |
| naive cycle 2 | 86.92 ± 3.37 | 87.85 ± 2.47 |

---

## 5. Assumptions and limitations

- **107-image test set** — the binding constraint. No split ratio fixes it; a 15%
  test set would give only 156 images (±4.7 pp).
- **3 seeds.** More would tighten the estimate but cannot overcome the test-set size.
- **Class order here is alphabetical** (ImageFolder on `data/07_turmeric_5cls`):
  `Aphids=0, Blotch=1, Dry Leaf=2, Healthy=3, Leaf_Spot=4`. This **differs** from the
  phase3 checkpoint's stored `class_to_idx` (`Healthy=0, Leaf_Spot=1, Blotch=2,
  Dry=3, Aphids=4`). Do not mix the two — a swap silently mislabels everything.
- **Cycle streams are random partitions of one dataset**, not genuinely new field
  observations. Real deployment streams would carry distribution shift this cannot
  simulate.
- Naive is a **reference condition**, not a competitor — it must be labelled as such
  wherever it appears.

---

## 6. Reproduce

```bash
python experiments/part5_turmeric_cl/code/run_all.py          # all 21 runs
python experiments/part5_turmeric_cl/code/analyze_results.py  # Table XIV + McNemar
```

Single stages:
```bash
python code/train_base_turmeric.py --seed 42
python code/train_cl_turmeric.py --method replay --cycle 1 --seed 42
```

Config: `configs/part5_turmeric_cl.yaml`. Environment: repo `AIOT` conda env
(torch 2.5.1+cu121). Runtime ~25 min for the full matrix on an RTX 4050.

| Output | Contents |
|---|---|
| `table_xiv.csv` | Aggregated mean ± std per stage |
| `pairwise_mcnemar.csv` | Paired tests, all method pairs × seeds |
| `metrics_summary.json` | Everything above + interpretation note |
| `<stage>/metrics/test_predictions.csv` | Per-image predictions (enables McNemar) |
| `<stage>/checkpoints/model.pth` | Trained weights |

---

## 7. What's missing

1. **More data** — the only real fix. ~400 images/class would give a ~60-image
   per-class test set and make method ranking possible.
2. **No CatB (class-expansion) run** for turmeric — only CatA (same-class streaming).
3. **Cycle streams are not real temporal data** (see §5).
4. **Single architecture** — no Case 1 / Case 3 comparison on turmeric.
5. The existing phase3 checkpoint is **not** integrated as a baseline row; it uses a
   different architecture and class order, so its 89.81% is not directly comparable
   to the numbers here.
