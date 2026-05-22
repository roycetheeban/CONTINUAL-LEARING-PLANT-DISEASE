# Continual Resource Report (All 3 Cases, Cat A + Cat B)

This report summarizes training-resource behavior for continual learning, focused on edge-oriented practicality.

All values below are taken from saved `metrics.json` files in each continuation output folder.

---

## 1) Training Device Details

- Training device in configs: `cuda`
- Detected GPU (`nvidia-smi`):
  - `NVIDIA GeForce RTX 4050 Laptop GPU`
  - GPU memory: `6141 MiB`
  - Driver: `595.79`

Notes:
- CPU model / total system RAM query via CIM was permission-blocked in this environment.
- Per-run process RAM and peak VRAM are still available from each `metrics.json`.

---

## 2) What Resource Metrics Were Collected

From each run:
- `runtime.total_wall_time_sec`
- `runtime.train_wall_time_sec`
- `runtime.peak_vram_mb`
- `runtime.process_ram_mb_end`
- `model_footprint.model_size_mb`
- `model_footprint.num_parameters`
- `config.device`

---

## 3) Model Footprint Summary (Important for Edge)

### Cat A (5-class heads)
- Standard methods (`ewc`, `experience_replay`, `naive_finetune`):
  - Params: `1,522,981`
  - Size: `5.8097 MB`
- `parameter_isolation` (grows across cycles):
  - Cycle1: `1,527,685` params, `5.8277 MB`
  - Cycle2: `1,532,389` params, `5.8456 MB`

### Cat B (5->7 expansion / branch methods)
- Standard Cat-B methods (`ewc/replay/hybrid/naive`):
  - Params: `1,525,031`
  - Size: `5.8175 MB`
- `cat_b_isolation_new_branch`:
  - Params: `1,563,511`
  - Size: `5.9643 MB`

Resource implication:
- Isolation gives best retention behavior in some settings, but has higher model growth.
- Replay/Naive keep size fixed and are more deployment-simple.

---

## 4) Cat A Resource Table (Cycle 2, Final Cycle)

| Case | Method | Train Time (s) | Peak VRAM (MB) | Process RAM End (MB) | Model Size (MB) |
|---|---|---:|---:|---:|---:|
| Case1 | ewc | 173.9541 | 169.51 | 1298.66 | 5.8097 |
| Case1 | experience_replay | 299.8844 | 155.58 | 1243.23 | 5.8097 |
| Case1 | parameter_isolation | 78.6674 | 122.14 | 1225.10 | 5.8456 |
| Case1 | naive_finetune | 93.9879 | 162.17 | 1229.80 | 5.8097 |
| Case2 | ewc (lambda1000) | 87.2660 | 169.51 | 1279.38 | 5.8097 |
| Case2 | experience_replay | 304.6246 | 155.58 | 1177.77 | 5.8097 |
| Case2 | parameter_isolation | 178.1234 | 122.14 | 1232.38 | 5.8456 |
| Case2 | naive_finetune | 160.4085 | 162.17 | 1229.56 | 5.8097 |
| Case3 | ewc | 85.9635 | 169.51 | 1297.55 | 5.8097 |
| Case3 | experience_replay | 197.4812 | 155.58 | 1253.34 | 5.8097 |
| Case3 | parameter_isolation | 76.3816 | 122.14 | 1202.17 | 5.8456 |
| Case3 | naive_finetune | 161.1195 | 162.17 | 1230.16 | 5.8097 |

Cat A resource pattern:
- Lowest VRAM: `parameter_isolation` (~122 MB).
- Fastest in many runs: `parameter_isolation` or `ewc`.
- Replay is usually the slowest in Cat A because mixed-batch replay overhead is higher.

---

## 5) Cat B Resource Table (Cycle 2, Final Cycle)

| Case | Method | Train Time (s) | Peak VRAM (MB) | Process RAM End (MB) | Model Size (MB) |
|---|---|---:|---:|---:|---:|
| Case1 | cat_b_ewc_head_expand | 390.6973 | 600.87 | 1375.38 | 5.8175 |
| Case1 | cat_b_replay_head_expand | 445.1587 | 325.39 | 1332.28 | 5.8175 |
| Case1 | cat_b_hybrid_replay_ewc | 275.3230 | 600.87 | 1379.83 | 5.8175 |
| Case1 | cat_b_isolation_new_branch | 108.7050 | 218.68 | 1341.38 | 5.9643 |
| Case1 | cat_b_naive_full_retrain_head_expand | 284.1518 | 325.39 | 1332.92 | 5.8175 |
| Case2 | cat_b_ewc_head_expand | 265.0872 | 600.87 | 1374.47 | 5.8175 |
| Case2 | cat_b_replay_head_expand | 259.4445 | 325.39 | 1330.42 | 5.8175 |
| Case2 | cat_b_hybrid_replay_ewc | 564.9888 | 600.87 | 1377.32 | 5.8175 |
| Case2 | cat_b_isolation_new_branch | 65.1010 | 218.68 | 1339.37 | 5.9643 |
| Case2 | cat_b_naive_full_retrain_head_expand | 169.5980 | 325.39 | 1241.66 | 5.8175 |
| Case3 | cat_b_ewc_head_expand | 295.5160 | 600.87 | 1378.53 | 5.8175 |
| Case3 | cat_b_replay_head_expand | 252.8694 | 325.39 | 1329.52 | 5.8175 |
| Case3 | cat_b_hybrid_replay_ewc | 369.6730 | 600.87 | 1379.34 | 5.8175 |
| Case3 | cat_b_isolation_new_branch | 107.5471 | 218.68 | 1341.89 | 5.9643 |
| Case3 | cat_b_naive_full_retrain_head_expand | 247.3422 | 325.39 | 1316.78 | 5.8175 |

Cat B resource pattern:
- Highest VRAM: `ewc` / `hybrid` (~600.87 MB peak).
- Mid VRAM: `replay` / `naive` (~325.39 MB).
- Lowest VRAM + shortest train time: `isolation` (~218.68 MB), but with larger model size and lower new-class quality in several runs.

---

## 6) Edge-Retraining Practical Conclusions

### Resource-friendliness
- Best VRAM efficiency: `parameter_isolation` (Cat A) and `cat_b_isolation_new_branch` (Cat B).
- Best overall balance (performance + resources) in Cat B is usually `replay` or `hybrid`:
  - Replay uses much less VRAM than EWC/Hybrid high-VRAM path and keeps model size fixed.
  - Hybrid often improves final accuracy tradeoff but can be slower/heavier in some cases.

### If edge retraining budget is tight
- Prefer:
  1. `replay` as primary practical CL method.
  2. `ewc` when replay memory policy is constrained and retention emphasis is high.
  3. `isolation` only when strict memory/VRAM for training is critical and lower plasticity is acceptable.

### Deployment-oriented metric readiness
- Already available in saved outputs:
  - wall time
  - train time
  - peak VRAM
  - process RAM
  - model size
  - parameter count
- For final edge paper section, add on-device (Jetson/edge CPU) inference benchmarks separately.

---

## 7) Raw Coverage Assurance

This report is based on **all discovered continuation `metrics.json` files** under:
- `part1_case1_scratch/Continuation/catA_same_classes`
- `part1_case1_scratch/Continuation/catB_new_classes`
- `part1_case2_imagenet_finetune/Continuation/catA_same_classes`
- `part1_case2_imagenet_finetune/Continuation/catB_new_classes`
- `part1_case3B_plantvillage_finetune/Continuation/catA_same_classes`
- `part1_case3B_plantvillage_finetune/Continuation/catB_new_classes`

So no saved run in those continuation folders was ignored for resource extraction.

---

## 8) Deployment-Readiness Comparison (From Saved Outputs)

### A) What is already comparable now

From current logs/metrics, we can already compare:
- `total_wall_time_sec` (end-to-end runtime)
- `train_wall_time_sec`
- `peak_vram_mb`
- `process_ram_mb_end`
- `model_size_mb`
- `num_parameters`

### B) Practical ranking (Cycle 2 trend, all cases)

#### Lowest training VRAM
1. `cat_b_isolation_new_branch` (~218.68 MB)
2. `cat_b_replay_head_expand` / `cat_b_naive_full_retrain_head_expand` (~325.39 MB)
3. `cat_b_ewc_head_expand` / `cat_b_hybrid_replay_ewc` (~600.87 MB)

#### Smallest model size
1. Standard methods (`ewc/replay/hybrid/naive`) ~5.8175 MB
2. Isolation branch method ~5.9643 MB (larger due to added branch)

#### Fastest retraining tendency
- Most frequently fastest in Cat-B cycle2: `cat_b_isolation_new_branch` and `naive_full_retrain` (case-dependent).
- Replay and EWC are moderate.
- Hybrid can be highest runtime depending on case/config.

### C) Combined deployment-view interpretation

- If **edge retraining memory budget** is strict:
  - Isolation is best on VRAM/time, but performance tradeoff on new classes is significant.
- If **balanced CL performance + reasonable resources** is needed:
  - Replay is usually the best compromise.
- If **peak accuracy balance** is priority and VRAM is acceptable:
  - Hybrid is strong but resource-heavier in several runs.

---

## 9) On-Device Benchmark Template (Jetson / Edge CPU)

Use this table in the final paper once on-device inference tests are run:

| Platform | Case | Category | Method | Precision | Batch | Latency (ms/img) | FPS | Peak RAM (MB) | Model Size (MB) | Old F1 | New F1 |
|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| Jetson Orin Nano | Case 1 | Cat B | replay | FP16 | 1 | TBD | TBD | TBD | 5.8175 | 0.6424 | 0.3193 |
| Jetson Orin Nano | Case 1 | Cat B | hybrid | FP16 | 1 | TBD | TBD | TBD | 5.8175 | 0.6746 | 0.3024 |
| Jetson Orin Nano | Case 1 | Cat B | isolation | FP16 | 1 | TBD | TBD | TBD | 5.9643 | 0.6865 | 0.1657 |
| Edge CPU | Case 2 | Cat B | replay | INT8 | 1 | TBD | TBD | TBD | 5.8175 | 0.6992 | 0.3294 |
| Edge CPU | Case 3 | Cat B | replay | INT8 | 1 | TBD | TBD | TBD | 5.8175 | 0.6952 | 0.3272 |

Recommended minimum device comparison set:
1. `replay` (resource/performance baseline CL)
2. `hybrid` (performance-leaning CL)
3. `isolation` (resource-leaning CL)
