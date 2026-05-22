# Continuation Report - All 3 Cases (Cat A + Cat B)

This report consolidates continuation results for:
- Case 1: `part1_case1_scratch`
- Case 2: `part1_case2_imagenet_finetune`
- Case 3: `part1_case3B_plantvillage_finetune`

---

## 1) Case 1 - Cat A (Same Classes)

| Type | Source Set | Method | Cycle | Old F1 | New F1 | Old Acc | New Acc | Joint F1 (Old+New)/2 |
|---|---|---|---:|---:|---:|---:|---:|---:|
| Selected CL | catA_same_classes | ewc | 1 | 0.9755 | - | 0.9780 | - | 0.9755 |
| Selected CL | catA_same_classes | ewc | 2 | 0.9705 | - | 0.9736 | - | 0.9705 |
| Selected CL | catA_same_classes | experience_replay | 1 | 0.9703 | - | 0.9745 | - | 0.9703 |
| Selected CL | catA_same_classes | experience_replay | 2 | 0.9666 | - | 0.9719 | - | 0.9666 |
| Selected CL | catA_same_classes | parameter_isolation | 1 | 0.9563 | - | 0.9631 | - | 0.9563 |
| Selected CL | catA_same_classes | parameter_isolation | 2 | 0.9596 | - | 0.9657 | - | 0.9596 |
| Baseline | catA_same_classes (normal) | naive_finetune | 1 | 0.9732 | - | 0.9763 | - | 0.9732 |
| Baseline | catA_same_classes (normal) | naive_finetune | 2 | 0.9713 | - | 0.9754 | - | 0.9713 |

## 2) Case 1 - Cat B (New Classes)

| Type | Source Set | Method | Cycle | Old F1 | New F1 | Old Acc | New Acc | Joint F1 (Old+New)/2 |
|---|---|---|---:|---:|---:|---:|---:|---:|
| Selected CL | catB_new_classes | cat_b_ewc_head_expand | 1 | 0.6587 | 0.2486 | 0.8804 | 0.7892 | 0.4537 |
| Selected CL | catB_new_classes | cat_b_ewc_head_expand | 2 | 0.6526 | 0.3079 | 0.8734 | 0.8665 | 0.4803 |
| Selected CL | catB_new_classes | cat_b_replay_head_expand | 1 | 0.6381 | 0.3191 | 0.8382 | 0.9323 | 0.4786 |
| Selected CL | catB_new_classes | cat_b_replay_head_expand | 2 | 0.6424 | 0.3193 | 0.8478 | 0.9246 | 0.4809 |
| Selected CL | catB_new_classes | cat_b_hybrid_replay_ewc | 1 | 0.6776 | 0.3059 | 0.9244 | 0.8530 | 0.4918 |
| Selected CL | catB_new_classes | cat_b_hybrid_replay_ewc | 2 | 0.6746 | 0.3024 | 0.9244 | 0.8356 | 0.4885 |
| Selected CL | catB_new_classes | cat_b_isolation_new_branch | 1 | 0.6899 | 0.0860 | 0.9639 | 0.1779 | 0.3880 |
| Selected CL | catB_new_classes | cat_b_isolation_new_branch | 2 | 0.6865 | 0.1657 | 0.9499 | 0.4101 | 0.4261 |
| Baseline | catB_new_classes (normal) | cat_b_naive_full_retrain_head_expand | 1 | 0.6450 | 0.3228 | 0.8522 | 0.9439 | 0.4839 |
| Baseline | catB_new_classes (normal) | cat_b_naive_full_retrain_head_expand | 2 | 0.6539 | 0.3207 | 0.8707 | 0.9342 | 0.4873 |

---

## 3) Case 2 - Cat A (Same Classes)

| Type | Source Set | Method | Cycle | Old F1 | New F1 | Old Acc | New Acc | Joint F1 (Old+New)/2 |
|---|---|---|---:|---:|---:|---:|---:|---:|
| Selected CL | catA_same_classes | ewc (lambda1000) | 1 | 0.9740 | - | 0.9771 | - | 0.9740 |
| Selected CL | catA_same_classes | ewc (lambda1000) | 2 | 0.9798 | - | 0.9815 | - | 0.9798 |
| Selected CL | catA_same_classes | experience_replay | 1 | 0.9781 | - | 0.9807 | - | 0.9781 |
| Selected CL | catA_same_classes | experience_replay | 2 | 0.9827 | - | 0.9842 | - | 0.9827 |
| Selected CL | catA_same_classes | parameter_isolation | 1 | 0.9686 | - | 0.9719 | - | 0.9686 |
| Selected CL | catA_same_classes | parameter_isolation | 2 | 0.9697 | - | 0.9727 | - | 0.9697 |
| Baseline | catA_same_classes (normal) | naive_finetune | 1 | 0.9733 | - | 0.9763 | - | 0.9733 |
| Baseline | catA_same_classes (normal) | naive_finetune | 2 | 0.9908 | - | 0.9921 | - | 0.9908 |

## 4) Case 2 - Cat B (New Classes)

| Type | Source Set | Method | Cycle | Old F1 | New F1 | Old Acc | New Acc | Joint F1 (Old+New)/2 |
|---|---|---|---:|---:|---:|---:|---:|---:|
| Selected CL | catB_new_classes | cat_b_ewc_head_expand | 1 | 0.6742 | 0.3221 | 0.9164 | 0.9362 | 0.4981 |
| Selected CL | catB_new_classes | cat_b_ewc_head_expand | 2 | 0.6917 | 0.3238 | 0.9595 | 0.9478 | 0.5077 |
| Selected CL | catB_new_classes | cat_b_replay_head_expand | 1 | 0.6901 | 0.4942 | 0.9525 | 0.9807 | 0.5922 |
| Selected CL | catB_new_classes | cat_b_replay_head_expand | 2 | 0.6992 | 0.3294 | 0.9719 | 0.9807 | 0.5143 |
| Selected CL | catB_new_classes | cat_b_hybrid_replay_ewc | 1 | 0.6959 | 0.3227 | 0.9692 | 0.9381 | 0.5093 |
| Selected CL | catB_new_classes | cat_b_hybrid_replay_ewc | 2 | 0.7027 | 0.3278 | 0.9824 | 0.9671 | 0.5152 |
| Selected CL | catB_new_classes | cat_b_isolation_new_branch | 1 | 0.6741 | 0.1765 | 0.9244 | 0.4545 | 0.4253 |
| Selected CL | catB_new_classes | cat_b_isolation_new_branch | 2 | 0.6762 | 0.2058 | 0.9226 | 0.5629 | 0.4410 |
| Baseline | catB_new_classes (normal) | cat_b_naive_full_retrain_head_expand | 1 | 0.6106 | 0.3870 | 0.7713 | 0.9458 | 0.4988 |
| Baseline | catB_new_classes (normal) | cat_b_naive_full_retrain_head_expand | 2 | 0.6598 | 0.3930 | 0.8751 | 0.9691 | 0.5264 |

---

## 5) Case 3 - Cat A (Same Classes)

| Type | Source Set | Method | Cycle | Old F1 | New F1 | Old Acc | New Acc | Joint F1 (Old+New)/2 |
|---|---|---|---:|---:|---:|---:|---:|---:|
| Selected CL | catA_same_classes | ewc | 1 | 0.9566 | - | 0.9613 | - | 0.9566 |
| Selected CL | catA_same_classes | ewc | 2 | 0.9635 | - | 0.9666 | - | 0.9635 |
| Selected CL | catA_same_classes | experience_replay | 1 | 0.9763 | - | 0.9780 | - | 0.9763 |
| Selected CL | catA_same_classes | experience_replay | 2 | 0.9835 | - | 0.9850 | - | 0.9835 |
| Selected CL | catA_same_classes | parameter_isolation | 1 | 0.9615 | - | 0.9648 | - | 0.9615 |
| Selected CL | catA_same_classes | parameter_isolation | 2 | 0.9633 | - | 0.9666 | - | 0.9633 |
| Baseline | catA_same_classes (normal) | naive_finetune | 1 | 0.9624 | - | 0.9648 | - | 0.9624 |
| Baseline | catA_same_classes (normal) | naive_finetune | 2 | 0.9798 | - | 0.9824 | - | 0.9798 |

## 6) Case 3 - Cat B (New Classes)

| Type | Source Set | Method | Cycle | Old F1 | New F1 | Old Acc | New Acc | Joint F1 (Old+New)/2 |
|---|---|---|---:|---:|---:|---:|---:|---:|
| Selected CL | catB_new_classes | cat_b_ewc_head_expand | 1 | 0.6875 | 0.3159 | 0.9464 | 0.9072 | 0.5017 |
| Selected CL | catB_new_classes | cat_b_ewc_head_expand | 2 | 0.6900 | 0.3156 | 0.9578 | 0.9052 | 0.5028 |
| Selected CL | catB_new_classes | cat_b_replay_head_expand | 1 | 0.6870 | 0.3265 | 0.9516 | 0.9632 | 0.5067 |
| Selected CL | catB_new_classes | cat_b_replay_head_expand | 2 | 0.6952 | 0.3272 | 0.9692 | 0.9652 | 0.5112 |
| Selected CL | catB_new_classes | cat_b_hybrid_replay_ewc | 1 | 0.6959 | 0.3197 | 0.9701 | 0.9265 | 0.5078 |
| Selected CL | catB_new_classes | cat_b_hybrid_replay_ewc | 2 | 0.6984 | 0.3204 | 0.9754 | 0.9304 | 0.5094 |
| Selected CL | catB_new_classes | cat_b_isolation_new_branch | 1 | 0.8040 | 0.0343 | 0.9675 | 0.0638 | 0.4192 |
| Selected CL | catB_new_classes | cat_b_isolation_new_branch | 2 | 0.8026 | 0.1083 | 0.9604 | 0.2437 | 0.4554 |
| Baseline | catB_new_classes (normal) | cat_b_naive_full_retrain_head_expand | 1 | 0.6765 | 0.3942 | 0.9384 | 0.9749 | 0.5353 |
| Baseline | catB_new_classes (normal) | cat_b_naive_full_retrain_head_expand | 2 | 0.6820 | 0.3265 | 0.9472 | 0.9632 | 0.5043 |

---

## Notes

- For Cat A (same classes), `New F1/New Acc` are not applicable and shown as `-`.
- Joint F1 for Cat A equals Old F1.
- Case 2 Cat A EWC line uses the selected `lambda1000` track.

---

## 7) Full Analysis

### A) Cat A (Same Classes) - Which Case Performs Best?

Using Cycle 2 (final cycle) as the main comparison:

- **Case 2 is strongest overall in Cat A.**
  - Best single result in all Cat A tables is:
    - `Case 2 + naive_finetune (cycle2)`: **Old F1 = 0.9908**, **Old Acc = 0.9921**
  - Strong CL alternatives in Case 2:
    - `experience_replay (cycle2)`: Old F1 = 0.9827
    - `ewc (cycle2)`: Old F1 = 0.9798

- **Case 3 is second-best in Cat A** (especially replay):
  - `experience_replay (cycle2)`: Old F1 = 0.9835
  - Better than Case 1 replay/ewc in Cat A.

- **Case 1 is the weakest among the three for Cat A** in final metrics (as expected from scratch base).

### B) Cat A - Which CL Method is Best?

- If baseline is allowed:
  - `naive_finetune` is top in Case 2 (0.9908) and very strong in Case 3 (0.9798).
- Among strict CL methods:
  - **Experience Replay** is the most consistently strong across cases.
  - `ewc` is good but usually slightly below replay in final cycle.
  - `parameter_isolation` is stable but generally lower than replay/ewc.

### C) Cat B (New Classes) - Which Case Performs Best?

Use Cycle 2 and Joint F1 (`(Old F1 + New F1)/2`) to measure plasticity-stability balance.

- **Best overall Cat B result in this report:**
  - `Case 2 + naive_full_retrain (cycle2)` -> **Joint F1 = 0.5264**
- Best strict CL result:
  - `Case 2 + hybrid (cycle2)` -> **Joint F1 = 0.5152**
  - closely followed by `Case 2 + replay (cycle2)` -> **0.5143**

Case-level view (best cycle2 joint):
- **Case 2** best (~0.5264 baseline / ~0.5152 CL)
- **Case 3** next (~0.5112 replay CL, 0.5353 baseline at cycle1 but cycle2 baseline 0.5043)
- **Case 1** lowest (~0.4885 hybrid CL, 0.4873 baseline cycle2)

### D) Cat B - Which CL Method is Best?

Across three cases, CL-method trend is:

- **Replay / Hybrid are the strongest practical CL choices** for Cat B.
  - They give the best balance of old retention + new acquisition.
- **EWC is competitive but usually below replay/hybrid** on joint outcome.
- **Isolation gives very strong old retention but weak new learning** (especially Case 3), so joint score drops.

### E) Stability vs Plasticity Interpretation

- **Replay/Hybrid**: best tradeoff (good old F1 and good new F1).
- **EWC**: good retention-oriented method, moderate plasticity.
- **Isolation**: retention-heavy, low plasticity in class-incremental setup without task-id routing.
- **Naive baseline**: often highest plasticity, but can risk forgetting depending on case/data.

### F) Practical Final Recommendation for Journal

1. **Cat A final method recommendation:** `experience_replay` (most consistent CL winner), with `ewc` as second.
2. **Cat B final method recommendation:** `hybrid` or `replay` as primary CL methods.
3. **Case ranking for continuation performance:** Case 2 > Case 3 > Case 1 (overall trend in final-cycle metrics).
4. Keep `naive` as baseline comparator (important reference), but frame CL conclusions using replay/hybrid/ewc.
