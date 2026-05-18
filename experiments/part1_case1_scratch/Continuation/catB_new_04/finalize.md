# Cat B Finalized Comparison Baseline (Case 1)

## Purpose
This file is the fixed reference for comparing future Cat B runs (Case 2/Case 3/new tuning) against the current best selected methods and baseline.

## Selected Reference Results (Finalized from new01/new02/new03 analysis)

| Type | Source Set | Method | Cycle | Old F1 | New F1 | Old Acc | New Acc | Joint F1 `(Old+New)/2` |
|---|---|---|---:|---:|---:|---:|---:|---:|
| Selected CL | `catB_new_03` | `cat_b_ewc_head_expand` | 1 | 0.6351 | 0.2696 | 0.8267 | 0.9014 | 0.4523 |
| Selected CL | `catB_new_03` | `cat_b_ewc_head_expand` | 2 | 0.6445 | 0.3156 | 0.8575 | 0.9110 | 0.4800 |
| Selected CL | `catB_new_03` | `cat_b_replay_head_expand` | 1 | 0.6381 | 0.3191 | 0.8382 | 0.9323 | 0.4786 |
| Selected CL | `catB_new_03` | `cat_b_replay_head_expand` | 2 | 0.6424 | 0.3193 | 0.8478 | 0.9246 | 0.4809 |
| Selected CL | `catB_new_03` | `cat_b_hybrid_replay_ewc` | 1 | 0.6347 | 0.3187 | 0.8329 | 0.9246 | 0.4767 |
| Selected CL | `catB_new_03` | `cat_b_hybrid_replay_ewc` | 2 | 0.6423 | 0.3183 | 0.8478 | 0.9207 | 0.4803 |
| Baseline | `catB_new_classes` (normal) | `cat_b_naive_full_retrain_head_expand` | 1 | 0.6450 | 0.3228 | 0.8522 | 0.9439 | 0.4839 |
| Baseline | `catB_new_classes` (normal) | `cat_b_naive_full_retrain_head_expand` | 2 | 0.6539 | 0.3207 | 0.8707 | 0.9342 | 0.4873 |

## How to Use This File
1. After each new run, add a new row in a "New Runs" section.
2. Compare directly against the baseline row (`naive_full_retrain`, cycle 2).
3. Promotion target:
- `Old F1 > 0.6539`
- `New F1 >= 0.3207`
- Prefer higher joint F1 than `0.4873`

## Source Summary Files
- `experiments/part1_case1_scratch/Continuation/catB_new_01/summary.md`
- `experiments/part1_case1_scratch/Continuation/catB_new_02/summary.md`
- `experiments/part1_case1_scratch/Continuation/catB_new_03/summary.md`
