# Cat B Finalized Comparison Baseline (Case 1)

## Purpose
This file is the fixed reference for comparing future Cat B runs (Case 2/Case 3/new tuning) against the current best selected methods and baseline.

## Selected Reference Results

| Type | Source Set | Method | Cycle | Old F1 | New F1 | Old Acc | New Acc | Joint F1 `(Old+New)/2` |
|---|---|---|---:|---:|---:|---:|---:|---:|
| Selected CL | `catB_new_classes` (normal) | `cat_b_ewc_head_expand` | 1 | 0.6263 | 0.2724 | 0.8074 | 0.9246 | 0.4494 |
| Selected CL | `catB_new_classes` (normal) | `cat_b_ewc_head_expand` | 2 | 0.6375 | 0.3163 | 0.8408 | 0.9149 | 0.4769 |
| Selected CL | `catB_new_classes tuned` | `cat_b_replay_head_expand` | 1 | 0.6227 | 0.3238 | 0.8135 | 0.9555 | 0.4732 |
| Selected CL | `catB_new_classes tuned` | `cat_b_replay_head_expand` | 2 | 0.6394 | 0.3229 | 0.8443 | 0.9478 | 0.4812 |
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
- `experiments/part1_case1_scratch/Continuation/catB_new_classes/summary.md`
- `experiments/part1_case1_scratch/Continuation/catB_new_classes tuned/summary.md`
