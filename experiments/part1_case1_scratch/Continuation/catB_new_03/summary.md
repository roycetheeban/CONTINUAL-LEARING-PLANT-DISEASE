# Cat B New_02 Summary (Case 1)

Generated: 2026-05-17

## Experiment Results

| Method | Cycle | Old F1 | New F1 | Old Acc | New Acc | Joint F1 `(Old+New)/2` |
|---|---:|---:|---:|---:|---:|---:|
| cat_b_ewc_head_expand | cycle1 | 0.6351 | 0.2696 | 0.8109 | 0.9168 | 0.4523 |
| cat_b_ewc_head_expand | cycle2 | 0.6445 | 0.3156 | 0.8549 | 0.9072 | 0.4800 |
| cat_b_replay_head_expand | cycle1 | 0.6381 | 0.3191 | 0.8382 | 0.9091 | 0.4786 |
| cat_b_replay_head_expand | cycle2 | 0.6424 | 0.3162 | 0.8514 | 0.9110 | 0.4793 |

## Baseline Reference (Finalized Target)

| Baseline | Cycle | Old F1 | New F1 | Old Acc | New Acc | Joint F1 |
|---|---:|---:|---:|---:|---:|---:|
| naive_full_retrain | 2 | 0.6539 | 0.3207 | 0.8707 | 0.9342 | 0.4873 |

## Quick Comparison

- Best `new_02` run by joint F1: `EWC cycle2` (`0.4800`).
- `Replay cycle2` is very close (`0.4793`).
- Both are still below the finalized naive full-retrain baseline joint (`0.4873`).

## Output Files Used

- `ewc_head_expand/outputs/cycle1/metrics/metrics.json`
- `ewc_head_expand/outputs/cycle2/metrics/metrics.json`
- `experience_replay_head_expand/outputs/cycle1/metrics/metrics.json`
- `experience_replay_head_expand/outputs/cycle2/metrics/metrics.json`
