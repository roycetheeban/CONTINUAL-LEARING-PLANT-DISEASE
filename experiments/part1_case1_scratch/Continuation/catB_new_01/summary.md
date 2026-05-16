# Cat B New_01 Summary (Case 1)

Generated: 2026-05-16

## Experiment Results

| Method | Cycle | Old F1 | New F1 | Old Acc | New Acc | Forgetting Score | Peak VRAM MB | Joint F1 `(Old+New)/2` |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| cat_b_ewc_head_expand | cycle1 | 0.6260 | 0.2721 | 0.8065 | 0.9246 | 0.3327 | 335.29 | 0.4490 |
| cat_b_ewc_head_expand | cycle2 | 0.6379 | 0.3167 | 0.8417 | 0.9168 | -0.0288 | 335.29 | 0.4773 |
| cat_b_replay_head_expand | cycle1 | 0.6257 | 0.3195 | 0.8039 | 0.9342 | NA | 325.39 | 0.4726 |
| cat_b_replay_head_expand | cycle2 | 0.6424 | 0.3162 | 0.8514 | 0.9110 | NA | 325.39 | 0.4793 |

## Baseline Reference (Naive Full Retrain, Cycle 2)

| Baseline Method | Old F1 | New F1 | Old Acc | New Acc | Joint F1 |
|---|---:|---:|---:|---:|---:|
| cat_b_naive_full_retrain_head_expand | 0.6539 | 0.3207 | 0.8707 | 0.9342 | 0.4873 |

## Quick Comparison

- Best in this run set: `Replay cycle2` (Joint F1 `0.4793`).
- Still below baseline joint F1 (`0.4873`), mainly due to lower old-class retention.
- EWC improved from cycle1 -> cycle2 and gave negative forgetting score in cycle2 (old-class recovery vs cycle start), but still below baseline.

## Output Metric Sources

- `ewc_head_expand/outputs/cycle1/metrics/metrics.json`
- `ewc_head_expand/outputs/cycle2/metrics/metrics.json`
- `experience_replay_head_expand/outputs/cycle1/metrics/metrics.json`
- `experience_replay_head_expand/outputs/cycle2/metrics/metrics.json`
