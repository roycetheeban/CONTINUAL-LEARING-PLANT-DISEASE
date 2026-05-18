# Cat B New_03 Summary (Case 1)

Generated: 2026-05-17

## Experiment Results

| Method | Cycle | Old F1 | New F1 | Old Acc | New Acc | Joint F1 `(Old+New)/2` | Epochs Run |
|---|---:|---:|---:|---:|---:|---:|---:|
| cat_b_ewc_head_expand | cycle1 | 0.6351 | 0.2696 | 0.8267 | 0.9014 | 0.4523 | 5 |
| cat_b_ewc_head_expand | cycle2 | 0.6445 | 0.3156 | 0.8575 | 0.9110 | 0.4800 | 12 |
| cat_b_replay_head_expand | cycle1 | 0.6381 | 0.3191 | 0.8382 | 0.9323 | 0.4786 | 9 |
| cat_b_replay_head_expand | cycle2 | 0.6424 | 0.3193 | 0.8478 | 0.9246 | 0.4809 | 12 |
| cat_b_hybrid_replay_ewc | cycle1 | 0.6347 | 0.3187 | 0.8329 | 0.9246 | 0.4767 | 10 |
| cat_b_hybrid_replay_ewc | cycle2 | 0.6423 | 0.3183 | 0.8478 | 0.9207 | 0.4803 | 12 |

## Best in New_03

- Best Cycle 2 joint F1: `Replay` (`0.4809`)
- Very close second: `Hybrid` (`0.4803`)
- EWC improved old retention but lower new-class score vs Replay.

## Baseline Comparison

### Against Naive Full Retrain Baseline (Case 1 Cycle 2)
- Baseline: `Old F1=0.6539`, `New F1=0.3207`, `Joint=0.4873`
- Best New_03 (Replay c2): `Old F1=0.6424`, `New F1=0.3193`, `Joint=0.4809`

Result:
- New_03 is **closer** but still **below** naive full retrain baseline.

### Against Cat B Finalized Best CL (Replay c2)
- Finalized Replay c2 joint: `0.4812`
- New_03 Replay c2 joint: `0.4809`

Result:
- New_03 is nearly tied, but slightly below finalized best CL.

## Output Files Used

- `ewc_head_expand/outputs/cycle1/metrics/metrics.json`
- `ewc_head_expand/outputs/cycle2/metrics/metrics.json`
- `experience_replay_head_expand/outputs/cycle1/metrics/metrics.json`
- `experience_replay_head_expand/outputs/cycle2/metrics/metrics.json`
- `hybrid_ewc_replay_head_expand/outputs/cycle1/metrics/metrics.json`
- `hybrid_ewc_replay_head_expand/outputs/cycle2/metrics/metrics.json`
