# Case 2 Output Set Selection for Next Continual Learning Stage

## Compared Runs
1. `case2_config_init.yaml` -> `experiments/part1_case2_imagenet_finetune/outputs`
2. `case2_tuned_config.yaml` -> `experiments/part1_case2_imagenet_finetune/outputs_tuned`

## Selection Rule for CL Start
Primary criterion: **highest test accuracy and test macro-F1**.
Timing is recorded as a secondary efficiency reference, not the deciding factor for CL start checkpoint selection.

## Key Comparison

| Metric | outputs (init cfg) | outputs_tuned (tuned cfg) | Better |
|---|---:|---:|---|
| Test Accuracy | 0.9719 | 0.9727 | tuned |
| Test Macro-F1 | 0.9685 | 0.9701 | tuned |
| Best Val Accuracy (Phase 2) | 0.9565 | 0.9551 | init |
| Best Val Macro-F1 (Phase 2) | 0.9472 | 0.9456 | init |
| Total Train Time (sec) | 593.8 | 983.8 | init |
| Phase 2 Time (sec) | 385.6 | 754.1 | init |
| Peak VRAM (MB) | 161.0 | 161.0 | tie |
| Model Size (MB) | 5.8097 | 5.8097 | tie |

## Decision
Use **`outputs_tuned`** as the main starting output set for the next continual learning stage, because it has the highest test accuracy and macro-F1.

## Selected Checkpoint for Next Stage
- `experiments/part1_case2_imagenet_finetune/outputs_tuned/checkpoints/M2.pth`

## Notes
- `outputs` remains a useful faster-run reference for efficiency analysis.
- Continue using identical test set for all CL methods to keep comparisons fair.
