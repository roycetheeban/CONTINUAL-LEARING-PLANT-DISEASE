# Cat A Naive Fine-tuning Conclusion (M2 Tuned Start)

## Scope
- Method: Naive Fine-tuning (no EWC, no replay)
- Task: Cat A (same 5 classes, new data stream)
- Runs:
  - Cycle 1: `outputs/catA_naive_cycle1`
  - Cycle 2: `outputs/catA_naive_cycle2`

## Final Results

| Cycle | Test Accuracy | Test Macro-F1 | Val Macro-F1 | Best Epoch |
|---|---:|---:|---:|---:|
| Cycle 1 | 0.9763 | 0.9733 | 0.9531 | 6 |
| Cycle 2 | 0.9921 | 0.9908 | 0.9601 | 11 |

## Resource Summary

| Cycle | Total Time (s) | Train Time (s) | Peak VRAM (MB) | End RAM (MB) | Model Size (MB) |
|---|---:|---:|---:|---:|---:|
| Cycle 1 | 155.6882 | 149.1135 | 161.02 | 1221.38 | 5.8097 |
| Cycle 2 | 166.7872 | 160.4085 | 162.17 | 1229.56 | 5.8097 |

- Parameters (both cycles): `1,522,981`
- No model growth across cycles (same architecture size).

## Key Observations
1. Cycle 2 improved strongly over Cycle 1 on the fixed test set.
2. No catastrophic forgetting signal is visible in this Cat A run (same-class stream).
3. Runtime increased in Cycle 2 mainly because training continued to later best epoch (11 vs 6).

## Interpretation
- In this same-class setup, Naive FT can still perform very well because Cycle 2 data appears to be compatible with existing class decision boundaries.
- This does **not** generalize to new-class continual learning by itself; Category B is still needed to test true stability-plasticity stress.

## Stored Artifacts (Confirmed)
- `checkpoints/model.pth`
- `logs/train_log.csv`
- `metrics/metrics.json`
- `metrics/classification_report.csv`
- `figures/train_val_curves.png`
- `figures/confusion_matrix.png`
