# Cat A Experience Replay Conclusion (M2 Tuned Start)

## Scope
- Method: Experience Replay
- Task: Cat A (same 5 classes, new data stream)
- Runs:
  - Cycle 1: `outputs/catA_replay_cycle1`
  - Cycle 2: `outputs/catA_replay_cycle2`
- Replay policy: immutable replay folder + manifest-only logical updates

## Final Results

| Cycle | Test Accuracy | Test Macro-F1 | Val Macro-F1 | Best Epoch |
|---|---:|---:|---:|---:|
| Cycle 1 | 0.9807 | 0.9781 | 0.9390 | 7 |
| Cycle 2 | 0.9842 | 0.9827 | 0.9484 | 7 |

## Resource Summary

| Cycle | Total Time (s) | Train Time (s) | Peak VRAM (MB) | End RAM (MB) | Model Size (MB) |
|---|---:|---:|---:|---:|---:|
| Cycle 1 | 310.5116 | 303.3799 | 155.58 | 1252.43 | 5.8097 |
| Cycle 2 | 311.2740 | 304.6246 | 155.58 | 1177.77 | 5.8097 |

- Parameters (both cycles): `1,522,981`
- No architecture growth across cycles.

## Replay Manifest Summary

| Cycle | Base Replay Count | Prior Added Count | Newly Added Count | Effective Old Count |
|---|---:|---:|---:|---:|
| Cycle 1 | 526 | 0 | 200 | 726 |
| Cycle 2 | 526 | 200 | 200 | 926 |

Notes:
- `replay_folder_immutable: true` in both manifests.
- Replay updates were applied by manifest only; no mutation of `data/02_tomato_5cls/replay_buffer/`.
- Fixed batch composition was used: `old=16`, `new=16`, total `32`.

## Key Insights
1. Cycle 2 improved over Cycle 1 on the same fixed test set:
   - Accuracy: `+0.0035` (0.9807 -> 0.9842)
   - Macro-F1: `+0.0046` (0.9781 -> 0.9827)
2. Replay remained stable in resource usage across cycles (similar runtime and identical peak VRAM).
3. Manifest-only replay expansion increased effective old memory (`726 -> 926`) and preserved reproducibility/auditability.

## Stored Artifacts (Confirmed Per Cycle)
- `checkpoints/model.pth`
- `logs/train_log.csv`
- `metrics/metrics.json`
- `metrics/classification_report.csv`
- `metrics/replay_buffer_manifest.json`
- `metrics/buffer_image_paths.txt`
- `figures/train_val_curves.png`
- `figures/confusion_matrix.png`
