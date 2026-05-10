# Part 1 - Case 1 (Scratch) - CatA Experience Replay Summary

Generated: 2026-05-10 16:49:40

## Base Training Reference
- Base checkpoint: `experiments/part1_case1_scratch/outputs/case1_scratch_weighted/checkpoints/model.pth`
- Base test accuracy: **0.9745**
- Base test macro-F1: **0.9712**

## Cycle Performance
| Stage | Test Acc | Test Macro-F1 | Delta Acc vs Base | Delta F1 vs Base |
|---|---:|---:|---:|---:|
| Cycle 1 | 0.9745 | 0.9703 | 0.0000 | -0.0008 |
| Cycle 2 | 0.9719 | 0.9666 | -0.0026 | -0.0045 |

## Hyperparameters
### Cycle 1
- Batch size: `32`
- Max epochs: `15`
- Min epochs: `5`
- Patience: `5`
- LR G3: `0.0001`
- LR Head: `0.0005`
- Replay old/new per batch: `16/16`
- Replay add per class: `40`
### Cycle 2
- Batch size: `32`
- Max epochs: `12`
- Min epochs: `5`
- Patience: `5`
- LR G3: `5e-05`
- LR Head: `0.0002`
- Replay old/new per batch: `16/16`
- Replay add per class: `40`

## Resource and Runtime
| Stage | Train Wall Time (s) | Total Wall Time (s) | Peak VRAM (MB) | RAM End (MB) | Model Size (MB) |
|---|---:|---:|---:|---:|---:|
| Cycle 1 | 273.4441 | 280.5577 | 155.5800 | 1249.0 | 5.8097 |
| Cycle 2 | 299.8844 | 306.9942 | 155.5800 | 1243.23 | 5.8097 |

## Artifact Checklist
- Replay manifest saved: `Yes`
- Buffer image list saved: `Yes`