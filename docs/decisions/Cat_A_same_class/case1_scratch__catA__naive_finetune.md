# Part 1 - Case 1 (Scratch) - CatA Naive Finetune Summary

Generated: 2026-05-10 16:49:40

## Base Training Reference
- Base checkpoint: `experiments/part1_case1_scratch/outputs/case1_scratch_weighted/checkpoints/model.pth`
- Base test accuracy: **0.9745**
- Base test macro-F1: **0.9712**

## Cycle Performance
| Stage | Test Acc | Test Macro-F1 | Delta Acc vs Base | Delta F1 vs Base |
|---|---:|---:|---:|---:|
| Cycle 1 | 0.9763 | 0.9732 | 0.0018 | 0.0020 |
| Cycle 2 | 0.9754 | 0.9713 | 0.0009 | 0.0001 |

## Hyperparameters
### Cycle 1
- Batch size: `32`
- Max epochs: `15`
- Min epochs: `5`
- Patience: `5`
- LR G3: `0.0001`
- LR Head: `0.0005`
### Cycle 2
- Batch size: `32`
- Max epochs: `12`
- Min epochs: `5`
- Patience: `5`
- LR G3: `5e-05`
- LR Head: `0.0002`

## Resource and Runtime
| Stage | Train Wall Time (s) | Total Wall Time (s) | Peak VRAM (MB) | RAM End (MB) | Model Size (MB) |
|---|---:|---:|---:|---:|---:|
| Cycle 1 | 119.9967 | 126.1573 | 161.0200 | 1220.88 | 5.8097 |
| Cycle 2 | 93.9879 | 100.8325 | 162.1700 | 1229.8 | 5.8097 |

## Artifact Checklist