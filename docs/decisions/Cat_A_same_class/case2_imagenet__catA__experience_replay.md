# Part 1 - Case 2 (ImageNet Finetune) - CatA Experience Replay Summary

Generated: 2026-05-10 16:49:40

## Base Training Reference
- Base checkpoint: `experiments/part1_case2_imagenet_finetune/outputs_tuned/checkpoints/M2.pth`
- Base test accuracy: **0.9727**
- Base test macro-F1: **0.9701**

## Cycle Performance
| Stage | Test Acc | Test Macro-F1 | Delta Acc vs Base | Delta F1 vs Base |
|---|---:|---:|---:|---:|
| Cycle 1 | 0.9807 | 0.9781 | 0.0079 | 0.0079 |
| Cycle 2 | 0.9842 | 0.9827 | 0.0114 | 0.0125 |

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
| Cycle 1 | 303.3799 | 310.5116 | 155.5800 | 1252.43 | 5.8097 |
| Cycle 2 | 304.6246 | 311.2740 | 155.5800 | 1177.77 | 5.8097 |

## Artifact Checklist
- Replay manifest saved: `Yes`
- Buffer image list saved: `Yes`