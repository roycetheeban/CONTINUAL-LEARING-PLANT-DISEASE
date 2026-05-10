# Part 1 - Case 3B (PlantVillage Finetune) - CatA Experience Replay Summary

Generated: 2026-05-10 16:49:40

## Base Training Reference
- Base checkpoint: `experiments/part1_case3B_plantvillage_finetune/outputs/checkpoints/model.pth`
- Base test accuracy: **0.9683**
- Base test macro-F1: **0.9652**

## Cycle Performance
| Stage | Test Acc | Test Macro-F1 | Delta Acc vs Base | Delta F1 vs Base |
|---|---:|---:|---:|---:|
| Cycle 1 | 0.9780 | 0.9763 | 0.0097 | 0.0111 |
| Cycle 2 | 0.9850 | 0.9835 | 0.0167 | 0.0184 |

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
| Cycle 1 | 271.8653 | 278.7440 | 155.5800 | 1253.94 | 5.8097 |
| Cycle 2 | 197.4812 | 204.0052 | 155.5800 | 1253.34 | 5.8097 |

## Artifact Checklist
- Replay manifest saved: `Yes`
- Buffer image list saved: `Yes`