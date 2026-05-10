# Part 1 - Case 3B (PlantVillage Finetune) - CatA Naive Finetune Summary

Generated: 2026-05-10 16:49:40

## Base Training Reference
- Base checkpoint: `experiments/part1_case3B_plantvillage_finetune/outputs/checkpoints/model.pth`
- Base test accuracy: **0.9683**
- Base test macro-F1: **0.9652**

## Cycle Performance
| Stage | Test Acc | Test Macro-F1 | Delta Acc vs Base | Delta F1 vs Base |
|---|---:|---:|---:|---:|
| Cycle 1 | 0.9648 | 0.9624 | -0.0035 | -0.0027 |
| Cycle 2 | 0.9824 | 0.9798 | 0.0141 | 0.0147 |

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
| Cycle 1 | 135.6634 | 142.0895 | 161.0200 | 1140.88 | 5.8097 |
| Cycle 2 | 161.1195 | 167.6975 | 162.1700 | 1230.16 | 5.8097 |

## Artifact Checklist