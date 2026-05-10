# Part 1 - Case 3B (PlantVillage Finetune) - CatA EWC Summary

Generated: 2026-05-10 16:49:40

## Base Training Reference
- Base checkpoint: `experiments/part1_case3B_plantvillage_finetune/outputs/checkpoints/model.pth`
- Base test accuracy: **0.9683**
- Base test macro-F1: **0.9652**

## Cycle Performance
| Stage | Test Acc | Test Macro-F1 | Delta Acc vs Base | Delta F1 vs Base |
|---|---:|---:|---:|---:|
| Cycle 1 | 0.9613 | 0.9566 | -0.0070 | -0.0086 |
| Cycle 2 | 0.9666 | 0.9635 | -0.0018 | -0.0017 |

## Hyperparameters
### Cycle 1
- Batch size: `32`
- Max epochs: `15`
- Min epochs: `5`
- Patience: `5`
- LR G3: `0.0001`
- LR Head: `0.0005`
- EWC lambda: `1000.0`
- Fisher max batches: `100`
### Cycle 2
- Batch size: `32`
- Max epochs: `12`
- Min epochs: `5`
- Patience: `5`
- LR G3: `5e-05`
- LR Head: `0.0002`
- EWC lambda: `1000.0`
- Fisher max batches: `100`

## Resource and Runtime
| Stage | Train Wall Time (s) | Total Wall Time (s) | Peak VRAM (MB) | RAM End (MB) | Model Size (MB) |
|---|---:|---:|---:|---:|---:|
| Cycle 1 | 99.1871 | 110.4102 | 169.5100 | 1291.16 | 5.8097 |
| Cycle 2 | 85.9635 | 96.2839 | 169.5100 | 1297.55 | 5.8097 |

## Artifact Checklist
- Fisher matrix saved: `Yes`
- Theta star saved: `Yes`