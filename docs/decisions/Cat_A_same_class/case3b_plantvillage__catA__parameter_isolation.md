# Part 1 - Case 3B (PlantVillage Finetune) - CatA Parameter Isolation Summary

Generated: 2026-05-10 16:49:40

## Base Training Reference
- Base checkpoint: `experiments/part1_case3B_plantvillage_finetune/outputs/checkpoints/model.pth`
- Base test accuracy: **0.9683**
- Base test macro-F1: **0.9652**

## Cycle Performance
| Stage | Test Acc | Test Macro-F1 | Delta Acc vs Base | Delta F1 vs Base |
|---|---:|---:|---:|---:|
| Cycle 1 | 0.9648 | 0.9615 | -0.0035 | -0.0036 |
| Cycle 2 | 0.9666 | 0.9633 | -0.0018 | -0.0018 |

## Hyperparameters
### Cycle 1
- Batch size: `32`
- Max epochs: `15`
- Min epochs: `5`
- Patience: `5`
- LR Adapter: `0.001`
### Cycle 2
- Batch size: `32`
- Max epochs: `12`
- Min epochs: `5`
- Patience: `5`
- LR Adapter: `0.0005`

## Resource and Runtime
| Stage | Train Wall Time (s) | Total Wall Time (s) | Peak VRAM (MB) | RAM End (MB) | Model Size (MB) |
|---|---:|---:|---:|---:|---:|
| Cycle 1 | 143.6844 | 149.5662 | 122.6300 | 1217.61 | 5.8277 |
| Cycle 2 | 76.3816 | 82.6552 | 122.1400 | 1202.17 | 5.8456 |

## Artifact Checklist
- Frozen mask saved: `Yes`
- Model size log saved: `Yes`