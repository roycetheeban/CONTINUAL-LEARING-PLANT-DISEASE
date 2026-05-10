# Part 1 - Case 2 (ImageNet Finetune) - CatA Parameter Isolation Summary

Generated: 2026-05-10 16:49:40

## Base Training Reference
- Base checkpoint: `experiments/part1_case2_imagenet_finetune/outputs_tuned/checkpoints/M2.pth`
- Base test accuracy: **0.9727**
- Base test macro-F1: **0.9701**

## Cycle Performance
| Stage | Test Acc | Test Macro-F1 | Delta Acc vs Base | Delta F1 vs Base |
|---|---:|---:|---:|---:|
| Cycle 1 | 0.9719 | 0.9686 | -0.0009 | -0.0015 |
| Cycle 2 | 0.9727 | 0.9697 | 0.0000 | -0.0005 |

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
| Cycle 1 | 92.7013 | 101.3614 | 122.6300 | 1228.2 | 5.8277 |
| Cycle 2 | 178.1234 | 184.8460 | 122.1400 | 1232.38 | 5.8456 |

## Artifact Checklist
- Frozen mask saved: `Yes`
- Model size log saved: `Yes`