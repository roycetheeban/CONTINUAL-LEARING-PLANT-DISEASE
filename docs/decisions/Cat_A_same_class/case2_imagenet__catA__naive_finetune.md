# Part 1 - Case 2 (ImageNet Finetune) - CatA Naive Finetune Summary

Generated: 2026-05-10 16:49:40

## Base Training Reference
- Base checkpoint: `experiments/part1_case2_imagenet_finetune/outputs_tuned/checkpoints/M2.pth`
- Base test accuracy: **0.9727**
- Base test macro-F1: **0.9701**

## Cycle Performance
| Stage | Test Acc | Test Macro-F1 | Delta Acc vs Base | Delta F1 vs Base |
|---|---:|---:|---:|---:|
| Cycle 1 | 0.9763 | 0.9733 | 0.0035 | 0.0031 |
| Cycle 2 | 0.9921 | 0.9908 | 0.0193 | 0.0207 |

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
| Cycle 1 | 149.1135 | 155.6882 | 161.0200 | 1221.38 | 5.8097 |
| Cycle 2 | 160.4085 | 166.7872 | 162.1700 | 1229.56 | 5.8097 |

## Artifact Checklist