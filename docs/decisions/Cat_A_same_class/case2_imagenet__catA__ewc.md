# Part 1 - Case 2 (ImageNet Finetune) - CatA EWC Summary

Generated: 2026-05-10 16:49:40

## Base Training Reference
- Base checkpoint: `experiments/part1_case2_imagenet_finetune/outputs_tuned/checkpoints/M2.pth`
- Base test accuracy: **0.9727**
- Base test macro-F1: **0.9701**

## Cycle Performance
| Stage | Test Acc | Test Macro-F1 | Delta Acc vs Base | Delta F1 vs Base |
|---|---:|---:|---:|---:|
| Cycle 1 | 0.9771 | 0.9740 | 0.0044 | 0.0039 |
| Cycle 2 | 0.9815 | 0.9798 | 0.0088 | 0.0096 |

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
| Cycle 1 | 76.5618 | 87.8587 | 169.5100 | 1293.51 | 5.8097 |
| Cycle 2 | 87.2660 | 98.0080 | 169.5100 | 1279.38 | 5.8097 |

## Artifact Checklist
- Fisher matrix saved: `Yes`
- Theta star saved: `Yes`

### EWC Lambda Sweep (Detected)
| Run | Test Acc | Test Macro-F1 |
|---|---:|---:|
| `experiments/part1_case2_imagenet_finetune/Continuation/catA_same_classes/ewc/outputs/cycle1_lambda1000/metrics/metrics.json` | 0.9771 | 0.9740 |
| `experiments/part1_case2_imagenet_finetune/Continuation/catA_same_classes/ewc/outputs/cycle1_lambda10000/metrics/metrics.json` | 0.9798 | 0.9779 |
| `experiments/part1_case2_imagenet_finetune/Continuation/catA_same_classes/ewc/outputs/cycle1_lambda15000/metrics/metrics.json` | 0.9789 | 0.9773 |
| `experiments/part1_case2_imagenet_finetune/Continuation/catA_same_classes/ewc/outputs/cycle1_lambda300/metrics/metrics.json` | 0.9763 | 0.9731 |
| `experiments/part1_case2_imagenet_finetune/Continuation/catA_same_classes/ewc/outputs/cycle1_lambda3000/metrics/metrics.json` | 0.9780 | 0.9756 |
| `experiments/part1_case2_imagenet_finetune/Continuation/catA_same_classes/ewc/outputs/cycle1_lambda5000/metrics/metrics.json` | 0.9763 | 0.9736 |
| `experiments/part1_case2_imagenet_finetune/Continuation/catA_same_classes/ewc/outputs/cycle1_lambda8000/metrics/metrics.json` | 0.9789 | 0.9769 |
| `experiments/part1_case2_imagenet_finetune/Continuation/catA_same_classes/ewc/outputs/cycle2_lambda1000/metrics/metrics.json` | 0.9815 | 0.9798 |
| `experiments/part1_case2_imagenet_finetune/Continuation/catA_same_classes/ewc/outputs/cycle2_lambda10000/metrics/metrics.json` | 0.9824 | 0.9807 |
| `experiments/part1_case2_imagenet_finetune/Continuation/catA_same_classes/ewc/outputs/cycle2_lambda15000/metrics/metrics.json` | 0.9815 | 0.9797 |
| `experiments/part1_case2_imagenet_finetune/Continuation/catA_same_classes/ewc/outputs/cycle2_lambda300/metrics/metrics.json` | 0.9807 | 0.9790 |
| `experiments/part1_case2_imagenet_finetune/Continuation/catA_same_classes/ewc/outputs/cycle2_lambda3000/metrics/metrics.json` | 0.9824 | 0.9807 |
| `experiments/part1_case2_imagenet_finetune/Continuation/catA_same_classes/ewc/outputs/cycle2_lambda5000/metrics/metrics.json` | 0.9824 | 0.9804 |
| `experiments/part1_case2_imagenet_finetune/Continuation/catA_same_classes/ewc/outputs/cycle2_lambda8000/metrics/metrics.json` | 0.9807 | 0.9787 |

Selected for this summary: `experiments/part1_case2_imagenet_finetune/Continuation/catA_same_classes/ewc/outputs/cycle1_lambda1000/metrics/metrics.json` and `experiments/part1_case2_imagenet_finetune/Continuation/catA_same_classes/ewc/outputs/cycle2_lambda1000/metrics/metrics.json`