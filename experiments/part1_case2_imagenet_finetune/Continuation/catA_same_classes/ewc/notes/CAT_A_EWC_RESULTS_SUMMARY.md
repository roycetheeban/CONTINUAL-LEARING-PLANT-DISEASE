# Cat A Same-Class EWC Results Summary (M2 Tuned Start)

## Baseline (Before Continual Learning)
- Initial model: `experiments/part1_case2_imagenet_finetune/outputs_tuned/checkpoints/M2.pth`
- Best validation (Phase 2): accuracy=0.9459, macro_f1=0.9357
- Test before CL: accuracy=0.9692, macro_f1=0.9660

## Major Hyperparameters (Cat A EWC)
- Model: MobileNetV3-Small (5 classes)
- Trainable layers: `features[9:]` (G3) + `classifier` (G4)
- Frozen layers: `features[:9]` (G1 + G2)
- Optimizer: Adam
- Batch size: 32
- Weight decay: 1e-4
- Scheduler: ReduceLROnPlateau (mode=max, factor=0.5, patience=3, min_lr=1e-6)
- EWC Fisher max batches: 100
- Early-stop forgetting guard: old val macro-F1 drop tolerance = 3%
- Data:
  - Cycle 1 train: `data/02_tomato_5cls/cl_cycle1_stream`
  - Cycle 2 train: `data/02_tomato_5cls/cl_cycle2_stream`
  - Fisher source: `data/02_tomato_5cls/replay_buffer`
  - Validation: `data/02_tomato_5cls/val`
  - Test: `data/02_tomato_5cls/test`

## Cycle 1 Results (Lambda Sweep)
| Lambda | Best Epoch | Fisher Batches | Val Acc | Val Macro-F1 | Test Acc | Test Macro-F1 | ? Test Acc vs Baseline | ? Test F1 vs Baseline |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 300 | 4 | 17 | 0.9551 | 0.9474 | 0.9754 | 0.9714 | +0.0062 | +0.0054 |
| 1000 | 4 | 17 | 0.9538 | 0.9451 | 0.9745 | 0.9704 | +0.0053 | +0.0043 |
| 3000 | 4 | 17 | 0.9565 | 0.9469 | 0.9754 | 0.9717 | +0.0062 | +0.0057 |

## Cycle 2 Results (Lambda Sweep)
| Lambda | Best Epoch | Fisher Batches | Val Acc | Val Macro-F1 | Test Acc | Test Macro-F1 | ? Test Acc vs Baseline | ? Test F1 vs Baseline |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 300 | 1 | 17 | 0.9604 | 0.9525 | 0.9789 | 0.9758 | +0.0097 | +0.0098 |
| 1000 | 1 | 17 | 0.9617 | 0.9537 | 0.9807 | 0.9782 | +0.0114 | +0.0122 |
| 3000 | 1 | 17 | 0.9617 | 0.9547 | 0.9798 | 0.9772 | +0.0106 | +0.0112 |

## Best After Cycle 2 (by Test Macro-F1)
- Lambda: 1000
- Test accuracy: 0.9807 (baseline 0.9692, change +0.0114)
- Test macro-F1: 0.9782 (baseline 0.9660, change +0.0122)

## Metrics File References
- `cycle1_lambda1000` (lambda=1000): `experiments/part1_case2_imagenet_finetune/Continuation/catA_same_classes/ewc/outputs/catA_ewc_cycle1_lambda1000/metrics/metrics.json`
- `cycle1_lambda300` (lambda=300): `experiments/part1_case2_imagenet_finetune/Continuation/catA_same_classes/ewc/outputs/catA_ewc_cycle1_lambda300/metrics/metrics.json`
- `cycle1_lambda3000` (lambda=3000): `experiments/part1_case2_imagenet_finetune/Continuation/catA_same_classes/ewc/outputs/catA_ewc_cycle1_lambda3000/metrics/metrics.json`
- `cycle2_lambda1000` (lambda=1000): `experiments/part1_case2_imagenet_finetune/Continuation/catA_same_classes/ewc/outputs/catA_ewc_cycle2_lambda1000/metrics/metrics.json`
- `cycle2_lambda300` (lambda=300): `experiments/part1_case2_imagenet_finetune/Continuation/catA_same_classes/ewc/outputs/catA_ewc_cycle2_lambda300/metrics/metrics.json`
- `cycle2_lambda3000` (lambda=3000): `experiments/part1_case2_imagenet_finetune/Continuation/catA_same_classes/ewc/outputs/catA_ewc_cycle2_lambda3000/metrics/metrics.json`
