# Cat A Same-Class - Parameter Isolation Run Plan

## Cycle 1
python experiments/part1_case2_imagenet_finetune/Continuation/catA_same_classes/parameter_isolation/code/train_catA_isolation.py --config experiments/part1_case2_imagenet_finetune/Continuation/catA_same_classes/parameter_isolation/configs/catA_isolation_cycle1.yaml

## Cycle 2
python experiments/part1_case2_imagenet_finetune/Continuation/catA_same_classes/parameter_isolation/code/train_catA_isolation.py --config experiments/part1_case2_imagenet_finetune/Continuation/catA_same_classes/parameter_isolation/configs/catA_isolation_cycle2.yaml

## Outputs per cycle
- checkpoints/model_expanded.pth
- logs/train_log.csv
- metrics/metrics.json
- metrics/classification_report.csv
- metrics/frozen_mask.pkl
- metrics/model_size_mb.txt
- figures/train_val_curves.png
- figures/confusion_matrix.png

## Isolation Policy
- All old parameters are frozen.
- Cycle 1 trains adapter C1 only (`lr=1e-3`).
- Cycle 2 stacks adapter C2 and trains C2 only (`lr=5e-4`).
- Classifier head remains frozen in Cat A isolation.

