# Cat A Same-Class - Naive Fine-tuning Run Plan

## Cycle 1
python experiments/part1_case2_imagenet_finetune/Continuation/catA_same_classes/naive_finetune/code/train_catA_naive.py --config experiments/part1_case2_imagenet_finetune/Continuation/catA_same_classes/naive_finetune/configs/catA_naive_cycle1.yaml

## Cycle 2
python experiments/part1_case2_imagenet_finetune/Continuation/catA_same_classes/naive_finetune/code/train_catA_naive.py --config experiments/part1_case2_imagenet_finetune/Continuation/catA_same_classes/naive_finetune/configs/catA_naive_cycle2.yaml

## Outputs per cycle
- checkpoints/model.pth
- logs/train_log.csv
- metrics/metrics.json
- metrics/classification_report.csv
- figures/train_val_curves.png
- figures/confusion_matrix.png

## Notes
- Same data splits as EWC for fair comparison.
- Same augmentation pipeline as EWC.
- Same layer strategy: freeze G1+G2, train G3+head.
- `metrics.json` now includes runtime/resource fields:
  - `runtime.total_wall_time_sec`
  - `runtime.train_wall_time_sec`
  - `runtime.peak_vram_mb`
  - `runtime.process_ram_mb_end`
  - `model_footprint.num_parameters`
  - `model_footprint.model_size_mb`
- `metrics.json` also includes per-class test metrics:
  - `test_per_class_precision`
  - `test_per_class_recall`
  - `test_per_class_f1`
