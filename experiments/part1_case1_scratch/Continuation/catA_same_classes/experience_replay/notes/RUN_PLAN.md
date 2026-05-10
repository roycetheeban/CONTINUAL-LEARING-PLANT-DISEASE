# Cat A Same-Class - Experience Replay Run Plan

## Cycle 1
python experiments/part1_case2_imagenet_finetune/Continuation/catA_same_classes/experience_replay/code/train_catA_replay.py --config experiments/part1_case2_imagenet_finetune/Continuation/catA_same_classes/experience_replay/configs/catA_replay_cycle1.yaml

## Cycle 2
python experiments/part1_case2_imagenet_finetune/Continuation/catA_same_classes/experience_replay/code/train_catA_replay.py --config experiments/part1_case2_imagenet_finetune/Continuation/catA_same_classes/experience_replay/configs/catA_replay_cycle2.yaml

## Outputs per cycle
- checkpoints/model.pth
- logs/train_log.csv
- metrics/metrics.json
- metrics/classification_report.csv
- metrics/replay_buffer_manifest.json
- metrics/buffer_image_paths.txt
- figures/train_val_curves.png
- figures/confusion_matrix.png

## Replay Policy (Important)
- `data/02_tomato_5cls/replay_buffer/` is immutable (no file mutation).
- Replay update is logical only via manifest (`replay_buffer_manifest.json`).
- Batch composition is fixed: old=16, new=16, total=32.

