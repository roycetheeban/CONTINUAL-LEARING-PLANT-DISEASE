# Cat A (Same Classes) - EWC Lambda Robustness Sweep (M2 tuned start)

## Cycle 1 (base checkpoint = M2 tuned)
python experiments/part1_case2_imagenet_finetune/Continuation/catA_same_classes/ewc/code/train_catA_ewc.py --config experiments/part1_case2_imagenet_finetune/Continuation/catA_same_classes/ewc/configs/catA_ewc_cycle1_lambda300.yaml
python experiments/part1_case2_imagenet_finetune/Continuation/catA_same_classes/ewc/code/train_catA_ewc.py --config experiments/part1_case2_imagenet_finetune/Continuation/catA_same_classes/ewc/configs/catA_ewc_cycle1_lambda1000.yaml
python experiments/part1_case2_imagenet_finetune/Continuation/catA_same_classes/ewc/code/train_catA_ewc.py --config experiments/part1_case2_imagenet_finetune/Continuation/catA_same_classes/ewc/configs/catA_ewc_cycle1_lambda3000.yaml

## Cycle 2 (each lambda continues from its own cycle1 result)
python experiments/part1_case2_imagenet_finetune/Continuation/catA_same_classes/ewc/code/train_catA_ewc.py --config experiments/part1_case2_imagenet_finetune/Continuation/catA_same_classes/ewc/configs/catA_ewc_cycle2_lambda300.yaml
python experiments/part1_case2_imagenet_finetune/Continuation/catA_same_classes/ewc/code/train_catA_ewc.py --config experiments/part1_case2_imagenet_finetune/Continuation/catA_same_classes/ewc/configs/catA_ewc_cycle2_lambda1000.yaml
python experiments/part1_case2_imagenet_finetune/Continuation/catA_same_classes/ewc/code/train_catA_ewc.py --config experiments/part1_case2_imagenet_finetune/Continuation/catA_same_classes/ewc/configs/catA_ewc_cycle2_lambda3000.yaml

## Output roots (auto-created)
- runs/cycle1_lambda300
- runs/cycle1_lambda1000
- runs/cycle1_lambda3000
- runs/cycle2_lambda300
- runs/cycle2_lambda1000
- runs/cycle2_lambda3000

## Stored per run
- checkpoints/model.pth
- checkpoints/theta_star.pt
- metrics/fisher_matrix.pt
- metrics/metrics.json
- logs/train_log.csv
